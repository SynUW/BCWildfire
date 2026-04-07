#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2-in-1：MOD11A1 / MYD11A1 掩膜 + MCD11A1 融合（保持原有掩膜标准，基于QC评分拼接）

输入影像波段顺序（6波段）：
  1: LST_Day_1km
  2: LST_Night_1km
  3: Emis_31
  4: Emis_32
  5: QC_Day
  6: QC_Night

掩膜（保持原有标准）：
  默认策略（conservative）：“只屏蔽特别低质量，保留中等质量”
  - Mandatory QA (bits 0-1): {2,3} -> mask
  - LST error (bits 6-7): >3K（值=3） -> mask；<=3K（0/1/2/3） -> keep（默认max=3）
  - Emissivity error (bits 4-5): >0.04（值=3） -> mask；<=0.04（0/1/2/3） -> keep（默认max=3）
  - Data quality (bits 2-3): 0=good, 1=other -> keep；2/3=TBD -> mask（可通过参数放宽）

融合：
  单侧有效→取该侧；两侧都无效→NoData；两侧有效→按QC评分择优：
  - LST_Day: 用 QC_Day 打分 (mandatory, lst_error, data_quality, emis_error)
  - LST_Night: 用 QC_Night 打分 (mandatory, lst_error, data_quality, emis_error)
  - Emis_31/32: 用 emissivity error 打分 (mandatory, emis_error, data_quality)
  平分时按 --prefer（默认 mod）。

NoData：统一使用 -32765（所有波段）。
"""

import re
import math
import shutil
import argparse
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import rasterio
from rasterio import windows
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, Resampling
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.simplefilter("ignore", NotGeoreferencedWarning)

# -------------------- 通用 --------------------
DATE_RE = re.compile(r"(?P<y>\d{4})[-_]?((?P<m>\d{2})[-_]?)(?P<d>\d{2})")
TILE_RE = re.compile(r"_tile(?P<tile>\d+)", re.IGNORECASE)

def parse_date_tile_key(p: Path) -> str | None:
    """解析文件名，返回 'YYYY_MM_DD_tileXX' 格式的键"""
    m_date = DATE_RE.search(p.stem)
    if not m_date: return None
    
    date_str = f"{m_date.group('y')}_{m_date.group('m')}_{m_date.group('d')}"
    
    # 尝试提取tile编号
    m_tile = TILE_RE.search(p.stem)
    if m_tile:
        tile_str = m_tile.group('tile')
        return f"{date_str}_tile{tile_str}"
    else:
        # 如果没有tile信息，返回日期（向后兼容）
        return date_str

def bitslice(arr, start, end):
    """Extract inclusive bit slice [start, end]."""
    width = end - start + 1
    mask = (1 << width) - 1
    return (arr >> start) & mask

def print_progress(prefix: str, ok: int, total: int, fail: int):
    print(f"\r{prefix}: OK={ok}/{total}, FAIL={fail}", end="", flush=True)

def finish_progress():
    print("", flush=True)

# -------------------- QC 字段解析 --------------------
def qc_fields(qc_arr):
    """
    解析 MOD11A1 QC_Day / QC_Night 位段为字典：
    0-1: mandatory
    2-3: data_quality
    4-5: emis_error
    6-7: lst_error
    """
    return dict(
        mandatory = bitslice(qc_arr, 0, 1),
        data_quality = bitslice(qc_arr, 2, 3),
        emis_error = bitslice(qc_arr, 4, 5),
        lst_error = bitslice(qc_arr, 6, 7),
    )

# -------------------- 掩膜判定（保持原有标准） --------------------
def build_bad_mask_from_qc(
    qc,
    mask_if_mandatory_23=True,
    mask_if_dq_tbd=True,
    dq_keep_1=True,
    max_keep_lst_error=3,     # 0..3 保留；>3 不存在，所以实际是全部保留
    max_keep_emis_error=3,    # 0..3 保留；>3 不存在，所以实际是全部保留
):
    """
    返回 boolean bad_mask：True 表示"质量特别低，需要屏蔽"。
    """
    q = qc_fields(qc)

    bad = np.zeros(qc.shape, dtype=bool)

    # 1) mandatory QA：2/3 -> 一定屏蔽（未生产）
    if mask_if_mandatory_23:
        bad |= np.isin(q["mandatory"], [2, 3])

    # 2) data quality：0=好，1=其它质量（保留），2/3=TBD -> 默认屏蔽
    if mask_if_dq_tbd:
        bad |= np.isin(q["data_quality"], [2, 3])
    if not dq_keep_1:
        # 如需更严格，可选择屏蔽 data_quality==1
        bad |= (q["data_quality"] == 1)

    # 3) LST error：>max_keep_lst_error -> 屏蔽；<=max_keep_lst_error -> 保留
    bad |= (q["lst_error"] > max_keep_lst_error)

    # 4) Emissivity error：>max_keep_emis_error -> 屏蔽；<=max_keep_emis_error -> 保留
    bad |= (q["emis_error"] > max_keep_emis_error)

    return bad

# -------------------- 评分系统（用于拼接） --------------------
def score_lst(qc_arr):
    """LST 评分：(mandatory, lst_error, data_quality, emis_error)"""
    q = qc_fields(qc_arr)
    return np.stack([
        q["mandatory"], q["lst_error"], q["data_quality"], q["emis_error"]
    ], axis=0)

def score_emis(qc_day_arr, qc_night_arr, mode="union"):
    """
    返回 emissivity 评分用的三元组数组 (mandatory, emis_error, data_quality)
    mode:
      - "union": 对每个像元，从 day/night 中挑更优（数值更小）的 mandatory、emis_error、data_quality
      - "day": 仅用 day
      - "night": 仅用 night
    """
    qd = qc_fields(qc_day_arr)
    qn = qc_fields(qc_night_arr)
    if mode == "day":
        mand = qd["mandatory"]; emis = qd["emis_error"]; dq = qd["data_quality"]
    elif mode == "night":
        mand = qn["mandatory"]; emis = qn["emis_error"]; dq = qn["data_quality"]
    else:
        mand = np.minimum(qd["mandatory"], qn["mandatory"])
        emis = np.minimum(qd["emis_error"], qn["emis_error"])
        dq   = np.minimum(qd["data_quality"], qn["data_quality"])
    return np.stack([mand, emis, dq], axis=0)

def lexicographic_choose(a_score, b_score, prefer_a=True):
    """
    a_score,b_score: shape=(K,H,W) 评分越小越好
    返回 bool mask: True 选择 A，False 选择 B；遇到完全相等按 prefer_a。
    """
    choose_a = np.zeros(a_score.shape[1:], dtype=bool)
    undecided = np.ones_like(choose_a, dtype=bool)
    for k in range(a_score.shape[0]):
        ak = a_score[k]; bk = b_score[k]
        better_a = ak < bk
        better_b = ak > bk
        choose_a |= (better_a & undecided)
        undecided &= ~(better_a | better_b)
        if not undecided.any():
            break
    if undecided.any():
        if prefer_a:
            choose_a |= undecided
    return choose_a

# -------------------- 几何对齐 --------------------
def pick_target_grid(mod_src, myd_src):
    """选择目标格网：CRS优先用 mod；分辨率取更细；范围取 union。"""
    crs = mod_src.crs or myd_src.crs
    if crs is None:
        raise ValueError("两个数据都缺少有效 CRS。")

    def res_of(src):
        a = src.transform
        return (abs(a.a), abs(a.e))
    mod_res = res_of(mod_src)
    myd_res = res_of(myd_src)
    xres = min(mod_res[0], myd_res[0])
    yres = min(mod_res[1], myd_res[1])

    mod_bounds = transform_bounds(mod_src.crs, crs, *mod_src.bounds, densify_pts=0) if mod_src.crs != crs else mod_src.bounds
    myd_bounds = transform_bounds(myd_src.crs, crs, *myd_src.bounds, densify_pts=0) if myd_src.crs != crs else myd_src.bounds
    xmin = min(mod_bounds[0], myd_bounds[0])
    ymin = min(mod_bounds[1], myd_bounds[1])
    xmax = max(mod_bounds[2], myd_bounds[2])
    ymax = max(mod_bounds[3], myd_bounds[3])

    from affine import Affine
    dst_transform = Affine(xres, 0.0, xmin, 0.0, -yres, ymax)
    width  = int(math.ceil((xmax - xmin) / xres))
    height = int(math.ceil((ymax - ymin) / yres))

    return crs, dst_transform, width, height

def ensure_valid_blocksize(b, max_size=1024, min_size=16):
    b = max(min(b, max_size), min_size)
    return (b // 16) * 16 or 16

# -------------------- 掩膜（单文件） --------------------
def process_single_file(
    path_in: Path,
    path_out: Path,
    out_nodata: float,
    overwrite: bool,
    allow_mandatory_23: bool,
    allow_dq_tbd: bool,
    mask_dq1: bool,
    max_keep_lst_error: int,
    max_keep_emis_error: int,
    emis_qc_mode_mask: str,  # 掩膜阶段的 emis_qc_mode
    compress: str,
):
    """处理单个文件的函数（用于并行处理）"""
    if path_out.exists() and not overwrite:
        return True, path_in.name, None

    try:
        with rasterio.open(path_in) as src:
            if src.count < 6:
                return False, path_in.name, f"需要 6 个波段（LST_Day, LST_Night, Emis_31, Emis_32, QC_Day, QC_Night）。"

            # 读取波段
            lst_day  = src.read(1)
            lst_nite = src.read(2)
            emis31   = src.read(3)
            emis32   = src.read(4)
            qc_day   = src.read(5)
            qc_nite  = src.read(6)

            # 基于 QC 生成"特别低质量"掩膜
            bad_day  = build_bad_mask_from_qc(
                qc_day,
                mask_if_mandatory_23=not allow_mandatory_23,
                mask_if_dq_tbd=not allow_dq_tbd,
                dq_keep_1=not mask_dq1,
                max_keep_lst_error=max_keep_lst_error,
                max_keep_emis_error=max_keep_emis_error,
            )
            bad_nite = build_bad_mask_from_qc(
                qc_nite,
                mask_if_mandatory_23=not allow_mandatory_23,
                mask_if_dq_tbd=not allow_dq_tbd,
                dq_keep_1=not mask_dq1,
                max_keep_lst_error=max_keep_lst_error,
                max_keep_emis_error=max_keep_emis_error,
            )

            # 3/4) Emis_31 / Emis_32：根据 emis_qc_mode_mask
            if emis_qc_mode_mask == "day":
                bad_emis = bad_day
            elif emis_qc_mode_mask == "night":
                bad_emis = bad_nite
            else:  # "union" (默认)：日/夜任一极差则屏蔽
                bad_emis = (bad_day | bad_nite)

            profile = src.profile.copy()
            bx = ensure_valid_blocksize(profile.get("blockxsize", 512))
            by = ensure_valid_blocksize(profile.get("blockysize", 512))
            profile.update(
                compress=compress,
                tiled=True,
                blockxsize=bx, blockysize=by,
                bigtiff="IF_SAFER",
            )

            # 统一使用 out_nodata
            path_out.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path_out, "w", **profile) as dst:
                # 1) LST_Day：按 bad_day 应用为 NoData
                arr = lst_day.astype(np.float32, copy=False)
                arr[bad_day] = out_nodata
                dst.write(arr, 1)
                dst.update_tags(1, nodata=str(int(out_nodata)))

                # 2) LST_Night：按 bad_nite
                arr = lst_nite.astype(np.float32, copy=False)
                arr[bad_nite] = out_nodata
                dst.write(arr, 2)
                dst.update_tags(2, nodata=str(int(out_nodata)))

                # 3) Emis_31
                arr = emis31.astype(np.float32, copy=False)
                arr[bad_emis] = out_nodata
                dst.write(arr, 3)
                dst.update_tags(3, nodata=str(int(out_nodata)))

                # 4) Emis_32
                arr = emis32.astype(np.float32, copy=False)
                arr[bad_emis] = out_nodata
                dst.write(arr, 4)
                dst.update_tags(4, nodata=str(int(out_nodata)))

                # 5/6) QA 原样写回
                dst.write(qc_day.astype(np.float32), 5)
                dst.write(qc_nite.astype(np.float32), 6)

        return True, path_in.name, None

    except Exception as e:
        return False, path_in.name, str(e)

# -------------------- 融合（单日期+tile） --------------------
def combine_one(
    date_tile_key: str,
    mod_path: Path | None,
    myd_path: Path | None,
    out_dir: Path,
    prefer_mod: bool,
    emis_qc_mode_combine: str,  # 拼接阶段的 emis_qc_mode
    chunk: int,
    compress: str,
    out_nodata: float,
    overwrite: bool,
):
    """融合单个日期+tile的MOD和MYD文件"""
    # 输出文件名：MCD11A1_YYYY_MM_DD_tileXX.tif
    out_filename = f"MCD11A1_{date_tile_key}.tif"
    out_path = out_dir / out_filename

    if out_path.exists() and not overwrite:
        return True, "skipped"

    # 单侧存在：直接复制
    if (mod_path is None) ^ (myd_path is None):
        shutil.copy2(mod_path or myd_path, out_path)
        return True, "copied"

    # 两侧都存在：融合
    with rasterio.open(mod_path) as mod, rasterio.open(myd_path) as myd:
        if mod.count < 6 or myd.count < 6:
            raise ValueError("输入需要 6 波段 (LST_Day, LST_Night, Emis_31, Emis_32, QC_Day, QC_Night)。")

        crs, dst_transform, width, height = pick_target_grid(mod, myd)

        # VRT：LST/Emis 用双线性，QC 用近邻
        vrt_opts_float = dict(crs=crs, transform=dst_transform, width=width, height=height,
                              resampling=Resampling.bilinear, add_alpha=False)
        vrt_opts_near  = dict(crs=crs, transform=dst_transform, width=width, height=height,
                              resampling=Resampling.nearest, add_alpha=False)

        # 统一输出 nodata
        out_nd = out_nodata

        bx = ensure_valid_blocksize(min(chunk, 1024))
        by = ensure_valid_blocksize(min(chunk, 1024))
        profile = dict(
            driver="GTiff", count=6, crs=crs, transform=dst_transform,
            width=width, height=height, dtype='float32',
            tiled=True, blockxsize=bx, blockysize=by,
            compress=compress, bigtiff="IF_SAFER",
        )

        with rasterio.open(out_path, "w", **profile) as dst, \
             WarpedVRT(mod, **vrt_opts_float) as mod_vrt_f, WarpedVRT(myd, **vrt_opts_float) as myd_vrt_f, \
             WarpedVRT(mod, **vrt_opts_near)  as mod_vrt_n, WarpedVRT(myd, **vrt_opts_near)  as myd_vrt_n:

            for row_off in range(0, height, chunk):
                h = min(chunk, height - row_off)
                for col_off in range(0, width, chunk):
                    w = min(chunk, width - col_off)
                    win = windows.Window(col_off, row_off, w, h)

                    # 读取数值波段（双线性）
                    m_day   = mod_vrt_f.read(1, window=win)
                    m_nite  = mod_vrt_f.read(2, window=win)
                    m_e31   = mod_vrt_f.read(3, window=win)
                    m_e32   = mod_vrt_f.read(4, window=win)

                    y_day   = myd_vrt_f.read(1, window=win)
                    y_nite  = myd_vrt_f.read(2, window=win)
                    y_e31   = myd_vrt_f.read(3, window=win)
                    y_e32   = myd_vrt_f.read(4, window=win)

                    # 读取 QC（近邻）
                    m_qd    = mod_vrt_n.read(5, window=win).astype(np.uint16, copy=False)
                    m_qn    = mod_vrt_n.read(6, window=win).astype(np.uint16, copy=False)
                    y_qd    = myd_vrt_n.read(5, window=win).astype(np.uint16, copy=False)
                    y_qn    = myd_vrt_n.read(6, window=win).astype(np.uint16, copy=False)

                    # 有效掩膜（基于统一 nodata）
                    mv_day  = (m_day != out_nd)
                    mv_nite = (m_nite != out_nd)
                    mv_e31  = (m_e31 != out_nd)
                    mv_e32  = (m_e32 != out_nd)

                    yv_day  = (y_day != out_nd)
                    yv_nite = (y_nite != out_nd)
                    yv_e31  = (y_e31 != out_nd)
                    yv_e32  = (y_e32 != out_nd)

                    # --- LST_Day 选择 ---
                    s_m_day = score_lst(m_qd)
                    s_y_day = score_lst(y_qd)
                    both = mv_day & yv_day
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_day[:, both], s_y_day[:, both], prefer_a=prefer_mod
                        )

                    out_day = np.full_like(m_day, out_nd)
                    only_m = mv_day & ~yv_day
                    only_y = ~mv_day & yv_day
                    out_day[only_m] = m_day[only_m]
                    out_day[only_y] = y_day[only_y]
                    aidx = both & choose_mod
                    bidx = both & ~choose_mod
                    out_day[aidx] = m_day[aidx]
                    out_day[bidx] = y_day[bidx]

                    # 对应 QC_Day：随 LST_Day 胜出者来源
                    out_qd = np.empty_like(m_qd)
                    out_qd[only_m] = m_qd[only_m]
                    out_qd[only_y] = y_qd[only_y]
                    out_qd[aidx]   = m_qd[aidx]
                    out_qd[bidx]   = y_qd[bidx]

                    # --- LST_Night 选择 ---
                    s_m_nite = score_lst(m_qn)
                    s_y_nite = score_lst(y_qn)
                    both = mv_nite & yv_nite
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_nite[:, both], s_y_nite[:, both], prefer_a=prefer_mod
                        )

                    out_nite = np.full_like(m_nite, out_nd)
                    only_m = mv_nite & ~yv_nite
                    only_y = ~mv_nite & yv_nite
                    out_nite[only_m] = m_nite[only_m]
                    out_nite[only_y] = y_nite[only_y]
                    aidx = both & choose_mod
                    bidx = both & ~choose_mod
                    out_nite[aidx] = m_nite[aidx]
                    out_nite[bidx] = y_nite[bidx]

                    # 对应 QC_Night：随 LST_Night 胜出者来源
                    out_qn = np.empty_like(m_qn)
                    out_qn[only_m] = m_qn[only_m]
                    out_qn[only_y] = y_qn[only_y]
                    out_qn[aidx]   = m_qn[aidx]
                    out_qn[bidx]   = y_qn[bidx]

                    # --- Emis 选择（使用 emis_qc_mode_combine） ---
                    s_m_emis = score_emis(m_qd, m_qn, mode=emis_qc_mode_combine)
                    s_y_emis = score_emis(y_qd, y_qn, mode=emis_qc_mode_combine)

                    # Emis_31
                    both = mv_e31 & yv_e31
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_emis[:, both], s_y_emis[:, both], prefer_a=prefer_mod
                        )
                    out_e31 = np.full_like(m_e31, out_nd)
                    only_m = mv_e31 & ~yv_e31
                    only_y = ~mv_e31 & yv_e31
                    out_e31[only_m] = m_e31[only_m]
                    out_e31[only_y] = y_e31[only_y]
                    aidx = both & choose_mod
                    bidx = both & ~choose_mod
                    out_e31[aidx] = m_e31[aidx]
                    out_e31[bidx] = y_e31[bidx]

                    # Emis_32
                    both = mv_e32 & yv_e32
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_emis[:, both], s_y_emis[:, both], prefer_a=prefer_mod
                        )
                    out_e32 = np.full_like(m_e32, out_nd)
                    only_m = mv_e32 & ~yv_e32
                    only_y = ~mv_e32 & yv_e32
                    out_e32[only_m] = m_e32[only_m]
                    out_e32[only_y] = y_e32[only_y]
                    aidx = both & choose_mod
                    bidx = both & ~choose_mod
                    out_e32[aidx] = m_e32[aidx]
                    out_e32[bidx] = y_e32[bidx]

                    # 写块
                    dst.write(out_day,  1, window=win)
                    dst.write(out_nite, 2, window=win)
                    dst.write(out_e31,  3, window=win)
                    dst.write(out_e32,  4, window=win)
                    dst.write(out_qd.astype(np.float32),   5, window=win)
                    dst.write(out_qn.astype(np.float32),   6, window=win)

            # 设置 per-band nodata
            for b in range(1, 7):
                dst.update_tags(b, nodata=str(int(out_nd)))

    return True, "merged"

# -------------------- 目录批处理 --------------------
def list_tifs(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in (".tif", ".tiff")])

def build_pairs(mod_dir: Path, myd_dir: Path):
    """构建配对：按日期和tile匹配MOD和MYD文件"""
    mod_map, myd_map = {}, {}
    for p in list_tifs(mod_dir):
        k = parse_date_tile_key(p)
        if k: mod_map[k] = p
    for p in list_tifs(myd_dir):
        k = parse_date_tile_key(p)
        if k: myd_map[k] = p
    keys = sorted(set(mod_map) | set(myd_map))
    return [(k, mod_map.get(k), myd_map.get(k)) for k in keys]

def mask_dir(in_dir: Path, out_dir: Path, out_nodata: float, args):
    """掩膜整个目录"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tifs = list_tifs(in_dir)
    total = len(tifs)
    prefix = f"[MASK] {in_dir.name} → {out_dir.name}"
    print_progress(prefix, 0, total, 0)

    tasks = [
        (
            p,
            out_dir / p.name,  # 保持原文件名
            out_nodata,
            args.overwrite,
            args.allow_mandatory_23,
            args.allow_dq_tbd,
            args.mask_dq1,
            args.max_keep_lst_error,
            args.max_keep_emis_error,
            args.emis_qc_mode_mask,
            args.compress,
        )
        for p in tifs
    ]

    ok, fail = 0, 0
    num_processes = max(1, cpu_count() // 4)
    with ProcessPoolExecutor(max_workers=num_processes) as ex:
        futs = {ex.submit(process_single_file, *task): task for task in tasks}
        for fut in as_completed(futs):
            try:
                success, fn, error_msg = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    if error_msg:
                        print(f"\n错误 {fn}: {error_msg}")
            except Exception as e:
                fail += 1
                task = futs[fut]
                print(f"\n异常 {task[0].name}: {e}")
            finally:
                print_progress(prefix, ok, total, fail)
    finish_progress()

def combine_dirs(mod_masked: Path, myd_masked: Path, out_dir: Path, out_nodata: float, args):
    """融合两个目录"""
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(mod_masked, myd_masked)
    total = len(pairs)
    prefix = f"[COMBINE] {mod_masked.name}+{myd_masked.name} → {out_dir.name}"
    print_progress(prefix, 0, total, 0)

    task = partial(
        combine_one,
        out_dir=out_dir,
        prefer_mod=(args.prefer=="mod"),
        emis_qc_mode_combine=args.emis_qc_mode_combine,
        chunk=args.chunk,
        compress=args.compress,
        out_nodata=out_nodata,
        overwrite=args.overwrite
    )
    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, k, mp, yp): k for (k, mp, yp) in pairs}
        for fut in as_completed(futs):
            try:
                fut.result(); ok += 1
            except Exception as e:
                fail += 1
                print(f"\n异常: {e}")
            finally:
                print_progress(prefix, ok, total, fail)
    finish_progress()

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="MOD11A1/MYD11A1 掩膜 + MCD11A1 融合（保持原有掩膜标准，基于QC评分拼接）")
    ap.add_argument("--mod_in",  type=Path, required=True, help="MOD11A1 输入目录")
    ap.add_argument("--myd_in",  type=Path, required=True, help="MYD11A1 输入目录")
    ap.add_argument("--mod_out", type=Path, required=True, help="MOD11A1 掩膜输出目录")
    ap.add_argument("--myd_out", type=Path, required=True, help="MYD11A1 掩膜输出目录")
    ap.add_argument("--mcd_out", type=Path, required=True, help="MCD11A1 融合输出目录")

    # 融合/性能
    ap.add_argument("--prefer", choices=["mod","myd"], default="mod", help="评分平分时优先（默认 mod）")
    ap.add_argument("--chunk", type=int, default=1024, help="块大小像素（默认 1024）")
    ap.add_argument("--workers", type=int, default=8, help="并行进程数（默认 8）")
    ap.add_argument("--compress", default="LZW", help="输出压缩（默认 LZW）")

    # NoData
    ap.add_argument("--out-nodata", type=float, default=-32765, help="统一 NoData 值（默认 -32765）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在输出")

    # ---------- 掩膜参数（保持原有标准） ----------
    ap.add_argument("--allow-mandatory-23", action="store_true",
                    help="放宽：不因 mandatory=2/3（未生产）而屏蔽（一般不建议）")
    ap.add_argument("--allow-dq-tbd", action="store_true",
                    help="放宽：不屏蔽 DataQuality=2/3（TBD）")
    ap.add_argument("--mask-dq1", action="store_true",
                    help="收紧：连 DataQuality=1（Other quality）也屏蔽（默认不屏蔽以保留中等质量）")
    ap.add_argument("--max-keep-lst-error", type=int, default=3, choices=[0,1,2,3],
                    help="允许保留的最大 LST 误差级别（0..3）；默认 3（全部保留）")
    ap.add_argument("--max-keep-emis-error", type=int, default=3, choices=[0,1,2,3],
                    help="允许保留的最大 Emissivity 误差级别（0..3）；默认 3（全部保留）")

    ap.add_argument("--emis-qc-mode-mask", choices=["union", "day", "night"], default="union",
                    help="掩膜阶段：发射率波段按哪个 QC 判定（默认 union）")
    ap.add_argument("--emis-qc-mode-combine", choices=["union", "day", "night"], default="union",
                    help="拼接阶段：Emis 评分用哪个 QC（默认 union）")

    args = ap.parse_args()

    # 掩膜
    mask_dir(args.mod_in, args.mod_out, args.out_nodata, args)
    mask_dir(args.myd_in, args.myd_out, args.out_nodata, args)

    # 融合
    combine_dirs(args.mod_out, args.myd_out, args.mcd_out, args.out_nodata, args)

if __name__ == "__main__":
    main()
