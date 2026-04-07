#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOD11A1 + MYD11A1 像元级融合脚本（并行 + 分块 + 地理对齐）
输出 6 波段：1 LST_Day_1km, 2 LST_Night_1km, 3 Emis_31, 4 Emis_32, 5 QC_Day, 6 QC_Night

融合规则（逐像元）：
- 若仅一侧有效（非 NoData），选该侧；
- 若两侧均无效，输出 NoData；
- 若两侧均有效：
  * LST_Day: 用 QC_Day 打分；得分更优者胜，QC_Day 取胜者对应像元
  * LST_Night: 用 QC_Night 打分；同上
  * Emis_31/32: 用 emissivity error 打分（默认从该源的 QC_Day/QC_Night 取“更优”的 emissivity error 及 mandatory/data_quality 作为评分；可配置）
- 打分元组（越小越好）：
  LST: (mandatory, lst_error, data_quality, emis_error)
  Emis: (mandatory, emis_error, data_quality)

几何与像元对齐：
- 目标格网：CRS 取 mod.crs（若缺则用 myd.crs）；分辨率取两者中更细像元尺寸；范围取两者 union；
- 通过 WarpedVRT 将两侧按同一格网分块读取，避免整图重投影内存炸裂；
- QC 波段使用 nearest 重采样；数值波段（LST/Emis）使用 bilinear 重采样。

并行：
- 按日期（文件名解析）多进程；每个任务内部再分块处理。
"""

import re
import sys
import math
import shutil
import argparse
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio import windows
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, calculate_default_transform, Resampling
from rasterio.errors import NotGeoreferencedWarning
import warnings

warnings.simplefilter("ignore", NotGeoreferencedWarning)

DATE_RE = re.compile(r"(?P<y>\d{4})[-_]?((?P<m>\d{2})[-_]?)(?P<d>\d{2})")

def parse_date_key(p: Path) -> str | None:
    m = DATE_RE.search(p.stem)
    if not m:
        return None
    y, mth, d = m.group("y"), m.group("m"), m.group("d")
    return f"{y}_{mth}_{d}"

def bitslice(arr, start, end):
    width = end - start + 1
    return (arr >> start) & ((1 << width) - 1)

def qc_fields(qc):
    return dict(
        mandatory   = bitslice(qc, 0, 1),
        data_quality= bitslice(qc, 2, 3),
        emis_error  = bitslice(qc, 4, 5),
        lst_error   = bitslice(qc, 6, 7),
    )

def score_lst(qc_arr):
    q = qc_fields(qc_arr)
    # (mandatory, lst_error, data_quality, emis_error)
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
    # 逐维比较
    choose_a = np.zeros(a_score.shape[1:], dtype=bool)
    undecided = np.ones_like(choose_a, dtype=bool)
    for k in range(a_score.shape[0]):
        ak = a_score[k]; bk = b_score[k]
        better_a = ak < bk
        better_b = ak > bk
        choose_a |= (better_a & undecided)
        undecided &= ~(better_a | better_b)
        # 继续循环，直到所有像元不再 undecided
        if not undecided.any():
            break
    # 平分：按偏好
    if undecided.any():
        if prefer_a:
            choose_a |= undecided
        # else: 默认 False 留给 B
    return choose_a

def read_nodata_per_band(src):
    nds = []
    for i in range(1, src.count + 1):
        nds.append(src.nodatavals[i-1])
    return nds

def pick_target_grid(mod_src, myd_src):
    """
    选择目标格网：CRS优先用 mod；分辨率取更细；范围取 union。
    若 mod 无 CRS 则用 myd 的。
    """
    crs = mod_src.crs or myd_src.crs
    if crs is None:
        raise ValueError("两个数据都缺少有效 CRS。")

    # 分辨率（绝对值更小者更细）
    def res_of(src):
        a = src.transform
        return (abs(a.a), abs(a.e))  # (xres, yres) 注意 yres 为负，取绝对值
    mod_res = res_of(mod_src)
    myd_res = res_of(myd_src)
    xres = min(mod_res[0], myd_res[0])
    yres = min(mod_res[1], myd_res[1])

    # union bounds (in crs)
    mod_bounds = transform_bounds(mod_src.crs, crs, *mod_src.bounds, densify_pts=0) if mod_src.crs != crs else mod_src.bounds
    myd_bounds = transform_bounds(myd_src.crs, crs, *myd_src.bounds, densify_pts=0) if myd_src.crs != crs else myd_src.bounds
    xmin = min(mod_bounds[0], myd_bounds[0])
    ymin = min(mod_bounds[1], myd_bounds[1])
    xmax = max(mod_bounds[2], myd_bounds[2])
    ymax = max(mod_bounds[3], myd_bounds[3])

    # 计算输出仿射与宽高
    dst_transform, width, height = calculate_default_transform(
        crs, crs,  # 仅用于宽高计算，这里传同一 CRS
        int((xmax - xmin) / xres), int((ymax - ymin) / yres),
        left=xmin, bottom=ymin, right=xmax, top=ymax,
        resolution=(xres, yres)
    )
    # 修正：calculate_default_transform 以上调用仅为快速生成，直接构造更简单
    from affine import Affine
    dst_transform = Affine(xres, 0.0, xmin, 0.0, -yres, ymax)
    width  = int(math.ceil((xmax - xmin) / xres))
    height = int(math.ceil((ymax - ymin) / yres))

    return crs, dst_transform, width, height, (xres, yres)

def band_types(src):
    return [np.dtype(dt) for dt in src.dtypes]

def is_nan_nodata(nd):
    return isinstance(nd, float) and np.isnan(nd)

def valid_mask(arr, nd):
    if is_nan_nodata(nd):
        return ~np.isnan(arr)
    return arr != nd

def write_profile_template(crs, transform, width, height, dtypes, compress="LZW"):
    # 输出 6 波段；逐波段 dtype 按输入 LST/Emis 倾向（QC 为整数）
    profile = dict(
        driver="GTiff",
        count=6,
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        tiled=True,
        blockxsize=min(1024, width),
        blockysize=min(1024, height),
        compress=compress,
        bigtiff="IF_SAFER"
    )
    # 若需要 per-band dtype，可写入后逐波段 cast；这里假定两侧 dtype 一致，按第一侧（mod）写
    profile["dtype"] = dtypes[0].name
    return profile

def choose_and_write(date_key, mod_path, myd_path, out_dir, prefer, emis_qc_mode,
                     chunk, compress):
    out_path = out_dir / f"{date_key}.tif"

    # 只有一侧存在：直接复制（不改变格网和元数据）
    if (mod_path is None) ^ (myd_path is None):
        src_path = mod_path or myd_path
        shutil.copy2(src_path, out_path)
        return True, "copied"

    # 两侧都存在：融合
    with rasterio.open(mod_path) as mod, rasterio.open(myd_path) as myd:
        if mod.count < 6 or myd.count < 6:
            raise ValueError("输入需要 6 波段 (LST_Day, LST_Night, Emis_31, Emis_32, QC_Day, QC_Night)。")

        # 目标格网（CRS 优先用 mod，分辨率取更细，范围取 union）
        crs, dst_transform, width, height, _ = pick_target_grid(mod, myd)

        # 建立对齐 VRT：LST/Emis 用双线性，QC 用近邻
        vrt_opts_float = dict(crs=crs, transform=dst_transform, width=width, height=height,
                              resampling=Resampling.bilinear, add_alpha=False)
        vrt_opts_near  = dict(crs=crs, transform=dst_transform, width=width, height=height,
                              resampling=Resampling.nearest, add_alpha=False)

        # 逐波段 nodata
        mod_nd = read_nodata_per_band(mod)
        myd_nd = read_nodata_per_band(myd)

        # 目标输出 dtype：沿用 mod 第一波段（通常为 int16 或 float32）
        dtypes = band_types(mod)
        profile = write_profile_template(crs, dst_transform, width, height, dtypes, compress=compress)

        # 统一输出 nodata（逐波段）：若源是浮点则 NaN，否则 -32768
        out_nd = []
        for i in range(6):
            dt = dtypes[i] if i < len(dtypes) else np.dtype("float32")
            if np.issubdtype(dt, np.floating):
                out_nd.append(np.nan)
            else:
                out_nd.append(np.int64(-32768))

        with rasterio.open(out_path, "w", **profile) as dst, \
             WarpedVRT(mod, **vrt_opts_float) as mod_vrt_f, WarpedVRT(myd, **vrt_opts_float) as myd_vrt_f, \
             WarpedVRT(mod, **vrt_opts_near)  as mod_vrt_n, WarpedVRT(myd, **vrt_opts_near)  as myd_vrt_n:

            # 块遍历
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

                    # 有效掩膜（基于 GeoTIFF nodata；重采样后 nodata 也按规则传播）
                    mv_day  = valid_mask(m_day,  mod_nd[0])
                    mv_nite = valid_mask(m_nite, mod_nd[1])
                    mv_e31  = valid_mask(m_e31,  mod_nd[2])
                    mv_e32  = valid_mask(m_e32,  mod_nd[3])

                    yv_day  = valid_mask(y_day,  myd_nd[0])
                    yv_nite = valid_mask(y_nite, myd_nd[1])
                    yv_e31  = valid_mask(y_e31,  myd_nd[3-1])  # index 2
                    yv_e32  = valid_mask(y_e32,  myd_nd[4-1])  # index 3

                    # --- LST_Day 选择 ---
                    s_m_day = score_lst(m_qd)
                    s_y_day = score_lst(y_qd)
                    # 仅在两侧都有效才比较打分
                    both = mv_day & yv_day
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_day[:, both], s_y_day[:, both], prefer_a=(prefer=="mod")
                        )

                    out_day = np.empty_like(m_day)
                    # 默认填 NoData
                    if is_nan_nodata(out_nd[0]):
                        out_day[:] = np.nan
                    else:
                        out_day[:] = out_nd[0]

                    # 单侧有效
                    only_m = mv_day & ~yv_day
                    only_y = ~mv_day & yv_day
                    out_day[only_m] = m_day[only_m]
                    out_day[only_y] = y_day[only_y]
                    # 双侧有效：按评分
                    aidx = both &  choose_mod
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
                            s_m_nite[:, both], s_y_nite[:, both], prefer_a=(prefer=="mod")
                        )

                    out_nite = np.empty_like(m_nite)
                    if is_nan_nodata(out_nd[1]):
                        out_nite[:] = np.nan
                    else:
                        out_nite[:] = out_nd[1]

                    only_m = mv_nite & ~yv_nite
                    only_y = ~mv_nite & yv_nite
                    out_nite[only_m] = m_nite[only_m]
                    out_nite[only_y] = y_nite[only_y]
                    aidx = both &  choose_mod
                    bidx = both & ~choose_mod
                    out_nite[aidx] = m_nite[aidx]
                    out_nite[bidx] = y_nite[bidx]

                    # 对应 QC_Night：随 LST_Night 胜出者来源
                    out_qn = np.empty_like(m_qn)
                    out_qn[only_m] = m_qn[only_m]
                    out_qn[only_y] = y_qn[only_y]
                    out_qn[aidx]   = m_qn[aidx]
                    out_qn[bidx]   = y_qn[bidx]

                    # --- Emis 选择（默认 union） ---
                    s_m_emis = score_emis(m_qd, m_qn, mode=emis_qc_mode)
                    s_y_emis = score_emis(y_qd, y_qn, mode=emis_qc_mode)

                    # Emis_31
                    both = mv_e31 & yv_e31
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_emis[:, both], s_y_emis[:, both], prefer_a=(prefer=="mod")
                        )
                    out_e31 = np.empty_like(m_e31)
                    if is_nan_nodata(out_nd[2]):
                        out_e31[:] = np.nan
                    else:
                        out_e31[:] = out_nd[2]
                    only_m = mv_e31 & ~yv_e31
                    only_y = ~mv_e31 & yv_e31
                    out_e31[only_m] = m_e31[only_m]
                    out_e31[only_y] = y_e31[only_y]
                    aidx = both &  choose_mod
                    bidx = both & ~choose_mod
                    out_e31[aidx] = m_e31[aidx]
                    out_e31[bidx] = y_e31[bidx]

                    # Emis_32
                    both = mv_e32 & yv_e32
                    choose_mod = np.zeros(both.shape, dtype=bool)
                    if both.any():
                        choose_mod[both] = lexicographic_choose(
                            s_m_emis[:, both], s_y_emis[:, both], prefer_a=(prefer=="mod")
                        )
                    out_e32 = np.empty_like(m_e32)
                    if is_nan_nodata(out_nd[3]):
                        out_e32[:] = np.nan
                    else:
                        out_e32[:] = out_nd[3]
                    only_m = mv_e32 & ~yv_e32
                    only_y = ~mv_e32 & yv_e32
                    out_e32[only_m] = m_e32[only_m]
                    out_e32[only_y] = y_e32[only_y]
                    aidx = both &  choose_mod
                    bidx = both & ~choose_mod
                    out_e32[aidx] = m_e32[aidx]
                    out_e32[bidx] = y_e32[bidx]

                    # 写块
                    dst.write(out_day,  1, window=win)
                    dst.write(out_nite, 2, window=win)
                    dst.write(out_e31,  3, window=win)
                    dst.write(out_e32,  4, window=win)
                    dst.write(out_qd,   5, window=win)
                    dst.write(out_qn,   6, window=win)

            # 设置 per-band nodata（GTiff 单值）
            for b in range(1, 7):
                nd = out_nd[b-1]
                if not is_nan_nodata(nd):
                    dst.update_tags(b, nodata=str(int(nd)))
                else:
                    # 对于 NaN，GTiff 无显式 nodata 值；多数栅格软件以 NaN 识别空值
                    pass
    return True, "merged"

def build_pairs(mod_dir: Path, myd_dir: Path):
    mod_map, myd_map = {}, {}
    for p in sorted(mod_dir.glob("*.tif")):
        k = parse_date_key(p)
        if k: mod_map[k] = p
    for p in sorted(myd_dir.glob("*.tif")):
        k = parse_date_key(p)
        if k: myd_map[k] = p
    keys = sorted(set(mod_map) | set(myd_map))
    pairs = [(k, mod_map.get(k), myd_map.get(k)) for k in keys]
    return pairs

def main():
    ap = argparse.ArgumentParser(description="MOD11A1/MYD11A1 按像元融合（并行 + 分块 + 地理对齐，保留 QC）")
    ap.add_argument("--mod", type=Path, 
                    default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/trash/MOD11A1_mosaic_QAapplied', help="MOD11A1 目录（文件名含日期）")
    ap.add_argument("--myd", type=Path, 
                    default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/trash/MYD11A1_mosaic_QAapplied', help="MYD11A1 目录（文件名含日期）")
    ap.add_argument("--out", type=Path, 
                    default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/trash/MCD11A1_mosaic_QAapplied', help="输出目录")
    ap.add_argument("--prefer", choices=["mod", "myd"], default="mod", help="评分完全相等时优先谁（默认 mod）")
    ap.add_argument("--emis-qc-mode", choices=["union", "day", "night"], default="union",
                    help="Emis 评分用哪个 QC（默认 union：取 day/night 更优者）")
    ap.add_argument("--chunk", type=int, default=1024, help="块大小（像素），默认 1024")
    ap.add_argument("--workers", type=int, default=8, help="并行进程数（默认 8）")
    ap.add_argument("--compress", default="LZW", help="输出压缩方式（默认 LZW）")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(args.mod, args.myd)
    if not pairs:
        print("未在输入目录中解析到日期匹配的文件名。", file=sys.stderr); sys.exit(1)

    task = partial(
        choose_and_write,
        out_dir=args.out, prefer=args.prefer,
        emis_qc_mode=args.emis_qc_mode,
        chunk=args.chunk, compress=args.compress
    )

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, k, mp, yp): k for (k, mp, yp) in pairs}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                _, msg = fut.result()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[{k}] FAILED: {e}", file=sys.stderr)
    print(f"All done. OK={ok}, FAILED={fail}")

if __name__ == "__main__":
    main()
