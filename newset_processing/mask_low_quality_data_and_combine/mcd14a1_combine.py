#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOD14A1 + MYD14A1 同日合成 (火点检测，Excel NoData + 有效性优先)
-------------------------------------------------
处理逻辑：
1. 输入：MOD14A1 (Terra, AM) 和 MYD14A1 (Aqua, PM)，波段顺序：
   ['FireMask','MaxFRP','QA']。
2. FireMask 合成：
   - 一方火点(7–9)，另一方非火点 → 保留火点
   - 双方火点 → 保留置信度更高(9 > 8 > 7)
   - 双方非火点 → 按优先级 Cloud(4) > Water(3) > Land(5) > Unknown(6) > Not processed(2/1)
   - 保留原始分类，不二值化
3. MaxFRP 合成：
   - 双方火点 → 取最大值
   - 一方火点 → 保留该值
   - 非火点 → 输出 0
4. QA 合成：
   - 陆水状态 (Bits 0–1): Land(2) > Coast(1) > Water(0) > Missing(3)
   - 昼夜 (Bit 2): 不同 → 保留 Day(1)
5. 输出：
   - 保留所有像元，不丢弃
   - NoData值与输入保持一致（优先使用MOD的NoData值，若MOD不存在则使用MYD的）

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ----------------- Excel NoData 读取 -----------------
def load_nodata_map(xlsx_path):
    """
    读取 Excel：返回 {folder_name(str) -> nodata_value(int)}。
    要求有列 'folder name', 'no data'。
    """
    import pandas as pd
    nd_map = {}
    if not xlsx_path:
        return nd_map
    if not os.path.exists(xlsx_path):
        print(f"[警告] Excel NoData 表不存在: {xlsx_path}")
        return nd_map

    try:
        df = pd.read_excel(xlsx_path)
        if 'folder name' not in df.columns or 'no data' not in df.columns:
            print(f"[警告] Excel 表缺少必要列: 'folder name', 'no data'")
            return nd_map
        for _, row in df.iterrows():
            folder_name = str(row['folder name']).strip()
            v = row['no data']
            if pd.notna(v):
                try:
                    nd_map[folder_name] = int(v)
                except Exception:
                    try:
                        nd_map[folder_name] = int(float(v))
                    except Exception:
                        pass
                if folder_name in nd_map:
                    print(f"[NoData] {folder_name} = {nd_map[folder_name]}")
    except Exception as e:
        print(f"[错误] 读取 Excel NoData 表失败: {e}")

    return nd_map

def nodata_for_folder(folder_path, nd_map, fallback=None):
    """用输入目录 basename 在 Excel 映射里查找 NoData；找不到返回 fallback。"""
    base = os.path.basename(os.path.normpath(folder_path)) if folder_path else None
    return nd_map.get(base, fallback)

def valid_mask_int(arr, nodata_value):
    """基于 Excel NoData 的有效像元判断（整型数组）。nodata_value 为 None 时认为全部有效。"""
    if nodata_value is None:
        return np.ones_like(arr, dtype=bool)
    return arr != nodata_value

# ---------- 工具函数 ----------
def parse_date(fname):
    m = re.search(r"(\d{4}_\d{2}_\d{2})", fname)
    return m.group(1) if m else None

def load_band(ds, band_idx, xoff=0, yoff=0, xsize=None, ysize=None):
    return ds.GetRasterBand(band_idx).ReadAsArray(xoff, yoff, xsize, ysize)

# ---------- 矢量化优先级 / 合成（与原逻辑一致，性能更好） ----------
# FireMask 合法类别（含 0..9 的可能值）；构建一个 0..255 的优先级表以纯向量化比较
_FIRE_PRIORITY = np.full(256, -999, dtype=np.int16)
# 置信度火点：9 > 8 > 7
_FIRE_PRIORITY[9] = 6
_FIRE_PRIORITY[8] = 5
_FIRE_PRIORITY[7] = 4
# 非火点优先级：Cloud(4) > Water(3) > Land(5) > Unknown(6) > Not processed(2/1)
_FIRE_PRIORITY[4] = 3
_FIRE_PRIORITY[3] = 2
_FIRE_PRIORITY[5] = 1
_FIRE_PRIORITY[6] = 0
_FIRE_PRIORITY[2] = -1
_FIRE_PRIORITY[1] = -2
# 你之前的代码里包含了 0，我们按较低优先级处理（如果你的数据从不出现 0，可以忽略它）
_FIRE_PRIORITY[0] = -3

_FIRE_CLASSES = np.array([7, 8, 9], dtype=np.int16)

def choose_firemask_vector(mod_fm, myd_fm):
    """
    纯矢量实现：与原 choose_firemask 等价
    返回最终 FireMask。
    """
    pri_mod = _FIRE_PRIORITY[mod_fm]
    pri_myd = _FIRE_PRIORITY[myd_fm]

    out = np.empty_like(mod_fm, dtype=mod_fm.dtype)

    # mod 优先
    mod_better = pri_mod > pri_myd
    if np.any(mod_better):
        out[mod_better] = mod_fm[mod_better]

    # myd 优先
    myd_better = pri_myd > pri_mod
    if np.any(myd_better):
        out[myd_better] = myd_fm[myd_better]

    # 相等：取 max
    equal_mask = ~(mod_better | myd_better)
    if np.any(equal_mask):
        out[equal_mask] = np.maximum(mod_fm[equal_mask], myd_fm[equal_mask])

    return out

def combine_qa_vector(mod_qa, myd_qa):
    """
    与原 combine_qa 等价（保持逻辑不变）：
      - Bits 0-1: land/water（Land(2) > Coast(1) > Water(0) > Missing(3)）
      - Bit 2: day/night（优先 Day(1)）
    纯矢量实现，避免 np.vectorize 的 Python 循环。
    """
    # Bits 0-1
    lw_mod = (mod_qa & 0b11).astype(np.uint8)
    lw_myd = (myd_qa & 0b11).astype(np.uint8)

    # 优先级：Land(2)=3 > Coast(1)=2 > Water(0)=1 > Missing(3)=0
    _LW_PRI = np.array([1, 2, 3, 0], dtype=np.int8)  # 索引即 0/1/2/3
    pri_mod = _LW_PRI[lw_mod]
    pri_myd = _LW_PRI[lw_myd]

    lw_out = np.where(pri_mod >= pri_myd, lw_mod, lw_myd).astype(np.uint8)

    # Bit 2: day/night（优先 Day=1）
    dn_mod = ((mod_qa >> 2) & 1).astype(np.uint8)
    dn_myd = ((myd_qa >> 2) & 1).astype(np.uint8)
    dn_out = ((dn_mod == 1) | (dn_myd == 1)).astype(np.uint8)  # 有任一为 day→1，否则 0

    return ((dn_out << 2) | lw_out).astype(mod_qa.dtype)

# ---------- 合成逻辑 ----------
def combine_day(mod_path, myd_path, out_path, block_size=512,
                nodata_mod=None, nodata_myd=None, output_nodata=None):
    ds_mod = gdal.Open(mod_path)
    ds_myd = gdal.Open(myd_path if myd_path else mod_path)
    driver = gdal.GetDriverByName("GTiff")

    xsize = ds_mod.RasterXSize
    ysize = ds_mod.RasterYSize
    bands = ds_mod.RasterCount  # 3: FireMask, MaxFRP, QA

    # 输出 NoData：优先 MOD；否则 MYD；再否则 -32768（不要用 0，避免与有效类别冲突）
    if output_nodata is None:
        if nodata_mod is not None:
            output_nodata = nodata_mod
        elif nodata_myd is not None:
            output_nodata = nodata_myd
        else:
            output_nodata = -32768

    out_ds = driver.Create(out_path, xsize, ysize, bands, gdal.GDT_Int16,
                           options=["COMPRESS=LZW", "TILED=YES",
                                    "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
    out_ds.SetProjection(ds_mod.GetProjection())
    out_ds.SetGeoTransform(ds_mod.GetGeoTransform())

    # —— 按你的要求：保证输出文件 NoData 与输入一致（使用上面确定的 output_nodata）
    #    三个波段统一设为同一个 NoData 值
    for b in range(1, bands + 1):
        out_ds.GetRasterBand(b).SetNoDataValue(int(output_nodata))

    for y in range(0, ysize, block_size):
        rows = min(block_size, ysize - y)
        for x in range(0, xsize, block_size):
            cols = min(block_size, xsize - x)

            # 读三波段
            mod_fm = load_band(ds_mod, 1, x, y, cols, rows).astype(np.int16)
            mod_frp = load_band(ds_mod, 2, x, y, cols, rows).astype(np.int16)
            mod_qa = load_band(ds_mod, 3, x, y, cols, rows).astype(np.int16)

            myd_fm = load_band(ds_myd, 1, x, y, cols, rows).astype(np.int16)
            myd_frp = load_band(ds_myd, 2, x, y, cols, rows).astype(np.int16)
            myd_qa = load_band(ds_myd, 3, x, y, cols, rows).astype(np.int16)

            # 有效性（基于 FireMask 的 Excel NoData）
            mod_valid = valid_mask_int(mod_fm, nodata_mod)
            myd_valid = valid_mask_int(myd_fm, nodata_myd)

            only_mod = mod_valid & (~myd_valid)
            only_myd = myd_valid & (~mod_valid)
            both_valid = mod_valid & myd_valid
            none_valid = (~mod_valid) & (~myd_valid)

            # 初始化输出（统一用 output_nodata 填充）
            fire_out = np.full_like(mod_fm, int(output_nodata), dtype=np.int16)
            frp_out  = np.full_like(mod_frp, int(output_nodata), dtype=np.int16)
            qa_out   = np.full_like(mod_qa, int(output_nodata), dtype=np.int16)

            # -------- FireMask 合成 --------
            if np.any(only_mod):
                fire_out[only_mod] = mod_fm[only_mod]
            if np.any(only_myd):
                fire_out[only_myd] = myd_fm[only_myd]
            if np.any(both_valid):
                fire_out[both_valid] = choose_firemask_vector(mod_fm[both_valid], myd_fm[both_valid])
            # none_valid 保持为 output_nodata

            # -------- MaxFRP 合成（保持你的规则：非火点 → 0，不是 NoData）--------
            # only_mod
            if np.any(only_mod):
                fire_mod_only = np.isin(mod_fm[only_mod], _FIRE_CLASSES)
                # 火点 → 取源 FRP；非火点 → 0
                frp_out[only_mod] = np.where(fire_mod_only, mod_frp[only_mod], 0).astype(np.int16)
            # only_myd
            if np.any(only_myd):
                fire_myd_only = np.isin(myd_fm[only_myd], _FIRE_CLASSES)
                frp_out[only_myd] = np.where(fire_myd_only, myd_frp[only_myd], 0).astype(np.int16)
            # both_valid
            if np.any(both_valid):
                fire_mod_b = np.isin(mod_fm[both_valid], _FIRE_CLASSES)
                fire_myd_b = np.isin(myd_fm[both_valid], _FIRE_CLASSES)
                both_fire  = fire_mod_b & fire_myd_b
                only_mfire = fire_mod_b & (~fire_myd_b)
                only_afire = (~fire_mod_b) & fire_myd_b
                neither    = (~fire_mod_b) & (~fire_myd_b)

                # 先填 0（非火点=0）
                block = np.zeros_like(mod_frp[both_valid], dtype=np.int16)
                # 双方火点 → 取最大
                if np.any(both_fire):
                    block[both_fire] = np.maximum(mod_frp[both_valid][both_fire],
                                                  myd_frp[both_valid][both_fire]).astype(np.int16)
                # 仅一方火点 → 取该值
                if np.any(only_mfire):
                    block[only_mfire] = mod_frp[both_valid][only_mfire]
                if np.any(only_afire):
                    block[only_afire] = myd_frp[both_valid][only_afire]
                # neither 已是 0
                frp_out[both_valid] = block

            # none_valid：保持 output_nodata（非火点=0 的规则仅在“有效输入”的像元内生效）

            # -------- QA 合成（保持你的原始逻辑不变：陆水优先、昼夜优先 Day）--------
            if np.any(only_mod):
                qa_out[only_mod] = mod_qa[only_mod]
            if np.any(only_myd):
                qa_out[only_myd] = myd_qa[only_myd]
            if np.any(both_valid):
                qa_out[both_valid] = combine_qa_vector(mod_qa[both_valid], myd_qa[both_valid])

            # 写出
            out_ds.GetRasterBand(1).WriteArray(fire_out, x, y)
            out_ds.GetRasterBand(2).WriteArray(frp_out,  x, y)
            out_ds.GetRasterBand(3).WriteArray(qa_out,   x, y)

    ds_mod = None
    ds_myd = None
    out_ds = None

# ---------- 批处理 ----------
def process_single_task(task):
    d, mod_path, myd_path, out_dir, block_size, nodata_mod, nodata_myd, output_nodata = task
    try:
        out_path = os.path.join(out_dir, f"{d}.tif")
        if mod_path and myd_path:
            combine_day(mod_path, myd_path, out_path, block_size,
                        nodata_mod=nodata_mod, nodata_myd=nodata_myd, output_nodata=output_nodata)
        else:
            # 单方存在 → 直接复制
            src = mod_path if mod_path else myd_path
            gdal.Translate(out_path, src, creationOptions=["COMPRESS=LZW", "TILED=YES",
                                                           "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
        return True
    except Exception as e:
        print(f"处理 {d} 失败: {e}")
        return False

def process_all(mod_dir, myd_dir, out_dir, workers=4, block_size=512,
                nodata_xlsx=None, output_nodata=None):
    os.makedirs(out_dir, exist_ok=True)

    # Excel NoData
    nd_map = load_nodata_map(nodata_xlsx) if nodata_xlsx else {}
    nodata_mod = nodata_for_folder(mod_dir, nd_map, fallback=None) if mod_dir else None
    nodata_myd = nodata_for_folder(myd_dir, nd_map, fallback=None) if myd_dir else None
    if mod_dir:
        print(f"[NoData] {os.path.basename(mod_dir)} = {nodata_mod}")
    if myd_dir:
        print(f"[NoData] {os.path.basename(myd_dir)} = {nodata_myd}")

    # 列出日期
    mod_files = {parse_date(f): os.path.join(mod_dir, f)
                 for f in os.listdir(mod_dir)} if mod_dir else {}
    mod_files = {k: v for k, v in mod_files.items() if k and v.lower().endswith(".tif")}
    myd_files = {parse_date(f): os.path.join(myd_dir, f)
                 for f in os.listdir(myd_dir)} if myd_dir else {}
    myd_files = {k: v for k, v in myd_files.items() if k and v.lower().endswith(".tif")}

    all_dates = sorted(set(mod_files.keys()) | set(myd_files.keys()))

    tasks = []
    for d in all_dates:
        mod_path = mod_files.get(d)
        myd_path = myd_files.get(d)
        if not mod_path and not myd_path:
            continue
        out_path = os.path.join(out_dir, f"{d}.tif")
        if os.path.exists(out_path):
            continue
        tasks.append((d, mod_path, myd_path, out_dir, block_size, nodata_mod, nodata_myd, output_nodata))

    if not tasks:
        print("没有可处理的任务（可能都已存在）。")
        return

    if workers <= 1:
        for task in tqdm(tasks, desc="处理文件"):
            process_single_task(task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(tqdm(ex.map(process_single_task, tasks),
                      total=len(tasks), desc="处理文件"))

# ----------------- CLI -----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MOD14A1 + MYD14A1 合成 (火点检测，Excel NoData + 有效性优先)")
    parser.add_argument("--mod_dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MOD14A1_mosaic', help="MOD14A1 文件夹路径")
    parser.add_argument("--myd_dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MYD14A1_mosaic', help="MYD14A1 文件夹路径")
    parser.add_argument("--out_dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MCD14A1_mosaic', help="输出文件夹路径")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")
    parser.add_argument("--block_size", type=int, default=512, help="分块大小")
    parser.add_argument("--nodata-xlsx", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/no_data_statistics.xlsx', help="含列 'folder name', 'no data' 的 Excel 文件路径")
    parser.add_argument("--output-nodata", type=int, default=None,
                        help="可选：强制指定输出 NoData；默认取 Excel 中 MOD/MYD 的值，若缺失则 -32768")
    args = parser.parse_args()

    process_all(args.mod_dir, args.myd_dir, args.out_dir,
                workers=args.workers, block_size=args.block_size,
                nodata_xlsx=args.nodata_xlsx, output_nodata=args.output_nodata)
