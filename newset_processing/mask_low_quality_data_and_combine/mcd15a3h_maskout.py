#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter low-quality pixels in MODIS MCD15A3H GeoTIFFs using QC bands.

Input band order (1-based):
  1: Lai
  2: Fpar
  3: FparLai_QC
  4: FparExtra_QC

For each input tif:
  - build a "good pixel" mask from QC bitfields
  - set Lai/Fpar to NoData for bad pixels (QC bands unchanged)
  - write to output dir with same filename

Dependencies:
  pip install gdal numpy tqdm
"""

import os
import re
import argparse
from typing import Set
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
from osgeo import gdal
from tqdm import tqdm

# 初始化GDAL
gdal.UseExceptions()

FNAME_RE = re.compile(r"^MCD15A3H_\d{4}_\d{2}_\d{2}_tile\d{2}\.tif$", re.IGNORECASE)


def bit_get(arr: np.ndarray, shift: int, width: int = 1) -> np.ndarray:
    """Extract bitfield of given width from integer array."""
    mask = (1 << width) - 1
    return (arr >> shift) & mask


def parse_set_ints(s: str) -> Set[int]:
    """Parse '0,1,3' -> {0,1,3}"""
    out = set()
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        out.add(int(part))
    return out


def build_good_mask(
    fparlai_qc: np.ndarray,
    fparextra_qc: np.ndarray,
    *,
    require_modland_good: bool,
    allow_scf_qc: Set[int],
    use_cloud_state: bool,
    allow_cloud_states: Set[int],
    use_dead_detector: bool,
    use_landsea: bool,
    allow_landsea: Set[int],
    use_snow: bool,
    use_aerosol: bool,
    use_cirrus: bool,
    use_internal_cloud: bool,
    use_shadow: bool,
    use_biome_interval: bool,
) -> np.ndarray:
    """
    Return boolean mask of good pixels (True = keep) based on QC bitmasks.
    """

    # Ensure unsigned to avoid right-shift surprises
    q1 = fparlai_qc.astype(np.uint8, copy=False)
    q2 = fparextra_qc.astype(np.uint8, copy=False)

    good = np.ones(q1.shape, dtype=bool)

    # ---- FparLai_QC ----
    # Bit 0: MODLAND_QC
    if require_modland_good:
        modland = bit_get(q1, 0, 1)  # 0 good, 1 other
        good &= (modland == 0)

    # Bit 2: Dead detector
    if use_dead_detector:
        dead = bit_get(q1, 2, 1)  # 0 ok, 1 dead detectors
        good &= (dead == 0)

    # Bits 3-4: Cloud state
    if use_cloud_state:
        cloud_state = bit_get(q1, 3, 2)  # 0..3
        good &= np.isin(cloud_state, list(allow_cloud_states))

    # Bits 5-7: SCF_QC (always applied)
    scf_qc = bit_get(q1, 5, 3)  # 0..7 (doc lists 0..4 common)
    good &= np.isin(scf_qc, list(allow_scf_qc))

    # ---- FparExtra_QC ----
    # Bits 0-1: Land-sea pass-thru
    if use_landsea:
        landsea = bit_get(q2, 0, 2)  # 0 land, 1 shore, 2 freshwater, 3 ocean
        good &= np.isin(landsea, list(allow_landsea))

    # Bit 2: Snow ice
    if use_snow:
        snow = bit_get(q2, 2, 1)
        good &= (snow == 0)

    # Bit 3: Aerosol
    if use_aerosol:
        aerosol = bit_get(q2, 3, 1)
        good &= (aerosol == 0)

    # Bit 4: Cirrus
    if use_cirrus:
        cirrus = bit_get(q2, 4, 1)
        good &= (cirrus == 0)

    # Bit 5: Internal cloud mask
    if use_internal_cloud:
        icloud = bit_get(q2, 5, 1)
        good &= (icloud == 0)

    # Bit 6: Cloud shadow
    if use_shadow:
        shadow = bit_get(q2, 6, 1)
        good &= (shadow == 0)

    # Bit 7: SCF biome mask (1 means biome in interval <1,4>)
    if use_biome_interval:
        biome = bit_get(q2, 7, 1)
        good &= (biome == 1)

    return good


def process_single_file(
    fn: str,
    input_dir: str,
    output_dir: str,
    nodata: float,
    overwrite: bool,
    require_modland_good: bool,
    allow_scf_qc: Set[int],
    use_cloud_state: bool,
    allow_cloud_states: Set[int],
    use_dead_detector: bool,
    use_landsea: bool,
    allow_landsea: Set[int],
    use_snow: bool,
    use_aerosol: bool,
    use_cirrus: bool,
    use_internal_cloud: bool,
    use_shadow: bool,
    use_biome_interval: bool,
):
    """处理单个文件的函数（用于并行处理）"""
    in_path = os.path.join(input_dir, fn)
    out_path = os.path.join(output_dir, fn)

    if (not overwrite) and os.path.exists(out_path):
        return True, fn, None

    try:
        # 使用GDAL打开输入文件
        src_ds = gdal.Open(in_path, gdal.GA_ReadOnly)
        if src_ds is None:
            return False, fn, f"无法打开文件: {in_path}"
        
        if src_ds.RasterCount < 4:
            raster_count = src_ds.RasterCount
            src_ds = None
            return False, fn, f"文件只有 {raster_count} 个波段，期望至少4个"
        
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        geotransform = src_ds.GetGeoTransform()
        projection = src_ds.GetProjection()
        
        # 读取4个波段
        lai_band = src_ds.GetRasterBand(1)
        fpar_band = src_ds.GetRasterBand(2)
        q1_band = src_ds.GetRasterBand(3)  # FparLai_QC
        q2_band = src_ds.GetRasterBand(4)  # FparExtra_QC
        
        # 读取数据
        lai = lai_band.ReadAsArray()
        fpar = fpar_band.ReadAsArray()
        q1 = q1_band.ReadAsArray()
        q2 = q2_band.ReadAsArray()
        
        # 获取源NoData值
        src_nodata = lai_band.GetNoDataValue()
        if src_nodata is not None:
            valid_src = (lai != src_nodata) & (fpar != src_nodata)
        else:
            valid_src = np.ones(lai.shape, dtype=bool)

        good = build_good_mask(
            q1, q2,
            require_modland_good=require_modland_good,
            allow_scf_qc=allow_scf_qc,
            use_cloud_state=use_cloud_state,
            allow_cloud_states=allow_cloud_states,
            use_dead_detector=use_dead_detector,
            use_landsea=use_landsea,
            allow_landsea=allow_landsea,
            use_snow=use_snow,
            use_aerosol=use_aerosol,
            use_cirrus=use_cirrus,
            use_internal_cloud=use_internal_cloud,
            use_shadow=use_shadow,
            use_biome_interval=use_biome_interval,
        )

        keep = good & valid_src

        # Apply: set bad pixels to nodata in LAI/FPAR only
        lai_out = lai.astype(np.float32, copy=False)
        fpar_out = fpar.astype(np.float32, copy=False)

        lai_out = np.where(keep, lai_out, nodata).astype(np.float32)
        fpar_out = np.where(keep, fpar_out, nodata).astype(np.float32)
        
        # QC波段保持原值，转换为float32
        q1_out = q1.astype(np.float32, copy=False)
        q2_out = q2.astype(np.float32, copy=False)

        # 使用GDAL创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        
        # 计算块大小：必须是16的倍数
        blockxsize = 256
        blockysize = 256
        if width < blockxsize:
            blockxsize = ((width + 15) // 16) * 16
        if height < blockysize:
            blockysize = ((height + 15) // 16) * 16
        
        creation_options = [
            'COMPRESS=LZW',
            'TILED=YES',
            f'BLOCKXSIZE={blockxsize}',
            f'BLOCKYSIZE={blockysize}',
            'BIGTIFF=IF_SAFER',
            'NUM_THREADS=ALL_CPUS'
        ]
        
        out_ds = driver.Create(
            out_path,
            width,
            height,
            4,  # 4个波段
            gdal.GDT_Float32,
            options=creation_options
        )
        
        if out_ds is None:
            src_ds = None
            return False, fn, f"无法创建输出文件: {out_path}"
        
        # 设置地理信息
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        
        # 写入数据
        out_band1 = out_ds.GetRasterBand(1)
        out_band1.SetNoDataValue(float(nodata))
        out_band1.WriteArray(lai_out)
        
        out_band2 = out_ds.GetRasterBand(2)
        out_band2.SetNoDataValue(float(nodata))
        out_band2.WriteArray(fpar_out)
        
        out_band3 = out_ds.GetRasterBand(3)
        # QC波段不设置NoData值（保持原样）
        out_band3.WriteArray(q1_out)
        
        out_band4 = out_ds.GetRasterBand(4)
        # QC波段不设置NoData值（保持原样）
        out_band4.WriteArray(q2_out)
        
        # 刷新缓存
        out_ds.FlushCache()
        
        # 清理资源
        out_ds = None
        src_ds = None
        
        return True, fn, None
        
    except Exception as e:
        return False, fn, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MCD15A3H', help="输入目录：包含 MCD15A3H_yyyy_mm_dd_tilexx.tif")
    ap.add_argument("--output_dir", type=str, default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MCD15A3H_QAapplied', help="输出目录")
    ap.add_argument("--nodata", type=float, default=-32768.0, help="写入 LAI/FPAR 的 NoData 值（坏像元置为该值）")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖已存在输出文件")

    # ---- Default QC policy (minimal & robust) ----
    # 默认只要求 SCF_QC∈{0,1,2,3}, modland为1表示可能是fill值或者backup模型结果，但是只有scf_qc为4时才确实是无效的
    
    ap.add_argument("--allow_scf_qc", type=str, default="0,1,2,3",
                    help="允许的 SCF_QC（FparLai_QC bits5-7），逗号分隔。默认 0,1（强烈建议保留）")

    ap.add_argument("--require_modland_good", action="store_true", default=False,
                    help="要求 MODLAND_QC bit0=0（默认启用；强烈建议保留）")
    ap.add_argument("--no_require_modland_good", action="store_false", dest="require_modland_good",
                    help="不要求 MODLAND_QC=0（不推荐）")

    # ---- Optional filters (all OFF by default) ----
    ap.add_argument("--use_cloud_state", action="store_true", default=False,
                    help="启用 cloud state 过滤（FparLai_QC bits3-4）。默认关闭。")
    ap.add_argument("--allow_cloud_states", type=str, default="0,3",
                    help="当 --use_cloud_state 启用时生效：允许的 cloud state，默认 0,3")

    ap.add_argument("--use_dead_detector", action="store_true", default=False,
                    help="启用 dead detector 过滤（FparLai_QC bit2=0）。默认关闭（一般不需要）。")

    ap.add_argument("--use_landsea", action="store_true", default=False,
                    help="启用 Land/Sea 过滤（FparExtra_QC bits0-1）。默认关闭（通常已有掩膜，不需要）。")
    ap.add_argument("--allow_landsea", type=str, default="0",
                    help="当 --use_landsea 启用时生效：允许的 land/sea，默认 0(LAND)")

    ap.add_argument("--use_snow", action="store_true", default=False,
                    help="启用 snow/ice 过滤（FparExtra_QC bit2=0）。默认关闭；做火季燃料 proxy 时建议开启。")

    ap.add_argument("--use_aerosol", action="store_true", default=False,
                    help="启用 aerosol 过滤（FparExtra_QC bit3=0）。默认关闭（通常不必要）。")

    ap.add_argument("--use_cirrus", action="store_true", default=False,
                    help="启用 cirrus 过滤（FparExtra_QC bit4=0）。默认关闭（通常不必要）。")

    ap.add_argument("--use_internal_cloud", action="store_true", default=False,
                    help="启用 internal cloud mask 过滤（FparExtra_QC bit5=0）。默认关闭（可与 cloud state 二选一）。")

    ap.add_argument("--use_shadow", action="store_true", default=False,
                    help="启用 cloud shadow 过滤（FparExtra_QC bit6=0）。默认关闭（容易误删）。")

    ap.add_argument("--use_biome_interval", action="store_true", default=False,
                    help="启用 biome interval 过滤（FparExtra_QC bit7=1）。默认关闭（不建议，可能引入偏差）。")

    args = ap.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    allow_scf_qc = parse_set_ints(args.allow_scf_qc)
    allow_cloud_states = parse_set_ints(args.allow_cloud_states) if args.use_cloud_state else set()
    allow_landsea = parse_set_ints(args.allow_landsea) if args.use_landsea else set()

    tifs = []
    for fn in os.listdir(input_dir):
        if fn.lower().endswith(".tif") and FNAME_RE.match(fn):
            tifs.append(fn)
    tifs.sort()

    if not tifs:
        raise RuntimeError(f"No matching MCD15A3H tif found in: {input_dir}")

    # 计算进程数：CPU数量的1/4
    num_processes = max(1, cpu_count() // 4)
    print(f"使用 {num_processes} 个进程并行处理 {len(tifs)} 个文件")
    
    # 准备任务参数
    tasks = [
        (
            fn,
            input_dir,
            output_dir,
            args.nodata,
            args.overwrite,
            args.require_modland_good,
            allow_scf_qc,
            args.use_cloud_state,
            allow_cloud_states,
            args.use_dead_detector,
            args.use_landsea,
            allow_landsea,
            args.use_snow,
            args.use_aerosol,
            args.use_cirrus,
            args.use_internal_cloud,
            args.use_shadow,
            args.use_biome_interval,
        )
        for fn in tifs
    ]
    
    # 并行处理
    total_success = 0
    total_failed = 0
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_file, *task): task for task in tasks}
        
        # 使用tqdm显示进度
        with tqdm(total=len(tifs), desc="QC filtering") as pbar:
            for future in as_completed(future_to_task):
                try:
                    success, fn, error_msg = future.result()
                    if success:
                        total_success += 1
                    else:
                        total_failed += 1
                        if error_msg:
                            print(f"\n错误 {fn}: {error_msg}")
                    pbar.update(1)
                except Exception as e:
                    total_failed += 1
                    task = future_to_task[future]
                    print(f"\n异常 {task[0]}: {e}")
                    pbar.update(1)
    
    # 输出统计信息
    print(f"\n处理完成: 成功 {total_success}, 失败 {total_failed}")
    if total_failed > 0:
        print(f"警告: {total_failed} 个文件处理失败")


if __name__ == "__main__":
    main()
