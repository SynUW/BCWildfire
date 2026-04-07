#!/usr/bin/env python3
"""
将指定文件夹中的GeoTIFF按通道拼接为一个多波段影像，并返回波段顺序。

特性：
- 从输入目录按文件名排序（可选自然排序）收集TIF文件；
- 严格校验所有输入的尺寸、投影、仿射一致；
- 将每个输入TIF的所有波段依次写入输出，多波段文件会被展平追加；
- 输出为单个多波段GTiff，保持首个输入的投影与仿射；
- 统一NoData为-9999：写入前将源波段的NoData（含NaN）映射为-9999，输出所有波段NoData统一设置为-9999；若首个文件数据类型为无符号整型，则输出dtype提升为Float32以容纳-9999。
- 输出同时生成一个sidecar映射文件（.bands.txt）列出 band_index -> source_file [band_index]；
- 如遇不同数据类型，会转换为输出数据类型并给出警告；
- 支持LZW压缩、瓦片、BIGTIFF；

示例：
python3 stack_tif_bands.py \
  --input-dir /path/to/src \
  --output /path/to/stacked.tif \
  --pattern "*.tif" \
  --natural-sort \
  --save-map
"""

import os
import re
import glob
import logging
import argparse
from typing import List, Tuple

import numpy as np
from osgeo import gdal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stack_tif_bands')

TARGET_NODATA = -32768.0

def list_tifs(folder: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, pattern)))

def natural_key(s: str):
    # 自然排序键：将数字部分作为整数排序
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def read_info(ds) -> Tuple[int, int, int, str, tuple, int]:
    w = ds.RasterXSize
    h = ds.RasterYSize
    b = ds.RasterCount
    prj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    dtype = ds.GetRasterBand(1).DataType
    return w, h, b, prj, gt, dtype

def dtype_name(dt: int) -> str:
    return gdal.GetDataTypeName(dt)

# 辅助：GDAL数据类型到numpy dtype
_numpy_from_gdal = {
    gdal.GDT_Byte: np.uint8,
    gdal.GDT_UInt16: np.uint16,
    gdal.GDT_Int16: np.int16,
    gdal.GDT_UInt32: np.uint32,
    gdal.GDT_Int32: np.int32,
    gdal.GDT_Float32: np.float32,
    gdal.GDT_Float64: np.float64,
}

def gdal_array_type_to_numpy(dt: int):
    return _numpy_from_gdal.get(dt, np.float32)

def is_unsigned(dt: int) -> bool:
    return dt in (gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_UInt32)

def choose_out_dtype(ref_dt: int) -> int:
    # 若首个为无符号整型，则用Float32以容纳-9999；否则沿用首个类型
    return gdal.GDT_Float32 if is_unsigned(ref_dt) else ref_dt

def map_src_nodata_to_target(arr: np.ndarray, src_nd) -> np.ndarray:
    out = arr
    if src_nd is None:
        return out
    try:
        # 处理NaN作为NoData
        if isinstance(src_nd, float) and np.isnan(src_nd):
            mask = np.isnan(out)
        else:
            mask = (out == src_nd)
        if np.any(mask):
            out = out.copy()
            out[mask] = TARGET_NODATA
    except Exception:
        pass
    return out

def main():
    ap = argparse.ArgumentParser(description='按通道拼接TIF为多波段影像，输出波段顺序（统一NoData=-9999）')
    ap.add_argument('--input-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/distance_map_interpolated_margin_nodata_norm', help='输入目录')
    ap.add_argument('--output', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/distance_maps_materials/distance_map_inter_norm_stack.tif', help='输出多波段TIF路径')
    ap.add_argument('--pattern', default='*.tif', help='文件匹配模式，默认*.tif')
    ap.add_argument('--natural-sort', action='store_true', help='使用自然排序（数字按数值排序）')
    ap.add_argument('--save-map', action='store_true', help='保存band映射到 .bands.txt sidecar 文件')
    args = ap.parse_args()

    files = list_tifs(args.input_dir, args.pattern)
    if not files:
        logger.error(f'输入目录无匹配TIF: {args.input_dir}/{args.pattern}')
        return
    files = sorted(files, key=natural_key if args.natural_sort else str.lower)

    # 打开首个文件作为参考
    ref_ds = gdal.Open(files[0], gdal.GA_ReadOnly)
    if ref_ds is None:
        logger.error(f'无法打开参考文件: {files[0]}')
        return
    ref_w, ref_h, ref_bands, ref_prj, ref_gt, ref_dtype = read_info(ref_ds)

    # 校验一致性与统计总波段
    total_bands = 0
    for fp in files:
        ds = gdal.Open(fp, gdal.GA_ReadOnly)
        if ds is None:
            logger.error(f'无法打开: {fp}')
            return
        w, h, b, prj, gt, dt = read_info(ds)
        ds = None
        if (w != ref_w) or (h != ref_h):
            logger.error(f'尺寸不一致: {os.path.basename(fp)} ({w}x{h}) != ({ref_w}x{ref_h})')
            return
        if prj != ref_prj:
            logger.error(f'投影不一致: {os.path.basename(fp)}')
            return
        if gt != ref_gt:
            logger.error(f'仿射参数不一致: {os.path.basename(fp)}')
            return
        total_bands += b

    out_dtype = choose_out_dtype(ref_dtype)
    if out_dtype != ref_dtype:
        logger.info(f'输出数据类型使用 Float32 以容纳统一NoData {TARGET_NODATA}（首个输入类型为 {dtype_name(ref_dtype)}）')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        args.output,
        ref_w,
        ref_h,
        total_bands,
        out_dtype,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
    )
    out_ds.SetProjection(ref_prj)
    out_ds.SetGeoTransform(ref_gt)

    # 记录映射
    mapping_lines: List[str] = []
    out_band_idx = 1

    for fp in files:
        ds = gdal.Open(fp, gdal.GA_ReadOnly)
        w, h, b, prj, gt, dt = read_info(ds)
        if dt != out_dtype:
            logger.warning(f'数据类型不一致: {os.path.basename(fp)} {dtype_name(dt)} -> 转为 {dtype_name(out_dtype)}')
        for bidx in range(1, b + 1):
            band = ds.GetRasterBand(bidx)
            arr = band.ReadAsArray()
            if arr is None:
                ds = None
                out_ds = None
                logger.error(f'读取失败: {fp} band {bidx}')
                return
            # 将源NoData映射为统一NoData
            src_nd = band.GetNoDataValue()
            arr = map_src_nodata_to_target(arr, src_nd)
            # 数据类型统一
            if band.DataType != out_dtype:
                arr = arr.astype(gdal_array_type_to_numpy(out_dtype))
            out_band = out_ds.GetRasterBand(out_band_idx)
            out_band.WriteArray(arr)
            # 统一设置目标NoData
            out_band.SetNoDataValue(TARGET_NODATA)
            mapping_lines.append(f'Band {out_band_idx} <- {os.path.basename(fp)} [band {bidx}]')
            out_band_idx += 1
        ds = None

    out_ds.FlushCache()
    out_ds = None

    # 保存映射
    if args.save_map:
        map_path = args.output + '.bands.txt'
        with open(map_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(mapping_lines) + '\n')
        logger.info(f'映射已保存: {map_path}')

    # 控制台输出映射（简短）
    logger.info(f'总波段: {total_bands} -> {os.path.basename(args.output)}; 统一NoData={TARGET_NODATA}')
    if len(mapping_lines) <= 20:
        for line in mapping_lines:
            logger.info(line)
    else:
        for line in mapping_lines[:10]:
            logger.info(line)
        logger.info('...')
        for line in mapping_lines[-10:]:
            logger.info(line)

if __name__ == '__main__':
    main()