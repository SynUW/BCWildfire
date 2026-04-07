#!/usr/bin/env python3
"""
单文件NoData统一工具：
- 若任一波段某像素为NoData，则所有波段该像素统一写为目标NoData（默认-9999）。
- 输出TIF与输入尺寸/投影/仿射一致，波段数不变；每个输出波段NoData统一设为目标值。
- 若输入数据类型为无符号整型（Byte/UInt16/UInt32），输出提升为Float32以容纳-9999；否则沿用输入首波段类型。
- 分块处理，LZW压缩、瓦片、BIGTIFF。

示例：
python3 homo_nodata.py \
  --src /path/to/in.tif \
  --dst /path/to/out.tif \
  --target-nodata -9999 \
  --block-size 1024
"""

import os
import logging
import argparse
from typing import Tuple

import numpy as np
from osgeo import gdal

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('homo_nodata_single')


def read_info(ds) -> Tuple[int, int, int, str, tuple, int]:
    w = ds.RasterXSize
    h = ds.RasterYSize
    b = ds.RasterCount
    prj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    dtype = ds.GetRasterBand(1).DataType
    return w, h, b, prj, gt, dtype

_numpy_from_gdal = {
    gdal.GDT_Byte: np.uint8,
    gdal.GDT_UInt16: np.uint16,
    gdal.GDT_Int16: np.int16,
    gdal.GDT_UInt32: np.uint32,
    gdal.GDT_Int32: np.int32,
    gdal.GDT_Float32: np.float32,
    gdal.GDT_Float64: np.float64,
}


def npdtype(dt: int):
    return _numpy_from_gdal.get(dt, np.float32)


def is_unsigned(dt: int) -> bool:
    return dt in (gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_UInt32)


def choose_out_dtype(ref_dt: int) -> int:
    # 目标NoData为-9999时，若输入为无符号整型，则提升为Float32
    return gdal.GDT_Float32 if is_unsigned(ref_dt) else ref_dt


def band_nodata_mask(arr: np.ndarray, nd_val) -> np.ndarray:
    if nd_val is None:
        return np.zeros(arr.shape, dtype=bool)
    if isinstance(nd_val, float) and np.isnan(nd_val):
        return np.isnan(arr)
    return arr == nd_val


def homogenize_single(src_path: str, dst_path: str, target_nd: float, block_size: int) -> bool:
    ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    if ds is None:
        logger.error(f'无法打开: {src_path}')
        return False
    width, height, bands, prj, gt, ref_dt = read_info(ds)
    out_dt = choose_out_dtype(ref_dt)
    out_np = npdtype(out_dt)

    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        dst_path, width, height, bands, out_dt,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
    )
    out_ds.SetProjection(prj)
    out_ds.SetGeoTransform(gt)
    for b in range(1, bands + 1):
        out_ds.GetRasterBand(b).SetNoDataValue(float(target_nd))

    # 预读各波段NoData
    band_nd = [ds.GetRasterBand(b).GetNoDataValue() for b in range(1, bands + 1)]

    # 分块
    for y in range(0, height, block_size):
        rows = min(block_size, height - y)
        for x in range(0, width, block_size):
            cols = min(block_size, width - x)
            blocks = []
            masks = []
            for b in range(1, bands + 1):
                arr = ds.GetRasterBand(b).ReadAsArray(x, y, cols, rows)
                if arr is None:
                    continue
                blocks.append(arr)
                masks.append(band_nodata_mask(arr, band_nd[b - 1]))
            if not blocks:
                continue
            any_nd = np.zeros(blocks[0].shape, dtype=bool)
            for m in masks:
                any_nd |= m
            for idx in range(bands):
                arr = blocks[idx].astype(out_np, copy=False)
                if np.any(any_nd):
                    arr = arr.copy()
                    arr[any_nd] = target_nd
                out_ds.GetRasterBand(idx + 1).WriteArray(arr, xoff=x, yoff=y)

    out_ds.FlushCache()
    ds = None
    out_ds = None
    return True


def main():
    ap = argparse.ArgumentParser(description='单文件NoData统一：若任一波段某像素为NoData，则所有波段该像素统一设为目标NoData')
    ap.add_argument('--src', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map/Topo_and_distance_map_stack.tif', help='输入多波段TIF')
    ap.add_argument('--dst', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map/Topo_and_distance_map_stack_nodata_homogenized.tif', help='输出TIF路径')
    ap.add_argument('--target-nodata', type=float, default=-9999.0, help='统一的输出NoData值，默认-9999')
    ap.add_argument('--block-size', type=int, default=1024, help='按块处理大小（像素）')
    args = ap.parse_args()

    ok = homogenize_single(args.src, args.dst, args.target_nodata, args.block_size)
    if ok:
        logger.info(f'完成: {os.path.basename(args.src)} -> {args.dst} (NoData={args.target_nodata})')
    else:
        logger.error('处理失败')


if __name__ == '__main__':
    main()