#!/usr/bin/env python3
"""
从MODIS多光谱反射率计算NDVI与EVI并输出为两波段GeoTIFF：
- 输入影像包含4个波段（MODIS Band1,2,3,7），命名形如 yyyy_mm_dd.tif。
- MODIS波段映射：Band1=RED(620-670nm), Band2=NIR(841-876nm), Band3=BLUE(459-479nm), Band4=SWIR(2105-2155nm)
- 输出两波段：Band1=NDVI，Band2=EVI。保存至新目录，文件名与输入一致。
- 可配置反射率缩放与偏移：out = in * scale_factor + offset（默认scale_factor=0.0001，offset=0）。
- NoData像元不参与计算；遇到分母为0的像元结果置为NoData。
- NDVI和EVI值会被clip到[0,1]范围内，确保植被指数的合理性。
- 投影、分辨率与尺寸不变；LZW压缩、瓦片化、支持BIGTIFF；分块并行处理。

NDVI = (NIR - RED) / (NIR + RED)
EVI  = G*(NIR - RED) / (NIR + C1*RED - C2*BLUE + L)
默认: G=2.5, L=1.0, C1=6.0, C2=7.5

示例：
python3 compute_ndvi_evi.py \
  --input-dir /path/to/reflectance \
  --output-dir /path/to/indices \
  --scale-factor 0.0001 --offset 0.0 \
  --max-workers 16 --block-size 1024
"""

import os
import glob
import logging
import argparse
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from osgeo import gdal
import re

# 初始化GDAL异常处理
gdal.UseExceptions()

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('compute_ndvi_evi')

def list_tifs(folder: str, pattern: str) -> List[str]:
    # 匹配 yyyy_mm_dd.tif 格式的文件
    all_files = glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.TIF"))
    matching_files = []
    for f in all_files:
        basename = os.path.basename(f)
        # 检查是否匹配 yyyy_mm_dd.tif 格式
        if re.match(r'^\d{4}_\d{1,2}_\d{1,2}\.tif$', basename, re.IGNORECASE):
            matching_files.append(f)
    return sorted(matching_files)


def read_info(ds) -> Tuple[int, int, int, str, tuple, int]:
    w = ds.RasterXSize
    h = ds.RasterYSize
    b = ds.RasterCount
    prj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    dtype = ds.GetRasterBand(1).DataType
    return w, h, b, prj, gt, dtype


def process_one(
    src_path: str,
    dst_path: str,
    scale_factor: float,
    offset: float,
    evi_G: float,
    evi_L: float,
    evi_C1: float,
    evi_C2: float,
    block_size: int,
    default_nodata: float,
) -> Tuple[str, bool]:
    try:
        # 若目标已存在则跳过
        if os.path.exists(dst_path):
            return src_path, True
        ds = gdal.Open(src_path, gdal.GA_ReadOnly)
        if ds is None:
            logger.error(f'无法打开文件: {src_path}')
            return src_path, False
        width, height, bands, projection, geotransform, _ = read_info(ds)
        if bands < 4:
            logger.error(f'波段数不足(需要>=4)，跳过: {src_path}')
            return src_path, False

        # 波段映射（MODIS）：Band1=RED, Band2=NIR, Band3=BLUE, Band4=SWIR
        # 对于NDVI/EVI计算：RED=Band1, NIR=Band2, BLUE=Band3
        band_red = ds.GetRasterBand(1)
        band_nir = ds.GetRasterBand(2)
        band_blue = ds.GetRasterBand(3)

        nd_red = band_red.GetNoDataValue()
        nd_nir = band_nir.GetNoDataValue()
        nd_blue = band_blue.GetNoDataValue()

        # 输出数据集（2波段）
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            dst_path,
            width,
            height,
            2,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
        )
        out_ds.SetProjection(projection)
        out_ds.SetGeoTransform(geotransform)

        nodata_out = default_nodata
        out_ds.GetRasterBand(1).SetNoDataValue(nodata_out)  # NDVI
        out_ds.GetRasterBand(2).SetNoDataValue(nodata_out)  # EVI

        for y in range(0, height, block_size):
            rows = min(block_size, height - y)
            for x in range(0, width, block_size):
                cols = min(block_size, width - x)
                r = band_red.ReadAsArray(x, y, cols, rows)
                n = band_nir.ReadAsArray(x, y, cols, rows)
                b = band_blue.ReadAsArray(x, y, cols, rows)
                if r is None or n is None or b is None:
                    continue

                # 有效掩膜（输入NoData不参与计算）
                valid = np.ones(r.shape, dtype=bool)
                if nd_red is not None:
                    valid &= (r != nd_red)
                if nd_nir is not None:
                    valid &= (n != nd_nir)
                if nd_blue is not None:
                    valid &= (b != nd_blue)

                # 缩放+偏移到物理反射率
                rf = r.astype(np.float32) * scale_factor + offset
                nf = n.astype(np.float32) * scale_factor + offset
                bf = b.astype(np.float32) * scale_factor + offset

                # NDVI 与 EVI 计算，遇到溢出/无效/分母为0均保持为NoData
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    # NDVI
                    ndvi = np.full(r.shape, nodata_out, dtype=np.float32)
                    num_ndvi = (nf - rf).astype(np.float64)
                    denom_ndvi = (nf + rf).astype(np.float64)
                    mask_ndvi = valid & (denom_ndvi != 0) & np.isfinite(num_ndvi) & np.isfinite(denom_ndvi)
                    if np.any(mask_ndvi):
                        ndvi_val = (num_ndvi[mask_ndvi] / denom_ndvi[mask_ndvi]).astype(np.float32)
                        # 仅写入有限值
                        finite_mask = np.isfinite(ndvi_val)
                        ndvi[mask_ndvi] = ndvi[mask_ndvi]  # no-op to keep shape
                        ndvi_idx = np.where(mask_ndvi)
                        # 对NDVI进行clip到[0,1]范围
                        ndvi_val_clipped = np.clip(ndvi_val[finite_mask], 0.0, 1.0)
                        ndvi[ndvi_idx[0][finite_mask], ndvi_idx[1][finite_mask]] = ndvi_val_clipped

                    # EVI
                    evi = np.full(r.shape, nodata_out, dtype=np.float32)
                    num_evi = (nf - rf).astype(np.float64)
                    denom_evi = (nf.astype(np.float64) + evi_C1 * rf.astype(np.float64) - evi_C2 * bf.astype(np.float64) + evi_L)
                    mask_evi = valid & (denom_evi != 0) & np.isfinite(num_evi) & np.isfinite(denom_evi)
                    if np.any(mask_evi):
                        evi_val = (evi_G * num_evi[mask_evi] / denom_evi[mask_evi]).astype(np.float32)
                        finite_mask_evi = np.isfinite(evi_val)
                        evi[mask_evi] = evi[mask_evi]
                        evi_idx = np.where(mask_evi)
                        # 对EVI进行clip到[0,1]范围
                        evi_val_clipped = np.clip(evi_val[finite_mask_evi], 0.0, 1.0)
                        evi[evi_idx[0][finite_mask_evi], evi_idx[1][finite_mask_evi]] = evi_val_clipped

                out_ds.GetRasterBand(1).WriteArray(ndvi, xoff=x, yoff=y)
                out_ds.GetRasterBand(2).WriteArray(evi, xoff=x, yoff=y)

        out_ds.FlushCache()
        ds = None
        out_ds = None
        return src_path, True
    except Exception as e:
        logger.error(f'处理失败 {src_path}: {e}')
        return src_path, False


def build_output_name(src_basename: str) -> str:
    # 直接使用输入文件名，保持一致性
    return src_basename


def main():
    ap = argparse.ArgumentParser(description='从MODIS多光谱反射率计算NDVI与EVI并输出为两波段GeoTIFF')
    ap.add_argument('--input-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/MCD09GA_b1237_mosaic_withQA_QAapplied_filled_downsampled', help='输入目录（含yyyy_mm_dd.tif格式文件）')
    ap.add_argument('--output-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/NDVI_EVI_withQA', help='输出目录（会创建）')
    ap.add_argument('--pattern', default='*.tif', help='输入匹配模式，默认为yyyy_mm_dd.tif格式')
    ap.add_argument('--scale-factor', type=float, default=0.0001, help='反射率缩放因子（原值*scale_factor+offset）')
    ap.add_argument('--offset', type=float, default=0, help='反射率偏移量')
    ap.add_argument('--default-nodata', type=float, default=-32768, help='输出NoData值')
    ap.add_argument('--evi-G', type=float, default=2.5)
    ap.add_argument('--evi-L', type=float, default=1.0)
    ap.add_argument('--evi-C1', type=float, default=6.0)
    ap.add_argument('--evi-C2', type=float, default=7.5)
    ap.add_argument('--block-size', type=int, default=1024, help='按块处理大小（像素）')
    ap.add_argument('--max-workers', type=int, default=8, help='并行进程数')

    args = ap.parse_args()

    tifs = list_tifs(args.input_dir, args.pattern)
    if not tifs:
        logger.error(f'输入目录无匹配TIF: {args.input_dir}/{args.pattern}')
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 预构建任务列表（断点续算：若输出已存在则跳过）
    tasks: List[Tuple[str, str]] = []
    skipped = 0
    for src in tifs:
        base = os.path.basename(src)
        out_name = build_output_name(base)
        dst = os.path.join(args.output_dir, out_name)
        if os.path.exists(dst):
            skipped += 1
            continue
        tasks.append((src, dst))

    total_all = len(tifs)
    total_run = len(tasks)
    done = 0
    ok_cnt = 0

    def progress():
        print(f"\r进度 {done}/{total_run} 成功:{ok_cnt} 跳过:{skipped}/{total_all}", end='', flush=True)

    if total_run == 0:
        logger.info(f'全部已存在，跳过 {total_all} 个文件')
        return

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = []
        for src, dst in tasks:
            futs.append(ex.submit(
                process_one,
                src,
                dst,
                args.scale_factor,
                args.offset,
                args.evi_G,
                args.evi_L,
                args.evi_C1,
                args.evi_C2,
                args.block_size,
                args.default_nodata,
            ))
        for f in as_completed(futs):
            _, ok = f.result()
            done += 1
            if ok:
                ok_cnt += 1
            progress()

    print()
    logger.info(f'完成: 新计算 {ok_cnt}/{total_run}，已跳过 {skipped}/{total_all} -> {args.output_dir}')


if __name__ == '__main__':
    main()