#!/usr/bin/env python3
"""
从DEM目录计算地形因子并与原始DEM拼接输出多波段：
- 输入：单波段DEM（支持任意GTiff），目录批处理；
- 计算：Slope(度)、Aspect(度)、Hillshade（0~255）
- 输出：4波段GeoTIFF：Band1=DEM，Band2=Slope，Band3=Aspect，Band4=Hillshade
- 外部掩膜：严格按绝对像素位置对齐（不使用地理参考、不做任何插值或重采样）。要求掩膜影像与DEM尺寸完全一致，否则报错。
- 投影/分辨率/地理范围保持一致；为所有波段设置相同NoData（沿用DEM的NoData或--default-nodata）
- 使用GDAL DEMProcessing计算，LZW压缩、瓦片化、支持BIGTIFF；并行处理
"""

import os
import glob
import logging
import argparse
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from osgeo import gdal, osr

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dem_to_terrain')

def list_tifs(folder: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, pattern)))


def read_info(ds) -> Tuple[int, int, int, str, tuple, int]:
    w = ds.RasterXSize
    h = ds.RasterYSize
    b = ds.RasterCount
    prj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    dtype = ds.GetRasterBand(1).DataType
    return w, h, b, prj, gt, dtype


def demprocess_to_array(src_ds, mode: str, options: List[str]) -> np.ndarray:
    vsipath = f"/vsimem/demproc_{mode}_{id(src_ds)}.tif"
    gdal.DEMProcessing(vsipath, src_ds, mode, options=options)
    out_ds = gdal.Open(vsipath, gdal.GA_ReadOnly)
    if out_ds is None:
        raise RuntimeError(f"DEMProcessing失败: {mode}")
    arr = out_ds.GetRasterBand(1).ReadAsArray()
    out_ds = None
    gdal.Unlink(vsipath)
    return arr


def pick_mask_source(mask_dir: Optional[str], mask_pattern: str, mask_file: Optional[str]) -> Optional[str]:
    if mask_file:
        if os.path.isfile(mask_file):
            return mask_file
        else:
            raise FileNotFoundError(f"指定的掩膜文件不存在: {mask_file}")
    if mask_dir:
        cands = list_tifs(mask_dir, mask_pattern)
        if cands:
            logger.info(f"使用掩膜参考: {cands[0]}")
            return cands[0]
        else:
            raise FileNotFoundError(f"掩膜目录无匹配TIF: {mask_dir}/{mask_pattern}")
    return None


def read_mask_array_strict(mask_src_path: str, dem_width: int, dem_height: int) -> np.ndarray:
    """读取掩膜数组，严格要求与DEM尺寸一致；不做任何插值或重采样。"""
    mds = gdal.Open(mask_src_path, gdal.GA_ReadOnly)
    if mds is None:
        raise FileNotFoundError(f"无法打开掩膜源: {mask_src_path}")
    mw, mh = mds.RasterXSize, mds.RasterYSize
    if (mw, mh) != (dem_width, dem_height):
        mds = None
        raise ValueError(f"掩膜尺寸({mw}x{mh})与DEM尺寸({dem_width}x{dem_height})不一致，且禁止重采样/插值。")
    marr = mds.GetRasterBand(1).ReadAsArray()
    mnd = mds.GetRasterBand(1).GetNoDataValue()
    mds = None
    return marr, mnd


def is_geographic_wkt(wkt: str) -> bool:
    try:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        return bool(srs.IsGeographic())
    except Exception:
        return False


def compute_center_lat_from_gt(gt: tuple, width: int, height: int) -> float:
    center_x_pix = width / 2.0
    center_y_pix = height / 2.0
    center_y = gt[3] + center_x_pix * gt[4] + center_y_pix * gt[5]
    return float(center_y)


def compute_auto_z_factor(projection: str, gt: tuple, width: int, height: int) -> float:
    if not is_geographic_wkt(projection):
        return 1.0
    lat = compute_center_lat_from_gt(gt, width, height)
    # 粗略按经向米/度（随纬度变化）近似：111320*cos(lat)
    meters_per_degree = 111320.0 * np.cos(np.deg2rad(lat))
    # 兜底：极端情况下避免为0
    if meters_per_degree <= 0 or not np.isfinite(meters_per_degree):
        meters_per_degree = 111320.0
    return float(meters_per_degree)


def process_one(
    src_path: str,
    dst_path: str,
    azimuth: float,
    altitude: float,
    z_factor: float,
    default_nodata: float,
    mask_src: Optional[str],
    mask_nodata_arg: Optional[float],
    mask_zero_invalid: bool,
) -> Tuple[str, bool]:
    try:
        ds = gdal.Open(src_path, gdal.GA_ReadOnly)
        if ds is None:
            logger.error(f"无法打开文件: {src_path}")
            return src_path, False
        width, height, bands, projection, geotransform, dtype = read_info(ds)
        if bands < 1:
            logger.error(f"波段数不足: {src_path}")
            return src_path, False
        band1 = ds.GetRasterBand(1)
        dem_nodata = band1.GetNoDataValue()
        if dem_nodata is None:
            dem_nodata = default_nodata

        # 自动z因子：若为地理坐标（度），按BC地区中心纬度近似修正水平单位
        z_auto = compute_auto_z_factor(projection, geotransform, width, height)
        z_use = z_factor if (z_factor and z_factor > 0) else z_auto

        slope = demprocess_to_array(
            ds,
            'slope',
            ['-compute_edges', '-s', str(z_use), '-of', 'GTiff', '-p']
        ).astype(np.float32)
        aspect = demprocess_to_array(
            ds,
            'aspect',
            ['-compute_edges', '-s', str(z_use), '-zero_for_flat', '-of', 'GTiff']
        ).astype(np.float32)
        hillshade = demprocess_to_array(
            ds,
            'hillshade',
            ['-compute_edges', '-s', str(z_use), '-az', str(azimuth), '-alt', str(altitude), '-of', 'GTiff']
        ).astype(np.float32)

        dem = band1.ReadAsArray().astype(np.float32)

        if dem_nodata is not None:
            dem_valid = (dem != dem_nodata)
        else:
            dem_valid = np.ones(dem.shape, dtype=bool)

        if mask_src:
            mask_arr, mask_nd_from_file = read_mask_array_strict(mask_src, width, height)
            mask_nd = mask_nodata_arg if (mask_nodata_arg is not None) else mask_nd_from_file
            if mask_nd is not None:
                valid_mask = (mask_arr != mask_nd)
            else:
                valid_mask = np.ones(mask_arr.shape, dtype=bool)
            if mask_zero_invalid:
                valid_mask &= (mask_arr != 0)
            dem_valid &= valid_mask

        dem[~dem_valid] = dem_nodata
        slope[~dem_valid] = dem_nodata
        aspect[~dem_valid] = dem_nodata
        hillshade[~dem_valid] = dem_nodata

        # 若DEM有效而aspect为无效（NaN/Inf），避免写NoData，使用0代替
        invalid_aspect = ~np.isfinite(aspect)
        fix_mask = dem_valid & invalid_aspect
        if np.any(fix_mask):
            aspect = aspect.copy()
            aspect[fix_mask] = 0.0

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            dst_path, width, height, 4, gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
        )
        out_ds.SetProjection(projection)
        out_ds.SetGeoTransform(geotransform)
        for b in range(1, 5):
            out_ds.GetRasterBand(b).SetNoDataValue(dem_nodata)

        out_ds.GetRasterBand(1).WriteArray(dem)
        out_ds.GetRasterBand(2).WriteArray(slope)
        out_ds.GetRasterBand(3).WriteArray(aspect)
        out_ds.GetRasterBand(4).WriteArray(hillshade)
        out_ds.FlushCache()

        ds = None
        out_ds = None
        return src_path, True
    except Exception as e:
        logger.error(f"处理失败 {src_path}: {e}")
        return src_path, False


def main():
    ap = argparse.ArgumentParser(description='计算DEM的Slope/Aspect/Hillshade并与DEM拼接输出为4波段GeoTIFF（支持外部掩膜，按像素绝对位置对齐，不插值）')
    ap.add_argument('--input-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked/before_final_products_and_previous_ill-processed_data/ASTER_GDEM_1km_aligned', help='输入DEM目录')
    ap.add_argument('--output-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map', help='输出目录（会创建）')
    ap.add_argument('--pattern', default='*.tif', help='输入匹配模式，默认*.tif')
    ap.add_argument('--azimuth', type=float, default=315.0, help='Hillshade方位角（度）')
    ap.add_argument('--altitude', type=float, default=35.0, help='Hillshade太阳高度角（度），BC省中高纬建议略低（如35°）')
    ap.add_argument('--z-factor', type=float, default=0.0, help='垂直比例因子（Z缩放）。<=0则自动按中心纬度为地理坐标系计算米/度比例')
    ap.add_argument('--default-nodata', type=float, default=-9999.0, help='当DEM无NoData时使用的NoData')
    ap.add_argument('--mask-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked/LULC_BCBoundingbox_resampled', help='掩膜参考目录（将从中挑选任意一张TIF）')
    ap.add_argument('--mask-pattern', default='*.tif', help='掩膜目录匹配模式')
    ap.add_argument('--mask-file', default=None, help='显式指定掩膜参考TIF（优先级高于目录）')
    ap.add_argument('--mask-nodata', type=float, default=None, help='掩膜参考的NoData值（当掩膜文件未定义NoData时可指定）')
    ap.add_argument('--mask-zero-invalid', action='store_true', help='当掩膜无NoData或未指定时，将像素值=0视为无效')
    ap.add_argument('--max-workers', type=int, default=16, help='并行进程数')

    args = ap.parse_args()

    tifs = list_tifs(args.input_dir, args.pattern)
    if not tifs:
        logger.error(f"输入目录无TIF: {args.input_dir}")
        return

    mask_src = pick_mask_source(args.mask_dir, args.mask_pattern, args.mask_file) if (args.mask_dir or args.mask_file) else None

    os.makedirs(args.output_dir, exist_ok=True)

    total = len(tifs)
    done = 0
    ok_cnt = 0

    def progress():
        print(f"\r进度 {done}/{total} 成功:{ok_cnt}", end='', flush=True)

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = []
        for src in tifs:
            dst = os.path.join(args.output_dir, os.path.basename(src))
            futs.append(ex.submit(
                process_one,
                src,
                dst,
                args.azimuth,
                args.altitude,
                args.z_factor,
                args.default_nodata,
                mask_src,
                args.mask_nodata,
                args.mask_zero_invalid,
            ))
        for f in as_completed(futs):
            _, ok = f.result()
            done += 1
            if ok:
                ok_cnt += 1
            progress()

    print()
    logger.info(f"完成: {ok_cnt}/{total} 个文件成功")


if __name__ == '__main__':
    main()