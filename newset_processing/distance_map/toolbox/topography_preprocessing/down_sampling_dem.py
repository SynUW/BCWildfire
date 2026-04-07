#!/usr/bin/env python3
"""
下采样并重算地形：
- 输入为7波段GeoTIFF：前4个波段任意（第4个为DEM），5/6/7为旧的slope/aspect/hillshade（将被替换）。
- 目标：
  1）将前4个波段使用双线性内插法按比例下采样到更粗栅格；
  2）使用下采样后的第4波段DEM，重新计算slope、aspect、hillshade，作为新的5/6/7波段；
  3）将得到的7个波段进行log变换（参考提供代码）：
     - signed模式：out = sign(x) * log1p(|x|)
     - 非signed：out = log1p(max(x, 0))
     - 非有限和NoData保持为NoData
  4）输出新7波段TIF，投影/范围跟随输入，像元大小按比例缩放。

用法示例：
python3 down_sampling_dem.py \
  --src /path/to/in_7bands.tif \
  --dst /path/to/out_7bands_downsampled_log.tif \
  --scale 2.0 \
  --target-nodata -9999 \
  --signed-log

说明：scale>1 表示像元变大（分辨率变粗）
"""

import os
import logging
import argparse
from typing import Tuple, Optional

import numpy as np
from osgeo import gdal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('down_sampling_dem')


def read_info(ds) -> Tuple[int, int, int, str, tuple]:
    w = ds.RasterXSize
    h = ds.RasterYSize
    b = ds.RasterCount
    prj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    return w, h, b, prj, gt


def compute_bounds_from_gt(gt: tuple, width: int, height: int) -> Tuple[float, float, float, float]:
    minx = gt[0]
    maxy = gt[3]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    miny = gt[3] + width * gt[4] + height * gt[5]
    return (min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))


def warp_single_band(src_ds, band_index: int, out_w: int, out_h: int, out_bounds, out_srs, src_nodata: Optional[float], dst_nodata: Optional[float]) -> np.ndarray:
    """将指定波段按双线性重采样到目标尺寸，返回数组。"""
    vrt_path = f"/vsimem/sel_band_{id(src_ds)}_{band_index}.vrt"
    gdal.Translate(vrt_path, src_ds, bandList=[band_index], format='VRT')
    out_path = f"/vsimem/warp_band_{id(src_ds)}_{band_index}.tif"
    warp_opts = gdal.WarpOptions(
        dstSRS=out_srs,
        outputBounds=out_bounds,
        width=out_w,
        height=out_h,
        resampleAlg=gdal.GRA_Bilinear,
        srcNodata=src_nodata,
        dstNodata=dst_nodata,
        format='GTiff'
    )
    out_ds = gdal.Warp(out_path, vrt_path, options=warp_opts)
    gdal.Unlink(vrt_path)
    if out_ds is None:
        raise RuntimeError('Warp失败')
    arr = out_ds.GetRasterBand(1).ReadAsArray()
    out_ds = None
    gdal.Unlink(out_path)
    return arr


def demprocess_to_array(src_arr: np.ndarray, prj: str, gt: tuple, mode: str, options: list) -> np.ndarray:
    """将数组写入临时GTiff后用DEMProcessing计算，返回数组。"""
    vsipath = "/vsimem/dem_base.tif"
    h, w = src_arr.shape
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(vsipath, w, h, 1, gdal.GDT_Float32)
    ds.SetProjection(prj)
    ds.SetGeoTransform(gt)
    band = ds.GetRasterBand(1)
    band.WriteArray(src_arr)
    # 不设置NoData为NaN，避免GDAL绑定类型错误；DEMProcessing不依赖这里的NoData
    ds.FlushCache()
    out_path = f"/vsimem/demproc_{mode}.tif"
    gdal.DEMProcessing(out_path, ds, mode, options=options)
    ds = None
    out_ds = gdal.Open(out_path, gdal.GA_ReadOnly)
    if out_ds is None:
        gdal.Unlink(vsipath)
        raise RuntimeError(f"DEMProcessing失败: {mode}")
    arr = out_ds.GetRasterBand(1).ReadAsArray()
    out_ds = None
    gdal.Unlink(vsipath)
    gdal.Unlink(out_path)
    return arr.astype(np.float32)


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def main():
    ap = argparse.ArgumentParser(description='7波段下采样并重算5/6/7坡度/坡向/晕渲，最终对7波段做log（支持signed模式）')
    ap.add_argument('--src', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map/Topo_and_distance_map_stack_nodata_homogenized.tif', help='输入7波段TIF，band4为DEM')
    ap.add_argument('--dst', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map/Topo_and_distance_map_stack_nodata_homogenized_downsampled10x_log.tif', help='输出TIF路径')
    ap.add_argument('--scale', type=float, default=10.0, help='下采样比例（>1 表示像元变大）')
    ap.add_argument('--target-nodata', type=float, default=-9999.0, help='输出NoData值，默认-9999')
    ap.add_argument('--signed-log', action='store_true', help='使用 signed log1p：sign(x)*log1p(|x|)')
    args = ap.parse_args()

    ds = gdal.Open(args.src, gdal.GA_ReadOnly)
    if ds is None:
        logger.error(f'无法打开输入: {args.src}')
        return
    width, height, bands, prj, gt = read_info(ds)
    if bands != 7:
        logger.error(f'输入必须为7波段，实际: {bands}')
        return
    if args.scale <= 1.0:
        logger.error('scale 必须 > 1.0 (下采样)')
        return

    out_w = max(1, int(width // args.scale))
    out_h = max(1, int(height // args.scale))
    bounds = compute_bounds_from_gt(gt, width, height)
    out_gt = list(gt)
    out_gt[1] = gt[1] * args.scale
    out_gt[5] = gt[5] * args.scale
    out_gt = tuple(out_gt)

    logger.info(f'下采样尺寸: {width}x{height} -> {out_w}x{out_h}')

    src_nd = [ds.GetRasterBand(i).GetNoDataValue() for i in range(1, 8)]

    down_bands = []
    for bi in range(1, 5):
        arr = warp_single_band(
            ds, bi, out_w, out_h, bounds, prj, src_nd[bi - 1], args.target_nodata
        )
        down_bands.append(arr.astype(np.float32))

    dem_ds_arr = down_bands[3]
    slope = demprocess_to_array(
        dem_ds_arr, prj, out_gt, 'slope', ['-compute_edges', '-p', '-of', 'GTiff']
    )
    aspect = demprocess_to_array(
        dem_ds_arr, prj, out_gt, 'aspect', ['-compute_edges', '-of', 'GTiff']
    )
    hillshade = demprocess_to_array(
        dem_ds_arr, prj, out_gt, 'hillshade', ['-compute_edges', '-az', '315', '-alt', '45', '-of', 'GTiff']
    )

    # 基于下采样DEM的有效掩膜
    nd = np.float32(args.target_nodata)
    dem_valid = np.isfinite(dem_ds_arr) & (dem_ds_arr != nd)

    # 在DEM无效处强制NoData
    slope[~dem_valid] = nd
    aspect[~dem_valid] = nd
    hillshade[~dem_valid] = nd

    # 规则：若DEM有效而aspect为无效（如NaN），避免写NoData，使用0代替
    invalid_aspect = ~np.isfinite(aspect)
    fix_mask = dem_valid & invalid_aspect
    if np.any(fix_mask):
        aspect = aspect.copy()
        aspect[fix_mask] = 0.0

    stack = [down_bands[0], down_bands[1], down_bands[2], dem_ds_arr, slope, aspect, hillshade]

    out_stack = []
    for band_arr in stack:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            out_arr = np.full(band_arr.shape, nd, dtype=np.float32)
            valid = np.isfinite(band_arr) & (band_arr != nd)
            if np.any(valid):
                if args.signed_log:
                    out_arr[valid] = signed_log1p(band_arr[valid].astype(np.float32))
                else:
                    nonneg = valid & (band_arr >= 0)
                    if np.any(nonneg):
                        out_arr[nonneg] = np.log1p(band_arr[nonneg].astype(np.float32))
        out_stack.append(out_arr)

    os.makedirs(os.path.dirname(args.dst) or '.', exist_ok=True)
    drv = gdal.GetDriverByName('GTiff')
    out_ds = drv.Create(args.dst, out_w, out_h, 7, gdal.GDT_Float32,
                        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
    out_ds.SetProjection(prj)
    out_ds.SetGeoTransform(out_gt)
    for i in range(7):
        rb = out_ds.GetRasterBand(i + 1)
        rb.SetNoDataValue(float(args.target_nodata))
        rb.WriteArray(out_stack[i])
    out_ds.FlushCache()
    out_ds = None
    ds = None
    logger.info(f'完成: {args.dst}')


if __name__ == '__main__':
    main()
