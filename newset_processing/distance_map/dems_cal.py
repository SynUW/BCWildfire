#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算坡度、坡向、坡度阴影
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine

def parse_args():
    ap = argparse.ArgumentParser("Units-safe slope/aspect/hillshade from DEM")
    ap.add_argument("--input-folder", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/distance_map_interpolated_margin_nodata', help="输入目录")
    ap.add_argument("--output-folder", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/distance_map_interpolated_margin_nodata', help="输出目录")
    ap.add_argument("--pattern", default="ASTER_GDEM_elevation_1km.tif")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument("--src-nodata", type=float, default=-32768)
    ap.add_argument("--dst-nodata", type=float, default=-32768)
    # 若 DEM 垂直单位不是米，可用这个把其换算到“米”（例如 cm→m 用 0.01；英尺→米用 0.3048）
    ap.add_argument("--z-unit-scale", type=float, default=1.0,
                    help="把 DEM 垂直单位缩放到米（默认1.0，表示本来就是米）")
    ap.add_argument("--slope-unit", choices=["degree","percent"], default="degree")
    ap.add_argument("--azimuth", type=float, default=315.0)
    ap.add_argument("--altitude", type=float, default=45.0)
    ap.add_argument("--compress", choices=["NONE","LZW","DEFLATE","ZSTD"], default="ZSTD")
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()

def is_dem(name: str) -> bool:
    n = name.lower()
    return ("dem" in n) and ("distance" not in n)

# 近似的“每度对应的米数”（随纬度变化）
def meters_per_degree_lat(lat_rad):
    # 常用球体近似，足以用于坡度尺度
    return (111132.954 - 559.822*np.cos(2*lat_rad) + 1.175*np.cos(4*lat_rad))

def meters_per_degree_lon(lat_rad):
    return (111412.84*np.cos(lat_rad) - 93.5*np.cos(3*lat_rad) + 0.118*np.cos(5*lat_rad))

def horn_gradients_unitsafe(dem, transform: Affine, nodata_val: float,
                            crs_is_geographic: bool, z_unit_scale: float):
    """
    计算单位安全的 dz/dx, dz/dy：
    - 若 crs 是地理坐标（度），按行估算“度→米”的像元大小（x、y分别换算），
      使得分母使用“米”而非“度”；
    - 若 crs 是投影坐标（米），直接用 transform 中的像元大小（米）。
    - 垂直单位通过 z_unit_scale 统一到米（例如 cm→m 用 0.01）。
    - 3×3 任一无效则输出 NaN（稍后再转 NoData）。
    """
    # Horn 核
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
    ky = np.array([[ 1,2,1],[ 0,0,0],[-1,-2,-1]], dtype=np.float64)

    h, w = dem.shape
    pad = 1
    dem_pad = np.pad(dem, pad, mode="edge")

    def shifted(arr, dy, dx):
        return arr[pad+dy:pad+dy+h, pad+dx:pad+dx+w]

    n00 = shifted(dem_pad,-1,-1); n01 = shifted(dem_pad,-1,0); n02 = shifted(dem_pad,-1,1)
    n10 = shifted(dem_pad, 0,-1); n11 = shifted(dem_pad, 0,0); n12 = shifted(dem_pad, 0,1)
    n20 = shifted(dem_pad, 1,-1); n21 = shifted(dem_pad, 1,0); n22 = shifted(dem_pad, 1,1)

    valid = (dem != nodata_val) & (~np.isnan(dem))
    valid_pad = np.pad(valid, pad, mode="edge")
    v00 = shifted(valid_pad,-1,-1); v01 = shifted(valid_pad,-1,0); v02 = shifted(valid_pad,-1,1)
    v10 = shifted(valid_pad, 0,-1); v11 = shifted(valid_pad, 0,0); v12 = shifted(valid_pad, 0,1)
    v20 = shifted(valid_pad, 1,-1); v21 = shifted(valid_pad, 1,0); v22 = shifted(valid_pad, 1,1)
    window_valid = v00 & v01 & v02 & v10 & v11 & v12 & v20 & v21 & v22

    gx_num = (n00*kx[0,0] + n01*kx[0,1] + n02*kx[0,2] +
              n10*kx[1,0] + n11*kx[1,1] + n12*kx[1,2] +
              n20*kx[2,0] + n21*kx[2,1] + n22*kx[2,2])

    gy_num = (n00*ky[0,0] + n01*ky[0,1] + n02*ky[0,2] +
              n10*ky[1,0] + n11*ky[1,1] + n12*ky[1,2] +
              n20*ky[2,0] + n21*ky[2,1] + n22*ky[2,2])

    # 水平像元大小（米）
    if crs_is_geographic:
        # 从 affine 求每行中心的纬度（度→弧度）
        # 行 y 的中心纬度（度） = y0 + (row+0.5)*pixelHeight；注意 pixelHeight 常为负
        rows = np.arange(h, dtype=np.float64)
        # 影像坐标到地理：lat = f + (row+0.5)*e，e 通常为负；这里取中心纬度
        lat_deg_center = transform.f + (rows + 0.5) * transform.e
        lat_rad_center = np.deg2rad(lat_deg_center)

        m_per_deg_x = meters_per_degree_lon(lat_rad_center)      # shape (h,)
        m_per_deg_y = meters_per_degree_lat(lat_rad_center)      # shape (h,)

        xres_deg = abs(transform.a)
        yres_deg = abs(transform.e)

        xres_m_row = xres_deg * m_per_deg_x                      # shape (h,)
        yres_m_row = yres_deg * m_per_deg_y                      # shape (h,)
    else:
        xres_m_row = np.full(h, abs(transform.a), dtype=np.float64)
        yres_m_row = np.full(h, abs(transform.e), dtype=np.float64)

    # Horn 分母：8 * cellsize(米)；再乘以 z_unit_scale（把垂直单位换算到米）
    denom_x = (8.0 * xres_m_row * (1.0 / z_unit_scale))  # 实际上等价于：将“dz（垂直单位）”换算到米后再除以“米”
    denom_y = (8.0 * yres_m_row * (1.0 / z_unit_scale))

    # 按行广播除法
    gx = gx_num / denom_x[:, None]
    gy = gy_num / denom_y[:, None]

    gx = np.where(window_valid, gx, np.nan)
    gy = np.where(window_valid, gy, np.nan)
    return gx, gy

def slope_aspect_hillshade_unitsafe(dem, transform, nodata_val, crs_is_geographic,
                                    z_unit_scale, slope_unit, azimuth_deg, altitude_deg):
    gx, gy = horn_gradients_unitsafe(dem, transform, nodata_val, crs_is_geographic, z_unit_scale)

    slope_rad = np.arctan(np.hypot(gx, gy))
    if slope_unit == "degree":
        slope_out = np.degrees(slope_rad).astype(np.float32)
    else:
        slope_out = (np.tan(slope_rad) * 100.0).astype(np.float32)

    aspect_rad = np.arctan2(gy, -gx)
    aspect_rad = np.where(aspect_rad < 0, aspect_rad + 2*np.pi, aspect_rad)
    aspect_out = np.degrees(aspect_rad).astype(np.float32)

    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    zen = np.pi/2.0 - alt
    hill = 255.0 * (np.cos(zen)*np.cos(slope_rad) + np.sin(zen)*np.sin(slope_rad)*np.cos(az - aspect_rad))
    hill = np.clip(hill, 0.0, 255.0).astype(np.float32)

    nan_mask = ~np.isfinite(slope_out) | ~np.isfinite(aspect_out) | ~np.isfinite(hill)
    slope_out[nan_mask]  = np.float32(nodata_val)
    aspect_out[nan_mask] = np.float32(nodata_val)
    hill[nan_mask]       = np.float32(nodata_val)
    return slope_out, aspect_out, hill

def write_gtiff(path: Path, arr, tpl_profile, compress="ZSTD", blocksize=512, nodata=None, overwrite=False):
    prof = tpl_profile.copy()
    prof.update(dtype="float32", count=1, nodata=nodata, tiled=True,
                blockxsize=blocksize, blockysize=blocksize)
    if compress != "NONE":
        prof.update(compress=compress)
        if compress == "ZSTD":
            prof.update(zstd_level=9)
    if path.exists():
        if overwrite: path.unlink()
        else: raise FileExistsError(f"Exists: {path}")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)

def main():
    args = parse_args()
    in_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.rglob(args.pattern) if args.recursive else in_dir.glob(args.pattern))
    if not files:
        print("未找到匹配影像"); return

    total=ok=skip=0
    for fp in files:
        if not is_dem(fp.name):
            skip += 1; continue
        total += 1
        try:
            with rasterio.open(fp) as src:
                dem = src.read(args.band).astype(np.float64, copy=False)
                if src.nodata is not None:
                    dem[np.isclose(dem, src.nodata, equal_nan=True)] = args.src_nodata
                dem[np.isnan(dem)] = args.src_nodata

                crs_is_geo = (src.crs is not None and src.crs.is_geographic)
                slope, aspect, hill = slope_aspect_hillshade_unitsafe(
                    dem=dem,
                    transform=src.transform,
                    nodata_val=args.src_nodata,
                    crs_is_geographic=crs_is_geo,
                    z_unit_scale=args.z_unit_scale,
                    slope_unit=args.slope_unit,
                    azimuth_deg=args.azimuth,
                    altitude_deg=args.altitude
                )

                # 将内部 nodata 标记替换为目标 nodata
                slope[slope == args.src_nodata]   = args.dst_nodata
                aspect[aspect == args.src_nodata] = args.dst_nodata
                hill[hill == args.src_nodata]     = args.dst_nodata

                rel = fp.relative_to(in_dir) if fp.is_relative_to(in_dir) else Path(fp.name)
                base = fp.stem
                prof = src.profile.copy(); prof.update(nodata=args.dst_nodata, dtype="float32")

                out_slope  = out_dir / rel.parent / f"{base}_slope_{args.slope_unit}.tif"
                out_aspect = out_dir / rel.parent / f"{base}_aspect.tif"
                out_hill   = out_dir / rel.parent / f"{base}_hillshade.tif"
                out_slope.parent.mkdir(parents=True, exist_ok=True)

                write_gtiff(out_slope,  slope,  prof, args.compress, args.blocksize, args.dst_nodata, args.overwrite)
                write_gtiff(out_aspect, aspect, prof, args.compress, args.blocksize, args.dst_nodata, args.overwrite)
                write_gtiff(out_hill,   hill,   prof, args.compress, args.blocksize, args.dst_nodata, args.overwrite)
                ok += 1
                print(f"[OK] {fp} -> slope/aspect/hillshade")
        except Exception as e:
            print(f"[失败] {fp} | {e}")

    print(f"完成。处理：{ok} | 跳过：{skip} | 总计：{total}")

if __name__ == "__main__":
    main()
