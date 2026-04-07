#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10× 下采样 DEM 与 Distance Map（按文件名自动识别）：
- 名称含 "distance"（不区分大小写） → 使用 Resampling.min
- 名称含 "dem"       （不区分大小写） → 使用 average/median/bilinear（默认 average）
- 其它文件跳过
- 仅单波段处理（用 --band 指定），递归可选

说明：
- 采用“按因子缩放网格”的方式：新像元大小 = 原像元大小 × factor（默认 10）
- 保持原始影像的左上角原点、范围对齐（可能出现最后一行/列部分覆盖，不额外扩边）
- 对窗口有效像元比例 < 阈值（如 0.2）的目标像元，写回 NoData，避免硬凑值
"""

import argparse
from pathlib import Path
import math
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

def parse_args():
    ap = argparse.ArgumentParser("10× downsample for DEM & Distance rasters")
    ap.add_argument("--input-folder", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/distance_map_interpolated_margin_nodata', help="输入目录")
    ap.add_argument("--output-folder", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/distance_map_interpolated_margin_nodata_new', help="输出目录")
    ap.add_argument("--recursive", action="store_true", help="递归遍历子目录")
    ap.add_argument("--pattern", default="*.tif", help="匹配模式（默认 *.tif）")
    ap.add_argument("--band", type=int, default=1, help="处理的波段（默认 1）")

    ap.add_argument("--factor", type=int, default=2, help="下采样因子（默认 10）")

    # DEM 选项
    ap.add_argument("--dem-method", choices=["average", "median", "bilinear"],
                    default="average", help="DEM 下采样方法（默认 average）")

    # NoData 与有效比例
    ap.add_argument("--src-nodata", type=float, default=-32768, help="源 NoData（如 -32768）")
    ap.add_argument("--dst-nodata", type=float, default=-32768, help="目标 NoData（如 -32768）")
    ap.add_argument("--valid-fraction", type=float, default=0.2,
                    help="窗口有效像元比例阈值（默认 0.2）")

    # 输出写盘
    ap.add_argument("--compress", default="ZSTD", choices=["NONE", "LZW", "DEFLATE", "ZSTD"],
                    help="压缩（默认 ZSTD）")
    ap.add_argument("--blocksize", type=int, default=512, help="瓦片块大小（默认 512）")
    ap.add_argument("--overwrite", action="store_true", help="已存在则覆盖")
    return ap.parse_args()

def pick_mode_from_name(name_lower: str):
    if "distance" in name_lower:
        return "distance"
    if "dem" in name_lower:
        return "dem"
    return None

def resampling_from_method(method: str) -> Resampling:
    if method == "average":  return Resampling.average
    if method == "median":   return Resampling.med
    if method == "bilinear": return Resampling.bilinear
    raise ValueError(method)

def build_x10_template(src, factor: int):
    """基于原始 transform 构造 10×（或其它因子）的新网格，保持左上角原点与范围对齐。"""
    t = src.transform
    # 原像元大小（注意 y 像元高通常为负）
    xres = abs(t.a)
    yres = abs(t.e)

    new_xres = xres * factor
    new_yres = yres * factor

    # 新尺寸（向上取整可覆盖全范围；这里用向上取整避免丢最后的行/列）
    new_w = math.ceil(src.width  / factor)
    new_h = math.ceil(src.height / factor)

    # 左上角原点不变；像元高保持负号（北向上）
    new_transform = Affine(new_xres, 0.0, t.c, 0.0, -new_yres, t.f)
    return new_transform, new_w, new_h, src.crs

def write_tif(path, arr, profile, compress="ZSTD", blocksize=512, nodata=None, overwrite=False):
    meta = profile.copy()
    meta.update(
        dtype=str(arr.dtype),
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        tiled=True,
        blockxsize=blocksize,
        blockysize=blocksize
    )
    if compress != "NONE":
        meta.update(compress=compress)
        if compress == "ZSTD":
            meta.update(zstd_level=9)
    if nodata is not None:
        meta.update(nodata=nodata)

    p = Path(path)
    if p.exists():
        if overwrite:
            p.unlink()
        else:
            raise FileExistsError(f"输出已存在（未指定 --overwrite）：{p}")

    with rasterio.open(p, "w", **meta) as dst:
        dst.write(arr, 1)

def downsample_by_factor(src, src_arr, src_nodata, tmpl_transform, tmpl_w, tmpl_h, mode, dem_method, valid_fraction, dst_nodata):
    """重投影 + 聚合：distance→min；DEM→average/median/bilinear；并按有效比例阈值筛除。"""
    dst_shape = (tmpl_h, tmpl_w)

    # 有效比例图：将有效掩膜(1/0) average 到目标网格，得到每个粗像元内有效比例
    valid_src = (src_arr != src_nodata) & (~np.isnan(src_arr))
    valid_src = valid_src.astype(np.uint8)
    valid_ratio = np.zeros(dst_shape, dtype=np.float32)

    reproject(
        source=valid_src,
        destination=valid_ratio,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=tmpl_transform,
        dst_crs=src.crs,
        resampling=Resampling.average,
        src_nodata=0,
        dst_nodata=0
    )

    # 主聚合
    out = np.full(dst_shape, dst_nodata, dtype=np.float32)
    if mode == "distance":
        rs = Resampling.min
    else:
        rs = resampling_from_method(dem_method)

    reproject(
        source=src_arr,
        destination=out,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=tmpl_transform,
        dst_crs=src.crs,
        resampling=rs,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata
    )

    # 有效比例过滤
    out[valid_ratio < float(valid_fraction)] = dst_nodata
    return out

def main():
    args = parse_args()
    in_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.rglob(args.pattern) if args.recursive else in_dir.glob(args.pattern))
    if not files:
        print("未找到匹配的 TIFF。"); return

    total = ok = skip = 0
    for f in files:
        mode = pick_mode_from_name(f.name.lower())
        if mode is None:
            skip += 1
            continue
        total += 1

        try:
            with rasterio.open(f) as src:
                arr = src.read(args.band).astype(np.float32, copy=False)

                # 统一源 NoData
                if src.nodata is not None:
                    arr[np.isclose(arr, src.nodata, equal_nan=True)] = args.src_nodata
                arr[np.isnan(arr)] = args.src_nodata
                src_nodata = args.src_nodata

                # 构造 10× 网格模板
                tform, w, h, crs = build_x10_template(src, args.factor)

                out_arr = downsample_by_factor(
                    src=src,
                    src_arr=arr,
                    src_nodata=src_nodata,
                    tmpl_transform=tform,
                    tmpl_w=w,
                    tmpl_h=h,
                    mode=mode,
                    dem_method=args.dem_method,
                    valid_fraction=args.valid_fraction,
                    dst_nodata=args.dst_nodata
                )

                # 输出路径（保持相对层级）
                rel = f.relative_to(in_dir) if f.is_relative_to(in_dir) else Path(f.name)
                out_path = out_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)

                profile = src.profile.copy()
                profile.update(
                    height=h, width=w, transform=tform, crs=crs,
                    nodata=args.dst_nodata, dtype="float32", count=1
                )
                write_tif(
                    out_path, out_arr, profile,
                    compress=args.compress, blocksize=args.blocksize,
                    nodata=args.dst_nodata, overwrite=args.overwrite
                )
                ok += 1
                print(f"[OK] {mode:<8} ×{args.factor} -> {out_path}")
        except Exception as e:
            print(f"[失败] {f} | {e}")

    print(f"完成。处理：{total} | 成功：{ok} | 跳过（非 dem/distance）：{skip} | 总计扫描：{len(files)}")

if __name__ == "__main__":
    main()
