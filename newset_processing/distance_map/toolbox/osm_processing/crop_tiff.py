"""
make_distance_rasters.py生成的距离栅格偏大，需要裁剪到边界内
裁剪之后，再拼接
"""

import argparse
import os
import math
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import fiona
from shapely.geometry import shape as shp_shape, mapping as shp_mapping
from shapely.ops import transform as shp_transform, unary_union
from pyproj import CRS, Transformer


def load_and_reproject_geometries(shapefile_path: str, target_crs: CRS, dissolve: bool, buffer_dist: float):
    """
    读取 Shapefile，并将几何投影到目标 CRS（通常为栅格的 CRS）。
    可选：将多几何 dissolve 成单一几何；按目标 CRS 单位进行 buffer。
    返回 GeoJSON 风格的几何列表（可直接喂给 rasterio.mask.mask）。
    """
    with fiona.open(shapefile_path, "r") as src:
        if src.crs_wkt:
            source_crs = CRS.from_wkt(src.crs_wkt)
        elif src.crs:
            source_crs = CRS.from_user_input(src.crs)
        else:
            raise ValueError("Shapefile 没有 CRS，无法进行重投影。")

        need_reproject = not source_crs.equals(target_crs)
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True) if need_reproject else None

        geoms = []
        for feat in src:
            if not feat.get("geometry"):
                continue
            geom = shp_shape(feat["geometry"])
            if need_reproject:
                geom = shp_transform(transformer.transform, geom)
            if buffer_dist and not math.isclose(buffer_dist, 0.0):
                geom = geom.buffer(buffer_dist)
            geoms.append(geom)

    if not geoms:
        raise ValueError("Shapefile 没有有效几何。")

    if dissolve:
        geoms = [unary_union(geoms)]

    return [shp_mapping(g) for g in geoms]


def decide_nodata(profile):
    """优先使用源栅格 nodata；否则根据 dtype 给出合理缺省值。"""
    if profile.get("nodata") is not None:
        return profile["nodata"]

    dtype = profile.get("dtype", "float32")
    if np.issubdtype(np.dtype(dtype), np.floating):
        return -9999.0
    return 0


def clip_raster_by_shapefile(
    input_tif: str,
    shapefile_path: str,
    output_tif: str,
    invert: bool = False,
    crop: bool = True,
    all_touched: bool = False,
    dissolve: bool = False,
    buffer_dist: float = 0.0,
):
    """
    使用 Shapefile 裁剪 GeoTIFF 并输出新文件。
    - invert: 反选（保留多边形外部）
    - crop: 裁剪至几何外接边界（减小输出尺寸）
    - all_touched: 与像元接触即算（更“宽松”的掩模）
    - dissolve: 多几何溶解为单一几何后再裁剪
    - buffer_dist: 在目标 CRS（即栅格 CRS）单位下对几何 buffer 的距离
    """
    with rasterio.open(input_tif) as src:
        target_crs = src.crs
        if target_crs is None:
            raise ValueError("输入栅格没有 CRS，无法与 Shapefile 进行对齐。")

        shapes = load_and_reproject_geometries(
            shapefile_path=shapefile_path,
            target_crs=target_crs,
            dissolve=dissolve,
            buffer_dist=buffer_dist,
        )

        nodata_value = decide_nodata(src.profile)

        clipped, out_transform = rio_mask(
            src,
            shapes=shapes,
            all_touched=all_touched,
            invert=invert,
            crop=crop,
            nodata=nodata_value,
            filled=True,
        )

        profile = src.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=out_transform,
            nodata=nodata_value,
            compress="lzw",
            tiled=True,
        )

        # 对浮点数据设置 predictor=3，可提升压缩效果
        if np.issubdtype(np.dtype(profile["dtype"]), np.floating):
            profile["predictor"] = 3

        os.makedirs(os.path.dirname(os.path.abspath(output_tif)), exist_ok=True)
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(clipped)


def parse_args():
    ap = argparse.ArgumentParser(description="使用 Shapefile 裁剪 GeoTIFF 并输出新文件")
    ap.add_argument("--input", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/canada-distance/distance_water.tif', help="输入 GeoTIFF 路径")
    ap.add_argument("--shapefile", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_boundary/british_columnbia_no_crs_boundingBox.shp', help="输入 Shapefile 路径（.shp 或其他 fiona 支持格式）")
    ap.add_argument("--output", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/canada-distance/distance_water_cropped.tif', help="输出 GeoTIFF 路径")
    ap.add_argument("--invert", action="store_true", help="反选（保留几何外部区域）")
    ap.add_argument("--no-crop", dest="crop", action="store_false", help="不裁剪至几何外接边界")
    ap.add_argument("--all-touched", action="store_true", help="像元与几何边界接触即算入掩模")
    ap.add_argument("--dissolve", action="store_true", help="将多几何溶解为单一几何后裁剪")
    ap.add_argument("--buffer", type=float, default=0.0, help="在目标 CRS 单位下对几何进行 buffer 的距离（默认 0）")
    ap.set_defaults(crop=True)
    return ap.parse_args()


def main():
    args = parse_args()
    clip_raster_by_shapefile(
        input_tif=args.input,
        shapefile_path=args.shapefile,
        output_tif=args.output,
        invert=args.invert,
        crop=args.crop,
        all_touched=args.all_touched,
        dissolve=args.dissolve,
        buffer_dist=args.buffer,
    )
    print(f"✅ 已生成: {args.output}")


if __name__ == "__main__":
    main()