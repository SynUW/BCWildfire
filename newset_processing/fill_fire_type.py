#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速 & 稳定：LULC 连通分量 + 种子重叠众数 赋 cluster
- 仅栅格计算（无 polygonize），省级数据可承受
- 允许区：AOI 内 ∧ (LULC != banned_class) ∧ LULC 非 nodata
- 对每个 LULC 类别的每个“连通分量”，用与 seed(来自 partial.shp 的 cluster) 的重叠众数决定 cluster
- 若某分量与 seed 无重叠，则保持 0（不赋值）
- 末尾：删小块（像素/面积阈值，可二选一或都用），再矢量化一次

依赖：numpy, rasterio, geopandas, scipy, shapely, fiona
pip install numpy rasterio geopandas scipy shapely fiona
"""

import argparse
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, shapes, geometry_mask
from scipy.ndimage import label as cc_label

def save_tif(out_path, ref, arr, dtype, nodata=0):
    prof = ref.profile.copy()
    prof.update(count=1, dtype=dtype, nodata=nodata, compress="deflate", tiled=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)

def rasterize_cluster(src, gdf, field="cluster", all_touched=False):
    if field not in gdf.columns:
        raise ValueError(f"partial.shp 缺少字段 {field}")
    return rasterize(
        shapes=list(zip(gdf.geometry, gdf[field].astype(int))),
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="int32",
        all_touched=all_touched
    )

def vectorize_labels(label_arr, transform, crs, out_shp, field="cluster"):
    import fiona
    schema = {"geometry":"Polygon", "properties":{field:"int"}}
    with fiona.open(out_shp, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as dst:
        for geom, val in shapes(label_arr.astype(np.int32), mask=(label_arr!=0), transform=transform):
            v = int(val)
            if v != 0:
                dst.write({"geometry": geom, "properties": {field: v}})

def remove_small_components_per_cluster(arr, transform, min_area_px=0, min_area_m2=0.0, connectivity=2):
    if min_area_px <= 0 and min_area_m2 <= 0:
        return arr
    out = arr.copy()
    # 像素面积（投影坐标下近似）
    a,b,c,d = transform.a, transform.b, transform.c, transform.d
    px_area = abs(a*d - b*c)
    structure = np.ones((3,3), np.uint8) if connectivity==2 else np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)

    for cl in np.unique(arr[arr>0]):
        mask = (arr==cl)
        comp_map, ncomp = cc_label(mask.astype(np.uint8), structure=structure)
        for cid in range(1, ncomp+1):
            m = (comp_map==cid)
            npix = int(m.sum())
            drop = (min_area_px>0 and npix < min_area_px)
            if (not drop) and (min_area_m2>0):
                if npix * px_area < min_area_m2:
                    drop = True
            if drop:
                out[m] = 0
    return out

def main():
    ap = argparse.ArgumentParser(description="LULC 连通分量 + 种子重叠众数 赋 cluster（快速版）")
    ap.add_argument("--lulc", required=True, help="LULC 栅格 tif")
    ap.add_argument("--partial", required=True, help="待插补矢量 shp（含 cluster 字段）")
    ap.add_argument("--cluster-field", default="Cluster")
    ap.add_argument("--aoi", default=None, help="AOI shp（可选）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--banned-class", type=int, default=17)
    ap.add_argument("--all-touched", action="store_true", help="栅格化 partial 时更宽松")
    ap.add_argument("--connectivity", type=int, default=2, choices=[1,2], help="连通性（1=4邻，2=8邻）")
    ap.add_argument("--min-area-px", type=int, default=0, help="删小块：像素阈值")
    ap.add_argument("--min-area-m2", type=float, default=0.0, help="删小块：面积阈值（投影坐标）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.lulc) as src:
        lulc = src.read(1)
        crs = src.crs
        transform = src.transform
        lulc_nodata = src.nodata

        # AOI 掩膜：True=AOI 内
        aoi_mask = None
        if args.aoi:
            aoi = gpd.read_file(args.aoi)
            if aoi.crs != crs: aoi = aoi.to_crs(crs)
            aoi_mask = geometry_mask([g for g in aoi.geometry],
                                     out_shape=(src.height, src.width),
                                     transform=transform, invert=True)

        # 允许区
        allowed = np.ones_like(lulc, dtype=bool)
        if lulc_nodata is not None: allowed &= (lulc != lulc_nodata)
        if args.banned_class is not None: allowed &= (lulc != args.banned_class)
        if aoi_mask is not None: allowed &= aoi_mask

        # 栅格化 seed（cluster）
        seeds = gpd.read_file(args.partial)
        if seeds.crs != crs: seeds = seeds.to_crs(crs)
        seed = rasterize_cluster(src, seeds, field=args.cluster_field, all_touched=args.all_touched)
        seed_mask = seed > 0

        # 结果初始化为 0（未赋值）
        result = np.zeros_like(seed, dtype=np.int32)

        # 只在“允许区”内处理；按 LULC 类别分治
        classes = np.unique(lulc[allowed])
        if lulc_nodata is not None:
            classes = classes[classes != lulc_nodata]
        if args.banned_class is not None:
            classes = classes[classes != args.banned_class]

        structure = np.ones((3,3), np.uint8) if args.connectivity==2 \
                    else np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)

        for c in classes:
            cls_mask = allowed & (lulc == c)
            if not np.any(cls_mask): continue

            comp_map, ncomp = cc_label(cls_mask.astype(np.uint8), structure=structure)
            if ncomp == 0: continue

            # 对该类的每个连通分量：计算与种子标签的重叠众数
            for comp_id in range(1, ncomp+1):
                comp = (comp_map == comp_id)
                # 与 seed 的重叠分布
                seeds_on_comp = seed[comp & seed_mask]
                if seeds_on_comp.size == 0:
                    # 无重叠 → 不赋值（保持 0）
                    continue
                # 众数 cluster
                uniq, cnt = np.unique(seeds_on_comp, return_counts=True)
                cl = int(uniq[np.argmax(cnt)])
                result[comp] = cl

        # 删小块（像素/面积阈值）
        result_clean = remove_small_components_per_cluster(
            result, transform, min_area_px=args.min_area_px, min_area_m2=args.min_area_m2,
            connectivity=args.connectivity
        )

        # 导出
        save_tif(out_dir/"filled_cluster.tif", src, result_clean, "int32", nodata=0)
        vectorize_labels(result_clean, transform, crs, str(out_dir/"filled_cluster.shp"), field="cluster")

        # 调试层（可选）
        save_tif(out_dir/"allowed.tif", src, allowed.astype(np.uint8), "uint8", nodata=0)
        save_tif(out_dir/"seed_mask.tif", src, seed_mask.astype(np.uint8), "uint8", nodata=0)

        print("✅ Done")
        print(f"  - filled_cluster.tif / .shp 写入 {out_dir}")
        print("  - 若仍见上下风格差异，适当提高 --min-area-px 或改用 4 邻接 (--connectivity 1) 以减少细碎度。")

if __name__ == "__main__":
    main()
