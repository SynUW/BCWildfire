import os
import math
import argparse
from typing import List, Iterable, Dict, Any, Set, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, binary_dilation
import fiona

# 图层分类
layers_needed = {
    "distance_buildings": ["buildings"],
    "distance_water": ["water_lines"],  # water_areas会导致几乎整张图都是水，需要进一步选择，所以暂时舍弃
    "distance_power": ["power_lines", "power_substations", "power_points"],
}

# 针对各主题，若存在这些字段，则按“非空”筛选；若字段不存在，则不做筛选（保留全部）
non_null_fields_map = {
    "distance_buildings": ["building"],
    "distance_water": ["waterway", "water", "natural", "landuse"],
    "distance_power": ["power"],
}

# 取值过滤：仅当指定时才应用；此处要求 waterway 仅保留 'river'
value_filters_map: Dict[str, Dict[str, Set[str]]] = {
    "distance_water": {"waterway": {"river"}},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="从一个或多个 GPKG 的主题图层生成距离栅格（按经纬度网格计算，像元值为到最近目标的距离-公里）"
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/canada-us-mosaic-thematics.gpkg",
        ],
        help="输入 GPKG 路径列表（同名图层自动汇总）",
    )
    ap.add_argument(
        "--output-dir",
        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/distance_map",
        help="输出目录",
    )
    ap.add_argument(
        "--resolution-m",
        type=float,
        default=1000.0,
        help="分辨率（米），用于换算到经纬度像元大小",
    )
    return ap.parse_args()


# 流式读取：按属性非空过滤，直接输出 GeoJSON 几何以供栅格化
def iter_shapes_from_layers(
    gpkg_paths: List[str],
    layer_names: List[str],
    candidate_cols: List[str],
) -> Iterable[Dict[str, Any]]:
    any_filter = False
    candidate_set = set(c.lower() for c in candidate_cols)
    for path in gpkg_paths:
        for layer in layer_names:
            try:
                with fiona.open(path, layer=layer) as src:
                    # 仅接受 WGS84/无CRS（默认按WGS84处理），否则跳过避免投影转换开销
                    try:
                        crs_wkt = src.crs_wkt or ""
                    except Exception:
                        crs_wkt = ""
                    if src.crs and str(src.crs).upper() not in ("EPSG:4326", "{'init': 'epsg:4326'}") and "WGS 84" not in crs_wkt:
                        # 若需要，也可在此加入 on-the-fly 重投影（为提速暂不做）
                        pass

                    # 判断是否需要按字段过滤
                    present_cols = set(k.lower() for k in (src.schema.get("properties", {}) or {}).keys())
                    do_filter = len(candidate_set & present_cols) > 0
                    any_filter = any_filter or do_filter

                    for feat in src:  # 流式读取
                        geom = feat.get("geometry")
                        if not geom:
                            continue
                        # 过滤空坐标
                        coords = geom.get("coordinates", None)
                        if coords is None or (isinstance(coords, (list, tuple)) and len(coords) == 0):
                            continue
                        if do_filter:
                            props = feat.get("properties") or {}
                            keep = False
                            for col in candidate_cols:
                                v = props.get(col)
                                if v is None:
                                    continue
                                s = str(v)
                                if len(s) > 0 and s.lower() != "none":
                                    keep = True
                                    break
                            if not keep:
                                continue
                        # 取值过滤（例如仅 waterway == 'river'）
                        # 若外层调用需要此过滤，则在 main 中包装一个带值过滤的生成器
                        yield geom
            except Exception:
                continue
    # 若没有任何图层具备待筛选字段，则等价于“不过滤”
    if not any_filter:
        # 已经在循环中 yield 了所有几何
        pass


# 计算联合范围：用 fiona.bounds 合并，避免把所有要素读入内存
def compute_union_bounds(gpkg_paths: List[str], all_layer_names: List[str]) -> tuple:
    minx, miny, maxx, maxy = (np.inf, np.inf, -np.inf, -np.inf)
    found = False
    for path in gpkg_paths:
        for layer in all_layer_names:
            try:
                with fiona.open(path, layer=layer) as src:
                    b = src.bounds  # (minx, miny, maxx, maxy)
                    if b is None:
                        continue
                    bx1, by1, bx2, by2 = b
                    if not (np.isfinite(bx1) and np.isfinite(by1) and np.isfinite(bx2) and np.isfinite(by2)):
                        continue
                    minx = min(minx, bx1)
                    miny = min(miny, by1)
                    maxx = max(maxx, bx2)
                    maxy = max(maxy, by2)
                    found = True
            except Exception:
                continue
    if not found:
        raise ValueError("输入的 GPKG 中没有可用的目标图层")
    return (minx, miny, maxx, maxy)


def main() -> None:
    args = parse_args()
    inputs: List[str] = args.inputs
    output_dir: str = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    RESOLUTION_DEG = float(args.resolution_m) / 111320.0

    # 计算联合范围（所有涉及的图层）
    all_layer_names = sorted({ln for v in layers_needed.values() for ln in v})
    minx, miny, maxx, maxy = compute_union_bounds(inputs, all_layer_names)

    width = int(math.ceil((maxx - minx) / RESOLUTION_DEG))
    height = int(math.ceil((maxy - miny) / RESOLUTION_DEG))
    width = max(width, 1)
    height = max(height, 1)

    maxx = minx + width * RESOLUTION_DEG
    maxy = miny + height * RESOLUTION_DEG
    transform = from_origin(minx, maxy, RESOLUTION_DEG, RESOLUTION_DEG)

    print(f"统一范围: ({minx}, {miny}, {maxx}, {maxy}) -> {width}x{height} 像元")

    # 逐类别生成距离图（流式栅格化 -> 二值化 -> EDT）
    for out_name, layer_list in layers_needed.items():
        candidate_cols = non_null_fields_map.get(out_name, [])

        # 栅格化（直接流式 shapes）。若需要按取值过滤，则在此包一层过滤器
        base_iter = iter_shapes_from_layers(inputs, layer_list, candidate_cols)
        vf: Optional[Dict[str, Set[str]]] = value_filters_map.get(out_name)
        if vf:
            vf_lower = {k.lower(): {v.lower() for v in vals} for k, vals in vf.items()}
            def filtered_iter():
                # 需要重新打开以获取属性，这里改为重新遍历数据源
                for path in inputs:
                    for layer in layer_list:
                        try:
                            with fiona.open(path, layer=layer) as src:
                                for feat in src:
                                    geom = feat.get("geometry")
                                    if not geom:
                                        continue
                                    coords = geom.get("coordinates", None)
                                    if coords is None or (isinstance(coords, (list, tuple)) and len(coords) == 0):
                                        continue
                                    props = feat.get("properties") or {}
                                    ok = True
                                    for fld, allowed in vf_lower.items():
                                        val = props.get(fld)
                                        if val is None or str(val).lower() not in allowed:
                                            ok = False
                                            break
                                    if not ok:
                                        continue
                                    # 同时保留非空字段的基本过滤逻辑
                                    if candidate_cols:
                                        present_cols = set(k.lower() for k in (src.schema.get("properties", {}) or {}).keys())
                                        do_filter = len(set(c.lower() for c in candidate_cols) & present_cols) > 0
                                        if do_filter:
                                            keep = False
                                            for col in candidate_cols:
                                                v = props.get(col)
                                                if v is None:
                                                    continue
                                                s = str(v)
                                                if len(s) > 0 and s.lower() != "none":
                                                    keep = True
                                                    break
                                            if not keep:
                                                continue
                                    yield geom
                        except Exception:
                            continue
            shapes_iter = ((geom, 1) for geom in filtered_iter())
        else:
            shapes_iter = ((geom, 1) for geom in base_iter)

        mask = features.rasterize(
            shapes=shapes_iter,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

        # 若全为0，直接输出全 NaN
        if mask.max() == 0:
            out_path = os.path.join(output_dir, f"{out_name}.tif")
            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=transform,
                nodata=np.nan,
                compress="DEFLATE",
            ) as dst:
                dst.write(np.full((height, width), np.nan, dtype=np.float32), 1)
            print(f"{out_name}: 无要素，已输出全 NaN -> {out_path}")
            continue

        # 栅格域 1 像素膨胀，增强细目标捕获
        mask = binary_dilation(mask.astype(bool), iterations=1).astype(np.uint8)

        # 距离（km）：注意现仍在经纬度网格上，km 为近似值
        inv = (mask == 0)
        dist_deg = distance_transform_edt(inv) * RESOLUTION_DEG
        dist_km = (dist_deg * 111320.0) / 1000.0

        out_path = os.path.join(output_dir, f"{out_name}.tif")
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs="EPSG:4326",
            transform=transform,
            nodata=np.nan,
            compress="DEFLATE",
        ) as dst:
            dst.write(dist_km.astype(np.float32), 1)

        print(f"{out_name} → {out_path}")

    print("✅ 距离图已完成（单位：km），范围与像元完全对齐。")


if __name__ == "__main__":
    main()
