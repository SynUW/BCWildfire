"""
将extract_osm_thematics.py生成的多个gpkg合并为一个gpkg，方便后续make_distance_rasters.py计算距离图使用
"""
import argparse
import os
import sys
from typing import List, Optional, Iterable, Set

import geopandas as gpd
from pyproj import CRS
import fiona


def ensure_directory_exists(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def read_layer_from_gpkg(gpkg_path: str, layer_name: str) -> gpd.GeoDataFrame:
    if not os.path.isfile(gpkg_path):
        raise FileNotFoundError(f"未找到文件: {gpkg_path}")
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    return gdf


def reproject_if_needed(gdf: gpd.GeoDataFrame, target_crs: CRS) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("输入数据缺少 CRS，无法进行坐标对齐。请先为该数据设置正确的坐标系。")
    if not CRS.from_user_input(gdf.crs).equals(target_crs):
        return gdf.to_crs(target_crs)
    return gdf


def align_columns(geodataframes: List[gpd.GeoDataFrame]) -> List[gpd.GeoDataFrame]:
    # 统一字段集合（保留 geometry）
    all_columns = []
    all_column_set = set()
    for gdf in geodataframes:
        for col in gdf.columns:
            if col == gdf.geometry.name:
                continue
            if col not in all_column_set:
                all_column_set.add(col)
                all_columns.append(col)

    aligned = []
    for gdf in geodataframes:
        cols = [c for c in all_columns if c in gdf.columns]
        # 添加缺失列为 None
        for missing in (set(all_columns) - set(cols)):
            gdf[missing] = None
        # 重新排序，保持 geometry 在最后由 GeoPandas 管理
        ordered = gdf[[*all_columns, gdf.geometry.name]]
        aligned.append(ordered)
    return aligned


def dissolve_geometries(
    gdf: gpd.GeoDataFrame,
    dissolve_by: Optional[str] = None,
) -> gpd.GeoDataFrame:
    if dissolve_by:
        # 聚合其他字段取第一个非空值
        return gdf.dissolve(by=dissolve_by, aggfunc="first", dropna=False)
    # 无字段：全部溶解为单一几何
    tmp = gdf.copy()
    tmp["__all__"] = 1
    res = tmp.dissolve(by="__all__", aggfunc="first", dropna=False)
    res = res.drop(columns=[c for c in res.columns if c == "__all__"], errors="ignore")
    return res


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将多个 GPKG 的同名图层合并为一个 GPKG 图层；或对所有同名图层分组合并")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/canada-thematics.gpkg',
                 '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/us-west-thematics.gpkg'],
        help="输入 GPKG 文件路径列表（空格分隔）",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="对所有图层执行分组合并（按同名合并到输出的同名图层）",
    )
    parser.add_argument(
        "--layer",
        required=False,
        default=None,
        help="需要合并的图层名；若不提供且未指定 --all-layers，则默认对所有图层按同名分组合并",
    )
    parser.add_argument(
        "--output",
        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/canada-us-mosaic-thematics.gpkg',
        help="输出 GPKG 路径",
    )
    parser.add_argument(
        "--out-layer",
        default=None,
        help="输出图层名（默认与 --layer 相同）",
    )
    parser.add_argument(
        "--target-crs",
        default=None,
        help="目标 CRS（如 EPSG:4326）。若不指定，则以第一个输入的 CRS 为准",
    )
    parser.add_argument(
        "--add-source-field",
        default="source_file",
        help="在输出中添加来源文件名字段（设为空字符串可禁用）",
    )
    parser.add_argument(
        "--dissolve",
        action="store_true",
        help="将所有要素溶解为单一（或按 --dissolve-by 字段分组溶解）",
    )
    parser.add_argument(
        "--dissolve-by",
        default=None,
        help="按该字段分组溶解（需要 --dissolve 生效）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths: List[str] = args.inputs
    output_path: str = args.output
    add_source_field: Optional[str] = (args.add_source_field or "").strip() or None

    if len(input_paths) == 0:
        raise ValueError("未提供任何输入 GPKG 文件。")

    # 判断是否为“合并单个图层”还是“合并所有图层”
    merge_all_layers: bool = bool(args.all_layers or not args.layer)

    if not merge_all_layers:
        # ===== 合并单个指定图层 =====
        layer_name: str = args.layer  # type: ignore[assignment]
        output_layer_name: str = args.out_layer or layer_name

        # 确定目标 CRS
        first_gdf = read_layer_from_gpkg(input_paths[0], layer_name)
        if first_gdf.crs is None and not args.target_crs:
            raise ValueError("第一个输入图层缺少 CRS，且未提供 --target-crs。请补充 CRS 信息。")
        if args.target_crs:
            target_crs = CRS.from_user_input(args.target_crs)
        else:
            target_crs = CRS.from_user_input(first_gdf.crs)

        geodataframes: List[gpd.GeoDataFrame] = []
        total = len(input_paths)
        for index, gpkg_path in enumerate(input_paths, start=1):
            sys.stdout.write(f"\r[{index}/{total}] 读取 {os.path.basename(gpkg_path)} …")
            sys.stdout.flush()
            try:
                gdf = read_layer_from_gpkg(gpkg_path, layer_name)
            except Exception:
                continue
            if gdf.empty:
                continue
            gdf = reproject_if_needed(gdf, target_crs)
            if add_source_field:
                gdf[add_source_field] = os.path.basename(gpkg_path)
            geodataframes.append(gdf)

        sys.stdout.write("\n")

        if not geodataframes:
            raise ValueError("所有输入图层均为空，或无法读取。")

        geodataframes = align_columns(geodataframes)
        merged_gdf = gpd.GeoDataFrame(pd_concat(geodataframes), crs=target_crs)
        if args.dissolve:
            merged_gdf = dissolve_geometries(merged_gdf, args.dissolve_by)

        ensure_directory_exists(output_path)
        # 首次写出将创建文件
        if os.path.exists(output_path):
            os.remove(output_path)
        merged_gdf.to_file(output_path, driver="GPKG", layer=output_layer_name)
        print(f"✅ 已输出: {output_path} (layer={output_layer_name})")

    else:
        # ===== 合并所有同名图层 =====
        # 计算所有输入的图层名并取并集
        layer_names: Set[str] = set()
        for gpkg_path in input_paths:
            try:
                for lyr in fiona.listlayers(gpkg_path):
                    layer_names.add(lyr)
            except Exception:
                continue

        if not layer_names:
            raise ValueError("未在输入文件中发现任何图层。")

        # 每个图层单独确定 target CRS：使用 --target-crs 或第一个包含该图层的文件的 CRS
        if os.path.exists(output_path):
            os.remove(output_path)

        names_sorted = sorted(layer_names)
        total_layers = len(names_sorted)
        for layer_idx, layer_name in enumerate(names_sorted, start=1):
            # 找到首个包含该图层的文件以确定默认 CRS
            target_crs: Optional[CRS] = None
            if args.target_crs:
                target_crs = CRS.from_user_input(args.target_crs)
            else:
                for p in input_paths:
                    try:
                        probe = read_layer_from_gpkg(p, layer_name)
                    except Exception:
                        continue
                    if probe is not None and probe.crs is not None:
                        target_crs = CRS.from_user_input(probe.crs)
                        break
            if target_crs is None:
                # 全部缺少 CRS，跳过该图层
                sys.stdout.write(f"\r[{layer_idx}/{total_layers}] 跳过 {layer_name}（无 CRS）")
                sys.stdout.flush()
                continue

            # 收集该图层的所有数据并重投影/对齐
            geodataframes: List[gpd.GeoDataFrame] = []
            for idx, gpkg_path in enumerate(input_paths, start=1):
                sys.stdout.write(
                    f"\r[{layer_idx}/{total_layers}] {layer_name}: 读取 {idx}/{len(input_paths)} {os.path.basename(gpkg_path)} …"
                )
                sys.stdout.flush()
                try:
                    gdf = read_layer_from_gpkg(gpkg_path, layer_name)
                except Exception:
                    continue
                if gdf.empty:
                    continue
                try:
                    gdf = reproject_if_needed(gdf, target_crs)
                except Exception:
                    continue
                if add_source_field:
                    gdf[add_source_field] = os.path.basename(gpkg_path)
                geodataframes.append(gdf)

            sys.stdout.write("\n")

            if not geodataframes:
                continue

            geodataframes = align_columns(geodataframes)
            merged_gdf = gpd.GeoDataFrame(pd_concat(geodataframes), crs=target_crs)
            if args.dissolve:
                merged_gdf = dissolve_geometries(merged_gdf, args.dissolve_by)

            # 写入同名图层
            ensure_directory_exists(output_path)
            merged_gdf.to_file(output_path, driver="GPKG", layer=layer_name)

        print(f"✅ 已输出: {output_path}（合并 {total_layers} 个图层）")


def pd_concat(gdfs: List[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    # 延迟导入，避免用户环境未装 pandas 时错误提示不清晰
    import pandas as pd

    return pd.concat(gdfs, ignore_index=True)


if __name__ == "__main__":
    main()

