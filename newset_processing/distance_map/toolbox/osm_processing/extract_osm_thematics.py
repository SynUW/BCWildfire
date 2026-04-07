"""
从已裁剪的 OSM .osm.pbf 提取主题图层，输出到单一 GeoPackage（建筑物、水体、道路、电力）
"""
import argparse
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple


def check_ogr2ogr() -> None:
    if shutil.which("ogr2ogr") is None:
        print("未找到 ogr2ogr，请先安装 gdal-bin：sudo apt-get install -y gdal-bin", file=sys.stderr)
        sys.exit(1)


def run_and_stream(cmd: List[str]) -> int:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line.rstrip())
    finally:
        return proc.wait()


def ogr2ogr_extract(
    input_pbf: str,
    output_gpkg: str,
    source_layer: str,
    output_layer_name: str,
    where_clause: str,
    disable_spatial_index: bool,
    transaction_size: int,
    is_first_layer: bool,
) -> int:
    cmd: List[str] = [
        "ogr2ogr",
        "-f",
        "GPKG",
    ]

    if not is_first_layer:
        cmd += ["-update"]

    # 覆盖同名图层（不影响其他图层）
    cmd += ["-overwrite"]

    if disable_spatial_index:
        cmd += ["-lco", "SPATIAL_INDEX=NO"]

    if transaction_size and transaction_size > 0:
        cmd += ["-gt", str(transaction_size)]

    cmd += [
        "-nln",
        output_layer_name,
        "-progress",
        output_gpkg,
        input_pbf,
        source_layer,
        "-where",
        where_clause,
    ]

    return run_and_stream(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从已裁剪的 OSM .osm.pbf 提取主题图层，输出到单一 GeoPackage（建筑物、水体、道路、电力）"
    )
    parser.add_argument(
        "--input_pbf",
        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/canada-latest-clip.osm.pbf",
        help="输入（已裁剪）.osm.pbf 路径",
    )
    parser.add_argument(
        "--output_gpkg",
        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/canada-thematics.gpkg",
        help="输出 GeoPackage 文件路径（包含多个图层）",
    )
    parser.add_argument(
        "--disable_spatial_index",
        action="store_true",
        help="写入时禁用 GPKG 空间索引（更快），需要时可后续再创建",
    )
    parser.add_argument(
        "--gt",
        dest="transaction_size",
        type=int,
        default=65536,
        help="事务批大小（-gt），如 65536，可提升写入速度",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若输出 .gpkg 已存在则先删除后重建",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_ogr2ogr()

    if os.path.exists(args.output_gpkg):
        if args.force:
            print(f"删除已存在的输出文件: {args.output_gpkg}")
            try:
                os.remove(args.output_gpkg)
            except Exception as exc:
                print(f"删除失败: {exc}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"输出文件已存在，将在其中覆盖同名图层: {args.output_gpkg}")

    # 定义主题: (源图层, 输出图层名, 过滤条件)
    thematics: Dict[str, Tuple[str, str]] = {
        # 建筑物（面）
        "buildings": ("multipolygons", "building IS NOT NULL OR other_tags LIKE '%\"building\"=>%'"),
        # 水体：线（河流/水道）
        "water_lines": (
            "lines",
            "waterway IN ('river','stream','canal','ditch','drain')",
        ),
        # 水体：面（湖泊/水库/湿地）
        "water_areas": (
            "multipolygons",
            "natural='water' OR landuse='reservoir' OR natural='wetland'",
        ),
        # 道路（所有类别）
        "roads": ("lines", "highway IS NOT NULL"),
        # 电力线路（仅在 other_tags 中匹配）
        "power_lines": (
            "lines",
            "other_tags LIKE '%\"power\"=>\"line\"%' OR other_tags LIKE '%\"power\"=>\"minor_line\"%' OR other_tags LIKE '%\"power\"=>\"cable\"%'",
        ),
        # 变电站（面，仅在 other_tags 中匹配）
        "power_substations": (
            "multipolygons",
            "other_tags LIKE '%\"power\"=>\"substation\"%'",
        ),
        # 铁塔/电线杆/变电站（点，仅在 other_tags 中匹配）
        "power_points": (
            "points",
            "other_tags LIKE '%\"power\"=>\"tower\"%' OR other_tags LIKE '%\"power\"=>\"pole\"%' OR other_tags LIKE '%\"power\"=>\"substation\"%'",
        ),
    }

    # 顺序稳定，便于进度显示
    order: List[str] = [
        "buildings",
        "water_lines",
        "water_areas",
        "roads",
        "power_lines",
        "power_substations",
        "power_points",
    ]

    total = len(order)
    for idx, name in enumerate(order, start=1):
        source_layer, where_clause = thematics[name]
        print(f"[{idx}/{total}] 提取 {name} -> {args.output_gpkg}")
        ret = ogr2ogr_extract(
            input_pbf=args.input_pbf,
            output_gpkg=args.output_gpkg,
            source_layer=source_layer,
            output_layer_name=name,
            where_clause=where_clause,
            disable_spatial_index=args.disable_spatial_index,
            transaction_size=args.transaction_size,
            is_first_layer=(idx == 1 and not os.path.exists(args.output_gpkg)),
        )
        if ret != 0:
            print(f"图层 {name} 提取失败（退出码 {ret})", file=sys.stderr)
            sys.exit(ret)

    print("全部主题图层已生成。")


if __name__ == "__main__":
    main()