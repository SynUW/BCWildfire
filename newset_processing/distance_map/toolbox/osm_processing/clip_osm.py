"""
使用 Shapefile 裁剪 OSM .osm.pbf 并输出裁剪后的 .osm.pbf（使用 GDAL 与 osmium）
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def check_ogr2ogr() -> None:
    if shutil.which("ogr2ogr") is None:
        print("未找到 ogr2ogr，请先安装 gdal-bin：sudo apt-get install -y gdal-bin", file=sys.stderr)
        sys.exit(1)


def check_osmium() -> None:
    if shutil.which("osmium") is None:
        print("未找到 osmium，请先安装 osmium-tool：sudo apt-get install -y osmium-tool", file=sys.stderr)
        sys.exit(1)


def run_and_stream(cmd: list) -> int:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line.rstrip())
    finally:
        return proc.wait()


def shapefile_to_geojson_wgs84(
    input_shapefile: str,
    output_geojson: str,
    assume_wgs84: bool = False,
) -> None:
    """将 Shapefile 转为 WGS84 GeoJSON。若 assume_wgs84=True，则先赋予 EPSG:4326。"""
    cmd = [
        "ogr2ogr",
        "-f",
        "GeoJSON",
        output_geojson,
        input_shapefile,
        "-t_srs",
        "EPSG:4326",
    ]
    if assume_wgs84:
        # 若源数据没有 .prj，但坐标确为经纬度（WGS84），为其赋予 EPSG:4326
        cmd += ["-a_srs", "EPSG:4326"]

    ret = run_and_stream(cmd)
    if ret != 0:
        print(f"Shapefile 转 GeoJSON 失败（退出码 {ret})", file=sys.stderr)
        sys.exit(ret)


def clip_pbf_with_polygon(
    input_pbf: str,
    polygon_geojson: str,
    output_pbf: str,
) -> None:
    """使用 osmium 根据多边形 GeoJSON 裁剪 PBF。"""
    os.makedirs(os.path.dirname(output_pbf) or ".", exist_ok=True)

    cmd = [
        "osmium",
        "extract",
        "-p",
        polygon_geojson,
        input_pbf,
        "-o",
        output_pbf,
        "--overwrite",
    ]
    ret = run_and_stream(cmd)
    if ret != 0:
        print(f"PBF 裁剪失败（退出码 {ret})", file=sys.stderr)
        sys.exit(ret)


def parse_args():
    parser = argparse.ArgumentParser(description="使用 Shapefile 裁剪 OSM .osm.pbf 并输出裁剪后的 .osm.pbf（使用 GDAL 与 osmium）")
    parser.add_argument("--input_pbf", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/us-west-latest.osm.pbf', help="输入 .osm.pbf 路径")
    parser.add_argument("--clipsrc", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_boundary/british_columnbia_no_crs_boundingBox.shp', help="用于裁剪的 Shapefile 路径（边界数据）")
    parser.add_argument("--output_pbf", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/us-west-latest-clip.osm.pbf', help="输出裁剪后的 .osm.pbf 路径")
    parser.add_argument("--assume_wgs84", action="store_true", help="若 Shapefile 无 .prj，但坐标是 WGS84，经纬度数值正确，则为其赋予 EPSG:4326")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_ogr2ogr()
    check_osmium()

    # 临时 GeoJSON 放到内存盘（若可用）以加速
    tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
    tmp_geojson = os.path.join(tmp_dir, f"clip_polygon_{os.getpid()}.geojson")

    print(f"[1/2] 转换裁剪边界为 WGS84 GeoJSON -> {tmp_geojson}")
    shapefile_to_geojson_wgs84(args.clipsrc, tmp_geojson, assume_wgs84=args.assume_wgs84)

    print(f"[2/2] 使用多边形裁剪 PBF -> {args.output_pbf}")
    clip_pbf_with_polygon(args.input_pbf, tmp_geojson, args.output_pbf)

    # 清理临时文件
    try:
        if os.path.exists(tmp_geojson):
            os.remove(tmp_geojson)
    except Exception:
        pass