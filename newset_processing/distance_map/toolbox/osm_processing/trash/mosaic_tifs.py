"""将裁剪后的两部分距离栅格拼接"""

import argparse
import os
import sys
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling


def list_cropped(path: str, suffix: str) -> Dict[str, str]:
    files = glob(os.path.join(path, f"*{suffix}"))
    return {os.path.basename(fp): fp for fp in files}


def choose_nodata(datasets: List[rasterio.io.DatasetReader], dtype: str):
    # 优先使用A的nodata，其次B；都没有则按dtype给定默认值
    for ds in datasets:
        if ds.nodata is not None:
            return ds.nodata
    if np.issubdtype(np.dtype(dtype), np.floating):
        return -9999.0
    return 0


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def merge_pair(a_path: str, b_path: str, out_path: str):
    with rasterio.open(a_path) as a_ds, rasterio.open(b_path) as b_ds:
        # 以A为基准的输出分辨率（像元尺寸），这样能保持与A同网格
        a_res = (abs(a_ds.transform.a), abs(a_ds.transform.e))
        dtype = a_ds.dtypes[0]
        nodata_value = choose_nodata([a_ds, b_ds], dtype)

        # 若B与A的CRS不同，先用 WarpedVRT 将B重投影到A的CRS
        b_src = b_ds
        if b_ds.crs != a_ds.crs:
            b_src = WarpedVRT(
                b_ds,
                crs=a_ds.crs,
                resampling=Resampling.nearest,
                src_nodata=b_ds.nodata,
                dst_nodata=nodata_value,
            )

        # 数据顺序 [A, B] 且 method='last' => 重叠处用B覆盖A
        mosaic, out_transform = rio_merge(
            [a_ds, b_src],
            method="last",
            nodata=nodata_value,
            res=a_res,
        )

        profile = a_ds.profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            nodata=nodata_value,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        # 浮点数据加预测器
        if np.issubdtype(np.dtype(dtype), np.floating):
            profile["predictor"] = 3

        ensure_dir(os.path.dirname(out_path))
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)


def find_pairs(dir_a: str, dir_b: str, suffix: str) -> List[Tuple[str, str, str]]:
    a_map = list_cropped(dir_a, suffix)
    b_map = list_cropped(dir_b, suffix)
    names = sorted(set(a_map.keys()) & set(b_map.keys()))
    return [(name, a_map[name], b_map[name]) for name in names]


def parse_args():
    ap = argparse.ArgumentParser(
        description="按地理位置拼接两个文件夹中同名的 *cropped.tif，重叠处用B覆盖A"
    )
    ap.add_argument("--dir-a", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/canada-distance', help="文件夹A路径")
    ap.add_argument("--dir-b", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/us-west-distance', help="文件夹B路径")
    ap.add_argument("--out-dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/osm_map/distance_mosaic', help="输出文件夹")
    ap.add_argument("--suffix", default="cropped.tif", help="文件名后缀（默认：cropped.tif）")
    return ap.parse_args()


def main():
    args = parse_args()
    pairs = find_pairs(args.dir_a, args.dir_b, args.suffix)

    total = len(pairs)
    if total == 0:
        print("未找到重名且以指定后缀结尾的影像对。")
        return

    for i, (name, a_path, b_path) in enumerate(pairs, 1):
        out_path = os.path.join(args.out_dir, name)
        # 动态单行进度：i/n 当前文件
        sys.stdout.write(f"\r[{i}/{total}] 拼接 {name} ...")
        sys.stdout.flush()
        merge_pair(a_path, b_path, out_path)

    sys.stdout.write("\n")
    print(f"完成。输出目录：{args.out_dir}")


if __name__ == "__main__":
    main()