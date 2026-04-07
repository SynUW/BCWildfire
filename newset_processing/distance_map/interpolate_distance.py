#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对距离图（distance raster）中的无效值进行插值，只填补无效像元。
默认使用 GDAL 的 FillNodata 算法（遵循邻近像元的变化规律，含平滑迭代），
支持批量处理文件夹、多线程与压缩输出。

依赖：
  - GDAL（Python 绑定）：pip install GDAL  或 使用系统自带的 osgeo.gdal
  - rasterio（可选，用于更稳健地读取/写出，非必须）

示例：
  python interpolate_distance_nodata.py \
      --input-folder /path/to/in_tifs \
      --output-folder /path/to/out_tifs \
      --nodata -9999 \
      --search-radius 50 \
      --smooth-iterations 2 \
      --threads 8 \
      --compress ZSTD \
      --blocksize 512
"""

import os
import sys
import math
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from osgeo import gdal
except ImportError as e:
    print("未找到 GDAL 的 Python 绑定（osgeo.gdal）。请先安装：pip install GDAL 或使用系统GDAL。")
    raise

# 可选：优先使用 rasterio 写出（更方便设置压缩/块大小），若没有则退回 GDAL
try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_OK = True
except Exception:
    RASTERIO_OK = False


def parse_args():
    ap = argparse.ArgumentParser(description="对距离图的 NoData 区域进行插值填补（仅填补无效像元）")
    ap.add_argument("--input-folder", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/distance_map', help="输入文件夹，包含待插值的 .tif")
    ap.add_argument("--output-folder", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/distance_map_interpolated_margin_nodata', help="输出文件夹")
    ap.add_argument("--nodata", default='-32768',
                    help="无效值（例如 -9999、-32768 或 'nan'）")
    ap.add_argument("--search-radius", type=float, default=50.0,
                    help="最大搜索半径（像素），GDAL FillNodata 的 maxSearchDist，默认 50")
    ap.add_argument("--smooth-iterations", type=int, default=2,
                    help="平滑迭代次数（smoothingIterations），默认 2；设为 0 关闭平滑")
    ap.add_argument("--threads", type=int, default=8, help="并行线程数，默认 8")
    ap.add_argument("--overwrite", action="store_true", help="若输出已存在则覆盖")
    ap.add_argument("--compress", default="ZSTD", choices=["NONE", "LZW", "ZSTD", "DEFLATE"],
                    help="输出压缩编码，默认 ZSTD")
    ap.add_argument("--blocksize", type=int, default=512, help="瓦片块大小，默认 512")
    ap.add_argument("--keep-dtype", action="store_true",
                    help="尽量保持原始 dtype（默认）。未开启时，统一写出为 float32")
    ap.add_argument("--gdal-only", action="store_true",
                    help="强制仅用 GDAL 写出（不依赖 rasterio）")
    return ap.parse_args()


def str_to_nodata(nodata_str: str):
    """将命令行 nodata 字符串解析为数值或 NaN。"""
    if nodata_str.lower() == "nan":
        return float("nan")
    try:
        return float(nodata_str)
    except ValueError:
        raise ValueError(f"无法解析 nodata='{nodata_str}'，请输入数字或 'nan'。")


def build_creation_options(compress: str, blocksize: int):
    co = []
    if compress and compress.upper() != "NONE":
        co.append(f"COMPRESS={compress.upper()}")
        # ZSTD 建议设置质量级别
        if compress.upper() == "ZSTD":
            co.append("ZSTD_LEVEL=9")
        if compress.upper() == "DEFLATE":
            co.append("PREDICTOR=2")
    if blocksize:
        co.append(f"TILED=YES")
        co.append(f"BLOCKXSIZE={blocksize}")
        co.append(f"BLOCKYSIZE={blocksize}")
    return co


def fill_nodata_gdal(in_path: Path, out_path: Path, user_nodata, max_search_dist: float,
                     smooth_iters: int, keep_dtype: bool, creation_options):
    """
    使用 GDAL.FillNodata 只填补 nodata 像元。尽量保持地理参考、投影与元数据。
    """
    # 先用 GDAL 打开源数据
    src_ds = gdal.Open(str(in_path), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"无法打开输入影像：{in_path}")

    band = src_ds.GetRasterBand(1)
    xsize = src_ds.RasterXSize
    ysize = src_ds.RasterYSize
    geotransform = src_ds.GetGeoTransform(can_return_null=True)
    projection = src_ds.GetProjectionRef()
    src_nodata = band.GetNoDataValue()

    # 决定 nodata 使用：优先采用用户指定；若用户为 NaN 且原来有 nodata，可沿用原值
    if isinstance(user_nodata, float) and math.isnan(user_nodata):
        # 用户输入 NaN：若源已有 nodata 就用源 nodata，否则用 NaN 流程（转 float32 并以 NaN 当作 nodata）
        use_nan = True if src_nodata is None else False
        nodata_value = src_nodata if src_nodata is not None else None
    else:
        use_nan = False
        nodata_value = float(user_nodata)

    # 确定输出 dtype
    gdal_dtype = band.DataType
    if not keep_dtype:
        # 统一 float32（GDAL：gdal.GDT_Float32）
        gdal_dtype = gdal.GDT_Float32

    # 创建输出
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(str(out_path), xsize, ysize, 1, gdal_dtype, creation_options)
    if out_ds is None:
        raise RuntimeError(f"无法创建输出影像：{out_path}")

    if geotransform:
        out_ds.SetGeoTransform(geotransform)
    if projection:
        out_ds.SetProjection(projection)

    out_band = out_ds.GetRasterBand(1)

    # 读取整幅数据（GDAL FillNodata 需要在目标 band 上操作）
    arr = band.ReadAsArray()
    if arr is None:
        out_ds = None
        src_ds = None
        raise RuntimeError(f"读取数据失败：{in_path}")

    import numpy as np

    # 将数据写入输出 band，并统一设置 nodata
    if use_nan:
        # 用户指定 nan；若源无 nodata，则我们把用户的“无效像元”视为 np.nan
        # 情况1：源有 nodata，则沿用源 nodata 值作为 band 的 nodata
        if nodata_value is not None:
            out_band.SetNoDataValue(nodata_value)
            arr_f = arr.astype("float32", copy=False)
        else:
            # 情况2：源无 nodata，用 NaN 作为占位（注意：band nodata 不能直接设为 NaN）
            # 我们临时将 NaN 用于算法识别，填充完成后再写出（band 不设 nodata 值）
            arr_f = arr.astype("float32", copy=False)
            # 将用户认为的无效值（此分支用户就是 nan，所以需要依据“原本可能没有 nodata”——没法识别）
            # 因此这里只处理原本已有的 NaN（若源数据就存在），否则 FillNodata 无法识别“哪是无效像元”。
            # 为更稳妥：如果源没有 nodata 且无 NaN，无需填充（直接写出）
            if not np.isnan(arr_f).any():
                # 无 NaN，无可填像元
                out_band.WriteArray(arr_f)
                out_band.FlushCache()
                out_ds = None
                src_ds = None
                return
    else:
        # 用户指定了明确数值 nodata
        out_band.SetNoDataValue(nodata_value)
        arr_f = arr.astype("float32", copy=False)
        # 将 nodata 像元保留为 nodata 值（若源 nodata 不同，统一转为用户 nodata）
        if src_nodata is not None and src_nodata != nodata_value:
            arr_f[arr == src_nodata] = nodata_value

    out_band.WriteArray(arr_f)
    out_band.FlushCache()

    # 使用 GDAL.FillNodata 填补
    # 仅在目标 band 中的 nodata 像元进行插值；maskBand 可选，这里不额外提供。
    # maxSearchDist 单位：像素；smoothingIterations：平滑次数
    gdal.ErrorReset()
    err = gdal.FillNodata(targetBand=out_band,
                          maskBand=None,
                          maxSearchDist=max_search_dist,
                          smoothingIterations=smooth_iters)
    if err != 0:
        # gdal.FillNodata 返回 0 表示成功，非 0 视为失败
        out_ds = None
        src_ds = None
        raise RuntimeError(f"GDAL.FillNodata 失败（代码 {err}）：{in_path}")

    out_band.FlushCache()
    out_ds.FlushCache()

    # 清理
    out_ds = None
    src_ds = None


def write_with_rasterio_like(in_path: Path, tmp_filled_path: Path, out_path: Path,
                             compress: str, blocksize: int, keep_dtype: bool):
    """
    如果启用 rasterio，则可将 GDAL 填充结果再标准化写出（确保压缩、tiling 等一致）。
    这里的实现是“读取 tmp_filled_path 并原样写到 out_path（带压缩/块大小）”。
    """
    if not RASTERIO_OK:
        # 直接把临时文件改名（或复制）为最终输出
        if out_path.exists():
            out_path.unlink()
        tmp_filled_path.replace(out_path)
        return

    import numpy as np
    co = {"compress": None} if compress == "NONE" else {"compress": compress.lower()}
    blocksize = int(blocksize) if blocksize else 512

    with rasterio.open(tmp_filled_path, "r") as src:
        meta = src.meta.copy()
        # 强制使用块
        meta.update(
            tiled=True,
            blockxsize=blocksize,
            blockysize=blocksize,
            **co
        )
        if not keep_dtype:
            meta.update(dtype="float32")
        # 删除旧文件
        if out_path.exists():
            out_path.unlink()
        with rasterio.open(out_path, "w", **meta) as dst:
            for ji, window in dst.block_windows(1):
                data = src.read(1, window=window)
                if not keep_dtype:
                    data = data.astype("float32", copy=False)
                dst.write(data, 1, window=window)
            # 继承 nodata
            nd = src.nodata
            if nd is not None:
                dst.nodata = nd


def process_one(in_path: Path,
                out_path: Path,
                nodata_value,
                search_radius: float,
                smooth_iterations: int,
                keep_dtype: bool,
                creation_options,
                use_rasterio_writer: bool):
    """
    处理单个文件：
      1) 用 GDAL.FillNodata 填补（仅 NoData 像元）
      2) 可选：用 rasterio 规范压缩/块写出
    """
    tmp_path = out_path.with_suffix(".tmp.tif")

    # 先在临时文件上执行填补
    # 这里用 GDAL 的 CreateCopy 再 FillNodata 会比较慢；我们直接 Create->Write->Fill。
    # 为了简化，我们先将其写到 tmp_path，然后再按需要标准化写出到 out_path。
    # —— 为了方便：直接把 tmp 当最终输出，最后再“标准化复制/转写”。
    fill_nodata_gdal(in_path=in_path,
                     out_path=tmp_path,
                     user_nodata=nodata_value,
                     max_search_dist=search_radius,
                     smooth_iters=smooth_iterations,
                     keep_dtype=keep_dtype,
                     creation_options=creation_options)

    # 按需要“规范化写出”
    write_with_rasterio_like(in_path=in_path,
                             tmp_filled_path=tmp_path,
                             out_path=out_path,
                             compress=creation_options_dict(creation_options).get("COMPRESS", "NONE"),
                             blocksize=int(creation_options_dict(creation_options).get("BLOCKXSIZE", 512)),
                             keep_dtype=keep_dtype)

    # 清理 tmp（rasterio 写出分支会在内部替换掉，不再存在；GDAL-only 分支会直接 rename）
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass


def creation_options_dict(co_list):
    """把 GDAL creation options 列表转成 dict 便于读取。"""
    d = {}
    for kv in co_list:
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k.upper()] = v
        else:
            d[kv.upper()] = True
    return d


def main():
    args = parse_args()
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    nodata_value = str_to_nodata(args.nodata)
    creation_options = build_creation_options(args.compress, args.blocksize)

    # 遍历输入文件
    tifs = sorted([p for p in input_folder.glob("*.tif")])
    if not tifs:
        print(f"输入目录中未找到 .tif：{input_folder}")
        sys.exit(1)

    tasks = []
    for tif in tifs:
        out_path = output_folder / tif.name
        if out_path.exists() and not args.overwrite:
            continue
        tasks.append((tif, out_path))

    total = len(tasks)
    skipped = len(tifs) - total
    print(f"待处理：{total} 个；跳过（已存在且未覆盖）：{skipped} 个")

    use_rasterio_writer = (RASTERIO_OK and not args.gdal_only)

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = []
        for in_path, out_path in tasks:
            futures.append(ex.submit(
                process_one,
                in_path,
                out_path,
                nodata_value,
                args.search_radius,
                args.smooth_iterations,
                args.keep_dtype,
                creation_options,
                use_rasterio_writer
            ))

        for f in as_completed(futures):
            try:
                f.result()
                ok += 1
                if (ok + fail) % 10 == 0:
                    print(f"[进度] 成功：{ok} | 失败：{fail} / {total}")
            except Exception as e:
                fail += 1
                print(f"[失败] {e}")

    print(f"完成。成功：{ok} | 失败：{fail} | 跳过：{skipped} | 总计：{len(tifs)}")


if __name__ == "__main__":
    main()
