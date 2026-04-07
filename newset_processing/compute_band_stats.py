#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历目录内所有 GeoTIFF，按波段统计：
- 全局最小/最大值（忽略无效值）
- 0.1% 与 99.9% 分位数（忽略无效值）

实现细节：
- 两遍扫描：
  1) 先求各波段全局 min/max
  2) 再在 [min,max] 上构建固定 bins 直方图，累计频数，最后从累计频数求分位数
- 按块读取（rasterio.block_windows），避免一次性载入整幅影像
- 无效值：优先使用文件的 band.nodata；可用 --default-nodata 指定默认无效值；始终忽略 NaN
"""

import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from osgeo import gdal
from tqdm import tqdm
import multiprocessing as mp


def list_tifs(root: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        return sorted(list(root.rglob('*.tif')) + list(root.rglob('*.TIF')))
    return sorted(list(root.glob('*.tif')) + list(root.glob('*.TIF')))


def is_valid(data: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    mask = np.isfinite(data)
    if nodata is not None:
        mask &= (data != nodata)
    return mask


def _gdal_pass1_one(tif_path: Path, default_nodata: Optional[float]) -> Tuple[int, List[float], List[float]]:
    band_count: Optional[int] = None
    global_min: List[float] = []
    global_max: List[float] = []
    ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开文件: {tif_path}")
    band_count = ds.RasterCount
    global_min = [math.inf] * band_count
    global_max = [-math.inf] * band_count
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    for bidx in range(1, band_count + 1):
        band = ds.GetRasterBand(bidx)
        nodata = band.GetNoDataValue()
        if nodata is None:
            nodata = default_nodata
        bx, by = band.GetBlockSize()
        if bx is None or bx <= 0:
            bx = 256
        if by is None or by <= 0:
            by = 256
        for yoff in range(0, ysize, by):
            ywin = min(by, ysize - yoff)
            for xoff in range(0, xsize, bx):
                xwin = min(bx, xsize - xoff)
                arr = band.ReadAsArray(xoff, yoff, xwin, ywin)
                if arr is None:
                    continue
                arr = np.asarray(arr)
                valid = is_valid(arr, nodata)
                if not valid.any():
                    continue
                vals = arr[valid]
                vmin = float(vals.min())
                vmax = float(vals.max())
                if vmin < global_min[bidx - 1]:
                    global_min[bidx - 1] = vmin
                if vmax > global_max[bidx - 1]:
                    global_max[bidx - 1] = vmax
    ds = None
    return band_count, global_min, global_max


def _gdal_pass1_one_star(args: Tuple[Path, Optional[float]]):
    """适配imap传参，避免lambda导致的pickle错误"""
    return _gdal_pass1_one(*args)


def pass1_min_max(tif_paths: List[Path], default_nodata: Optional[float], workers: int) -> Tuple[int, List[float], List[float]]:
    """返回：(band_count, mins, maxs) 全部为按索引的列表（长度为波段数）"""
    band_count: Optional[int] = None
    global_min: List[float] = []
    global_max: List[float] = []

    # GDAL全局配置（线程/缓存）
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    gdal.SetCacheMax(1024 * 1024 * 1024)

    # 并行：每个进程处理一部分文件并返回局部min/max，主进程做规约
    results = []
    with mp.Pool(processes=max(1, workers)) as pool:
        args_iter = ((p, default_nodata) for p in tif_paths)
        for res in tqdm(
            pool.imap_unordered(_gdal_pass1_one_star, args_iter),
            total=len(tif_paths), desc="Pass1 min/max", ncols=100, unit="file"
        ):
            results.append(res)

    # 规约
    for idx, (bc, mins_part, maxs_part) in enumerate(results):
        if band_count is None:
            band_count = bc
            global_min = [math.inf] * band_count
            global_max = [-math.inf] * band_count
        elif bc != band_count:
            raise RuntimeError(f"波段数不一致：第一个为{band_count}，某文件返回{bc}")
        for i in range(band_count):
            if math.isfinite(mins_part[i]) and mins_part[i] < global_min[i]:
                global_min[i] = mins_part[i]
            if math.isfinite(maxs_part[i]) and maxs_part[i] > global_max[i]:
                global_max[i] = maxs_part[i]

    if band_count is None:
        raise RuntimeError("未在目录中找到任何TIF文件")

    # 处理全为无效值的极端情况
    for i in range(band_count):
        if not math.isfinite(global_min[i]) or not math.isfinite(global_max[i]):
            global_min[i] = math.inf
            global_max[i] = -math.inf

    return band_count, global_min, global_max


def _gdal_pass2_one(
    tif_path: Path,
    band_count: int,
    mins: List[float],
    maxs: List[float],
    bins: int,
    default_nodata: Optional[float]
) -> Tuple[List[np.ndarray], List[int]]:
    hist_list = [np.zeros(bins, dtype=np.int64) for _ in range(band_count)]  # noqa: F841
    valid_counts = [0] * band_count  # noqa: F841
    # 预计算 scale/offset
    scales: List[float] = []
    offsets: List[float] = []
    for i in range(band_count):
        lo, hi = mins[i], maxs[i]
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            scales.append(0.0)
            offsets.append(0.0)
        else:
            scales.append((bins - 1) / (hi - lo))
            offsets.append(-lo)

    ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开文件: {tif_path}")
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    for bidx in range(1, band_count + 1):
        lo, hi = mins[bidx - 1], maxs[bidx - 1]
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            continue
        band = ds.GetRasterBand(bidx)
        nodata = band.GetNoDataValue()
        if nodata is None:
            nodata = default_nodata
        bx, by = band.GetBlockSize()
        if bx is None or bx <= 0:
            bx = 256
        if by is None or by <= 0:
            by = 256
        scale = scales[bidx - 1]
        offset = offsets[bidx - 1]
        for yoff in range(0, ysize, by):
            ywin = min(by, ysize - yoff)
            for xoff in range(0, xsize, bx):
                xwin = min(bx, xsize - xoff)
                arr = band.ReadAsArray(xoff, yoff, xwin, ywin)
                if arr is None:
                    continue
                arr = np.asarray(arr)
                valid = is_valid(arr, nodata)
                if not valid.any():
                    continue
                vals = arr[valid].astype(np.float64)
                idx = ((vals + offset) * scale).astype(np.int64)
                idx[idx < 0] = 0
                max_bin = bins - 1
                idx[idx > max_bin] = max_bin
                bincount = np.bincount(idx, minlength=bins)
                hist_list[bidx - 1] += bincount
                valid_counts[bidx - 1] += idx.size
    ds = None
    return hist_list, valid_counts


def _gdal_pass2_one_star(args: Tuple[Path, int, List[float], List[float], int, Optional[float]]):
    """适配imap传参，避免lambda导致的pickle错误"""
    return _gdal_pass2_one(*args)


def pass2_histogram(
    tif_paths: List[Path],
    band_count: int,
    mins: List[float],
    maxs: List[float],
    bins: int,
    default_nodata: Optional[float],
    workers: int
) -> Tuple[List[np.ndarray], List[int]]:
    """返回：(直方图列表 per band, 有效像元数列表 per band)"""
    hist_list = [np.zeros(bins, dtype=np.int64) for _ in range(band_count)]
    valid_counts = [0] * band_count

    # 预计算每个 band 的缩放系数
    scales: List[float] = []
    offsets: List[float] = []
    for i in range(band_count):
        lo, hi = mins[i], maxs[i]
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            scales.append(0.0)
            offsets.append(0.0)
        else:
            scales.append((bins - 1) / (hi - lo))
            offsets.append(-lo)

    # GDAL全局配置
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    gdal.SetCacheMax(1024 * 1024 * 1024)

    # 并行：每文件局部直方图与count，主进程规约求和
    total_hist = [np.zeros(bins, dtype=np.int64) for _ in range(band_count)]
    total_counts = [0] * band_count
    args_iter = ((p, band_count, mins, maxs, bins, default_nodata) for p in tif_paths)
    with mp.Pool(processes=max(1, workers)) as pool:
        for h_part, c_part in tqdm(
                pool.imap_unordered(_gdal_pass2_one_star, args_iter),
                total=len(tif_paths), desc="Pass2 hist", ncols=100, unit="file"):
            for i in range(band_count):
                total_hist[i] += h_part[i]
                total_counts[i] += c_part[i]

    return total_hist, total_counts


def percentile_from_hist(lo: float, hi: float, hist: np.ndarray, q: float) -> float:
    """从直方图估计分位数（线性插值）。q in [0,1]"""
    total = int(hist.sum())
    if total == 0 or not (math.isfinite(lo) and math.isfinite(hi)):
        return float('nan')
    target = q * (total - 1)
    cdf = np.cumsum(hist)
    # 找到第一个 >= target+1 的 bin
    k = int(np.searchsorted(cdf, target + 1))
    k = max(0, min(k, len(hist) - 1))
    # bin 边界
    width = (hi - lo) / len(hist)
    bin_lo = lo + k * width
    # 估计 bin 内位置
    prev_cum = 0 if k == 0 else int(cdf[k - 1])
    bin_count = int(hist[k])
    if bin_count <= 0:
        return bin_lo  # 罕见情况
    within = (target - prev_cum) / bin_count
    return float(bin_lo + within * width)


def main():
    parser = argparse.ArgumentParser(description="统计目录内TIF的每波段min/max/0.1%/99.9%（忽略无效值），GDAL并行版")
    parser.add_argument("--input_dir",  
                        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/ERA5_consistent_mosaic_withnodata_downsampled',
                        help="输入目录路径")
    parser.add_argument("--bins", type=int, default=2048, help="直方图bins数，越大越精细，越慢")
    parser.add_argument("--recursive", action="store_true", help="递归搜索子目录")
    parser.add_argument("--default-nodata", type=float, default=None, help="默认无效值（当文件未设置nodata时使用）")
    parser.add_argument("--output", default=None, help="结果保存为JSON路径（可选）")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2), help="并行进程数（文件级并行）")

    args = parser.parse_args()
    root = Path(args.input_dir)
    if not root.exists() or not root.is_dir():
        print(f"目录不存在：{root}")
        sys.exit(1)

    tif_paths = list_tifs(root, recursive=args.recursive)
    if not tif_paths:
        print("未找到任何TIF文件")
        sys.exit(1)

    # 第一遍：全局 min/max（并行）
    band_count, mins, maxs = pass1_min_max(tif_paths, args.default_nodata, args.workers)

    # 第二遍：直方图（并行）
    hists, counts = pass2_histogram(tif_paths, band_count, mins, maxs, args.bins, args.default_nodata, args.workers)

    # 结果汇总
    results: Dict[str, Dict[str, float]] = {}
    for i in range(band_count):
        lo, hi = mins[i], maxs[i]
        total = int(counts[i])
        p001 = percentile_from_hist(lo, hi, hists[i], 0.001)
        p999 = percentile_from_hist(lo, hi, hists[i], 0.999)
        results[f"band_{i+1}"] = {
            "count": total,
            "min": None if not math.isfinite(lo) else lo,
            "max": None if not math.isfinite(hi) else hi,
            "p0_1": None if not np.isfinite(p001) else p001,
            "p99_9": None if not np.isfinite(p999) else p999,
        }

    # 打印
    print("统计结果（忽略无效值）：")
    for k in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        r = results[k]
        print(f"- {k}: count={r['count']}, min={r['min']}, p0.1%={r['p0_1']}, p99.9%={r['p99_9']}, max={r['max']}")

    # 保存
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"已保存结果 -> {out_path}")


if __name__ == "__main__":
    main()


