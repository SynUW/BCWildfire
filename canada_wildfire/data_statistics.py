#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import math
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

# -------------------- 工具函数 --------------------

def find_tifs(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return [p for p in root.rglob("*") if p.suffix.lower() in (".tif", ".tiff")]

def find_subfolder_tifs(root: Path) -> Dict[str, List[Path]]:
    """
    查找包含多个子文件夹的父文件夹中的 tif 文件
    返回: {子文件夹名: [tif文件列表]}
    """
    if root.is_file():
        return {"single_file": [root]}

    subfolder_tifs: Dict[str, List[Path]] = {}
    for subfolder in root.iterdir():
        if subfolder.is_dir():
            tifs = [p for p in subfolder.rglob("*") if p.suffix.lower() in (".tif", ".tiff")]
            if tifs:
                subfolder_tifs[subfolder.name] = tifs

    return subfolder_tifs

def discover_inputs(root: Path) -> Tuple[str, List[Path], Dict[str, List[Path]]]:
    """
    返回 (mode, all_tifs, subfolder_map)
    - mode: "single" | "flat" | "hier"
    - all_tifs: 统一的 tif 列表（single/flat 直接列出；hier 汇总所有子文件夹内）
    - subfolder_map: 仅在 hier 模式下有效
    """
    if root.is_file():
        return "single", [root], {}

    # 扁平：根目录直接包含 tif
    flat_tifs = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]
    if flat_tifs:
        return "flat", sorted(flat_tifs), {}

    # 分层：按子文件夹聚合
    sub_map = find_subfolder_tifs(root)
    if sub_map:
        all_tifs: List[Path] = []
        for tlist in sub_map.values():
            all_tifs.extend(tlist)
        return "hier", sorted(all_tifs), sub_map

    return "empty", [], {}

def _iter_block_windows(ds: rasterio.io.DatasetReader):
    """优先使用内置块窗口；若无块信息则退化为整图或大窗口。"""
    try:
        # block_windows(1) 返回 ( (ji), Window )
        for _, win in ds.block_windows(1):
            yield win
    except Exception:
        # 退化：整图一个窗口
        yield Window(0, 0, ds.width, ds.height)

def _valid_mask(arr, nodata_value: Optional[float], extra_invalid: List[float]):
    # 这里 arr 可能是 masked array，优先用掩膜
    if np.ma.isMaskedArray(arr):
        mask = ~arr.mask
        data = arr.data
    else:
        data = arr
        mask = np.isfinite(data)
    if nodata_value is not None:
        mask &= (data != nodata_value)
    if extra_invalid:
        for v in extra_invalid:
            mask &= (data != v)
    return mask

def _update_stats(run: Dict[str, Any], valid: np.ndarray):
    if valid.size == 0:
        return
    # 用 float64 累加器，避免溢出
    run["count"] += valid.size
    s = valid.sum(dtype=np.float64)
    ss = np.square(valid, dtype=np.float64).sum(dtype=np.float64)
    run["sum"] += float(s)
    run["sumsq"] += float(ss)
    vmin = valid.min()
    vmax = valid.max()
    run["min"] = float(vmin) if run["min"] is None else min(run["min"], float(vmin))
    run["max"] = float(vmax) if run["max"] is None else max(run["max"], float(vmax))

def _reservoir_append(reservoir: np.ndarray, chunk: np.ndarray, cap: int) -> np.ndarray:
    if chunk.size == 0:
        return reservoir
    if reservoir.size == 0:
        if chunk.size <= cap:
            return chunk.astype(np.float64, copy=True)
        idx = np.linspace(0, chunk.size - 1, cap, dtype=int)
        return chunk[idx].astype(np.float64, copy=False)
    total = reservoir.size + chunk.size
    if total <= cap:
        return np.concatenate([reservoir, chunk.astype(np.float64, copy=False)], axis=0)
    # 线性下采样（近似均匀）
    combined = np.concatenate([reservoir, chunk.astype(np.float64, copy=False)], axis=0)
    keep_idx = np.linspace(0, combined.size - 1, cap, dtype=int)
    return combined[keep_idx]

def _hist_accumulate(counts: np.ndarray, edges: np.ndarray, valid: np.ndarray):
    """将一段数据累加进固定边界的直方图计数。"""
    if valid.size == 0:
        return
    c, _ = np.histogram(valid, bins=edges)
    counts += c

def _finalize_stats(run: Dict[str, Any]) -> Tuple[float, float]:
    if run["count"] == 0:
        return None, None
    mean = run["sum"] / run["count"]
    var = max(run["sumsq"] / run["count"] - mean * mean, 0.0)
    return float(mean), float(math.sqrt(var))

# -------------------- 子进程：处理单文件 --------------------

def process_single_file(
    tif_path: str,
    extra_invalid: List[float],
    sample_stride: int,
    per_band_reservoir_cap: int,
    want_quantiles: bool,
    want_hist: bool,
    hist_edges_by_band: Optional[Dict[int, np.ndarray]],
    gdal_cachemax: Optional[int],
) -> Dict[str, Any]:
    """
    返回：
      {
        "file": path,
        "bands": {
          band_index: {
             "count":, "sum":, "sumsq":, "min":, "max":,
             "dtype":, "crs":, "nodata":,
             "samples": np.ndarray (可选),
             "hist_counts": np.ndarray (可选)  # bins-1 长度
          },
          ...
        }
      }
    """
    # 每个进程独立设置 GDAL 环境
    env_kwargs = {}
    if gdal_cachemax:
        env_kwargs["GDAL_CACHEMAX"] = gdal_cachemax
    env_kwargs["NUM_THREADS"] = "ALL_CPUS"

    out = {"file": tif_path, "bands": {}}
    with rasterio.Env(**env_kwargs):
        with rasterio.open(tif_path) as ds:
            nodata = ds.nodata
            crs = str(ds.crs) if ds.crs else ""
            bands = ds.indexes

            for b in bands:
                # 初始化
                stats = dict(count=0, sum=0.0, sumsq=0.0, min=None, max=None,
                             dtype=ds.dtypes[b-1], crs=crs, nodata=nodata)
                samples = np.array([], dtype=np.float64) if want_quantiles else None

                # 直方图初始化
                if want_hist and hist_edges_by_band and b in hist_edges_by_band:
                    edges = hist_edges_by_band[b]
                    hist_counts = np.zeros(len(edges) - 1, dtype=np.int64)
                else:
                    hist_counts = None
                    edges = None

                # 遍历块
                for _, win in ds.block_windows(b):
                    # masked 读取，自动考虑 nodata（对很多格式生效）
                    arr = ds.read(b, window=win, masked=True)
                    if sample_stride > 1:
                        arr = arr[::sample_stride, ::sample_stride]

                    # 快速掩膜/取值（尽量避免拷贝）
                    mask = _valid_mask(arr, nodata, extra_invalid)
                    if not np.any(mask):
                        continue
                    valid = arr.data[mask].astype(np.float64, copy=False)

                    # 统计
                    _update_stats(stats, valid)

                    # 直方图累加
                    if hist_counts is not None:
                        _hist_accumulate(hist_counts, edges, valid)

                    # 分位数所需样本（小容量）
                    if want_quantiles:
                        # 从当前块再做一步下采样以控内存：块级最多 8k
                        if valid.size > 8192:
                            sel = np.linspace(0, valid.size - 1, 8192, dtype=int)
                            chunk = valid[sel]
                        else:
                            chunk = valid
                        samples = _reservoir_append(samples, chunk, per_band_reservoir_cap)

                # 汇总
                band_out = stats
                if want_quantiles:
                    band_out["samples"] = samples
                if hist_counts is not None:
                    band_out["hist_counts"] = hist_counts
                    band_out["hist_edges"] = edges  # 只带回边界用于 sanity check
                out["bands"][b] = band_out

    return out

# -------------------- 主流程 --------------------

def plot_overall_hist_from_counts(counts: np.ndarray, edges: np.ndarray, out_png: Path, title: str):
    if counts is None or counts.sum() == 0:
        plt.figure(figsize=(8,5))
        plt.title("No valid data to plot")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150); plt.close(); return
    plt.figure(figsize=(10,6))
    # 还原中点坐标用于可视化
    mids = 0.5*(edges[:-1] + edges[1:])
    plt.bar(mids, counts, width=np.diff(edges), align="center")
    plt.yscale("log")
    plt.xlabel("Value"); plt.ylabel("Count (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_overall_hist_from_samples(samples: np.ndarray, out_png: Path, bins: int, clip_low: float, clip_high: float, title: str):
    if samples.size == 0:
        plt.figure(figsize=(8,5))
        plt.title("No valid data to plot")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150); plt.close(); return
    lo = np.percentile(samples, clip_low) if clip_low is not None else np.min(samples)
    hi = np.percentile(samples, clip_high) if clip_high is not None else np.max(samples)
    if hi <= lo:
        lo, hi = np.min(samples), np.max(samples)
    plt.figure(figsize=(10,6))
    plt.hist(np.clip(samples, lo, hi), bins=bins, range=(lo, hi))
    plt.yscale("log")
    plt.xlabel("Value"); plt.ylabel("Count (log scale)")
    plt.title(title)
    for q in [5, 50, 95]:
        v = np.percentile(samples, q)
        if lo <= v <= hi:
            plt.axvline(v, linestyle="--")
            plt.text(v, plt.ylim()[1]*0.9, f"P{q}:{v:.3g}", rotation=90, va="top", ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="高速统计 GeoTIFF（单文件、文件夹或包含多个子文件夹的父文件夹）数值分布，支持多进程与直方图累加。子文件夹模式下每个波段输出一行统计。"
    )
    parser.add_argument("--input", type=str, default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_10x_without_qa_masked_clip_min_max_normalized/MOD21A1DN_multibands_withoutFiltering_merged_10x', help="输入 .tif 文件、文件夹或包含多个子文件夹的父文件夹")
    parser.add_argument("--ignore", type=float, action="append", default=[-9999], help="额外无效值，可多次指定")
    parser.add_argument("--sample-stride", type=int, default=1, help="块内抽样步距（>1 更快，默认 2）")
    parser.add_argument("--bins", type=int, default=256, help="绘图 bins（样本法时生效）")
    parser.add_argument("--clip-low", type=float, default=0.0, help="样本法直方图下分位（%）")
    parser.add_argument("--clip-high", type=float, default=100.0, help="样本法直方图上分位（%）")
    parser.add_argument("--out-png", type=str, default="value_distribution.png", help="输出直方图 PNG")
    parser.add_argument("--out-csv", type=str, default="value_stats.csv", help="输出统计 CSV")
    parser.add_argument("--global-stats", action="store_true", help="输入为文件夹时，按波段聚合全局统计")
    parser.add_argument("--workers", type=int, default=max(1, (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 4) - 1),
                        help="并行进程数（默认 CPU-1）")
    parser.add_argument("--enable-quantiles", action="store_true",
                        help="启用分位数统计（会构建小容量样本池，稍慢）")
    parser.add_argument("--reservoir-cap", type=int, default=200_000,
                        help="每波段样本池上限（启用分位数时生效，默认 200k）")
    parser.add_argument("--enable-hist", action="store_true",
                        help="启用总体直方图逐块累加（无需大样本，更快更省内存）")
    parser.add_argument("--hist-min", type=float, default=None, help="总体直方图最小值（不设则自动估计）")
    parser.add_argument("--hist-max", type=float, default=None, help="总体直方图最大值（不设则自动估计）")
    parser.add_argument("--gdal-cachemax", type=int, default=10240, help="GDAL 缓存（MB），如 1024")
    args = parser.parse_args()

    root = Path(args.input).expanduser().resolve()

    mode, all_tifs, subfolder_tifs = discover_inputs(root)
    if mode == "empty" or not all_tifs:
        print(f"未找到 GeoTIFF 文件：{root}")
        sys.exit(1)

    subfolder_mode = (mode == "hier")
    # 文件夹输入统一聚合统计；单文件时由 --global-stats 控制
    use_global_stats = True if mode in ("flat", "hier") else bool(args.global_stats)

    if mode == "hier":
        print(f"发现 {len(subfolder_tifs)} 个子文件夹；共 {len(all_tifs)} 个 tif 文件。每个波段将输出一行统计")
    elif mode == "flat":
        print(f"发现扁平文件夹，共 {len(all_tifs)} 个 tif 文件。每个波段将输出一行统计")

    # -------- 预估直方图边界（可选：两步法先扫极小样本） --------
    hist_edges_by_band = None
    if args.enable_hist:
        # 用一次快速扫描确定每个 band 的全局 min/max（stride 更大，且不保留样本）
        per_band_minmax = {}
        quick_stride = max(args.sample_stride, 8)  # 更粗的抽样
        
        # 直接用统一的 all_tifs 列表
        for tif in tqdm(all_tifs, desc="Quick scan for hist edges"):
            with rasterio.Env(NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=args.gdal_cachemax):
                try:
                    with rasterio.open(tif) as ds:
                        nodata = ds.nodata
                        for b in ds.indexes:
                            r = per_band_minmax.get(b, {"min": None, "max": None})
                            for _, win in ds.block_windows(b):
                                arr = ds.read(b, window=win, masked=True)
                                arr = arr[::quick_stride, ::quick_stride]
                                mask = _valid_mask(arr, nodata, args.ignore)
                                if not np.any(mask):
                                    continue
                                v = arr.data[mask]
                                vmin = float(v.min())
                                vmax = float(v.max())
                                r["min"] = vmin if r["min"] is None else min(r["min"], vmin)
                                r["max"] = vmax if r["max"] is None else max(r["max"], vmax)
                            per_band_minmax[b] = r
                except Exception as e:
                    print(f"[警告] 快速扫描失败 {tif}: {e}")

        hist_edges_by_band = {}
        for b, mm in per_band_minmax.items():
            if mm["min"] is None or mm["max"] is None or mm["min"] >= mm["max"]:
                # 兜底
                edges = np.linspace(0, 1, args.bins + 1, dtype=np.float64)
            else:
                lo = args.hist_min if args.hist_min is not None else mm["min"]
                hi = args.hist_max if args.hist_max is not None else mm["max"]
                if hi <= lo:
                    hi = lo + 1.0
                edges = np.linspace(lo, hi, args.bins + 1, dtype=np.float64)
            hist_edges_by_band[b] = edges

    # -------- 并行处理 --------
    results = []
    rows_out: List[Dict[str, Any]] = []
    all_samples = np.array([], dtype=np.float64)  # 仅用于样本法绘图
    # 直方图累计（全局）
    global_hist_counts = None
    global_hist_edges = None

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork" if sys.platform != "win32" else "spawn")) as ex:
        futs = []

        for tif in all_tifs:
            fut = ex.submit(
                process_single_file,
                str(tif),
                args.ignore,
                max(1, args.sample_stride),
                args.reservoir_cap,
                args.enable_quantiles,
                args.enable_hist,
                hist_edges_by_band,
                args.gdal_cachemax,
            )
            futs.append(fut)

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing files (parallel)"):
            r = fut.result()
            results.append(r)

    # -------- 汇总全局统计 --------
    # band -> aggregator
    agg: Dict[int, Dict[str, Any]] = {}

    for r in results:
        for b, st in r["bands"].items():
            g = agg.get(b, None)
            if g is None:
                g = {
                    "dtype": st["dtype"], "crs": st["crs"], "nodata_value": st["nodata"],
                    "count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None
                }
                # 合并样本池（用于分位数）
                if args.enable_quantiles:
                    g["samples"] = np.array([], dtype=np.float64)
                # 合并直方图
                if args.enable_hist and "hist_counts" in st and st["hist_counts"] is not None:
                    g["hist_counts"] = st["hist_counts"].astype(np.int64, copy=True)
                    g["hist_edges"] = st["hist_edges"]
                agg[b] = g

            # 累加统计
            g["count"] += st["count"]
            g["sum"] += st["sum"]
            g["sumsq"] += st["sumsq"]
            if st["min"] is not None:
                g["min"] = st["min"] if g["min"] is None else min(g["min"], st["min"])
            if st["max"] is not None:
                g["max"] = st["max"] if g["max"] is None else max(g["max"], st["max"])

            # 合并样本
            if args.enable_quantiles and "samples" in st and st["samples"] is not None:
                g["samples"] = _reservoir_append(g["samples"], st["samples"], args.reservoir_cap)

            # 合并直方图
            if args.enable_hist and "hist_counts" in st and st["hist_counts"] is not None:
                if "hist_counts" not in g or g["hist_counts"] is None:
                    g["hist_counts"] = st["hist_counts"].astype(np.int64, copy=True)
                    g["hist_edges"] = st["hist_edges"]
                else:
                    g["hist_counts"] += st["hist_counts"]

    # 生成最终行
    if use_global_stats:
        for b in sorted(agg.keys()):
            g = agg[b]
            mean, std = _finalize_stats(g)
            row = {
                "band": b,
                "dtype": g["dtype"], "crs": g["crs"], "nodata_value": g["nodata_value"],
                "count": g["count"], "valid_count": g["count"],
                "min": g["min"], "max": g["max"], "mean": mean, "std": std,
                "p1": None, "p5": None, "p25": None, "p50": None, "p75": None, "p95": None, "p99": None,
            }
            
            # 如果是子文件夹模式，添加子文件夹信息
            if subfolder_mode:
                # 找到包含这个波段的所有子文件夹
                subfolders_with_band = []
                for subfolder_name, tifs in subfolder_tifs.items():
                    for tif in tifs:
                        try:
                            with rasterio.open(tif) as ds:
                                if b in ds.indexes:
                                    subfolders_with_band.append(subfolder_name)
                                    break
                        except Exception:
                            continue
                row["subfolders"] = ",".join(sorted(set(subfolders_with_band)))
                row["num_subfolders"] = len(set(subfolders_with_band))
            if args.enable_quantiles and g.get("samples", None) is not None and g["samples"].size > 0:
                for q, key in [(1,"p1"),(5,"p5"),(25,"p25"),(50,"p50"),(75,"p75"),(95,"p95"),(99,"p99")]:
                    row[key] = float(np.percentile(g["samples"], q))
                all_samples = _reservoir_append(all_samples, g["samples"], cap=1_000_000)  # 仅用于绘图的小池
            rows_out.append(row)

            # 同步全局直方图（若启用）
            if args.enable_hist and g.get("hist_counts", None) is not None:
                if global_hist_counts is None:
                    global_hist_counts = g["hist_counts"].copy()
                    global_hist_edges = g["hist_edges"]
                else:
                    global_hist_counts += g["hist_counts"]
    else:
        # 非全局模式：按文件+波段逐行（仍然受益于并行）
        for r in results:
            for b, st in r["bands"].items():
                mean, std = _finalize_stats(st)
                row = {
                    "file": r["file"],
                    "band": b,
                    "dtype": st["dtype"], "crs": st["crs"], "nodata_value": st["nodata"],
                    "count": st["count"], "valid_count": st["count"],
                    "min": st["min"], "max": st["max"], "mean": mean, "std": std,
                    "p1": None, "p5": None, "p25": None, "p50": None, "p75": None, "p95": None, "p99": None,
                }
                if args.enable_quantiles and "samples" in st and st["samples"] is not None and st["samples"].size > 0:
                    for q, key in [(1,"p1"),(5,"p5"),(25,"p25"),(50,"p50"),(75,"p75"),(95,"p95"),(99,"p99")]:
                        row[key] = float(np.percentile(st["samples"], q))
                    all_samples = _reservoir_append(all_samples, st["samples"], cap=1_000_000)
                rows_out.append(row)

                if args.enable_hist and "hist_counts" in st and st["hist_counts"] is not None:
                    if global_hist_counts is None:
                        global_hist_counts = st["hist_counts"].copy()
                        global_hist_edges = st["hist_edges"]
                    else:
                        global_hist_counts += st["hist_counts"]

    # 写 CSV
    if rows_out:
        df = pd.DataFrame(rows_out)
        if use_global_stats:
            if subfolder_mode:
                sort_cols = ["band"]
            else:
                sort_cols = ["band"]
        else:
            sort_cols = ["file", "band"]
        rest = [c for c in df.columns if c not in sort_cols]
        df = df[sort_cols + rest]
        df.to_csv(args.out_csv, index=False)
        print(f"统计表已保存：{args.out_csv}（共 {len(df)} 行）")
        if subfolder_mode:
            print(f"每个波段一行，共 {len(df)} 个波段")
    else:
        print("没有可用的统计结果（可能全部为无效像元）。")

    # 绘图
    title = "Global Value Distribution (aggregated across all files, valid pixels)" if use_global_stats \
            else "Overall Value Distribution (all files & bands, valid pixels)"

    if args.enable_hist and global_hist_counts is not None:
        plot_overall_hist_from_counts(global_hist_counts, global_hist_edges, Path(args.out_png), title)
    else:
        # 回退：用小样本池画图（耗时略大，但已大幅缩小池子）
        plot_overall_hist_from_samples(all_samples, Path(args.out_png),
                                       bins=args.bins, clip_low=args.clip_low, clip_high=args.clip_high, title=title)
    print(f"直方图已保存：{args.out_png}")

if __name__ == "__main__":
    main()
