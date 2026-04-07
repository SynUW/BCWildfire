#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import argparse
import warnings
from pathlib import Path
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=UserWarning)

# GDAL/RasterIO 线程优化
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.shutil import copy as rio_copy
from tqdm import tqdm


def list_tifs(folder: Path):
    exts = (".tif", ".tiff")
    return {p.stem: p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts and p.is_file()}


def grids_compatible(A: rasterio.DatasetReader, B: rasterio.DatasetReader) -> bool:
    try:
        return (A.crs == B.crs) and (A.transform == B.transform) and (A.width == B.width) and (A.height == B.height)
    except Exception:
        return False


def choose_resample(dtype, force=None, auto=True):
    if force is not None:
        m = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "average": Resampling.average,
            "mode": Resampling.mode,
        }
        return m.get(force, Resampling.nearest)
    if not auto:
        return Resampling.nearest
    return Resampling.nearest if np.issubdtype(dtype, np.integer) else Resampling.bilinear


def band_valid_mask(ds: rasterio.DatasetReader, band_idx: int, arr: np.ndarray, window=None) -> np.ndarray:
    """
    True=有效。
    与 arr 同窗口、同形状生成掩膜：
      - 初始 mask = read_masks(window)>0 （若失败，则全 True）
      - 若 ds.nodata 不为 None，则与 (arr != nodata) 取交集
    """
    try:
        mask = ds.read_masks(band_idx, window=window) > 0
        if mask.shape != arr.shape:
            # 形状不一致（例如驱动返回整幅 mask）→ 安全回退：不替换
            return np.ones_like(arr, dtype=bool)
    except Exception:
        mask = np.ones_like(arr, dtype=bool)

    nd = ds.nodata
    if nd is not None:
        try:
            valid_nd = (arr != nd)
            if valid_nd.shape != arr.shape:
                return np.ones_like(arr, dtype=bool)
            mask = mask & valid_nd
        except Exception:
            # 任意异常 → 不替换
            return np.ones_like(arr, dtype=bool)
    return mask


def reproject_to_template(src: rasterio.DatasetReader, tmpl: rasterio.DatasetReader,
                          resampling: Resampling):
    """将 src 重投影到 tmpl 网格，返回 (count, H, W) 的数组。"""
    H, W = tmpl.height, tmpl.width
    out = np.zeros((src.count, H, W), dtype=np.dtype(src.dtypes[0]))
    dst_nd = tmpl.nodata if tmpl.nodata is not None else src.nodata
    for b in range(1, src.count + 1):
        src_arr = src.read(b)
        out_band = out[b - 1]
        if dst_nd is not None:
            out_band.fill(dst_nd)
        reproject(
            source=src_arr,
            destination=out_band,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tmpl.transform,
            dst_crs=tmpl.crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=dst_nd,
        )
    return out


def fuse_pair(b_path: Path,
              a_path: Path,
              out_path: Path,
              suffix: str = "",
              force_resample: bool = False,
              resample_policy: str | None = None,
              resample_auto: bool = True,
              copy_when_b_missing: bool = True) -> tuple[str, bool, str]:
    """
    用 A 填补 B 的无效像元（逐波段）。异常或不匹配时按“较小者视为无效 → 直接复制较大者”的策略，不做替换。
    返回: (basename, success, message)
    """
    name = (b_path or a_path).stem

    if b_path is None:
        if not copy_when_b_missing or a_path is None:
            return name, False, "B missing and A missing or copy disabled"
        out_path = out_path.with_name(out_path.stem + suffix + out_path.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rio_copy(a_path, out_path, driver="GTiff", copy_src_overviews=True)
        return name, True, "copied A (B missing)"

    with rasterio.open(b_path) as B:
        profile = B.profile.copy()
        profile.update(
            driver="GTiff",
            tiled=True,
            compress="LZW",
            BIGTIFF="IF_SAFER",
            num_threads="ALL_CPUS",
        )
        # predictor
        if np.issubdtype(np.dtype(B.dtypes[0]), np.floating):
            profile.update(predictor=3)
        else:
            profile.update(predictor=2)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        dst_fn = out_path.with_name(out_path.stem + (suffix or "") + ".tif")

        if a_path is None:
            rio_copy(b_path, dst_fn, driver="GTiff", copy_src_overviews=True)
            return name, True, "copied B (A missing)"

        with rasterio.open(a_path) as A:
            need_reproj = force_resample or (not grids_compatible(A, B))
            if need_reproj:
                resampling = choose_resample(np.dtype(A.dtypes[0]), force=resample_policy, auto=resample_auto)
                A_on_B = reproject_to_template(A, B, resampling=resampling)  # (count,H,W)
                # 用于有效性判断的 nodata
                a_nd = B.nodata if B.nodata is not None else A.nodata

            with rasterio.open(dst_fn, "w", **profile) as dst:
                if B.nodata is not None:
                    dst.nodata = B.nodata

                # 逐波段
                for b in range(1, B.count + 1):
                    # 读 B
                    out_band = np.zeros((B.height, B.width), dtype=np.dtype(B.dtypes[b - 1]))

                    # 分窗口，避免一次性整图
                    for ji, win in B.block_windows(b):
                        try:
                            b_arr = B.read(b, window=win)
                            b_valid = band_valid_mask(B, b, b_arr, window=win)
                        except Exception:
                            # 任意异常：认为 B 窗口有效，直接写回
                            out_band[win.row_off:win.row_off+win.height,
                                     win.col_off:win.col_off+win.width] = B.read(b, window=win)
                            continue

                        # 准备 A 的同窗口数据与有效掩膜
                        if need_reproj:
                            try:
                                a_arr = A_on_B[b - 1][win.row_off:win.row_off+win.height,
                                                      win.col_off:win.col_off+win.width]
                                if a_arr.shape != b_arr.shape:
                                    # 形状不一致：不替换
                                    fill_mask = np.zeros_like(b_arr, dtype=bool)
                                else:
                                    if a_nd is not None:
                                        a_valid = (a_arr != a_nd)
                                    else:
                                        # 没有 nodata：保守，不用 A 替换
                                        a_valid = np.zeros_like(a_arr, dtype=bool)
                                    fill_mask = (~b_valid) & a_valid
                            except Exception:
                                # 任意异常：不替换
                                fill_mask = np.zeros_like(b_arr, dtype=bool)
                                a_arr = b_arr  # 避免未定义
                        else:
                            # 网格一致：直接同窗口读取
                            try:
                                a_arr = A.read(b, window=win)
                                if a_arr.shape != b_arr.shape:
                                    # 形状不一致：不替换（较小者视为无效）
                                    fill_mask = np.zeros_like(b_arr, dtype=bool)
                                else:
                                    a_valid = band_valid_mask(A, b, a_arr, window=win)
                                    # 仍做一次形状防御
                                    if a_valid.shape != b_arr.shape:
                                        fill_mask = np.zeros_like(b_arr, dtype=bool)
                                    else:
                                        fill_mask = (~b_valid) & a_valid
                            except Exception:
                                fill_mask = np.zeros_like(b_arr, dtype=bool)
                                a_arr = b_arr

                        # 合成：默认复制 B；仅在 fill_mask=True 的像元用 A 替换
                        out_slice = b_arr.copy()
                        if fill_mask.any():
                            out_slice[fill_mask] = a_arr[fill_mask]

                        out_band[win.row_off:win.row_off+win.height,
                                 win.col_off:win.col_off+win.width] = out_slice

                    dst.write(out_band, b)

        return name, True, "fused"


def _worker(args):
    return fuse_pair(*args)


def main():
    ap = argparse.ArgumentParser(description="用 A(MYD) 填补 B(MOD) 的无效像元（逐波段，稳健回退版）")
    ap.add_argument("--b_dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MOD11A2_mosaic', help="B 基底目录（MOD）")
    ap.add_argument("--a_dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MYD11A2_mosaic', help="A 填补源目录（MYD）")
    ap.add_argument("--out_dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MCD11A2_mosaic', help="输出目录")
    ap.add_argument("--suffix", default="", help="输出文件名后缀（如 _fused）")
    ap.add_argument("--workers", type=int, default=4, help="并行进程数")
    ap.add_argument("--resample_auto", action="store_true", help="自动选择重采样（整型=nearest；浮点=bilinear）")
    ap.add_argument("--resample_policy", default=None,
                    choices=[None, "nearest", "bilinear", "cubic", "average", "mode"],
                    help="强制重采样策略")
    ap.add_argument("--force_resample", action="store_true", help="即使网格一致也重采样 A→B")
    ap.add_argument("--copy_when_b_missing", action="store_true", help="当 B 缺失且 A 存在时，复制 A 到输出")
    args = ap.parse_args()

    b_dir = Path(args.b_dir)
    a_dir = Path(args.a_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b_map = list_tifs(b_dir)
    a_map = list_tifs(a_dir)
    keys = sorted(set(b_map.keys()) | set(a_map.keys()))

    tasks = []
    for k in keys:
        b_path = b_map.get(k)
        a_path = a_map.get(k)
        out_path = out_dir / f"{k}.tif"
        tasks.append((
            b_path,
            a_path,
            out_path,
            args.suffix,
            args.force_resample,
            args.resample_policy,
            args.resample_auto,
            args.copy_when_b_missing
        ))

    with Pool(processes=max(1, args.workers)) as pool:
        results = []
        for r in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), desc="Fusing"):
            results.append(r)

    ok = sum(1 for _, s, _ in results if s)
    fail = len(results) - ok
    print(f"\nDone. Success: {ok}, Failed: {fail}")
    for name, s, msg in results:
        if not s:
            print(f"[FAIL] {name}: {msg}")


if __name__ == "__main__":
    main()
