#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# ==================== 配置 ====================
CSV_PATH = "min_max_clip.csv"  # 含 folder name, band, nan, min, max
INPUT_ROOT = Path("/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_10x_without_qa_masked_clip")  # 已裁切的数据根目录
OUTPUT_ROOT = Path("/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_10x_without_qa_masked_clip_min_max_normalized")  # 归一化输出根目录
WORKERS = max(1, (os.cpu_count() or 4) - 1)
OUTPUT_NODATA = -9999.0         # 输出栅格的 nodata（float32）
DENOM_EPS = 1e-12                # 防止除0
ENABLE_COMPRESS = True           # 输出压缩（QGIS 打开更快）

# ==================== 读取规则 ====================
def parse_band_to_int(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str):
        m = re.search(r"(\d+)", x)
        if m:
            return int(m.group(1))
    raise ValueError(f"无法解析 band 字段: {x}")

def load_rules(csv_path: str):
    # 根据文件扩展名选择读取方法
    if csv_path.endswith('.xlsx') or csv_path.endswith('.xls'):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)
    
    # 容错列名 - 处理带引号的列名
    cols = {}
    for c in df.columns:
        # 移除单引号并转换为小写作为键
        clean_name = c.strip().strip("'").lower()
        cols[clean_name] = c
    
    need = ["folder name", "band", "nan", "min", "max"]
    for k in need:
        if k not in cols:
            raise ValueError(f"CSV 缺少列：{k} (实际列名: {list(df.columns)})")

    # { folder: { band(1-based): (nodata_in_input, vmin, vmax) } }
    rules = {}
    for _, row in df.iterrows():
        folder = str(row[cols["folder name"]]).strip()
        band = parse_band_to_int(row[cols["band"]])  # 注意：1-based
        nodata_in = float(row[cols["nan"]]) if pd.notna(row[cols["nan"]]) else None
        vmin = float(row[cols["min"]])
        vmax = float(row[cols["max"]])
        rules.setdefault(folder, {})[band] = (nodata_in, vmin, vmax)
    return rules

# ==================== 归一化（按文件） ====================
def normalize_one_file(in_tif: Path, out_tif: Path, band_rules: dict):
    """
    band_rules: {band_index (1-based): (nodata_in_input, vmin, vmax)}
    将每个在规则内的波段做 0-1 归一化（保持 nodata 不变），输出 float32。
    """
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    if out_tif.exists():
        return True, "exists"

    with rasterio.Env(NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=1024):
        with rasterio.open(in_tif) as src:
            profile = src.profile.copy()
            # 输出为 float32，nodata 统一为 OUTPUT_NODATA
            profile.update(dtype="float32", nodata=OUTPUT_NODATA)

            if ENABLE_COMPRESS:
                profile.update(
                    compress="DEFLATE",
                    predictor=1,      # 使用 predictor=1 避免 libtiff 版本兼容性问题
                    zlevel=6,
                    tiled=True,
                    blockxsize=max(16, min(512, src.width) // 16 * 16),  # 确保是 16 的倍数，最小 16
                    blockysize=max(16, min(512, src.height) // 16 * 16),  # 确保是 16 的倍数，最小 16
                )

            with rasterio.open(out_tif, "w", **profile) as dst:
                # 对每个波段逐块处理
                for b in src.indexes:
                    has_rule = (b in band_rules)
                    nodata_in = vmin = vmax = None
                    if has_rule:
                        nodata_in, vmin, vmax = band_rules[b]

                    for _, win in src.block_windows(b):
                        arr = src.read(b, window=win)

                        # 先构造输出数组（float32），默认填 nodata
                        out = np.full(arr.shape, OUTPUT_NODATA, dtype=np.float32)

                        if not has_rule:
                            # 没有归一化规则的波段：直接拷贝并强制为 float32（nodata 同步映射）
                            if nodata_in is None:
                                valid = np.isfinite(arr)
                            else:
                                valid = arr != nodata_in
                            if np.any(valid):
                                out[valid] = arr[valid].astype(np.float32, copy=False)
                            # 写块
                            dst.write(out, b, window=win)
                            continue

                        # 有归一化规则的波段
                        if nodata_in is None:
                            valid = np.isfinite(arr)
                        else:
                            valid = arr != nodata_in

                        if not np.any(valid):
                            dst.write(out, b, window=win)
                            continue

                        denom = max(float(vmax - vmin), DENOM_EPS)
                        # 注意：输入已裁切到 [vmin, vmax]，这里再稳妥 clip 一下
                        work = arr.astype(np.float64, copy=False)
                        work = np.clip(work, vmin, vmax, out=work)
                        norm = (work - vmin) / denom

                        out[valid] = norm[valid].astype(np.float32, copy=False)
                        # 无效像元保持 OUTPUT_NODATA
                        dst.write(out, b, window=win)

    return True, "ok"

# ==================== 任务生成 ====================
def build_tasks(rules: dict):
    """
    根据 rules 在 INPUT_ROOT 下匹配文件，生成待处理任务列表
    """
    tasks = []
    for folder, band_rules in rules.items():
        in_dir = INPUT_ROOT / folder
        if not in_dir.exists():
            print(f"[跳过] 输入目录不存在：{in_dir}")
            continue
        out_dir = OUTPUT_ROOT / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        for tif in in_dir.glob("*.tif"):
            tasks.append((tif, out_dir / tif.name, band_rules))
    return tasks

# ==================== 主流程 ====================
def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rules = load_rules(CSV_PATH)
    tasks = build_tasks(rules)
    if not tasks:
        print("未发现可处理的 tif 文件。")
        return

    print(f"共 {len(tasks)} 个文件，将使用 {WORKERS} 进程归一化。")
    ok, fail = 0, 0
    ctx = mp.get_context("fork" if os.name != "nt" else "spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        futures = [ex.submit(normalize_one_file, in_tif, out_tif, band_rules) for (in_tif, out_tif, band_rules) in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Normalizing"):
            try:
                success, _ = fut.result()
                ok += int(success)
                fail += int(not success)
            except Exception as e:
                fail += 1
                print("[错误]", e)

    print(f"完成：成功 {ok}，失败 {fail}。输出目录：{OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
