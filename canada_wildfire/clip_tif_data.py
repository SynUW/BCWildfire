#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

# -------------------- 配置 --------------------
CSV_PATH = "min_max_clip.csv"  # 你的Excel另存为CSV后的路径
INPUT_ROOT = Path("/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_10x_without_qa_masked")
OUTPUT_ROOT = Path("/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_10x_without_qa_masked_clip")
OUTPUT_COMPRESS = True  # 是否在输出时开启压缩（推荐）
WORKERS = max(1, (os.cpu_count() or 4) - 1)  # 进程数

# -------------------- 读取规则 --------------------
def parse_band_to_int(x):
    """
    支持 'band1' 或 1 两种写法；注意：band是从1开始计数（你的表就是1-based）
    """
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str):
        m = re.search(r"(\d+)", x)
        if m:
            return int(m.group(1))
    raise ValueError(f"无法解析band字段: {x}")

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
            raise ValueError(f"CSV缺少列: {k} (实际列名: {list(df.columns)})")

    rules = {}
    for _, row in df.iterrows():
        folder = str(row[cols["folder name"]]).strip()
        band = parse_band_to_int(row[cols["band"]])  # 1-based
        nodata = float(row[cols["nan"]]) if pd.notna(row[cols["nan"]]) else None
        vmin = float(row[cols["min"]])
        vmax = float(row[cols["max"]])
        rules.setdefault(folder, {})[band] = (nodata, vmin, vmax)
    return rules

# -------------------- 裁剪核心（按文件） --------------------
def clip_one_file(in_tif: Path, out_tif: Path, band_rules: dict):
    """
    band_rules: {band_index (1-based): (nodata, vmin, vmax)}
    """
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    if out_tif.exists():
        return True, "exists"

    # 为每个进程设置GDAL参数（多核读取）
    with rasterio.Env(NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=1024):
        with rasterio.open(in_tif) as src:
            profile = src.profile.copy()

            # 开启压缩（可选）
            if OUTPUT_COMPRESS:
                profile.update(
                    compress="DEFLATE",
                    predictor=1,  # 使用 predictor=1 避免 libtiff 版本兼容性问题
                    zlevel=6,
                    tiled=True,
                    blockxsize=max(16, min(512, src.width) // 16 * 16),  # 确保是 16 的倍数，最小 16
                    blockysize=max(16, min(512, src.height) // 16 * 16),  # 确保是 16 的倍数，最小 16
                )

            with rasterio.open(out_tif, "w", **profile) as dst:
                # 先写入原数据（避免无规则波段漏写）
                # 采用块方式复制更省内存
                for b in src.indexes:
                    for _, win in src.block_windows(b):
                        arr = src.read(b, window=win)
                        dst.write(arr, b, window=win)

                # 对有规则的波段做裁剪
                for b, (nodata, vmin, vmax) in band_rules.items():
                    if b > src.count:
                        # 表里写了比影像更多的band，跳过
                        continue

                    for _, win in src.block_windows(b):
                        arr = src.read(b, window=win)  # 保持原dtype
                        if nodata is None:
                            valid = np.isfinite(arr)  # 兼容浮点
                        else:
                            valid = arr != nodata

                        if not np.any(valid):
                            continue

                        # 在有效处裁剪；保持dtype一致
                        # 先转float进行裁剪，再cast回原dtype
                        work = arr.astype(np.float64, copy=False)
                        work[valid] = np.clip(work[valid], vmin, vmax)

                        # cast回原dtype
                        clipped = work.astype(arr.dtype, copy=False)

                        # 确保nodata像元保持不变
                        if nodata is not None:
                            clipped[~valid] = arr[~valid]

                        dst.write(clipped, b, window=win)

    return True, "ok"

# -------------------- 任务生成 --------------------
def build_tasks(rules: dict):
    """
    生成 (in_tif, out_tif, band_rules) 列表
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

# -------------------- 主流程 --------------------
def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rules = load_rules(CSV_PATH)
    tasks = build_tasks(rules)
    if not tasks:
        print("未发现可处理的tif文件。")
        return

    print(f"共 {len(tasks)} 个文件，将使用 {WORKERS} 进程处理。")
    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("fork" if os.name != "nt" else "spawn")) as ex:
        futures = [ex.submit(clip_one_file, in_tif, out_tif, band_rules) for (in_tif, out_tif, band_rules) in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Clipping"):
            try:
                success, msg = fut.result()
                ok += int(success)
                fail += int(not success)
            except Exception as e:
                fail += 1
                print("[错误]", e)

    print(f"完成：成功 {ok}，失败 {fail}。输出目录：{OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
