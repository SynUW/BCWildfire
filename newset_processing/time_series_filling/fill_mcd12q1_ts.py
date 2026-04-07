#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LULC 类型
MCD12Q1 年度→日度“插值”（向后复制填满时间序列；闭区间）
========================================================

输入目录中每个年份有一个 TIF，文件名形如 yyyy_mm_dd.tif（通常是 yyyy_01_01.tif）。
脚本会在给定的闭区间 [--start, --end] 内，为每一天生成一个输出 TIF：
- 在该日所属的年份内，使用该“年份的影像”（从该年的日期起向后复制，直至下一年文件出现前的所有日期）。
- 对于时间序列开头若早于最早年份（例如没有 2000 年文件），使用最早年份影像（例如 2001）填充。
- 对于时间序列末尾若晚于最晚年份，使用最晚年份影像填充。
- 输出默认用“硬链接”节省空间；也可选择 copy 或 symlink。

MCD09GA_b1237_mosaic_withQA_QAapplied
MCD09GA_b1237_mosaic_withQA_QAapplied

示例
----
python mcd12q1_to_daily.py \
  --input_dir /path/to/yearly \
  --output_dir /path/to/daily \
  --start 2000-01-01 \
  --end   2005-12-31 \
  --mode hardlink --overwrite

参数
----
--input_dir   : 包含 yyyy_mm_dd.tif 的目录。
--output_dir  : 逐日输出目录（自动创建）。
--start       : 起始日期（闭区间），格式 YYYY-MM-DD。
--end         : 结束日期（闭区间），格式 YYYY-MM-DD。
--mode        : 输出方式：hardlink | copy | symlink（默认 hardlink）。
--overwrite   : 若目标已存在则覆盖。
"""

import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
import re
import os
import shutil
from typing import Dict, List, Tuple
from tqdm import tqdm

DATE_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})\.tif$", re.IGNORECASE)

def parse_args():
    p = argparse.ArgumentParser(description="MCD12Q1 年度影像转逐日（向后复制）")
    p.add_argument("--input_dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/MCD12Q1_mosaic_downsampled', type=Path, help="输入目录（包含 yyyy_mm_dd.tif）")
    p.add_argument("--output_dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/MCD12Q1_mosaic_downsampled', type=Path, help="输出目录（逐日 TIF）")
    p.add_argument("--start", default='2000-01-01', type=str, help="起始日期（闭区间），YYYY-MM-DD")
    p.add_argument("--end", default='2024-12-31', type=str, help="结束日期（闭区间），YYYY-MM-DD")
    p.add_argument("--mode", default="copy", choices=["hardlink", "copy", "symlink"],
                   help="输出方式（节省空间优先推荐 hardlink）")
    p.add_argument("--overwrite", action="store_true", help="如存在则覆盖")
    return p.parse_args()

def find_yearly_files(input_dir: Path) -> Dict[int, Path]:
    """
    扫描目录，找出形如 yyyy_mm_dd.tif 的文件；若同一年出现多个文件，按日期最早的为准。
    返回 year -> filepath 的映射。
    """
    year_to_file: Dict[int, Tuple[date, Path]] = {}
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        m = DATE_RE.match(p.name)
        if not m:
            continue
        y, mm, dd = map(int, m.groups())
        d = date(y, mm, dd)
        if y not in year_to_file or d < year_to_file[y][0]:
            year_to_file[y] = (d, p)

    # 只保留路径
    return {y: tup[1] for y, tup in year_to_file.items()}

def daterange_closed(d0: date, d1: date):
    """闭区间 [d0, d1] 的逐日生成器。"""
    cur = d0
    one = timedelta(days=1)
    while cur <= d1:
        yield cur
        cur += one

def select_source_for_day(d: date, year_keys_sorted: List[int], year_to_file: Dict[int, Path]) -> Path:
    """
    对于给定日 d，选择应该使用的“年份影像”：
      - 若存在同年 y=d.year 的文件，用该年文件；
      - 否则使用最近的“已知年份 ≤ d.year 的最大年份”；
      - 若 d 早于最早年份，则用最早年份文件；
      - 若 d 晚于最晚年份，则用最晚年份文件。
    """
    # 精确命中该年
    if d.year in year_to_file:
        return year_to_file[d.year]

    # 寻找 <= d.year 的最大年份
    candidate = None
    for y in year_keys_sorted:
        if y <= d.year:
            candidate = y
        else:
            break

    if candidate is not None:
        return year_to_file[candidate]

    # 早于最早年份
    return year_to_file[year_keys_sorted[0]]

def ensure_output(p: Path, mode: str, src: Path, overwrite: bool):
    """
    以指定方式在 p 位置生成文件，来源为 src。
    mode: hardlink | copy | symlink
    """
    if p.exists():
        if overwrite:
            p.unlink()
        else:
            return  # 跳过

    p.parent.mkdir(parents=True, exist_ok=True)

    if mode == "hardlink":
        try:
            os.link(src, p)
        except OSError:
            # 跨文件系统硬链失败则退化为复制
            shutil.copy2(src, p)
    elif mode == "symlink":
        try:
            os.symlink(src, p)
        except OSError:
            # 某些环境可能禁止软链，退化为复制
            shutil.copy2(src, p)
    else:  # copy
        shutil.copy2(src, p)

def main():
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end, "%Y-%m-%d").date()
    assert start <= end, "start 必须早于或等于 end（闭区间）"

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    year_to_file = find_yearly_files(input_dir)
    if not year_to_file:
        raise RuntimeError(f"输入目录中未找到任何形如 yyyy_mm_dd.tif 的文件：{input_dir}")

    # 排序年份
    years_sorted = sorted(year_to_file.keys())

    # 关键信息输出
    print(f"发现年份文件：{len(years_sorted)} 个；年份范围：{years_sorted[0]}—{years_sorted[-1]}")
    for y in years_sorted:
        print(f"  {y} -> {year_to_file[y].name}")

    # 遍历闭区间逐日输出
    total_days = (end - start).days + 1
    for d in tqdm(daterange_closed(start, end), total=total_days, desc="生成逐日"):
        src = select_source_for_day(d, years_sorted, year_to_file)
        out_name = f"{d.year:04d}_{d.month:02d}_{d.day:02d}.tif"
        out_path = output_dir / out_name
        ensure_output(out_path, args.mode, src, args.overwrite)

    print("完成！")

if __name__ == "__main__":
    main()
