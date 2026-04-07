#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCD15A3H 时间序列"前值填充"补齐脚本（逐日）
- 直接在源目录下操作，无需输出目录
- 输入目录下 GeoTIFF 命名格式：yyyy_mm_dd.tif（如 2023_02_03.tif）
- 在给定的起止日期 [start_date, end_date] 内，逐日生成 GeoTIFF：
  * 若当天文件已存在：跳过
  * 若缺失：用"最近的既往日期"的影像填充（前值填充）
  * 若在最早可用日期之前：一直用最早日期影像回填
  * 若在最晚可用日期之后：一直用最晚日期影像外推

用法示例：
python fill_mcd15a3h_ts.py \
  --folder /path/to/mcd15a3h \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --workers 8
"""

import argparse
import re
import sys
import shutil
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm 未安装：pip install tqdm", file=sys.stderr)
    tqdm = None

DATE_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})\.tif$", re.IGNORECASE)

def parse_args():
    ap = argparse.ArgumentParser(description="MCD15A3H 前值填充补齐（逐日）")
    ap.add_argument("--folder", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/MCD15A3H_mosaic_249_filled_downsampled', help="操作目录（直接在原目录内填充）")
    ap.add_argument("--start-date", default='2000-01-01', help="起始日期，YYYY-MM-DD")
    ap.add_argument("--end-date", default='2024-12-31', help="终止日期，YYYY-MM-DD（含）")
    ap.add_argument("--workers", type=int, default=8, help="并发线程数（I/O 复制并行）")
    return ap.parse_args()

def to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def scan_input(input_dir: Path) -> Dict[date, Path]:
    """扫描输入目录，解析符合 yyyy_mm_dd.tif 的文件，返回 {日期: 路径}"""
    mapping: Dict[date, Path] = {}
    for p in input_dir.glob("*.tif"):
        m = DATE_RE.match(p.name)
        if not m:
            continue
        y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dt = date(y, mth, d)
        except ValueError:
            continue
        # 若同一天有多个文件，保留第一个，或可替换为更复杂的选择策略
        mapping.setdefault(dt, p)
    return mapping

def daterange(d0: date, d1: date):
    """包含端点的逐日迭代"""
    cur = d0
    one = timedelta(days=1)
    while cur <= d1:
        yield cur
        cur += one

def build_fill_plan(available: List[date], start: date, end: date) -> Dict[date, date]:
    """
    为 [start, end] 每一天确定“来源日期”（即用哪天的影像来填）。
    规则：
      - 若当天存在：用当天
      - 若缺失：用最近的既往日期
      - 若在最早可用日期之前：用最早日期
      - 若在最晚可用日期之后：用最晚日期
    """
    if not available:
        raise ValueError("输入目录中未发现任何符合 yyyy_mm_dd.tif 的文件。")

    available_sorted = sorted(available)
    earliest = available_sorted[0]
    latest = available_sorted[-1]

    plan: Dict[date, date] = {}
    prev_seen: Optional[date] = None

    for d in daterange(start, end):
        if d in available:
            prev_seen = d
            plan[d] = d
        else:
            if prev_seen is not None and prev_seen <= d:
                # 用最近的既往日期
                plan[d] = prev_seen
            else:
                # 还没遇到任何可用日期（在序列最前端）
                if d < earliest:
                    plan[d] = earliest  # 回填为最早日期
                elif d > latest:
                    plan[d] = latest    # 外推为最晚日期
                else:
                    # 理论上到不了这里；为稳妥起见做一次向左搜索
                    # （极端情况下 prev_seen 为空但 d 已经在 available 范围内且右侧有数据）
                    # 这里实现一次线性回溯：从 d-1 往前找第一天可用数据
                    back = d - timedelta(days=1)
                    while back >= earliest and back not in available:
                        back -= timedelta(days=1)
                    plan[d] = back if back >= earliest else earliest
    return plan


def materialize(src: Path, dst: Path):
    """复制文件到目标位置"""
    if dst.exists():
        return  # 跳过已存在的文件
    
    # 直接复制文件
    shutil.copy2(src, dst)  # 保留时间戳

def main():
    args = parse_args()
    folder = Path(args.folder)

    start = to_date(args.start_date)
    end = to_date(args.end_date)
    if end < start:
        raise SystemExit("end-date 早于 start-date。")

    mapping = scan_input(folder)
    if not mapping:
        raise SystemExit("操作目录中没有符合 yyyy_mm_dd.tif 命名的文件。")

    plan = build_fill_plan(sorted(mapping.keys()), start, end)

    tasks = []
    for d, src_d in plan.items():
        src_path = mapping.get(src_d)
        if src_path is None:
            # 极少数边界情况（并发/外部删除）下可能为 None
            raise SystemExit(f"找不到来源文件：{src_d}")
        
        # 如果目标文件已存在，跳过
        dst_name = f"{d:%Y_%m_%d}.tif"
        dst_path = folder / dst_name
        if dst_path.exists():
            continue
            
        tasks.append((src_path, dst_path))

    total = len(tasks)
    print(f"📅 时间范围：{start} 至 {end}")
    print(f"📁 操作目录：{folder}")
    print(f"📦 找到 {len(mapping)} 个现有文件")
    print(f"🚀 需要填充 {total} 个缺失文件")

    def _worker(item):
        src, dst = item
        materialize(src, dst)
        return dst.name

    if tqdm is None:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            completed = 0
            for _ in as_completed(ex.submit(_worker, t) for t in tasks):
                completed += 1
                print(f"\r[{completed}/{total}] 填充进度", end='', flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for _ in tqdm(as_completed(ex.submit(_worker, t) for t in tasks),
                          total=total, desc="填充缺失文件"):
                pass

    print(f"\n✅ [完成] 已在 {folder} 填充 {total} 个缺失文件")
    print(f"📊 时间范围：{start} 至 {end}")

if __name__ == "__main__":
    main()
