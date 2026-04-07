#!/usr/bin/env python3
"""
按指定日期范围，在输出目录中生成一批以日期命名的ASTER影像文件：
- 从指定的单个模板TIF复制为每个日期的文件：ASTER_yyyy_mm_dd.tif
- 仅重命名/复制，不改动文件内容与投影信息
- 支持并行复制、跳过已存在文件

示例：
python3 interpolate_files.py \
  --src-file /path/to/template.tif \
  --output-dir /path/to/out \
  --start-date 2000-01-01 --end-date 2000-12-31 \
  --max-workers 16 --overwrite
"""

import os
import shutil
import logging
import argparse
from datetime import datetime, timedelta
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('interpolate_files')


def build_dates(start: str, end: str) -> List[datetime]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    if e < s:
        raise ValueError("end-date 早于 start-date")
    days = (e - s).days + 1
    return [s + timedelta(days=i) for i in range(days)]


def out_name_for_date(d: datetime, prefix: str = "ASTER") -> str:
    return f"{prefix}_{d.year:04d}_{d.month:02d}_{d.day:02d}.tif"


def copy_one(template: str, out_path: str, overwrite: bool) -> bool:
    if os.path.exists(out_path) and not overwrite:
        return True  # 视为成功，跳过
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy2(template, out_path)
    return True


def main():
    ap = argparse.ArgumentParser(description='按日期范围复制模板TIF为 ASTER_yyyy_mm_dd.tif')
    ap.add_argument('--src-file', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map/Topo_and_distance_map_stack_nodata_homogenized_downsampled10x_log.tif', help='模板TIF文件路径（将复制为各日期文件）')
    ap.add_argument('--output-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked/DEM_and_distance_map_interpolated_10x', help='输出目录（会创建）')
    ap.add_argument('--start-date', default='2000-01-01', help='起始日期 YYYY-MM-DD（含）')
    ap.add_argument('--end-date', default='2024-12-31', help='结束日期 YYYY-MM-DD（含）')
    ap.add_argument('--prefix', default='TopoDistance', help='输出文件名前缀，默认ASTER')
    ap.add_argument('--max-workers', type=int, default=16, help='并行进程数')
    ap.add_argument('--overwrite', action='store_true', help='若目标文件已存在则覆盖')

    args = ap.parse_args()

    template = args.src_file
    if not os.path.isfile(template):
        logger.error(f"模板文件不存在: {template}")
        return

    dates = build_dates(args.start_date, args.end_date)

    os.makedirs(args.output_dir, exist_ok=True)

    total = len(dates)
    done = 0
    ok = 0

    def progress():
        print(f"\r进度 {done}/{total} 成功:{ok}", end='', flush=True)

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = []
        for d in dates:
            name = out_name_for_date(d, args.prefix)
            dst = os.path.join(args.output_dir, name)
            futs.append(ex.submit(copy_one, template, dst, args.overwrite))
        for f in as_completed(futs):
            res = f.result()
            done += 1
            if res:
                ok += 1
            progress()

    print()
    logger.info(f"完成: {ok}/{total} 个文件生成 -> {args.output_dir}")


if __name__ == '__main__':
    main()
