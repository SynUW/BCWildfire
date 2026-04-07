#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
时间序列TIFF数据重建工具
======================

功能：
1. 扫描输入文件夹中以日期命名的TIFF文件（yyyy_mm_dd.tif）
2. 在指定时间范围内，为缺失的日期生成TIFF文件
3. 填充策略：
   - 优先使用前向填充（forward fill）：用最近的过去数据填充
   - 如果前面没有数据，使用后向填充（backward fill）：用最近的未来数据填充
4. 保持TIFF元数据（地理信息、投影、NoData等）

用法示例：
  python rebuild_timeseries.py \
    --input-dir /path/to/sparse_tiffs \
    --output-dir /path/to/dense_tiffs \
    --start-date 2000-01-01 \
    --end-date 2024-12-31

  # 覆盖已有文件
  python rebuild_timeseries.py \
    --input-dir /path/to/sparse_tiffs \
    --output-dir /path/to/dense_tiffs \
    --start-date 2000-01-01 \
    --end-date 2024-12-31 \
    --overwrite
"""

import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import shutil

from osgeo import gdal
from tqdm import tqdm

# 禁用GDAL错误输出
gdal.PushErrorHandler('CPLQuietErrorHandler')


def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """
    从文件名解析日期（yyyy_mm_dd.tif 或 yyyy-mm-dd.tif）
    """
    patterns = [
        r"(\d{4})[_-](\d{2})[_-](\d{2})\.tif",  # yyyy_mm_dd.tif or yyyy-mm-dd.tif
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            year, month, day = match.groups()
            try:
                return datetime(int(year), int(month), int(day))
            except ValueError:
                continue
    return None


def scan_existing_files(input_dir: Path) -> Dict[datetime, Path]:
    """
    扫描输入目录，建立日期->文件路径的映射
    """
    date_file_map = {}
    
    tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    
    for tiff_file in tiff_files:
        date = parse_date_from_filename(tiff_file.name)
        if date:
            date_file_map[date] = tiff_file
    
    return date_file_map


def generate_date_range(start_date: datetime, end_date: datetime):
    """
    生成日期范围内的所有日期
    """
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def find_fill_source(target_date: datetime, 
                     existing_dates: Dict[datetime, Path],
                     prefer_forward: bool = True) -> Optional[Path]:
    """
    为目标日期找到填充数据源
    
    策略：
    1. 优先前向填充：找最近的过去日期
    2. 如果没有过去数据，使用后向填充：找最近的未来日期
    
    Args:
        target_date: 目标日期
        existing_dates: 已有数据的日期字典
        prefer_forward: 优先前向填充
    
    Returns:
        源文件路径，如果找不到返回None
    """
    if not existing_dates:
        return None
    
    sorted_dates = sorted(existing_dates.keys())
    
    # 前向填充：找最近的过去日期
    past_dates = [d for d in sorted_dates if d < target_date]
    if past_dates:
        nearest_past = max(past_dates)  # 最近的过去日期
        return existing_dates[nearest_past]
    
    # 后向填充：找最近的未来日期
    future_dates = [d for d in sorted_dates if d > target_date]
    if future_dates:
        nearest_future = min(future_dates)  # 最近的未来日期
        return existing_dates[nearest_future]
    
    return None


def copy_tiff_file(src_path: Path, dst_path: Path):
    """
    复制TIFF文件（保持所有元数据）
    """
    shutil.copy2(src_path, dst_path)


def format_date_filename(date: datetime) -> str:
    """
    将日期格式化为文件名（yyyy_mm_dd.tif）
    """
    return f"{date.year:04d}_{date.month:02d}_{date.day:02d}.tif"


def main():
    parser = argparse.ArgumentParser(
        description="重建时间序列TIFF数据，填充缺失日期",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--input-dir", type=str, required=True,
                       help="输入TIFF文件夹路径")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="输出TIFF文件夹路径")
    parser.add_argument("--start-date", type=str, default='2000-01-01',
                       help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default='2024-12-31',
                       help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--overwrite", action="store_true",
                       help="覆盖已存在的输出文件")
    parser.add_argument("--dry-run", action="store_true",
                       help="模拟运行，不实际复制文件")
    
    args = parser.parse_args()
    
    # 解析参数
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析日期范围
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"日期格式错误（应为YYYY-MM-DD）: {e}")
    
    if start_date > end_date:
        raise ValueError("起始日期不能晚于结束日期")
    
    print("=" * 70)
    print("时间序列数据重建工具")
    print("=" * 70)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"时间范围: {start_date.date()} 至 {end_date.date()}")
    print(f"覆盖模式: {'是' if args.overwrite else '否'}")
    print(f"模拟运行: {'是' if args.dry_run else '否'}")
    print("=" * 70)
    
    # 扫描已有文件
    print("\n[1/3] 扫描输入文件...")
    existing_dates = scan_existing_files(input_dir)
    
    if not existing_dates:
        raise FileNotFoundError(f"输入目录中未找到任何有效的TIFF文件")
    
    sorted_dates = sorted(existing_dates.keys())
    print(f"✓ 找到 {len(existing_dates)} 个有效文件")
    print(f"  最早日期: {min(sorted_dates).date()}")
    print(f"  最晚日期: {max(sorted_dates).date()}")
    
    # 生成完整日期范围
    print("\n[2/3] 分析缺失日期...")
    total_days = (end_date - start_date).days + 1
    all_dates = list(generate_date_range(start_date, end_date))
    
    missing_dates = [d for d in all_dates if d not in existing_dates]
    
    print(f"✓ 总天数: {total_days}")
    print(f"  已有数据: {len(existing_dates)} 天")
    print(f"  缺失数据: {len(missing_dates)} 天")
    print(f"  覆盖率: {len(existing_dates)/total_days*100:.1f}%")
    
    # 填充缺失日期
    print("\n[3/3] 填充缺失日期...")
    
    stats = {
        'copied': 0,      # 从输入复制的已有文件
        'forward': 0,     # 前向填充
        'backward': 0,    # 后向填充
        'skipped': 0,     # 已存在，跳过
        'failed': 0       # 失败
    }
    
    with tqdm(total=total_days, desc="处理进度", ncols=100) as pbar:
        for date in all_dates:
            output_file = output_dir / format_date_filename(date)
            
            # 检查输出文件是否已存在
            if output_file.exists() and not args.overwrite:
                stats['skipped'] += 1
                pbar.update(1)
                continue
            
            # 如果是已有数据，直接复制
            if date in existing_dates:
                src_file = existing_dates[date]
                
                if not args.dry_run:
                    try:
                        copy_tiff_file(src_file, output_file)
                        stats['copied'] += 1
                    except Exception as e:
                        print(f"\n⚠️  复制失败 {date.date()}: {e}")
                        stats['failed'] += 1
                else:
                    stats['copied'] += 1
            
            # 如果是缺失数据，找填充源
            else:
                src_file = find_fill_source(date, existing_dates)
                
                if src_file is None:
                    print(f"\n⚠️  无法找到填充源: {date.date()}")
                    stats['failed'] += 1
                    pbar.update(1)
                    continue
                
                # 判断是前向还是后向填充
                src_date = parse_date_from_filename(src_file.name)
                is_forward = src_date < date
                
                if not args.dry_run:
                    try:
                        copy_tiff_file(src_file, output_file)
                        if is_forward:
                            stats['forward'] += 1
                        else:
                            stats['backward'] += 1
                    except Exception as e:
                        print(f"\n⚠️  填充失败 {date.date()}: {e}")
                        stats['failed'] += 1
                else:
                    if is_forward:
                        stats['forward'] += 1
                    else:
                        stats['backward'] += 1
            
            pbar.update(1)
    
    # 输出统计信息
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print(f"总文件数: {total_days}")
    print(f"  直接复制（已有数据）: {stats['copied']}")
    print(f"  前向填充（过去→未来）: {stats['forward']}")
    print(f"  后向填充（未来→过去）: {stats['backward']}")
    print(f"  跳过（已存在）: {stats['skipped']}")
    print(f"  失败: {stats['failed']}")
    print("=" * 70)
    
    if args.dry_run:
        print("\n⚠️  这是模拟运行，实际文件未被修改")
    else:
        print(f"\n✅ 输出目录: {output_dir}")
        print(f"✅ 时间序列已完整重建 ({start_date.date()} 至 {end_date.date()})")


if __name__ == "__main__":
    main()

