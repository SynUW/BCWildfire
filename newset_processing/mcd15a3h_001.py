#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值处理脚本 - 将小于指定阈值的像元值转为0

处理文件夹内所有TIFF文件，将小于阈值的像元值转为0，输出到新文件夹
支持多波段文件，对所有波段进行处理

阈值规则：
- 小于阈值的像元值（不包括阈值本身）转为0
- 大于等于阈值的像元值保持不变
- NoData值转为0，输出文件不再设置NoData值
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import rasterio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings

warnings.filterwarnings('ignore')

class ThreadSafeCounter:
    """线程安全的计数器"""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"

def print_progress(current: int, total: int, start_time: float, prefix: str = ""):
    """打印进度条"""
    current_time = time.time()
    progress_percent = current / total * 100
    
    should_update = (
        current_time - start_time >= 2.0 or
        progress_percent - int(progress_percent) == 0 or
        current == total
    )
    
    if not should_update and current < total:
        return
    
    elapsed = current_time - start_time
    if current > 0:
        avg_time_per_item = elapsed / current
        remaining_items = total - current
        eta = remaining_items * avg_time_per_item
        eta_str = format_time(eta)
    else:
        eta_str = "?:??:??"
    
    elapsed_str = format_time(elapsed)
    total_estimated = elapsed / current * total if current > 0 else 0
    total_str = format_time(total_estimated)
    
    print(f"\r{prefix}[{current}/{total}], {elapsed_str}/{total_str}, ETA: {eta_str}", end='', flush=True)
    
    if current == total:
        print()

def apply_threshold(data: np.ndarray, threshold: float, nodata_value: float = None) -> np.ndarray:
    """
    应用阈值处理
    
    Args:
        data: 输入数据数组
        threshold: 阈值
        nodata_value: NoData值
    
    Returns:
        处理后的数据数组
    
    阈值规则：
    - 小于阈值的像元值（不包括阈值本身）转为0
    - 大于等于阈值的像元值保持不变
    - NoData值转为0
    """
    # 创建输出数组的副本
    result = data.copy().astype(data.dtype)
    
    # 处理NoData值：将NoData值转为0
    if nodata_value is not None:
        nodata_mask = data == nodata_value
        result[nodata_mask] = 0
    
    # 应用阈值规则：小于阈值的值转为0
    # 注意：不包括阈值本身，所以使用 < 而不是 <=
    # 同时排除已经是0的值
    threshold_mask = (data < threshold) & (data > 0)
    result[threshold_mask] = 0
    
    return result

def process_single_file(input_path: Path, output_path: Path, threshold: float) -> bool:
    """
    处理单个TIFF文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        threshold: 阈值
    
    Returns:
        处理是否成功
    """
    try:
        # 读取输入文件
        with rasterio.open(input_path) as src:
            # 获取文件信息
            profile = src.profile.copy()
            nodata_value = src.nodata
            
            # 修改profile：移除NoData值设置
            profile['nodata'] = None
            
            # 读取所有波段
            data = src.read()
            
            # 对每个波段应用阈值处理
            processed_data = []
            for band_idx in range(data.shape[0]):
                band_data = data[band_idx]
                processed_band = apply_threshold(band_data, threshold, nodata_value)
                processed_data.append(processed_band)
            
            # 写入输出文件
            with rasterio.open(output_path, 'w', **profile) as dst:
                for band_idx, band_data in enumerate(processed_data):
                    dst.write(band_data, band_idx + 1)
                
                # 复制元数据
                if src.descriptions:
                    dst.descriptions = src.descriptions
                if src.colorinterp:
                    dst.colorinterp = src.colorinterp
                
                # 安全地复制tags
                try:
                    src_tags = src.tags()
                    if src_tags:
                        dst.update_tags(**src_tags)
                except Exception:
                    # 如果tags复制失败，忽略错误继续处理
                    pass
        
        return True
        
    except Exception as e:
        print(f"\n处理文件 {input_path.name} 失败: {e}")
        return False

def process_single_file_parallel(args_tuple) -> tuple:
    """并行处理的包装函数"""
    input_path, output_path, threshold = args_tuple
    try:
        success = process_single_file(input_path, output_path, threshold)
        return success, input_path, ""
    except Exception as e:
        return False, input_path, str(e)

def find_tiff_files(folder: Path) -> list:
    """查找文件夹内的所有TIFF文件"""
    tiff_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    tiff_files = []
    for ext in tiff_extensions:
        tiff_files.extend(folder.glob(f'*{ext}'))
    return sorted(tiff_files)

def main():
    parser = argparse.ArgumentParser(
        description="阈值处理：将小于指定阈值的像元值转为0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
阈值规则说明：
- 小于阈值的像元值（不包括阈值本身）转为0
- 大于等于阈值的像元值保持不变  
- NoData值转为0，输出文件不再设置NoData值

示例：
- 阈值=0.1：0.05→0, 0.1→0.1, 0.15→0.15, NoData(-9999)→0
- 阈值=0.5：0.3→0, 0.5→0.5, 0.8→0.8, NoData(-9999)→0
        """
    )
    
    parser.add_argument('--input_folder', type=Path, 
                        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_10x_norm/MCD15A3H_mosaic_downsampled_daily', 
                        help='输入文件夹路径')
    parser.add_argument('--output_folder', type=Path, 
                        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_10x_norm/MCD15A3H_mosaic_downsampled_daily_001', 
                        help='输出文件夹路径')
    parser.add_argument('--threshold', type=float, default=0.1, help='阈值（默认: 0.1，小于此值的像元转为0，不包括阈值本身）')
    parser.add_argument('--max_workers', type=int, default=16, help='并行处理的最大线程数（默认: CPU核心数）')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出文件')
    parser.add_argument('--skip_existing', action='store_true', help='跳过已存在的输出文件')
    
    args = parser.parse_args()
    
    if not args.input_folder.exists():
        print(f"错误: 输入文件夹不存在: {args.input_folder}", file=sys.stderr)
        sys.exit(1)
    
    # 创建输出文件夹
    args.output_folder.mkdir(parents=True, exist_ok=True)
    
    input_files = find_tiff_files(args.input_folder)
    if not input_files:
        print(f"错误: 在文件夹 {args.input_folder} 中未找到TIFF文件", file=sys.stderr)
        sys.exit(1)
    
    max_workers = args.max_workers if args.max_workers else min(32, (len(input_files) or 1) + 4)
    
    print(f"输入目录: {args.input_folder}")
    print(f"输出目录: {args.output_folder}")
    print(f"阈值: {args.threshold}")
    print(f"阈值规则: 小于 {args.threshold} 的像元值转为0（不包括阈值本身）")
    print(f"NoData处理: NoData值转为0，输出文件不再设置NoData值")
    print(f"并行线程数: {max_workers}")
    print(f"找到 {len(input_files)} 个TIFF文件")
    print()
    
    # 准备任务
    tasks = []
    skip_count = 0
    
    for input_file in input_files:
        output_file = args.output_folder / input_file.name
        
        if output_file.exists():
            if args.overwrite:
                tasks.append((input_file, output_file, args.threshold))
            elif args.skip_existing:
                skip_count += 1
                continue
            else:
                print(f"警告: 输出文件已存在: {output_file}。使用 --overwrite 覆盖或 --skip_existing 跳过。")
                continue
        else:
            tasks.append((input_file, output_file, args.threshold))
    
    if not tasks:
        print("没有需要处理的文件")
        return
    
    print(f"需要处理 {len(tasks)} 个文件，跳过 {skip_count} 个文件")
    print()
    
    success_count = 0
    error_count = 0
    processed_count = ThreadSafeCounter()
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_single_file_parallel, task): task 
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            success, input_path, error_msg = future.result()
            current = processed_count.increment()
            
            print_progress(current, len(tasks), start_time, prefix="阈值处理 ")
            
            if success:
                success_count += 1
            else:
                error_count += 1
                if error_msg:
                    print(f"\n处理文件 {input_path.name} 失败: {error_msg}")
    
    elapsed_time = time.time() - start_time
    print(f"\n处理完成! 总耗时: {format_time(elapsed_time)}")
    print(f"总计: {len(input_files)}, 成功: {success_count}, 跳过: {skip_count}, 失败: {error_count}")
    print(f"输出目录: {args.output_folder}")
    
    if success_count > 0:
        avg_time = elapsed_time / success_count
        print(f"平均处理时间: {avg_time:.2f}秒/文件")
        print(f"并行加速比: {max_workers:.1f}x")
    
    # 显示阈值处理统计
    print(f"\n阈值处理规则:")
    print(f"• 小于 {args.threshold} 的像元值 → 0")
    print(f"• 大于等于 {args.threshold} 的像元值 → 保持不变")
    print(f"• NoData值 → 0")
    print(f"• 输出文件不再设置NoData值")

if __name__ == "__main__":
    main()
