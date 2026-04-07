#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIRMS数据二值化脚本（简化版）

专门处理FIRMS数据的第一个波段：
- 只读取第一个波段
- 阈值8：≥8设为1，<8设为0
- 保存为新的TIFF文件

使用方法:
python firms_binary_simple.py --input_folder /path/to/input --output_folder /path/to/output
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import rasterio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 设置日志级别，只显示警告和错误
warnings.filterwarnings('ignore')

# 线程安全的计数器
class ThreadSafeCounter:
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
    
    # 控制更新频率
    should_update = (
        current_time - start_time >= 2.0 or  # 至少2秒间隔
        progress_percent - int(progress_percent) == 0 or  # 每1%进度
        current == total  # 完成时
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
        print()  # 完成后换行

def process_single_file_parallel(args_tuple) -> tuple:
    """
    并行处理单个文件的包装函数
    
    Args:
        args_tuple: (input_path, output_path, threshold) 元组
    
    Returns:
        (success: bool, input_path: Path, error_msg: str)
    """
    input_path, output_path, threshold = args_tuple
    try:
        success = process_single_file(input_path, output_path, threshold)
        return success, input_path, ""
    except Exception as e:
        return False, input_path, str(e)

def process_single_file(input_path: Path, output_path: Path, threshold: int = 8) -> bool:
    """
    处理单个TIFF文件的第一个波段
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        threshold: 二值化阈值
    
    Returns:
        处理是否成功
    """
    try:
        with rasterio.open(input_path) as src:
            # 只读取第一个波段
            band_data = src.read(1)
            
            # 获取源文件信息
            profile = src.profile.copy()
            
            # 检查NoData值是否与uint8兼容
            nodata_value = src.nodata
            if nodata_value is not None:
                if nodata_value < 0 or nodata_value > 255:
                    # 如果NoData值超出uint8范围，使用255作为NoData值
                    output_nodata = 255
                else:
                    output_nodata = int(nodata_value)
            else:
                output_nodata = None
            
            # 二值化处理
            binary_data = np.zeros_like(band_data, dtype=np.uint8)
            
            # 标记NoData位置
            if nodata_value is not None:
                nodata_mask = (band_data == nodata_value)
            else:
                nodata_mask = np.zeros_like(band_data, dtype=bool)
            
            # 二值化：大于等于阈值设为1，小于阈值设为0
            binary_data[band_data >= threshold] = 1
            binary_data[band_data < threshold] = 0
            
            # 保持NoData值不变（使用255作为uint8的NoData值）
            if nodata_value is not None:
                binary_data[nodata_mask] = 255
            
            # 更新profile用于输出
            profile.update({
                'dtype': 'uint8',
                'count': 1,  # 只有一个波段
                'compress': 'LZW',
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'BIGTIFF': 'IF_SAFER',
                'nodata': output_nodata
            })
            
            # 写入输出文件
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(binary_data, 1)
                
                # 复制元数据
                dst.descriptions = src.descriptions[:1] if src.descriptions else None
                dst.colorinterp = src.colorinterp[:1] if src.colorinterp else None
        
        return True
        
    except Exception as e:
        print(f"\n处理文件 {input_path.name} 失败: {e}")
        return False

def find_tiff_files(folder: Path) -> list:
    """查找文件夹中的所有TIFF文件"""
    tiff_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    tiff_files = []
    
    for ext in tiff_extensions:
        tiff_files.extend(folder.glob(f'*{ext}'))
    
    return sorted(tiff_files)

def main():
    parser = argparse.ArgumentParser(
        description="FIRMS数据第一波段二值化处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python firms_binary_simple.py --input_folder /path/to/input --output_folder /path/to/output
  python firms_binary_simple.py --input_folder /path/to/input --output_folder /path/to/output --threshold 8
  python firms_binary_simple.py --input_folder /path/to/input --output_folder /path/to/output --max_workers 16
        """
    )
    
    parser.add_argument('--input_folder', type=Path, 
                        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/MCD14A1_mosaic',
                       help='输入文件夹路径')
    parser.add_argument('--output_folder', type=Path, 
                        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/MCD14A1_mosaic_binary',
                       help='输出文件夹路径')
    parser.add_argument('--threshold', type=int, default=8,
                       help='二值化阈值（默认: 8）')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='并行处理的最大线程数（默认: CPU核心数）')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的输出文件')
    parser.add_argument('--skip_existing', action='store_true',
                       help='跳过已存在的输出文件')
    
    args = parser.parse_args()
    
    # 检查输入文件夹
    if not args.input_folder.exists():
        print(f"错误: 输入文件夹不存在: {args.input_folder}")
        sys.exit(1)
    
    # 自动创建输出文件夹
    args.output_folder.mkdir(parents=True, exist_ok=True)
    
    # 查找输入文件
    input_files = find_tiff_files(args.input_folder)
    if not input_files:
        print(f"错误: 在文件夹 {args.input_folder} 中未找到TIFF文件")
        sys.exit(1)
    
    # 确定并行线程数
    max_workers = args.max_workers if args.max_workers else min(32, (len(input_files) or 1) + 4)
    
    print(f"输入目录: {args.input_folder}")
    print(f"输出目录: {args.output_folder}")
    print(f"二值化阈值: {args.threshold}")
    print(f"并行线程数: {max_workers}")
    print(f"找到 {len(input_files)} 个TIFF文件")
    print("处理模式: 只处理第一个波段")
    print()
    
    # 预处理：检查输出文件存在性
    tasks = []
    skip_count = 0
    
    for input_file in input_files:
        output_file = args.output_folder / input_file.name
        
        # 检查输出文件是否已存在
        if output_file.exists():
            if args.overwrite:
                # 覆盖模式，添加到任务列表
                tasks.append((input_file, output_file, args.threshold))
            elif args.skip_existing:
                # 跳过已存在文件
                skip_count += 1
                continue
            else:
                # 默认模式，报错
                print(f"警告: 输出文件已存在: {output_file}")
                print("使用 --overwrite 覆盖或 --skip_existing 跳过")
                continue
        else:
            # 文件不存在，添加到任务列表
            tasks.append((input_file, output_file, args.threshold))
    
    if not tasks:
        print("没有需要处理的文件")
        return
    
    print(f"需要处理 {len(tasks)} 个文件，跳过 {skip_count} 个文件")
    print()
    
    # 并行处理文件
    success_count = 0
    error_count = 0
    processed_count = ThreadSafeCounter()
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_single_file_parallel, task): task 
            for task in tasks
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            success, input_path, error_msg = future.result()
            current = processed_count.increment()
            
            # 更新进度条
            print_progress(current, len(tasks), start_time, prefix="二值化 ")
            
            if success:
                success_count += 1
            else:
                error_count += 1
                if error_msg:
                    print(f"\n处理文件 {input_path.name} 失败: {error_msg}")
    
    # 输出统计信息
    elapsed_time = time.time() - start_time
    print(f"\n处理完成! 总耗时: {format_time(elapsed_time)}")
    print(f"总计: {len(input_files)}, 成功: {success_count}, 跳过: {skip_count}, 失败: {error_count}")
    print(f"输出目录: {args.output_folder}")
    
    if success_count > 0:
        avg_time = elapsed_time / success_count
        print(f"平均处理时间: {avg_time:.2f}秒/文件")
        print(f"并行加速比: {max_workers:.1f}x")

if __name__ == "__main__":
    main()
