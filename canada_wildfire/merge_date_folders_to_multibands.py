#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 设置日志 - 只显示错误，忽略警告
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 禁用 GDAL 警告信息
import os
os.environ['CPL_LOG_ERRORS'] = 'OFF'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'

def is_valid_date_folder(folder_name: str) -> bool:
    """检查文件夹名是否为有效的日期格式 yyyy_mm_dd"""
    pattern = r'^\d{4}_\d{2}_\d{2}$'
    return bool(re.match(pattern, folder_name))

def get_tif_files_sorted(folder_path: Path) -> List[Path]:
    """获取文件夹内所有 tif 文件，排除包含 QC 的文件，按名称排序"""
    tif_files = []
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ['.tif', '.tiff']:
            # 排除包含 QC 的文件
            if 'QC' not in file_path.name.upper():
                tif_files.append(file_path)
    
    # 按文件名排序
    tif_files.sort(key=lambda x: x.name)
    return tif_files

def merge_tifs_to_multiband(input_files: List[Path], output_file: Path, gdal_cachemax: int = 1024) -> bool:
    """将多个单波段 tif 文件合并为一个多波段 tif 文件（优化版本）"""
    if not input_files:
        logger.warning(f"没有找到有效的 tif 文件")
        return False
    
    try:
        # 读取第一个文件获取参考信息
        with rasterio.open(input_files[0]) as src:
            profile = src.profile.copy()
            height, width = src.height, src.width
            crs = src.crs
            transform = src.transform
            nodata = src.nodata
        
        # 更新 profile 为多波段
        profile.update(
            count=len(input_files),
            dtype=rasterio.float32,  # 统一使用 float32
            nodata=-9999,  # 统一 nodata 值
            compress="DEFLATE",
            predictor=1,  # 避免 libtiff 版本兼容性问题
            zlevel=6,
            tiled=True,
            blockxsize=max(16, min(512, width) // 16 * 16),  # 确保是 16 的倍数
            blockysize=max(16, min(512, height) // 16 * 16),  # 确保是 16 的倍数
        )
        
        # 预读取所有数据到内存（向量化处理）
        all_data = []
        for input_file in input_files:
            with rasterio.open(input_file) as src:
                # 检查尺寸是否匹配
                if src.height != height or src.width != width:
                    logger.error(f"文件 {input_file.name} 的尺寸 ({src.width}x{src.height}) 与参考文件 ({width}x{height}) 不匹配")
                    return False
                
                # 读取数据
                data = src.read(1)  # 读取第一个波段
                
                # 处理 nodata
                if src.nodata is not None:
                    data = np.where(data == src.nodata, -9999, data)
                
                all_data.append(data)
        
        # 批量写入所有波段
        with rasterio.Env(NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=gdal_cachemax, CPL_LOG_ERRORS='OFF'):
            with rasterio.open(output_file, 'w', **profile) as dst:
                # 一次性写入所有波段
                dst.write(np.array(all_data))
                
                # 设置波段描述
                for band_idx, input_file in enumerate(input_files, 1):
                    dst.set_band_description(band_idx, input_file.stem)
        
        return True
        
    except Exception as e:
        logger.error(f"合并文件时出错: {e}")
        return False

def process_date_folder_worker(args):
    """并行处理单个日期文件夹的工作函数"""
    date_folder, output_folder, gdal_cachemax = args
    
    # 获取排序后的 tif 文件
    tif_files = get_tif_files_sorted(date_folder)
    
    if not tif_files:
        return date_folder.name, False, "没有找到有效的 tif 文件"
    
    # 创建输出文件名
    output_file = output_folder / f"{date_folder.name}.tif"
    
    # 如果输出文件已存在，跳过
    if output_file.exists():
        return date_folder.name, True, "文件已存在"
    
    # 合并文件
    try:
        success = merge_tifs_to_multiband(tif_files, output_file, gdal_cachemax)
        if success:
            return date_folder.name, True, "成功"
        else:
            return date_folder.name, False, "合并失败"
    except Exception as e:
        return date_folder.name, False, str(e)

def process_date_folder(date_folder: Path, output_folder: Path, gdal_cachemax: int) -> bool:
    """处理单个日期文件夹（兼容性函数）"""
    result = process_date_folder_worker((date_folder, output_folder, gdal_cachemax))
    return result[1]

def main():
    parser = argparse.ArgumentParser(
        description="将日期文件夹内的多个 tif 文件合成为多波段 tif 文件"
    )
    parser.add_argument("--input", type=str, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/MODIS_LST_29_31_32_QC/MOD21A1N",
                        help="输入文件夹路径，子文件夹为日期格式 (yyyy_mm_dd)")
    parser.add_argument("--gdal-cachemax", type=int, default=102400,
                        help="GDAL 缓存大小 (MB)")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行工作进程数（默认使用 CPU 核心数）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示将要处理的文件，不实际执行")
    
    args = parser.parse_args()
    
    input_folder = Path(args.input).expanduser().resolve()
    
    if not input_folder.exists():
        logger.error(f"输入文件夹不存在: {input_folder}")
        sys.exit(1)
    
    if not input_folder.is_dir():
        logger.error(f"输入路径不是文件夹: {input_folder}")
        sys.exit(1)
    
    # 创建输出文件夹
    output_folder_name = f"{input_folder.name}_multibands_withoutFiltering"
    output_folder = input_folder.parent / output_folder_name
    output_folder.mkdir(exist_ok=True)
    
    # 获取所有日期子文件夹
    date_folders = []
    for item in input_folder.iterdir():
        if item.is_dir() and is_valid_date_folder(item.name):
            date_folders.append(item)
    
    if not date_folders:
        logger.error(f"在 {input_folder} 中没有找到有效的日期文件夹 (格式: yyyy_mm_dd)")
        sys.exit(1)
    
    # 按日期排序
    date_folders.sort(key=lambda x: x.name)
    
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"找到 {len(date_folders)} 个日期文件夹")
    
    if args.dry_run:
        print("=== DRY RUN 模式 ===")
        for date_folder in date_folders:
            tif_files = get_tif_files_sorted(date_folder)
            print(f"日期 {date_folder.name}: {len(tif_files)} 个 tif 文件")
            for tif_file in tif_files:
                print(f"  - {tif_file.name}")
        return
    
    # 设置并行工作进程数
    if args.workers is None:
        args.workers = min(mp.cpu_count(), 8)  # 限制最大进程数避免内存不足
    
    print(f"使用 {args.workers} 个并行进程")
    
    # 并行处理每个日期文件夹
    success_count = 0
    total_count = len(date_folders)
    failed_folders = []
    
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork" if sys.platform != "win32" else "spawn")) as executor:
        # 提交所有任务
        future_to_folder = {
            executor.submit(process_date_folder_worker, (date_folder, output_folder, args.gdal_cachemax)): date_folder
            for date_folder in date_folders
        }
        
        # 处理完成的任务
        for future in tqdm(as_completed(future_to_folder), total=len(future_to_folder), desc="处理日期文件夹"):
            folder_name, success, message = future.result()
            if success:
                success_count += 1
            else:
                failed_folders.append((folder_name, message))
                logger.error(f"处理失败 {folder_name}: {message}")
    
    print(f"\n处理完成: {success_count}/{total_count} 个日期文件夹成功")
    if failed_folders:
        print(f"失败 {len(failed_folders)} 个文件夹:")
        for folder_name, message in failed_folders[:10]:  # 只显示前10个失败
            print(f"  - {folder_name}: {message}")
        if len(failed_folders) > 10:
            print(f"  ... 还有 {len(failed_folders) - 10} 个失败")
    print(f"输出文件夹: {output_folder}")

if __name__ == "__main__":
    main() 