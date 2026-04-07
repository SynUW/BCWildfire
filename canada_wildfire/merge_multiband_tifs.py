#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple
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

def find_multiband_tifs(folders: List[Path]) -> Dict[str, List[Path]]:
    """
    在所有文件夹中查找多波段 tif 文件，按文件名分组
    返回: {文件名: [文件路径列表]}
    """
    tif_files = {}
    
    for folder in folders:
        if not folder.exists() or not folder.is_dir():
            logger.warning(f"文件夹不存在或不是目录: {folder}")
            continue
            
        for file_path in folder.rglob("*.tif"):
            if file_path.is_file():
                # 检查是否为多波段文件
                try:
                    with rasterio.open(file_path) as src:
                        if src.count > 1:  # 多波段文件
                            filename = file_path.name
                            if filename not in tif_files:
                                tif_files[filename] = []
                            tif_files[filename].append(file_path)
                except Exception as e:
                    logger.warning(f"无法读取文件 {file_path}: {e}")
                    continue
    
    return tif_files

def merge_multiband_tifs_worker(args):
    """并行处理单个多波段 tif 文件的拼接工作函数"""
    filename, file_paths, output_folder, gdal_cachemax = args
    
    try:
        if len(file_paths) < 2:
            return filename, False, "文件数量不足，需要至少2个文件"
        
        # 读取第一个文件获取参考信息
        with rasterio.open(file_paths[0]) as src:
            profile = src.profile.copy()
            height, width = src.height, src.width
            crs = src.crs
            transform = src.transform
            nodata = src.nodata
        
        # 计算总波段数
        total_bands = 0
        for file_path in file_paths:
            with rasterio.open(file_path) as src:
                total_bands += src.count
        
        # 更新 profile
        profile.update(
            count=total_bands,
            dtype=rasterio.float32,  # 统一使用 float32
            nodata=-9999,  # 统一 nodata 值
            compress="DEFLATE",
            predictor=1,  # 避免 libtiff 版本兼容性问题
            zlevel=6,
            tiled=True,
            blockxsize=max(16, min(512, width) // 16 * 16),  # 确保是 16 的倍数
            blockysize=max(16, min(512, height) // 16 * 16),  # 确保是 16 的倍数
        )
        
        # 创建输出文件名
        output_filename = f"Thermal_{filename}"
        output_file = output_folder / output_filename
        
        # 如果输出文件已存在，跳过
        if output_file.exists():
            return filename, True, "文件已存在"
        
        # 预读取所有数据到内存（向量化处理）
        all_data = []
        band_descriptions = []
        
        for file_path in file_paths:
            with rasterio.open(file_path) as src:
                # 检查尺寸是否匹配
                if src.height != height or src.width != width:
                    return filename, False, f"文件 {file_path.name} 的尺寸不匹配"
                
                # 读取所有波段
                data = src.read()  # 读取所有波段
                
                # 处理 nodata
                if src.nodata is not None:
                    data = np.where(data == src.nodata, -9999, data)
                
                # 添加到数据列表
                for band_idx in range(data.shape[0]):
                    all_data.append(data[band_idx])
                    band_descriptions.append(f"{file_path.stem}_band{band_idx+1}")
        
        # 批量写入所有波段
        with rasterio.Env(NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=gdal_cachemax, CPL_LOG_ERRORS='OFF'):
            with rasterio.open(output_file, 'w', **profile) as dst:
                # 一次性写入所有波段
                dst.write(np.array(all_data))
                
                # 设置波段描述
                for band_idx, description in enumerate(band_descriptions, 1):
                    dst.set_band_description(band_idx, description)
        
        return filename, True, "成功"
        
    except Exception as e:
        return filename, False, str(e)

def main():
    parser = argparse.ArgumentParser(
        description="将不同文件夹内的同名多波段 tif 图像拼接成一个多波段 tif 图像"
    )
    parser.add_argument("--folders", type=str, nargs='+', 
                        default=["/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/MODIS_LST_29_31_32_QC/MOD21A1D_multibands_withoutFiltering",
                                 "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/MODIS_LST_29_31_32_QC/MOD21A1N_multibands_withoutFiltering"],
                        help="输入文件夹路径列表")
    parser.add_argument("--output", type=str, 
                        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/MODIS_LST_29_31_32_QC/MOD21A1DN_multibands_withoutFiltering_merged",
                        help="输出文件夹路径")
    parser.add_argument("--gdal-cachemax", type=int, default=102400,
                        help="GDAL 缓存大小 (MB)")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行工作进程数（默认使用 CPU 核心数）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示将要处理的文件，不实际执行")
    
    args = parser.parse_args()
    
    # 解析输入文件夹
    input_folders = [Path(folder).expanduser().resolve() for folder in args.folders]
    output_folder = Path(args.output).expanduser().resolve()
    
    # 检查输入文件夹
    for folder in input_folders:
        if not folder.exists():
            logger.error(f"输入文件夹不存在: {folder}")
            sys.exit(1)
        if not folder.is_dir():
            logger.error(f"输入路径不是文件夹: {folder}")
            sys.exit(1)
    
    # 创建输出文件夹
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"输入文件夹: {[str(f) for f in input_folders]}")
    print(f"输出文件夹: {output_folder}")
    
    # 查找所有多波段 tif 文件
    print("正在扫描多波段 tif 文件...")
    tif_files = find_multiband_tifs(input_folders)
    
    if not tif_files:
        print("没有找到多波段 tif 文件")
        sys.exit(1)
    
    # 过滤出有多个文件的文件名
    multi_file_tifs = {filename: file_paths for filename, file_paths in tif_files.items() 
                      if len(file_paths) >= 2}
    
    if not multi_file_tifs:
        print("没有找到同名的多波段 tif 文件")
        sys.exit(1)
    
    print(f"找到 {len(multi_file_tifs)} 个同名多波段 tif 文件")
    
    if args.dry_run:
        print("=== DRY RUN 模式 ===")
        for filename, file_paths in multi_file_tifs.items():
            print(f"文件: {filename}")
            print(f"  来源文件夹:")
            for file_path in file_paths:
                print(f"    - {file_path.parent}")
            print(f"  输出: Thermal_{filename}")
            print()
        return
    
    # 设置并行工作进程数
    if args.workers is None:
        args.workers = min(mp.cpu_count(), 8)  # 限制最大进程数避免内存不足
    
    print(f"使用 {args.workers} 个并行进程")
    
    # 并行处理每个多波段 tif 文件
    success_count = 0
    total_count = len(multi_file_tifs)
    failed_files = []
    
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork" if sys.platform != "win32" else "spawn")) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(merge_multiband_tifs_worker, (filename, file_paths, output_folder, args.gdal_cachemax)): filename
            for filename, file_paths in multi_file_tifs.items()
        }
        
        # 处理完成的任务
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="处理多波段文件"):
            filename, success, message = future.result()
            if success:
                success_count += 1
            else:
                failed_files.append((filename, message))
                logger.error(f"处理失败 {filename}: {message}")
    
    print(f"\n处理完成: {success_count}/{total_count} 个文件成功")
    if failed_files:
        print(f"失败 {len(failed_files)} 个文件:")
        for filename, message in failed_files[:10]:  # 只显示前10个失败
            print(f"  - {filename}: {message}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个失败")
    print(f"输出文件夹: {output_folder}")

if __name__ == "__main__":
    main() 