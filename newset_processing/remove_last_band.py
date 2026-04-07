#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
移除TIF图像最后几个波段脚本
- 将指定文件夹内所有TIF文件的最后几个波段移除
- 支持指定删除的波段数量（如1表示删除最后一个，2表示删除最后两个）
- 支持单文件或批量处理
- 保持原始数据类型、地理信息和NoData设置
- 自动跳过已处理的文件
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple
import rasterio
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def process_single_file(input_file: str, output_file: str, bands_to_remove: int) -> bool:
    """
    处理单个TIF文件，移除最后几个波段
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        bands_to_remove: 要移除的波段数量（1表示最后一个，2表示最后两个）
        
    Returns:
        处理是否成功
    """
    try:
        with rasterio.open(input_file) as src:
            # 检查波段数量
            if src.count <= bands_to_remove:
                logger.warning(f"文件 {input_file} 只有 {src.count} 个波段，无法移除 {bands_to_remove} 个波段")
                return False
            
            # 获取原始profile
            profile = src.profile.copy()
            
            # 更新波段数量
            new_band_count = src.count - bands_to_remove
            profile.update({
                'count': new_band_count
            })
            
            # 读取要保留的波段数据
            processed_data = []
            processed_descriptions = []
            
            for i in range(1, new_band_count + 1):  # 只读取要保留的波段
                band_data = src.read(i)
                processed_data.append(band_data)
                
                # 保留波段描述（如果有的话）
                if src.descriptions and i-1 < len(src.descriptions):
                    processed_descriptions.append(src.descriptions[i-1])
            
            # 写入输出文件
            with rasterio.open(output_file, 'w', **profile) as dst:
                for i, band_data in enumerate(processed_data):
                    dst.write(band_data, i + 1)
                
                # 设置波段描述
                if processed_descriptions:
                    dst.descriptions = tuple(processed_descriptions)
            
            return True
            
    except Exception as e:
        logger.error(f"处理文件 {input_file} 时出错: {str(e)}")
        return False

def process_folder(input_dir: str, output_dir: str, bands_to_remove: int, skip_existing: bool = True) -> Tuple[int, int]:
    """
    处理文件夹中的所有TIF文件
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        bands_to_remove: 要移除的波段数量
        skip_existing: 是否跳过已存在的文件
        
    Returns:
        (成功数量, 总数量)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有TIF文件
    tif_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.TIF"))
    if not tif_files:
        logger.warning(f"在目录 {input_dir} 中未找到TIF文件")
        return 0, 0
    
    logger.info(f"找到 {len(tif_files)} 个TIF文件")
    
    success_count = 0
    
    for tif_file in tqdm(tif_files, desc="处理文件", ncols=100):
        output_file = output_path / tif_file.name
        
        # 检查是否跳过已存在的文件
        if skip_existing and output_file.exists():
            success_count += 1
            continue
        
        if process_single_file(str(tif_file), str(output_file), bands_to_remove):
            success_count += 1
    
    return success_count, len(tif_files)

def main():
    parser = argparse.ArgumentParser(description="移除TIF图像的最后几个波段")
    parser.add_argument("--input", default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/MCD12Q1_mosaic_downsampled", help="输入文件或文件夹路径")
    parser.add_argument("--output", default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/MCD12Q1_mosaic_downsampledc_right", help="输出文件或文件夹路径")
    parser.add_argument("--bands", "-b", type=int, default=1, help="要移除的波段数量（1表示最后一个，2表示最后两个，以此类推）")
    parser.add_argument("--no-skip-existing", action="store_true", 
                       help="不跳过已存在的输出文件")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    bands_to_remove = args.bands
    skip_existing = not args.no_skip_existing
    
    # 验证参数
    if bands_to_remove <= 0:
        logger.error("要移除的波段数量必须大于0")
        return
    
    if not input_path.exists():
        logger.error(f"输入路径不存在: {input_path}")
        return
    
    logger.info(f"将移除每个文件的最后 {bands_to_remove} 个波段")
    
    if input_path.is_file():
        # 处理单个文件
        if process_single_file(str(input_path), str(output_path), bands_to_remove):
            logger.info("文件处理完成")
        else:
            logger.error("文件处理失败")
    elif input_path.is_dir():
        # 处理文件夹
        success_count, total_count = process_folder(
            str(input_path), str(output_path), bands_to_remove, skip_existing
        )
        logger.info(f"处理完成: {success_count}/{total_count} 个文件成功")
    else:
        logger.error(f"输入路径既不是文件也不是文件夹: {input_path}")

if __name__ == "__main__":
    main()
