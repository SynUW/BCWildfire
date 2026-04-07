#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局有效位置掩膜检查工具
使用全局掩膜文件检查所有tif图像在有效位置处是否有无效值
"""

import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
from osgeo import gdal

# 优化GDAL性能配置
gdal.UseExceptions()
os.environ['GDAL_CACHEMAX'] = '102400'  # 100GB缓存

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    # 如果没有tqdm，创建一个简单的替代
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_mask_file(mask_path: str) -> np.ndarray:
    """
    加载全局有效位置掩膜文件
    
    Args:
        mask_path: 掩膜文件路径
        
    Returns:
        掩膜数组，有效位置为1，无效位置为0
    """
    try:
        ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"无法打开掩膜文件: {mask_path}")
        
        # 读取掩膜数据
        band = ds.GetRasterBand(1)
        mask = band.ReadAsArray().astype(np.uint8)
        
        height, width = ds.RasterYSize, ds.RasterXSize
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        
        ds = None
        
        logger.info(f"加载掩膜文件成功: {width}x{height}, 有效像素数: {np.sum(mask == 1)}")
        
        return mask, (width, height), geotransform, projection
        
    except Exception as e:
        logger.error(f"加载掩膜文件失败: {e}")
        raise


def check_tif_file_optimized(tif_path: str, mask: np.ndarray, mask_info: Dict, 
                             chunk_size: int = 2048) -> Dict:
    """
    优化版：检查单个tif文件在有效位置处是否有无效值
    使用分块读取提高性能，避免加载整个大文件到内存
    
    Args:
        tif_path: tif文件路径
        mask: 全局掩膜数组
        mask_info: 掩膜元信息 (width, height, geotransform, projection)
        chunk_size: 分块大小（默认2048像素）
        
    Returns:
        检查结果字典
    """
    mask_width, mask_height = mask_info['size']
    
    result = {
        'file': tif_path,
        'status': 'unknown',
        'errors': [],
        'valid_pixels_checked': 0,
        'invalid_pixels_found': 0
    }
    
    try:
        ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
        if ds is None:
            result['status'] = 'failed'
            result['errors'].append('无法打开文件')
            return result
        
        # 快速检查空间尺寸是否匹配
        tif_width, tif_height = ds.RasterXSize, ds.RasterYSize
        if tif_width != mask_width or tif_height != mask_height:
            result['status'] = 'failed'
            result['errors'].append(f'尺寸不匹配: {tif_width}x{tif_height} vs {mask_width}x{mask_height}')
            ds = None
            return result
        
        # 获取有效位置掩膜（只计算一次）
        valid_mask = (mask == 1)
        num_valid_pixels = np.sum(valid_mask)
        result['valid_pixels_checked'] = num_valid_pixels
        
        # 如果没有任何有效像素，快速返回
        if num_valid_pixels == 0:
            ds = None
            result['status'] = 'passed'
            return result
        
        # 检查每个波段
        bands = ds.RasterCount
        found_invalid = False  # 只记录是否有无效值，不统计数量
        
        for band_idx in range(1, bands + 1):
            if found_invalid:
                break  # 如果已经发现无效值，直接跳出波段循环
            
            band = ds.GetRasterBand(band_idx)
            nodata = band.GetNoDataValue()
            
            # 使用分块读取优化性能
            for y in range(0, tif_height, chunk_size):
                y_end = min(y + chunk_size, tif_height)
                height = y_end - y
                
                # 读取一个分块
                data = band.ReadAsArray(0, y, tif_width, height)
                
                # 获取该分块对应的掩膜
                mask_chunk = valid_mask[y:y_end, :]
                
                # 如果这个分块没有有效像素，跳过
                if not np.any(mask_chunk):
                    continue
                
                # 在有效位置处检查无效值
                valid_data = data[mask_chunk]
                
                # 快速判断无效值（向量化操作）
                invalid_mask = ~np.isfinite(valid_data)  # NaN 或 Inf
                
                if nodata is not None:
                    invalid_mask |= (valid_data == float(nodata))
                
                # 只要发现任何一个无效值，立即标记并跳出
                if np.any(invalid_mask):
                    found_invalid = True
                    result['errors'].append(f'波段{band_idx}: 在有效位置处发现无效值 (nodata={nodata})')
                    break  # 跳出分块循环
        
        ds = None
        
        result['invalid_pixels_found'] = 1 if found_invalid else 0  # 只记录有无
        
        if found_invalid:
            result['status'] = 'failed'
        else:
            result['status'] = 'passed'
        
        return result
        
    except Exception as e:
        result['status'] = 'failed'
        result['errors'].append(f'检查过程出错: {str(e)}')
        return result


def find_all_tif_files(root_dir: str) -> List[str]:
    """
    递归查找根目录下的所有tif文件
    支持两种情况：
    1. 根目录下直接是tif文件
    2. 根目录下是文件夹，文件夹内包含tif文件
    
    Args:
        root_dir: 根目录路径
        
    Returns:
        tif文件路径列表
    """
    tif_files = []
    
    # 首先检查根目录本身是否是tif文件
    if os.path.isfile(root_dir):
        if root_dir.lower().endswith('.tif') or root_dir.lower().endswith('.tiff'):
            return [root_dir]
        else:
            logger.warning(f"指定路径是一个文件但不是tif: {root_dir}")
            return []
    
    # 递归查找根目录下的所有tif文件
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                tif_files.append(os.path.join(root, file))
    
    return sorted(tif_files)


def process_files_parallel(tif_files: List[str], mask: np.ndarray, mask_info: Dict, 
                          max_workers: int = 8, log_path: str = None) -> List[Dict]:
    """
    并行处理多个tif文件（优化版，动态写入日志）
    
    Args:
        tif_files: tif文件路径列表
        mask: 掩膜数组
        mask_info: 掩膜元信息
        max_workers: 最大并行线程数
        log_path: 日志文件路径（用于动态写入）
        
    Returns:
        检查结果列表
    """
    results = []
    failed_count = 0
    
    # 过滤掉掩膜文件本身
    mask_file_path = mask_info.get('mask_file', '')
    filtered_files = [f for f in tif_files if os.path.abspath(f) != os.path.abspath(mask_file_path)]
    
    logger.info(f"开始检查 {len(filtered_files)} 个tif文件 (使用 {max_workers} 个线程)")
    
    # 初始化日志文件
    if log_path:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("全局有效位置掩膜检查结果（动态更新）\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总文件数: {len(filtered_files)}\n\n")
            f.write("失败文件列表:\n")
            f.write("-" * 80 + "\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(check_tif_file_optimized, tif_file, mask, mask_info): tif_file 
            for tif_file in filtered_files
        }
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="检查进度", ncols=100):
            result = future.result()
            results.append(result)
            
            # 动态写入失败文件到日志
            if result['status'] == 'failed' and log_path:
                failed_count += 1
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n#{failed_count} 文件: {result['file']}\n")
                    f.write(f"  状态: {result['status']}\n")
                    f.write(f"  检查的有效像素数: {result['valid_pixels_checked']}\n")
                    if result['errors']:
                        f.write("  错误信息:\n")
                        for error in result['errors']:
                            f.write(f"    - {error}\n")
            
            # 动态进度指示器（减少输出频率）
            completed = len(results)
            total = len(filtered_files)
            if completed % 100 == 0 or completed == total:  # 每100个或最后一个才输出
                passed = sum(1 for r in results if r['status'] == 'passed')
                failed = sum(1 for r in results if r['status'] == 'failed')
                if HAS_TQDM:
                    tqdm.write(f"进度: {completed}/{total}, 通过: {passed}, 失败: {failed}")
                else:
                    print(f"进度: {completed}/{total}, 通过: {passed}, 失败: {failed}")
                
                # 同时更新日志统计
                if log_path and completed == total:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("检查完成\n")
                        f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"检查通过: {passed}\n")
                        f.write(f"检查失败: {failed}\n")
                        f.write("=" * 80 + "\n")
    
    return sorted(results, key=lambda x: x['file'])


def write_log(results: List[Dict], log_path: str):
    """
    将检查结果写入日志文件
    
    Args:
        results: 检查结果列表
        log_path: 日志文件路径
    """
    total_files = len(results)
    passed_files = sum(1 for r in results if r['status'] == 'passed')
    failed_files = sum(1 for r in results if r['status'] == 'failed')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("全局有效位置掩膜检查结果\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总文件数: {total_files}\n")
        f.write(f"检查通过: {passed_files}\n")
        f.write(f"检查失败: {failed_files}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # 按状态分组写入
        failed_results = [r for r in results if r['status'] == 'failed']
        passed_results = [r for r in results if r['status'] == 'passed']
        
        if failed_files > 0:
            f.write(f"\n失败文件列表 ({failed_files} 个):\n")
            f.write("-" * 80 + "\n")
            for result in failed_results:
                f.write(f"\n文件: {result['file']}\n")
                f.write(f"状态: {result['status']}\n")
                f.write(f"检查的有效像素数: {result['valid_pixels_checked']}\n")
                if result['errors']:
                    f.write("错误信息:\n")
                    for error in result['errors']:
                        f.write(f"  - {error}\n")
                f.write("\n")
        
        if passed_files > 0:
            f.write(f"\n通过文件列表 ({passed_files} 个):\n")
            f.write("-" * 80 + "\n")
            for result in passed_results[:100]:  # 只显示前100个通过的
                f.write(f"{result['file']}\n")
            
            if passed_files > 100:
                f.write(f"... (还有 {passed_files - 100} 个文件通过检查)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("检查完成\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"日志已保存到: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="使用全局掩膜检查所有tif文件在有效位置处的无效值"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/global_valid_mask_v5.tif',
        help="全局有效位置掩膜文件路径"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/distance_maps',
        help="包含tif文件的根目录或单个tif文件路径"
    )
    parser.add_argument(
        "--output-log",
        type=str,
        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/mask_check_log.txt',
        help="输出日志文件路径 (默认: <root_dir>/mask_check_log.txt)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=f"最大并行线程数 (默认: CPU核心数 {multiprocessing.cpu_count()})"
    )
    
    args = parser.parse_args()
    
    # 设置默认并行线程数
    if args.max_workers is None:
        args.max_workers = multiprocessing.cpu_count()
    
    # 检查掩膜文件是否存在
    if not os.path.exists(args.mask):
        logger.error(f"掩膜文件不存在: {args.mask}")
        return
    
    # 检查根目录或文件是否存在
    if not os.path.exists(args.root_dir):
        logger.error(f"指定路径不存在: {args.root_dir}")
        return
    
    # 设置默认日志路径
    if args.output_log is None:
        # 如果root_dir是文件，使用文件所在目录；如果是目录，直接使用
        if os.path.isfile(args.root_dir):
            root_dir_for_log = os.path.dirname(args.root_dir)
        else:
            root_dir_for_log = args.root_dir
        args.output_log = os.path.join(root_dir_for_log, 'mask_check_log.txt')
    
    logger.info("=" * 80)
    logger.info("开始全局有效位置掩膜检查")
    logger.info("=" * 80)
    logger.info(f"掩膜文件: {args.mask}")
    logger.info(f"检查路径: {args.root_dir}")
    logger.info(f"输出日志: {args.output_log}")
    logger.info(f"并行线程数: {args.max_workers}")
    logger.info("")
    
    # 加载掩膜文件
    logger.info("步骤 1/4: 加载全局掩膜文件...")
    mask, size, geotransform, projection = load_mask_file(args.mask)
    mask_info = {
        'size': size,
        'geotransform': geotransform,
        'projection': projection,
        'mask_file': args.mask
    }
    
    # 查找所有tif文件
    logger.info("步骤 2/3: 查找所有tif文件...")
    tif_files = find_all_tif_files(args.root_dir)
    logger.info(f"找到 {len(tif_files)} 个tif文件")
    
    # 检查文件（动态写入日志）
    logger.info("步骤 3/3: 检查tif文件（日志文件实时更新中）...")
    results = process_files_parallel(tif_files, mask, mask_info, args.max_workers, args.output_log)
    
    # 输出统计信息
    total_files = len(results)
    passed_files = sum(1 for r in results if r['status'] == 'passed')
    failed_files = sum(1 for r in results if r['status'] == 'failed')
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("检查完成")
    logger.info("=" * 80)
    logger.info(f"总文件数: {total_files}")
    logger.info(f"检查通过: {passed_files}")
    logger.info(f"检查失败: {failed_files}")
    logger.info(f"详细结果已实时写入: {args.output_log}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
