# -*- coding: utf-8 -*-
"""
检查TIFF文件是否能被GDAL读取，如果无法读取则删除该文件

使用方法：
    python check_and_remove_invalid_files.py --input-dir /path/to/input
    python check_and_remove_invalid_files.py --input-dir /path/to/input --dry-run  # 仅检查，不删除
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from osgeo import gdal

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("check_files")

# 初始化GDAL
gdal.UseExceptions()


def format_time(seconds):
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def check_file_readable(file_path):
    """
    检查文件是否能被GDAL读取
    
    Returns:
        tuple: (is_valid, error_message)
        - is_valid: True表示文件可读，False表示不可读
        - error_message: 如果不可读，返回错误信息；否则返回None
    """
    try:
        ds = gdal.Open(str(file_path), gdal.GA_ReadOnly)
        if ds is None:
            error_msg = gdal.GetLastErrorMsg()
            if not error_msg:
                error_msg = "无法打开文件（GDAL返回None）"
            return False, error_msg
        
        # 尝试读取基本信息以确保文件完整
        try:
            width = ds.RasterXSize
            height = ds.RasterYSize
            bands = ds.RasterCount
            if width <= 0 or height <= 0 or bands <= 0:
                ds = None
                return False, f"无效的栅格尺寸: {width}x{height}, 波段数: {bands}"
            
            # 尝试读取地理变换信息
            gt = ds.GetGeoTransform()
            proj = ds.GetProjection()
            
            # 尝试读取第一个波段的一小块数据（验证数据完整性）
            band = ds.GetRasterBand(1)
            if band is None:
                ds = None
                return False, "无法读取第一个波段"
            
            # 读取左上角1x1像素验证数据可读性
            sample_data = band.ReadAsArray(0, 0, 1, 1)
            
            ds = None
            return True, None
            
        except Exception as e:
            ds = None
            return False, f"读取文件信息时出错: {str(e)}"
            
    except RuntimeError as e:
        error_msg = str(e)
        if "not recognized as a supported file format" in error_msg:
            return False, "不支持的文件格式"
        return False, f"RuntimeError: {error_msg}"
    except Exception as e:
        return False, f"未知错误: {str(e)}"


def remove_file(file_path, dry_run=False):
    """删除文件"""
    if dry_run:
        logger.info(f"[DRY-RUN] 将删除: {file_path}")
        return True
    
    try:
        os.remove(str(file_path))
        return True
    except Exception as e:
        logger.error(f"删除文件失败 {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='检查TIFF文件是否能被GDAL读取，无法读取则删除')
    
    parser.add_argument('--input-dir', 
                       default=r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD09GA_b1237',
                       help='包含TIFF文件的输入目录')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅检查，不实际删除文件')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证输入目录
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"输入目录不存在：{args.input_dir}")
        sys.exit(1)
    
    if not input_path.is_dir():
        logger.error(f"输入路径不是目录：{args.input_dir}")
        sys.exit(1)
    
    # 查找所有TIFF文件
    patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    all_files = []
    for pattern in patterns:
        all_files.extend(input_path.glob(pattern))
    
    if not all_files:
        logger.warning(f"在目录 {args.input_dir} 中未找到TIFF文件")
        sys.exit(0)
    
    total_files = len(all_files)
    logger.info(f"找到 {total_files} 个TIFF文件")
    
    if args.dry_run:
        logger.info("=== DRY-RUN 模式：仅检查，不会删除文件 ===")
    
    # 统计信息
    valid_files = 0
    invalid_files = 0
    deleted_files = 0
    failed_deletions = 0
    
    start_time = time.time()
    
    # 检查每个文件
    for idx, file_path in enumerate(sorted(all_files), 1):
        # 显示进度
        if idx % 100 == 0 or idx == total_files:
            elapsed = time.time() - start_time
            progress = idx / total_files * 100
            print(f"\r进度: [{idx}/{total_files}] ({progress:.1f}%) | "
                  f"有效: {valid_files} | 无效: {invalid_files} | "
                  f"已删除: {deleted_files} | 耗时: {format_time(elapsed)}", 
                  end='', flush=True)
        
        # 检查文件是否可读
        is_valid, error_msg = check_file_readable(file_path)
        
        if is_valid:
            valid_files += 1
            if args.verbose:
                logger.debug(f"✓ {file_path.name}")
        else:
            invalid_files += 1
            logger.warning(f"✗ {file_path.name}: {error_msg}")
            
            # 删除无效文件
            if remove_file(file_path, dry_run=args.dry_run):
                deleted_files += 1
            else:
                failed_deletions += 1
    
    print()  # 换行
    
    elapsed_time = time.time() - start_time
    
    # 输出统计信息
    logger.info("=" * 60)
    logger.info("检查完成!")
    logger.info(f"总耗时: {format_time(elapsed_time)}")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"有效文件: {valid_files} ({valid_files/total_files*100:.1f}%)")
    logger.info(f"无效文件: {invalid_files} ({invalid_files/total_files*100:.1f}%)")
    
    if args.dry_run:
        logger.info(f"[DRY-RUN] 将删除的文件数: {invalid_files}")
    else:
        logger.info(f"成功删除: {deleted_files}")
        if failed_deletions > 0:
            logger.warning(f"删除失败: {failed_deletions}")
    
    logger.info("=" * 60)
    
    if invalid_files > 0 and not args.dry_run:
        logger.info(f"已清理 {deleted_files} 个无效文件")
    elif invalid_files > 0 and args.dry_run:
        logger.info(f"发现 {invalid_files} 个无效文件（使用 --dry-run 查看详情）")


if __name__ == "__main__":
    main()

