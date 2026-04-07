# -*- coding: utf-8 -*-
"""
合并两个文件夹中的TIFF文件，按波段拼接

功能：
- 接收两个文件夹输入，都包含tif文件
- 文件命名格式：*_YYYY_MM_DD_tilexx.tif（xx为00、01、02、03等）
- 匹配相同日期和tile的文件进行合并
- 合并方式：第二个文件夹的波段拼接在第一个文件夹对应文件的波段后面
- 分辨率处理：以第二个图像的分辨率为基准，使用最近邻法下采样第一个图像

使用方法：
    python merge_two_folders_bands.py --input-dir1 /path/to/folder1 --input-dir2 /path/to/folder2 --output-dir /path/to/output
"""

import os
import sys
import argparse
import logging
import time
import re
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from osgeo import gdal

# ================== 默认配置 ==================
# 并行处理配置
DEFAULT_MAX_WORKERS = max(1, os.cpu_count() // 2)  # 默认使用一半CPU核心数

# GDAL优化配置
GDAL_CACHE_SIZE    = 102400       # GDAL缓存大小 (MB)
GDAL_NUM_THREADS   = "ALL_CPUS"   # GDAL线程数
COMPRESS           = "LZW"        # 输出压缩
TILED              = True          # 输出块瓦片标志
BIGTIFF            = "IF_SAFER"    # 自动切换 BigTIFF
BLOCK_SIZE         = 256           # 块大小

# 线程安全的计数器
progress_lock = Lock()

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merge_bands")

# 初始化GDAL
gdal.UseExceptions()
gdal.SetCacheMax(GDAL_CACHE_SIZE * 1024 * 1024)
gdal.SetConfigOption('GDAL_NUM_THREADS', GDAL_NUM_THREADS)
gdal.SetConfigOption('GDAL_CACHEMAX', str(GDAL_CACHE_SIZE))

# 文件名匹配模式：*_YYYY_MM_DD_tilexx.tif
FILENAME_PATTERN = re.compile(r'(.+)_(\d{4})[_-](\d{2})[_-](\d{2})_tile(\d+)\.tif$', re.IGNORECASE)


def format_time(seconds):
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def parse_filename(filename):
    """
    解析文件名，提取日期和tile编号
    
    Args:
        filename: 文件名（不含路径）
    
    Returns:
        tuple: (date_str, tile_num) 或 None
        - date_str: YYYY_MM_DD格式的日期字符串
        - tile_num: tile编号字符串（如"00", "01"）
    """
    match = FILENAME_PATTERN.match(filename)
    if match:
        prefix, year, month, day, tile_num = match.groups()
        date_str = f"{year}_{month}_{day}"
        return date_str, tile_num
    return None


def group_files_by_date_tile(directory):
    """
    按日期和tile分组文件
    
    Args:
        directory: 输入目录路径
    
    Returns:
        dict: {(date_str, tile_num): file_path}
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    file_dict = {}
    patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    
    for pattern in patterns:
        for file_path in dir_path.glob(pattern):
            parsed = parse_filename(file_path.name)
            if parsed:
                date_str, tile_num = parsed
                key = (date_str, tile_num)
                if key in file_dict:
                    logger.warning(f"发现重复的日期和tile组合: {file_path.name}，将使用第一个文件")
                else:
                    file_dict[key] = file_path
    
    return file_dict


def get_file_info(ds):
    """获取GDAL数据集的基本信息"""
    return {
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'bands': ds.RasterCount,
        'geotransform': ds.GetGeoTransform(),
        'projection': ds.GetProjection(),
        'datatype': ds.GetRasterBand(1).DataType,
    }


def resample_to_target(ds_source, ds_target, output_path=None):
    """
    将源数据集重采样到目标数据集的分辨率和地理范围
    
    Args:
        ds_source: 源GDAL数据集（第一个文件夹的文件）
        ds_target: 目标GDAL数据集（第二个文件夹的文件，作为基准）
        output_path: 临时输出路径（可选，如果提供则保存到文件）
    
    Returns:
        GDAL数据集对象或None（如果失败）
    """
    try:
        # 获取目标数据集的信息
        target_info = get_file_info(ds_target)
        target_gt = target_info['geotransform']
        target_proj = target_info['projection']
        target_width = target_info['width']
        target_height = target_info['height']
        
        # 计算目标地理范围
        pixel_size_x = abs(target_gt[1])
        pixel_size_y = abs(target_gt[5])
        target_min_x = target_gt[0]
        target_max_y = target_gt[3]
        target_max_x = target_min_x + target_width * pixel_size_x
        target_min_y = target_max_y - target_height * pixel_size_y
        
        # 使用gdal.Warp进行重采样
        # 关键：明确指定输出宽度和高度，确保尺寸完全匹配
        warp_options = gdal.WarpOptions(
            format='GTiff' if output_path else 'MEM',
            width=target_width,  # 明确指定输出宽度
            height=target_height,  # 明确指定输出高度
            outputBounds=(target_min_x, target_min_y, target_max_x, target_max_y),  # 目标地理范围
            resampleAlg=gdal.GRA_NearestNeighbour,  # 最近邻重采样
            dstSRS=target_proj,  # 目标投影
            outputType=ds_source.GetRasterBand(1).DataType,  # 保持原始数据类型
            creationOptions=[
                f'COMPRESS={COMPRESS}',
                'TILED=YES' if TILED else 'TILED=NO',
                f'BLOCKXSIZE={BLOCK_SIZE}',
                f'BLOCKYSIZE={BLOCK_SIZE}',
                f'BIGTIFF={BIGTIFF}',
                'NUM_THREADS=ALL_CPUS'
            ] if output_path else None,
            multithread=True,
            warpOptions=['NUM_THREADS=ALL_CPUS']
        )
        
        # 执行重采样
        warp_dest = output_path if output_path else '/vsimem/temp_resampled'
        resampled_ds = gdal.Warp(
            warp_dest,
            ds_source,
            options=warp_options
        )
        
        if resampled_ds is None:
            gdal_error = gdal.GetLastErrorMsg()
            error_msg = "重采样失败"
            if gdal_error:
                error_msg += f": {gdal_error}"
            logger.error(error_msg)
            return None
        
        # 验证输出尺寸（应该完全匹配，因为明确指定了width和height）
        if resampled_ds.RasterXSize != target_width or resampled_ds.RasterYSize != target_height:
            logger.warning(f"重采样后尺寸不匹配: {resampled_ds.RasterXSize}x{resampled_ds.RasterYSize} vs {target_width}x{target_height}")
            logger.warning("这不应该发生，因为明确指定了width和height")
            # 如果仍然不匹配，强制调整
            resampled_ds = None
            return resample_to_target_force(ds_source, ds_target, output_path)
        
        # 确保地理变换完全匹配（包括所有6个参数）
        resampled_gt = resampled_ds.GetGeoTransform()
        if resampled_gt != target_gt:
            logger.debug("调整地理变换以匹配目标")
            resampled_ds.SetGeoTransform(target_gt)
            # 重新获取以验证
            resampled_gt = resampled_ds.GetGeoTransform()
        
        # 验证像素大小完全一致
        resampled_pixel_x = abs(resampled_gt[1])
        resampled_pixel_y = abs(resampled_gt[5])
        target_pixel_x = abs(target_gt[1])
        target_pixel_y = abs(target_gt[5])
        
        if abs(resampled_pixel_x - target_pixel_x) > 1e-10 or abs(resampled_pixel_y - target_pixel_y) > 1e-10:
            logger.warning(f"像素大小不完全匹配: {resampled_pixel_x:.10f}x{resampled_pixel_y:.10f} vs {target_pixel_x:.10f}x{target_pixel_y:.10f}")
            # 强制设置地理变换
            resampled_ds.SetGeoTransform(target_gt)
        
        # 验证空间覆盖范围
        resampled_min_x = resampled_gt[0]
        resampled_max_y = resampled_gt[3]
        target_min_x = target_gt[0]
        target_max_y = target_gt[3]
        
        if abs(resampled_min_x - target_min_x) > 1e-6 or abs(resampled_max_y - target_max_y) > 1e-6:
            logger.warning(f"空间覆盖不完全匹配: 左上角 ({resampled_min_x:.6f}, {resampled_max_y:.6f}) vs ({target_min_x:.6f}, {target_max_y:.6f})")
            # 强制设置地理变换
            resampled_ds.SetGeoTransform(target_gt)
        
        return resampled_ds
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"重采样过程出错: {e}")
        logger.error(f"详细错误信息:\n{error_details}")
        gdal_error = gdal.GetLastErrorMsg()
        if gdal_error:
            logger.error(f"GDAL错误信息: {gdal_error}")
        # 如果上述方法失败，尝试强制方法
        try:
            return resample_to_target_force(ds_source, ds_target, output_path)
        except Exception as e2:
            import traceback
            error_details2 = traceback.format_exc()
            logger.error(f"强制重采样也失败: {e2}")
            logger.error(f"强制重采样详细错误信息:\n{error_details2}")
            return None


def resample_to_target_force(ds_source, ds_target, output_path=None):
    """
    强制重采样方法：直接创建指定尺寸的输出数据集并手动复制数据
    用于处理尺寸不匹配的情况
    """
    try:
        target_info = get_file_info(ds_target)
        target_gt = target_info['geotransform']
        target_proj = target_info['projection']
        target_width = target_info['width']
        target_height = target_info['height']
        target_datatype = ds_source.GetRasterBand(1).DataType
        
        # 先使用Warp进行重采样（不指定width/height，让GDAL自动计算）
        pixel_size_x = abs(target_gt[1])
        pixel_size_y = abs(target_gt[5])
        target_min_x = target_gt[0]
        target_max_y = target_gt[3]
        target_max_x = target_min_x + target_width * pixel_size_x
        target_min_y = target_max_y - target_height * pixel_size_y
        
        warp_options = gdal.WarpOptions(
            format='MEM',
            outputBounds=(target_min_x, target_min_y, target_max_x, target_max_y),
            xRes=pixel_size_x,
            yRes=pixel_size_y,
            resampleAlg=gdal.GRA_NearestNeighbour,
            dstSRS=target_proj,
            outputType=target_datatype,
            multithread=True
        )
        
        temp_ds = gdal.Warp('/vsimem/temp_warp', ds_source, options=warp_options)
        if temp_ds is None:
            return None
        
        # 创建正确尺寸的输出数据集
        driver = gdal.GetDriverByName('GTiff' if output_path else 'MEM')
        creation_options = [
            f'COMPRESS={COMPRESS}',
            'TILED=YES' if TILED else 'TILED=NO',
            f'BLOCKXSIZE={BLOCK_SIZE}',
            f'BLOCKYSIZE={BLOCK_SIZE}',
            f'BIGTIFF={BIGTIFF}',
            'NUM_THREADS=ALL_CPUS'
        ] if output_path else None
        
        out_ds = driver.Create(
            output_path if output_path else '/vsimem/temp_resampled',
            target_width,
            target_height,
            temp_ds.RasterCount,
            target_datatype,
            options=creation_options
        )
        
        if out_ds is None:
            temp_ds = None
            return None
        
        out_ds.SetGeoTransform(target_gt)
        out_ds.SetProjection(target_proj)
        
        # 计算需要复制的区域（从temp_ds的中心或左上角开始）
        src_width = temp_ds.RasterXSize
        src_height = temp_ds.RasterYSize
        
        # 如果源尺寸大于目标，从左上角裁剪
        # 如果源尺寸小于目标，从左上角开始填充
        copy_width = min(target_width, src_width)
        copy_height = min(target_height, src_height)
        src_x = 0
        src_y = 0
        dst_x = 0
        dst_y = 0
        
        # 复制数据
        for band_idx in range(1, temp_ds.RasterCount + 1):
            src_band = temp_ds.GetRasterBand(band_idx)
            dst_band = out_ds.GetRasterBand(band_idx)
            
            # 设置NoData值
            nodata = src_band.GetNoDataValue()
            if nodata is not None:
                dst_band.SetNoDataValue(nodata)
                # 先填充NoData值
                dst_band.Fill(nodata)
            
            # 读取源数据并写入目标
            if copy_width > 0 and copy_height > 0:
                src_data = src_band.ReadAsArray(src_x, src_y, copy_width, copy_height)
                dst_band.WriteArray(src_data, dst_x, dst_y)
            
            dst_band.FlushCache()
        
        temp_ds = None
        return out_ds
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"强制重采样失败: {e}")
        logger.error(f"详细错误信息:\n{error_details}")
        gdal_error = gdal.GetLastErrorMsg()
        if gdal_error:
            logger.error(f"GDAL错误信息: {gdal_error}")
        return None


def merge_bands(ds1, ds2, output_path, overwrite=False):
    """
    合并两个数据集的波段
    
    Args:
        ds1: 第一个数据集（将被重采样）
        ds2: 第二个数据集（作为分辨率基准）
        output_path: 输出文件路径
        overwrite: 是否覆盖已存在的文件
    
    Returns:
        bool: 成功返回True，失败返回False
    """
    if not overwrite and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info(f"文件已存在，跳过: {output_path}")
        return True
    
    try:
        # 获取第二个数据集的信息（作为基准）
        target_info = get_file_info(ds2)
        target_width = target_info['width']
        target_height = target_info['height']
        target_gt = target_info['geotransform']
        target_proj = target_info['projection']
        target_datatype = target_info['datatype']
        
        # 获取波段数
        bands1 = ds1.RasterCount
        bands2 = ds2.RasterCount
        total_bands = bands1 + bands2
        
        logger.debug(f"合并波段: {bands1} + {bands2} = {total_bands}")
        logger.debug(f"输出尺寸: {target_width} x {target_height}")
        
        # 重采样第一个数据集到第二个数据集的分辨率
        logger.debug("开始重采样第一个数据集...")
        logger.debug(f"源数据集信息: {ds1.RasterXSize}x{ds1.RasterYSize}, {ds1.RasterCount}波段")
        logger.debug(f"目标数据集信息: {target_width}x{target_height}, {ds2.RasterCount}波段")
        resampled_ds1 = resample_to_target(ds1, ds2)
        if resampled_ds1 is None:
            logger.error("重采样第一个数据集失败")
            logger.error(f"源文件: {ds1.GetDescription() if hasattr(ds1, 'GetDescription') else 'N/A'}")
            logger.error(f"目标文件: {ds2.GetDescription() if hasattr(ds2, 'GetDescription') else 'N/A'}")
            return False
        
        # 验证重采样后的尺寸
        if resampled_ds1.RasterXSize != target_width or resampled_ds1.RasterYSize != target_height:
            logger.error(f"重采样后尺寸不匹配: {resampled_ds1.RasterXSize}x{resampled_ds1.RasterYSize} vs {target_width}x{target_height}")
            resampled_ds1 = None
            return False
        
        # 创建输出数据集
        driver = gdal.GetDriverByName('GTiff')
        creation_options = [
            f'COMPRESS={COMPRESS}',
            'TILED=YES' if TILED else 'TILED=NO',
            f'BLOCKXSIZE={BLOCK_SIZE}',
            f'BLOCKYSIZE={BLOCK_SIZE}',
            f'BIGTIFF={BIGTIFF}',
            'NUM_THREADS=ALL_CPUS'
        ]
        
        out_ds = driver.Create(
            output_path,
            target_width,
            target_height,
            total_bands,
            target_datatype,
            options=creation_options
        )
        
        if out_ds is None:
            gdal_error = gdal.GetLastErrorMsg()
            error_msg = f"无法创建输出文件: {output_path}"
            if gdal_error:
                error_msg += f"\nGDAL错误: {gdal_error}"
            logger.error(error_msg)
            resampled_ds1 = None
            return False
        
        # 设置地理信息
        out_ds.SetGeoTransform(target_gt)
        out_ds.SetProjection(target_proj)
        
        # 统一NoData值为-32768
        UNIFIED_NODATA = -32768
        
        # GDAL数据类型到numpy数据类型的映射
        dtype_map = {
            gdal.GDT_Byte: np.uint8,
            gdal.GDT_UInt16: np.uint16,
            gdal.GDT_Int16: np.int16,
            gdal.GDT_UInt32: np.uint32,
            gdal.GDT_Int32: np.int32,
            gdal.GDT_Float32: np.float32,
            gdal.GDT_Float64: np.float64,
        }
        target_np_dtype = dtype_map.get(target_datatype, np.int16)  # 默认int16
        
        # 使用分块处理以提高性能（特别是对大图像）
        # 块大小：使用256x256或更大的块，但不超过图像大小
        block_size = min(512, target_width, target_height)  # 512x512像素的块
        
        # 复制第一个数据集的波段（使用分块处理）
        logger.debug(f"复制第一个数据集的 {bands1} 个波段（分块处理）...")
        for band_idx in range(1, bands1 + 1):
            src_band = resampled_ds1.GetRasterBand(band_idx)
            dst_band = out_ds.GetRasterBand(band_idx)
            
            # 获取源NoData值
            src_nodata = src_band.GetNoDataValue()
            need_nodata_replace = (src_nodata is not None and src_nodata != UNIFIED_NODATA)
            
            # 设置统一的NoData值
            dst_band.SetNoDataValue(UNIFIED_NODATA)
            
            # 分块读取和写入
            for y in range(0, target_height, block_size):
                y_size = min(block_size, target_height - y)
                for x in range(0, target_width, block_size):
                    x_size = min(block_size, target_width - x)
                    
                    # 读取块数据
                    data = src_band.ReadAsArray(x, y, x_size, y_size)
                    
                    # 处理NoData值
                    if need_nodata_replace:
                        data = np.where(data == src_nodata, UNIFIED_NODATA, data)
                    
                    # 确保数据类型匹配
                    if data.dtype != target_np_dtype:
                        data = data.astype(target_np_dtype)
                    
                    # 写入块数据
                    dst_band.WriteArray(data, x, y)
        
        # 复制第二个数据集的波段（使用分块处理）
        logger.debug(f"复制第二个数据集的 {bands2} 个波段（分块处理）...")
        for band_idx in range(1, bands2 + 1):
            src_band = ds2.GetRasterBand(band_idx)
            dst_band = out_ds.GetRasterBand(bands1 + band_idx)
            
            # 获取源NoData值
            src_nodata = src_band.GetNoDataValue()
            need_nodata_replace = (src_nodata is not None and src_nodata != UNIFIED_NODATA)
            
            # 设置统一的NoData值
            dst_band.SetNoDataValue(UNIFIED_NODATA)
            
            # 分块读取和写入
            for y in range(0, target_height, block_size):
                y_size = min(block_size, target_height - y)
                for x in range(0, target_width, block_size):
                    x_size = min(block_size, target_width - x)
                    
                    # 读取块数据
                    data = src_band.ReadAsArray(x, y, x_size, y_size)
                    
                    # 处理NoData值
                    if need_nodata_replace:
                        data = np.where(data == src_nodata, UNIFIED_NODATA, data)
                    
                    # 确保数据类型匹配
                    if data.dtype != target_np_dtype:
                        data = data.astype(target_np_dtype)
                    
                    # 写入块数据
                    dst_band.WriteArray(data, x, y)
        
        # 最后统一刷新缓存（而不是每个波段都刷新）
        out_ds.FlushCache()
        
        # 清理资源
        out_ds = None
        resampled_ds1 = None
        
        logger.debug(f"成功合并并保存: {output_path}")
        return True
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"合并波段时出错: {e}")
        logger.error(f"详细错误信息:\n{error_details}")
        logger.error(f"输出路径: {output_path}")
        # 尝试获取GDAL错误信息
        gdal_error = gdal.GetLastErrorMsg()
        if gdal_error:
            logger.error(f"GDAL错误信息: {gdal_error}")
        return False


def main():
    parser = argparse.ArgumentParser(description='合并两个文件夹中的TIFF文件，按波段拼接')
    
    parser.add_argument('--input-dir1', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD09GA_b1237',
                       help='第一个输入文件夹路径')
    parser.add_argument('--input-dir2', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD09GA_QA_state1km',
                       help='第二个输入文件夹路径（1km QA, 作为分辨率基准）')
    parser.add_argument('--output-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MOD09GA_b1237_withQA',
                       help='输出文件夹路径')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的输出文件')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细信息')
    parser.add_argument('--workers', type=int, default=DEFAULT_MAX_WORKERS//4,
                       help=f'并行工作线程数 (默认: {DEFAULT_MAX_WORKERS})')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证输入目录
    for dir_path, name in [(args.input_dir1, '第一个输入目录'), 
                           (args.input_dir2, '第二个输入目录')]:
        if not os.path.isdir(dir_path):
            logger.error(f"{name}不存在：{dir_path}")
            sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 分组文件
    logger.info("扫描第一个文件夹...")
    files1 = group_files_by_date_tile(args.input_dir1)
    logger.info(f"第一个文件夹找到 {len(files1)} 个文件")
    
    logger.info("扫描第二个文件夹...")
    files2 = group_files_by_date_tile(args.input_dir2)
    logger.info(f"第二个文件夹找到 {len(files2)} 个文件")
    
    # 找到匹配的文件对
    matched_pairs = []
    for key in files1:
        if key in files2:
            matched_pairs.append((key, files1[key], files2[key]))
    
    if not matched_pairs:
        logger.warning("未找到匹配的文件对（相同日期和tile）")
        sys.exit(0)
    
    logger.info(f"找到 {len(matched_pairs)} 对匹配的文件")
    logger.info(f"使用 {args.workers} 个并行工作线程")
    
    # 处理函数（用于并行处理）
    def process_file_pair(pair_data):
        """处理单个文件对的函数"""
        idx, key, file1_path, file2_path, output_dir, overwrite = pair_data
        date_str, tile_num = key
        # 使用第一个文件夹的文件名作为输出文件名
        output_filename = file1_path.name
        output_path = output_dir / output_filename
        
        try:
            # 打开文件
            ds1 = gdal.Open(str(file1_path), gdal.GA_ReadOnly)
            if ds1 is None:
                gdal_error = gdal.GetLastErrorMsg()
                error_msg = f"无法打开文件1: {file1_path.name}"
                if gdal_error:
                    error_msg += f" (GDAL错误: {gdal_error})"
                return False, error_msg
            
            ds2 = gdal.Open(str(file2_path), gdal.GA_ReadOnly)
            if ds2 is None:
                gdal_error = gdal.GetLastErrorMsg()
                error_msg = f"无法打开文件2: {file2_path.name}"
                if gdal_error:
                    error_msg += f" (GDAL错误: {gdal_error})"
                ds1 = None
                return False, error_msg
            
            # 合并波段
            success = merge_bands(ds1, ds2, str(output_path), overwrite)
            
            # 清理资源
            ds1 = None
            ds2 = None
            
            if success:
                return True, None
            else:
                # merge_bands已经记录了详细错误信息，这里只返回简要信息
                return False, f"合并失败: {date_str} tile{tile_num} (文件1: {file1_path.name}, 文件2: {file2_path.name}, 输出: {output_filename})"
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"处理文件对时出错 {date_str} tile{tile_num}: {e}\n详细错误:\n{error_details}"
            return False, error_msg
    
    # 准备任务数据
    tasks = [
        (idx, key, file1_path, file2_path, output_dir, args.overwrite)
        for idx, (key, file1_path, file2_path) in enumerate(matched_pairs, 1)
    ]
    
    # 并行处理
    total_success = 0
    total_failed = 0
    start_time = time.time()
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_file_pair, task): task for task in tasks}
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            try:
                success, error_msg = future.result()
                with progress_lock:
                    completed_count += 1
                    if success:
                        total_success += 1
                    else:
                        total_failed += 1
                        if error_msg:
                            logger.error(error_msg)
                
                # 显示进度
                if completed_count % 10 == 0 or completed_count == len(matched_pairs):
                    elapsed = time.time() - start_time
                    progress = completed_count / len(matched_pairs) * 100
                    avg_time = elapsed / completed_count if completed_count > 0 else 0
                    eta = avg_time * (len(matched_pairs) - completed_count)
                    print(f"\r进度: [{completed_count}/{len(matched_pairs)}] ({progress:.1f}%) | "
                          f"成功: {total_success} | 失败: {total_failed} | "
                          f"耗时: {format_time(elapsed)} | ETA: {format_time(eta)}", 
                          end='', flush=True)
            except Exception as e:
                with progress_lock:
                    completed_count += 1
                    total_failed += 1
                    logger.error(f"任务执行异常: {e}")
    
    print()  # 换行
    
    elapsed_time = time.time() - start_time
    
    # 输出统计信息
    logger.info("=" * 60)
    logger.info("处理完成!")
    logger.info(f"总耗时: {format_time(elapsed_time)}")
    logger.info(f"成功: {total_success}/{len(matched_pairs)}")
    if total_failed > 0:
        logger.warning(f"失败: {total_failed}/{len(matched_pairs)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

