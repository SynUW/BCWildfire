#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIF文件掩膜工具
使用模板文件的有效值范围掩膜目标文件夹中的所有TIF图像
"""

import os
import glob
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from osgeo import gdal, osr
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_tiff_with_metadata(file_path):
    """
    读取TIF文件并返回数据、地理变换、投影和NoData值
    """
    try:
        ds = gdal.Open(file_path, gdal.GA_ReadOnly)
        if ds is None:
            logger.error(f"无法打开文件: {file_path}")
            return None, None, None, None
        
        # 获取基本信息
        width = ds.RasterXSize
        height = ds.RasterYSize
        bands = ds.RasterCount
        
        # 获取地理变换和投影
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        
        # 读取数据
        if bands == 1:
            data = ds.GetRasterBand(1).ReadAsArray()
            nodata = ds.GetRasterBand(1).GetNoDataValue()
        else:
            data = np.array([ds.GetRasterBand(i+1).ReadAsArray() for i in range(bands)])
            nodata = [ds.GetRasterBand(i+1).GetNoDataValue() for i in range(bands)]
        
        ds = None
        return data, geotransform, projection, nodata
        
    except Exception as e:
        logger.error(f"读取文件失败 {file_path}: {e}")
        return None, None, None, None

def get_template_mask_and_nodata(template_file):
    """
    从模板文件中获取NoData值和空间掩膜
    """
    logger.info(f"从模板文件获取NoData值和空间掩膜: {template_file}")
    
    data, _, _, nodata = read_tiff_with_metadata(template_file)
    if data is None:
        logger.error("无法读取模板文件")
        return None, None, None
    
    # 处理多波段数据
    if data.ndim == 3:
        # 多波段：使用第一个波段确定掩膜
        band_data = data[0]
        band_nodata = nodata[0] if isinstance(nodata, list) else nodata
    else:
        # 单波段
        band_data = data
        band_nodata = nodata
    
    # 创建有效掩膜（非NoData且有限值）
    valid_mask = np.isfinite(band_data)
    if band_nodata is not None:
        valid_mask &= (band_data != band_nodata)
    
    # 详细调试信息
    logger.info(f"模板文件数据形状: {band_data.shape}")
    logger.info(f"模板文件数据类型: {band_data.dtype}")
    logger.info(f"模板文件NoData值: {band_nodata}")
    logger.info(f"模板文件数据范围: {np.nanmin(band_data):.3f} ~ {np.nanmax(band_data):.3f}")
    logger.info(f"模板文件有限值数量: {np.sum(np.isfinite(band_data))}")
    if band_nodata is not None:
        logger.info(f"模板文件NoData值数量: {np.sum(band_data == band_nodata)}")
    logger.info(f"模板文件有效像素数: {np.sum(valid_mask)} / {valid_mask.size} ({100*np.sum(valid_mask)/valid_mask.size:.1f}%)")
    
    if not np.any(valid_mask):
        logger.error("模板文件中没有有效数据")
        return None, None, None
    
    return valid_mask, band_nodata, data

def apply_template_mask_to_tiff(input_file, output_file, template_mask, template_nodata, template_gt, template_proj):
    """
    使用模板文件的空间掩膜和NoData值对TIF文件进行掩膜
    """
    try:
        # 读取输入文件
        data, geotransform, projection, nodata = read_tiff_with_metadata(input_file)
        if data is None:
            return False
        
        # 检查地理信息是否匹配
        if not np.allclose(geotransform, template_gt, rtol=1e-6):
            logger.warning(f"地理变换不匹配: {input_file}")
            return False
        
        if projection != template_proj:
            logger.warning(f"投影不匹配: {input_file}")
            return False
        
        # 检查数据尺寸是否匹配
        if data.ndim == 3:
            if data.shape[1:] != template_mask.shape:
                logger.warning(f"数据尺寸不匹配: {input_file} {data.shape[1:]} vs {template_mask.shape}")
                return False
        else:
            if data.shape != template_mask.shape:
                logger.warning(f"数据尺寸不匹配: {input_file} {data.shape} vs {template_mask.shape}")
                return False
        
        # 确保NoData值的数据类型与数据兼容
        if data.dtype != np.float32 and data.dtype != np.float64:
            # 对于整数类型，确保NoData值在有效范围内
            if data.dtype == np.uint8:
                if template_nodata < 0 or template_nodata > 255:
                    template_nodata = 255
            elif data.dtype == np.uint16:
                if template_nodata < 0 or template_nodata > 65535:
                    template_nodata = 65535
        
        # 应用模板掩膜
        if data.ndim == 3:
            # 多波段数据
            masked_data = data.copy()
            for i in range(data.shape[0]):
                # 使用模板掩膜
                masked_data[i][~template_mask] = template_nodata
        else:
            # 单波段数据
            masked_data = data.copy()
            # 使用模板掩膜
            masked_data[~template_mask] = template_nodata
        
        
        
        # 写入输出文件，使用模板的NoData值
        write_tiff_with_metadata(output_file, masked_data, geotransform, projection, template_nodata)
        return True
        
    except Exception as e:
        logger.error(f"处理文件失败 {input_file}: {e}")
        return False

def write_tiff_with_metadata(output_file, data, geotransform, projection, nodata):
    """
    将数据写入TIF文件
    """
    try:
        # 创建输出目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 确定数据类型
        if data.dtype == np.float32 or data.dtype == np.float64:
            dtype = gdal.GDT_Float32
        elif data.dtype == np.uint8:
            dtype = gdal.GDT_Byte
        elif data.dtype == np.uint16:
            dtype = gdal.GDT_UInt16
        elif data.dtype == np.int16:
            dtype = gdal.GDT_Int16
        else:
            dtype = gdal.GDT_Float32
        
        # 创建数据集
        if data.ndim == 3:
            bands, height, width = data.shape
        else:
            bands = 1
            height, width = data.shape
            data = data[np.newaxis, ...]
        
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(output_file, width, height, bands, dtype)
        
        if ds is None:
            logger.error(f"无法创建输出文件: {output_file}")
            return False
        
        # 设置地理信息
        ds.SetGeoTransform(geotransform)
        ds.SetProjection(projection)
        
        # 写入数据
        for i in range(bands):
            band = ds.GetRasterBand(i + 1)
            band.WriteArray(data[i])
            
            # 设置NoData值
            if isinstance(nodata, list) and i < len(nodata):
                if nodata[i] is not None:
                    band.SetNoDataValue(nodata[i])
            elif not isinstance(nodata, list) and nodata is not None:
                band.SetNoDataValue(nodata)
        
        ds = None
        return True
        
    except Exception as e:
        logger.error(f"写入文件失败 {output_file}: {e}")
        return False

def process_single_file(args_tuple):
    """
    处理单个文件的函数，用于多线程
    """
    input_file, output_file, template_mask, template_nodata, template_gt, template_proj = args_tuple
    
    # 应用模板掩膜
    success = apply_template_mask_to_tiff(input_file, output_file, template_mask, template_nodata, template_gt, template_proj)
    
    if success:
        return input_file, True, None
    else:
        return input_file, False, f"处理失败: {input_file}"

def main():
    parser = argparse.ArgumentParser(description='使用模板文件的有效值范围掩膜TIF文件')
    parser.add_argument('--template', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/ERA5_consistent_mosaic_withnodata_downsampled/2024_12_31.tif', help='模板TIF文件路径')
    parser.add_argument('--input_dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/MCD14A1_mosaic_binary_downsampled', help='输入TIF文件目录')
    parser.add_argument('--output_dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/MCD14A1_mosaic_binary_downsampled_masked', help='输出掩膜后文件目录')
    parser.add_argument('--pattern', default='*.tif', help='文件匹配模式 (默认: *.tif)')
    parser.add_argument('--recursive', action='store_true', help='递归搜索子目录')
    parser.add_argument('--threads', type=int, default=4, help='线程数 (默认: 4)')
    
    args = parser.parse_args()
    
    # 检查模板文件
    if not os.path.exists(args.template):
        logger.error(f"模板文件不存在: {args.template}")
        return
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取模板文件的掩膜和NoData值
    template_mask, template_nodata, template_data = get_template_mask_and_nodata(args.template)
    if template_mask is None:
        logger.error("无法获取模板文件的掩膜信息")
        return
    
    # 获取模板文件的地理信息
    _, template_gt, template_proj, _ = read_tiff_with_metadata(args.template)
    if template_gt is None:
        logger.error("无法获取模板文件的地理信息")
        return
    
    # 查找输入文件
    if args.recursive:
        pattern = os.path.join(args.input_dir, '**', args.pattern)
        input_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(args.input_dir, args.pattern)
        input_files = glob.glob(pattern)
    
    if not input_files:
        logger.error(f"在 {args.input_dir} 中未找到匹配 {args.pattern} 的文件")
        return
    
    logger.info(f"找到 {len(input_files)} 个输入文件")
    logger.info(f"使用 {args.threads} 个线程进行处理")
    
    # 准备多线程参数
    file_args = []
    for input_file in input_files:
        # 生成输出文件路径
        rel_path = os.path.relpath(input_file, args.input_dir)
        output_file = os.path.join(args.output_dir, rel_path)
        
        # 创建输出文件的目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        file_args.append((input_file, output_file, template_mask, template_nodata, template_gt, template_proj))
    
    # 使用多线程处理文件
    success_count = 0
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_single_file, file_arg): file_arg[0] for file_arg in file_args}
        
        # 使用tqdm显示进度
        with tqdm(total=len(input_files), desc="处理文件") as pbar:
            for future in as_completed(future_to_file):
                input_file, success, error_msg = future.result()
                
                if success:
                    success_count += 1
                else:
                    failed_files.append(input_file)
                    logger.warning(error_msg)
                
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': len(failed_files),
                    '进度': f"{success_count + len(failed_files)}/{len(input_files)}"
                })
    
    logger.info(f"处理完成: {success_count}/{len(input_files)} 个文件成功")
    if failed_files:
        logger.warning(f"失败的文件: {failed_files}")

if __name__ == "__main__":
    main()