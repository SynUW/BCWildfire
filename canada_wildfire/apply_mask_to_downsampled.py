#!/usr/bin/env python3
"""
掩膜应用工具
- 从ERA5_multi_bands文件夹的第一个文件的第一个波段创建掩膜（值>220的像元）
- 将掩膜应用到所有下采样数据
- 保持原始文件结构
- 使用绝对坐标进行掩膜匹配
"""

import os
import glob
import numpy as np
from osgeo import gdal, gdalconst, osr
import logging
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('MaskApplier')

# 设置GDAL配置
gdal.SetConfigOption('GDAL_CACHEMAX', '20480')
gdal.SetConfigOption('GDAL_NUM_THREADS', '16')
gdal.UseExceptions()

class MaskApplier:
    def __init__(self, downsampled_root, output_root, max_workers=8):
        """
        初始化掩膜应用器
        
        Args:
            downsampled_root: 下采样数据根目录
            output_root: 输出根目录
            max_workers: 最大线程数
        """
        self.downsampled_root = downsampled_root
        self.output_root = output_root
        self.max_workers = max_workers
        
        # 掩膜相关属性
        self.mask_data = None
        self.mask_geotransform = None
        self.mask_projection = None
        self.mask_width = None
        self.mask_height = None
        
    def create_mask_from_era5(self):
        """
        从ERA5数据创建掩膜
        """
        era5_folder = os.path.join(self.downsampled_root, 'ERA5_multi_bands_10x')
        
        if not os.path.exists(era5_folder):
            logger.error(f"ERA5文件夹不存在: {era5_folder}")
            return False
        
        # 获取第一个tif文件
        tif_files = glob.glob(os.path.join(era5_folder, '*.tif'))
        if not tif_files:
            logger.error(f"ERA5文件夹中没有找到tif文件: {era5_folder}")
            return False
        
        # 按文件名排序，取第一个
        tif_files.sort()
        first_file = tif_files[0]
        
        logger.info(f"使用文件创建掩膜: {os.path.basename(first_file)}")
        
        try:
            # 打开文件
            ds = gdal.Open(first_file, gdalconst.GA_ReadOnly)
            if ds is None:
                logger.error(f"无法打开文件: {first_file}")
                return False
            
            # 获取地理信息
            self.mask_geotransform = ds.GetGeoTransform()
            self.mask_projection = ds.GetProjection()
            self.mask_width = ds.RasterXSize
            self.mask_height = ds.RasterYSize
            
            # 读取第一个波段
            band = ds.GetRasterBand(1)
            data = band.ReadAsArray()
            
            # 创建掩膜：值>220的像元为1，其他为0
            self.mask_data = (data > 200).astype(np.uint8)
            
            # 统计掩膜信息
            total_pixels = self.mask_data.size
            valid_pixels = np.sum(self.mask_data)
            valid_ratio = valid_pixels / total_pixels * 100
            
            logger.info(f"掩膜创建完成:")
            logger.info(f"  - 掩膜尺寸: {self.mask_width} x {self.mask_height}")
            logger.info(f"  - 有效像元: {valid_pixels:,} / {total_pixels:,} ({valid_ratio:.2f}%)")
            logger.info(f"  - 使用像素位置直接匹配，无需地理坐标转换")
            
            ds = None
            return True
            
        except Exception as e:
            logger.error(f"创建掩膜时出错: {e}")
            return False
    
    def apply_pixel_mask(self, target_width, target_height):
        """
        直接使用像素位置应用掩膜（无需地理坐标转换）
        
        Args:
            target_width, target_height: 目标图像尺寸
            
        Returns:
            掩膜数组
        """
        # 检查图像尺寸是否一致
        if target_width != self.mask_width or target_height != self.mask_height:
            logger.error(f"图像尺寸不匹配！")
            logger.error(f"  掩膜尺寸: {self.mask_width} x {self.mask_height}")
            logger.error(f"  目标尺寸: {target_width} x {target_height}")
            # 如果尺寸不匹配，返回全零掩膜
            return np.zeros((target_height, target_width), dtype=np.uint8)
        
        # 尺寸一致，直接返回掩膜
        return self.mask_data.copy()
    
    def apply_mask_to_file(self, input_file, output_file):
        """
        对单个文件应用掩膜
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 打开输入文件
            src_ds = gdal.Open(input_file, gdalconst.GA_ReadOnly)
            if src_ds is None:
                logger.error(f"无法打开文件: {input_file}")
                return False
            
            # 获取文件信息
            src_width = src_ds.RasterXSize
            src_height = src_ds.RasterYSize
            src_bands = src_ds.RasterCount
            src_geotransform = src_ds.GetGeoTransform()
            src_projection = src_ds.GetProjection()
            src_datatype = src_ds.GetRasterBand(1).DataType
            
            # 直接使用像素位置应用掩膜
            pixel_mask = self.apply_pixel_mask(src_width, src_height)
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 创建输出文件
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                output_file, src_width, src_height, src_bands, src_datatype,
                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
            )
            
            if dst_ds is None:
                logger.error(f"无法创建输出文件: {output_file}")
                src_ds = None
                return False
            
            # 设置地理信息
            dst_ds.SetGeoTransform(src_geotransform)
            dst_ds.SetProjection(src_projection)
            
            # 处理每个波段
            for band_idx in range(1, src_bands + 1):
                src_band = src_ds.GetRasterBand(band_idx)
                dst_band = dst_ds.GetRasterBand(band_idx)
                
                # 读取数据
                data = src_band.ReadAsArray()
                nodata_value = src_band.GetNoDataValue()
                
                # 根据数据类型选择合适的NoData值
                if nodata_value is None:
                    if src_datatype == gdal.GDT_Byte:
                        # 对于uint8类型的数据（如FIRMS），使用255作为NoData
                        nodata_value = 255
                    else:
                        # 对于其他数据类型，使用-9999
                        nodata_value = -9999
                
                # 确保NoData值在数据类型范围内
                if src_datatype == gdal.GDT_Byte:
                    # 对于uint8，确保NoData值在0-255范围内
                    if nodata_value < 0 or nodata_value > 255:
                        nodata_value = 255
                        logger.warning(f"FIRMS数据使用255作为NoData值: {os.path.basename(input_file)}")
                
                # 掩膜为0的地方设为nodata
                masked_data = np.where(pixel_mask == 1, data, nodata_value)
                
                # 写入数据
                dst_band.WriteArray(masked_data)
                dst_band.SetNoDataValue(nodata_value)
                
                # 安全地计算统计信息
                try:
                    dst_band.ComputeStatistics(False)
                except Exception:
                    pass
            
            # 清理
            src_ds = None
            dst_ds = None
            
            return True
            
        except Exception as e:
            logger.error(f"处理文件时出错 {input_file}: {e}")
            return False
    
    def process_folder(self, folder_path):
        """
        处理单个文件夹
        
        Args:
            folder_path: 文件夹路径
        """
        folder_name = os.path.basename(folder_path)
        
        # 跳过ERA5文件夹（用于创建掩膜）
        # if folder_name == 'ERA5_multi_bands':
        #    logger.info(f"跳过ERA5文件夹: {folder_name}")
        #    return
        
        logger.info(f"开始处理文件夹: {folder_name}")
        
        # 获取所有tif文件
        tif_files = glob.glob(os.path.join(folder_path, '*.tif'))
        
        if not tif_files:
            logger.warning(f"文件夹 {folder_name} 中没有找到tif文件")
            return
        
        # 创建输出文件夹
        relative_path = os.path.relpath(folder_path, self.downsampled_root)
        output_folder = os.path.join(self.output_root, relative_path)
        os.makedirs(output_folder, exist_ok=True)
        
        # 准备任务
        tasks = []
        for tif_file in tif_files:
            filename = os.path.basename(tif_file)
            output_file = os.path.join(output_folder, filename)
            
            # 跳过已存在的文件
            if os.path.exists(output_file):
                continue
            
            tasks.append((tif_file, output_file))
        
        if not tasks:
            logger.info(f"文件夹 {folder_name} 中所有文件已存在，跳过处理")
            return
        
        # 多线程处理
        logger.info(f"使用 {min(self.max_workers, len(tasks))} 个线程处理 {len(tasks)} 个文件")
        start_time = time.time()
        
        def worker(task):
            input_file, output_file = task
            return self.apply_mask_to_file(input_file, output_file)
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tasks))) as executor:
            results = list(tqdm(
                executor.map(worker, tasks),
                total=len(tasks),
                desc=f"掩膜 {folder_name}",
                ncols=120
            ))
        
        success_count = sum(results)
        elapsed_time = time.time() - start_time
        
        logger.info(f"文件夹 {folder_name} 处理完成: {success_count}/{len(tasks)} 个文件成功，"
                   f"耗时 {elapsed_time:.1f} 秒")
    
    def process_all(self):
        """
        处理所有文件夹
        """
        logger.info(f"开始处理根目录: {self.downsampled_root}")
        logger.info(f"输出目录: {self.output_root}")
        
        # 首先创建掩膜
        if not self.create_mask_from_era5():
            logger.error("创建掩膜失败，停止处理")
            return
        
        # 创建输出根目录
        os.makedirs(self.output_root, exist_ok=True)
        
        # 获取所有子文件夹
        subfolders = [
            os.path.join(self.downsampled_root, d) 
            for d in os.listdir(self.downsampled_root) 
            if os.path.isdir(os.path.join(self.downsampled_root, d))
        ]
        
        if not subfolders:
            logger.error("没有找到子文件夹")
            return
        
        logger.info(f"找到 {len(subfolders)} 个子文件夹")
        
        # 处理每个文件夹
        total_start_time = time.time()
        for folder in subfolders:
            self.process_folder(folder)
        
        total_elapsed = time.time() - total_start_time
        logger.info(f"所有文件夹处理完成，总耗时: {total_elapsed:.1f} 秒")

def main():
    """主函数"""
    # 配置路径
    downsampled_root = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_10x_without_qa'
    output_root = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_10x_without_qa_masked'
    
    # 创建掩膜应用器
    mask_applier = MaskApplier(
        downsampled_root=downsampled_root,
        output_root=output_root,
        max_workers=4
    )
    
    # 开始处理
    mask_applier.process_all()

if __name__ == "__main__":
    main() 