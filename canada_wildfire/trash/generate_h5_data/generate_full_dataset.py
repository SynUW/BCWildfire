#!/usr/bin/env python3
"""
完整数据集生成器 - 年份范围版本
只考虑指定年份的时间范围，不考虑过去和未来天数
最新的h5数据生成代码, 2025年7月21日
"""

import os
import sys
import h5py
import numpy as np
import glob
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
from functools import lru_cache
import gc
from multiprocessing import Pool, cpu_count, Manager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pickle
import time
import ast

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YearOnlyDatasetGenerator:
    def __init__(self, data_dir, output_dir, target_years=[2024], 
                 max_workers=12, batch_size=10000):
        """
        初始化年份范围数据集生成器
        
        Args:
            data_dir: 数据根目录
            output_dir: 输出目录
            target_years: 目标年份列表
            max_workers: 最大并行工作进程数（现在主要用于数据加载）
            batch_size: 像素批处理大小
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.target_years = target_years if isinstance(target_years, list) else [target_years]
        self.max_workers = min(max_workers, cpu_count())
        self.batch_size = batch_size
        
        # 固定的驱动因素顺序（与merge_pixel_samples.py一致）
        # self.driver_order = [
        #     'Firms_Detection_resampled_10x',  # 1
        #     'ERA5_multi_bands_10x',  # 2-13
        #     'LULC_BCBoundingbox_resampled_10x',  # 14
        #     'DEM_and_distance_map_interpolated_10x',  # 15-20
        #     'NDVI_EVI_with_qa_applied_10x', # 21-22
        #     'Reflection_500_merge_TerraAquaWGS84_clip_qa_masked_cloud_final_10x', # 23-26
        #     'MODIS_Terra_Aqua_B20_21_merged_resampled_qa_masked_cloud_nodata-unified_final_10x', # 27-30
        #     'MOD21A1DN_multibands_filtered_resampled_10x', # 31-38
        #     'LAI_BCBoundingbox_resampled_right_interpolation_clip_10x' # 39
        # ]
        
        # self.driver_order = [
        #     'Firms_Detection_resampled',  # 1  1234
        #     'ERA5_multi_bands',  # 2-13 1234
        #     'LULC_BCBoundingbox_resampled',  # 14 1234
        #     'DEM_and_distance_map_interpolated',  # 15-20 1234
        #     'with_qa_applied/NDVI_EVI_with_qa_applied', # 21-22 1234
        #     'with_qa_applied/Reflection_500_merge_TerraAquaWGS84_clip_qa_masked_cloud_final', # 23-26  1234
        #     'with_qa_applied/MODIS_Terra_Aqua_B20_21_merged_resampled_qa_masked_cloud_nodata_unified_final', # 27-30  1234
        #     'MOD21A1DN_multibands_filtered_resampled', # 31-38 1234
        #     'LAI_BCBoundingbox_resampled_right_interpolation_clip' # 39  1234
        # ]
        
        # self.driver_order = [
        #     'Firms_Detection_resampled_10x',  # 1
        #     'ERA5_multi_bands_10x',  # 2-13
        #     'LULC_BCBoundingbox_resampled_10x',  # 14
        #     'DEM_and_distance_map_interpolated_10x',  # 15-20
        #     'NDVI_EVI_10x', # 21-22
        #     'Reflection_500_merge_TerraAquaWGS84_clip_scaled_10x', # 23-26
        #     'MODIS_Terra_Aqua_B20_21_merged_resampled_10x', # 27-30
        #     # 'MOD21A1DN_multibands_filtered_resampled_10x', # 31-38
        #     'MOD21A1DN_multibands_withoutFiltering_merged_10x', # 31-38 without filtering
        #     'LAI_BCBoundingbox_resampled_right_interpolation_clip_10x' # 39
        # ]
        
        self.driver_order = [
            'Firms_Detection_resampled',  # 1
            'ERA5_multi_bands',  # 2-13
            'LULC_BCBoundingbox_resampled',  # 14
            'DEM_and_distance_map_interpolated',  # 15-20
            'before_final_products_and_previous_ill-processed_data/NDVI_EVI', # 21-22
            'Reflection_500_merge_TerraAquaWGS84_clip_scaled', # 23-26
            'MODIS_Terra_Aqua_B20_21_merged_resampled', # 27-30
            'MOD21A1DN_multibands_filtered_resampled', # 31-38 without filtering
            'LAI_BCBoundingbox_resampled_right_interpolation_clip' # 39
        ]
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取驱动因素目录
        self.driver_dirs = self._get_driver_directories()
        
        # 计算总通道数和通道映射
        self.total_channels, self.channel_mapping = self._calculate_channels_and_mapping()
        
        logger.info(f"数据目录: {data_dir}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"目标年份: {self.target_years}")
        logger.info(f"找到 {len(self.driver_dirs)} 个驱动因素")
        logger.info(f"总通道数: {self.total_channels}")
        logger.info(f"并行工作进程: {self.max_workers}")
        logger.info(f"批处理大小: {batch_size}")
    
    def _get_driver_directories(self):
        """获取所有驱动因素目录（支持多级目录）"""
        driver_dirs = {}
        
        # 使用os.walk递归扫描所有子目录
        for root, dirs, files in os.walk(self.data_dir):
            # 检查当前目录是否包含tif文件
            tif_files = [f for f in files if f.endswith('.tif')]
            if tif_files:
                # 计算相对于data_dir的路径
                rel_path = os.path.relpath(root, self.data_dir)
                if rel_path == '.':
                    # 如果是根目录，跳过
                    continue
                
                # 将路径分隔符统一为'/'（跨平台兼容）
                rel_path = rel_path.replace(os.sep, '/')
                driver_dirs[rel_path] = root
        
        logger.info(f"找到驱动因素目录:")
        for name, path in driver_dirs.items():
            logger.info(f"  {name}: {path}")
        
        return driver_dirs
    
    def _calculate_channels_and_mapping(self):
        """计算总通道数并创建通道映射"""
        total_channels = 0
        channel_mapping = {}
        
        # 按照固定顺序处理驱动因素
        for driver_name in self.driver_order:
            if driver_name in self.driver_dirs:
                driver_dir = self.driver_dirs[driver_name]
                # 获取一个样本文件
                tif_files = glob.glob(os.path.join(driver_dir, '*.tif'))
                if tif_files:
                    sample_file = tif_files[0]
                    try:
                        data, _ = self._load_single_file_with_gdal(sample_file)
                        if data is not None:
                            if len(data.shape) == 3:  # 多波段
                                channels = data.shape[0]
                            else:  # 单波段
                                channels = 1
                            
                            # 记录通道映射
                            channel_mapping[driver_name] = {
                                'start_idx': total_channels,
                                'channels': channels,
                                'end_idx': total_channels + channels
                            }
                            
                            total_channels += channels
                            logger.info(f"  {driver_name}: {channels} 个通道 (索引 {channel_mapping[driver_name]['start_idx']}-{channel_mapping[driver_name]['end_idx']-1})")
                    except Exception as e:
                        logger.warning(f"检查驱动因素 {driver_name} 的通道数失败: {e}")
        
        return total_channels, channel_mapping
    
    @lru_cache(maxsize=1000)
    def _get_date_from_filename(self, filename):
        """从文件名提取日期"""
        try:
            # 查找日期模式 YYYY_MM_DD
            import re
            date_pattern = r'(\d{4})_(\d{2})_(\d{2})'
            match = re.search(date_pattern, filename)
            if match:
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day))
        except Exception as e:
            logger.debug(f"从文件名 {filename} 提取日期失败: {e}")
        return None
    
    def _load_single_file_with_gdal(self, file_path):
        """使用GDAL加载单个文件"""
        try:
            from osgeo import gdal
            
            dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
            if dataset is None:
                return None, None
            
            bands = dataset.RasterCount
            if bands == 1:
                band = dataset.GetRasterBand(1)
                data = band.ReadAsArray()
            else:
                data = []
                for i in range(1, bands + 1):
                    band = dataset.GetRasterBand(i)
                    band_data = band.ReadAsArray()
                    data.append(band_data)
                data = np.array(data)
            
            geo_info = {
                'geotransform': dataset.GetGeoTransform(),
                'projection': dataset.GetProjection(),
                'raster_size': (dataset.RasterXSize, dataset.RasterYSize),
                'raster_count': bands
            }
            
            dataset = None
            return data, geo_info
            
        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {e}")
            return None, None
    
    def _get_all_required_dates(self, target_year):
        """获取指定年份的所有日期"""
        # 年份的起始和结束日期
        year_start = datetime(target_year, 1, 1)
        year_end = datetime(target_year, 12, 31)
        
        # 生成该年份的所有日期
        all_dates = []
        current_date = year_start
        while current_date <= year_end:
            all_dates.append(current_date)
            current_date += timedelta(days=1)
        
        # 计算总天数（考虑闰年）
        total_days = len(all_dates)
        
        logger.info(f"年份 {target_year} 日期范围: {year_start.strftime('%Y-%m-%d')} 到 {year_end.strftime('%Y-%m-%d')}")
        logger.info(f"总天数: {total_days}")
        
        return all_dates
    
    def _build_global_date_mapping(self):
        """构建全局日期到文件的映射"""
        logger.info("构建全局日期到文件映射...")
        
        all_date_to_file = {}
        
        with tqdm(total=len(self.driver_dirs), desc="扫描驱动因素") as pbar:
            for driver_name, driver_dir in self.driver_dirs.items():
                pbar.set_description(f"扫描: {driver_name}")
                
                tif_files = glob.glob(os.path.join(driver_dir, '*.tif'))
                
                for file_path in tif_files:
                    filename = os.path.basename(file_path)
                    date = self._get_date_from_filename(filename)
                    
                    if date:
                        if date not in all_date_to_file:
                            all_date_to_file[date] = {}
                        all_date_to_file[date][driver_name] = file_path
                
                pbar.update(1)
        
        logger.info(f"映射了 {len(all_date_to_file)} 个日期的文件")
        return all_date_to_file
    
    def _get_valid_pixels(self, all_date_to_file, target_year):
        """获取所有有效像素位置"""
        logger.info(f"获取年份 {target_year} 的有效像素位置...")
        
        # 使用目标年份中间的某一天的FIRMS数据来确定有效像素
        target_date = datetime(target_year, 6, 15)  # 年中的日期
        
        firms_data = None
        era5_data = None
        geo_reference = None
        
        logger.info(f"检查FIRMS数据可用性...")
        logger.info(f"可用的驱动因素: {list(self.driver_dirs.keys())}")
        
        # 自动查找FIRMS相关的驱动因素
        firms_driver = None
        for driver_name in self.driver_order:
            if 'Firms' in driver_name or 'firms' in driver_name:
                firms_driver = driver_name
                break
        
        if firms_driver and firms_driver in self.driver_dirs:
            logger.info(f"找到FIRMS驱动因素: {firms_driver}")
            # 查找最接近的日期
            closest_date = None
            min_diff = float('inf')
            
            for date in all_date_to_file.keys():
                if date.year == target_year:
                    diff = abs((date - target_date).days)
                    if diff < min_diff:
                        min_diff = diff
                        closest_date = date
            
            if closest_date and firms_driver in all_date_to_file[closest_date]:
                firms_file = all_date_to_file[closest_date][firms_driver]
                firms_data, geo_reference = self._load_single_file_with_gdal(firms_file)
                logger.info(f"使用 {closest_date.strftime('%Y-%m-%d')} 的{firms_driver}数据确定有效像素")
        
        # 同时加载ERA5数据用于额外限制
        era5_driver = None
        for driver_name in self.driver_order:
            if 'ERA5' in driver_name or 'era5' in driver_name:
                era5_driver = driver_name
                break
        
        if era5_driver and era5_driver in self.driver_dirs:
            if closest_date and era5_driver in all_date_to_file[closest_date]:
                era5_file = all_date_to_file[closest_date][era5_driver]
                era5_data, _ = self._load_single_file_with_gdal(era5_file)
                logger.info(f"使用 {closest_date.strftime('%Y-%m-%d')} 的{era5_driver}数据确定有效像素")
        
        if firms_data is None:
            logger.error("无法加载FIRMS数据来确定有效像素")
            return [], None
        
        # 如果是多波段，取第一个波段
        if len(firms_data.shape) == 3:
            firms_data = firms_data[0]
        
        # 找到所有非NaN且不等于NoData值的像素
        # 对于FIRMS数据，0表示无火灾，1-100表示有火灾，255是背景值，-9999是NoData值
        valid_mask = (~np.isnan(firms_data)) & (firms_data != 255) & (firms_data != -9999)
        
        # 如果ERA5数据可用，添加ERA5数值大于200的限制
        if era5_data is not None:
            if len(era5_data.shape) == 3:
                era5_data = era5_data[0]  # 取第一个波段
            era5_valid_mask = (era5_data > 0) & (~np.isnan(era5_data))
            valid_mask = valid_mask & era5_valid_mask
            logger.info(f"应用ERA5数值>200的限制条件")
        
        valid_pixels = [(int(row), int(col)) for row, col in zip(*np.where(valid_mask))]
        
        logger.info(f"找到 {len(valid_pixels)} 个有效像素")
        
        return valid_pixels, geo_reference
    
    def _load_date_data_worker(self, args):
        """并行加载单个日期的数据"""
        date, date_files = args
        
        date_data = {}
        for driver_name in self.driver_order:
            if driver_name in date_files:
                file_path = date_files[driver_name]
                data, _ = self._load_single_file_with_gdal(file_path)
                if data is not None:
                    date_data[driver_name] = data
        
        return date, date_data
    
    def _preload_all_data_parallel(self, all_dates, all_date_to_file):
        """并行预加载该年份的所有数据"""
        logger.info("并行预加载年份数据...")
        
        # 准备参数
        args_list = []
        for date in all_dates:
            if date in all_date_to_file:
                args_list.append((date, all_date_to_file[date]))
        
        preloaded_data = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._load_date_data_worker, args) for args in args_list]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="预加载数据"):
                try:
                    date, date_data = future.result()
                    if date_data:
                        preloaded_data[date] = date_data
                except Exception as e:
                    logger.error(f"预加载数据失败: {e}")
        
        logger.info(f"成功预加载 {len(preloaded_data)}/{len(all_dates)} 个日期的数据")
        return preloaded_data
    
    def _extract_pixel_time_series_from_preloaded(self, pixel_coord, all_dates, preloaded_data):
        """从预加载数据中提取像素时序"""
        row, col = pixel_coord
        pixel_time_series = np.full((self.total_channels, len(all_dates)), np.nan, dtype=np.float32)
        
        for time_idx, date in enumerate(all_dates):
            if date in preloaded_data:
                date_data = preloaded_data[date]
                
                channel_offset = 0
                for driver_name in self.driver_order:
                    if driver_name in date_data:
                        driver_data = date_data[driver_name]
                        
                        # 处理多波段或单波段数据
                        if len(driver_data.shape) == 3:  # 多波段
                            channels = driver_data.shape[0]
                            for c in range(channels):
                                if row < driver_data.shape[1] and col < driver_data.shape[2]:
                                    pixel_time_series[channel_offset + c, time_idx] = driver_data[c, row, col]
                        else:  # 单波段
                            if row < driver_data.shape[0] and col < driver_data.shape[1]:
                                pixel_time_series[channel_offset, time_idx] = driver_data[row, col]
                            channels = 1
                        
                        channel_offset += channels
        
        return pixel_time_series
    
    def _reorganize_preloaded_data(self, preloaded_data, all_dates):
        """将预加载数据重组为更高效的格式"""
        logger.info("重组预加载数据为向量化格式...")
        
        # 获取数据维度
        sample_date = next(iter(preloaded_data.keys()))
        sample_data = preloaded_data[sample_date]
        
        # 确定数据的空间维度
        height, width = None, None
        for driver_name in self.driver_order:
            if driver_name in sample_data:
                driver_data = sample_data[driver_name]
                if len(driver_data.shape) == 3:  # 多波段
                    height, width = driver_data.shape[1], driver_data.shape[2]
                else:  # 单波段
                    height, width = driver_data.shape[0], driver_data.shape[1]
                break
        
        if height is None or width is None:
            raise ValueError("无法确定数据的空间维度")
        
        # 创建统一的数据数组 [channels, time, height, width]
        organized_data = np.full((self.total_channels, len(all_dates), height, width), 
                               np.nan, dtype=np.float32)
        
        # 填充数据
        for time_idx, date in enumerate(tqdm(all_dates, desc="重组数据")):
            if date in preloaded_data:
                date_data = preloaded_data[date]
                
                channel_offset = 0
                for driver_name in self.driver_order:
                    if driver_name in date_data:
                        driver_data = date_data[driver_name]
                        
                        if len(driver_data.shape) == 3:  # 多波段
                            channels = driver_data.shape[0]
                            organized_data[channel_offset:channel_offset+channels, time_idx, :, :] = driver_data
                        else:  # 单波段
                            organized_data[channel_offset, time_idx, :, :] = driver_data
                            channels = 1
                        
                        channel_offset += channels
        
        logger.info(f"数据重组完成，形状: {organized_data.shape}")
        return organized_data
    
    def _extract_pixels_vectorized(self, organized_data, valid_pixels):
        """使用向量化操作提取所有像素的时间序列"""
        logger.info("使用向量化操作提取像素时间序列...")
        
        # 提取所有有效像素的坐标
        rows = np.array([pixel[0] for pixel in valid_pixels])
        cols = np.array([pixel[1] for pixel in valid_pixels])
        
        # 使用高级索引一次性提取所有像素的数据
        # organized_data: [channels, time, height, width]
        # 结果: [channels, time, num_pixels]
        pixel_data = organized_data[:, :, rows, cols]
        
        # 重新排列为 [num_pixels, channels, time]
        pixel_data = pixel_data.transpose(2, 0, 1)
        
        logger.info(f"向量化提取完成，形状: {pixel_data.shape}")
        return pixel_data
    
    def _process_pixels_batch_vectorized(self, valid_pixels, pixel_data, batch_size=10000):
        """批量处理像素数据，返回有效的像素结果"""
        logger.info("批量处理像素数据...")
        
        pixel_results = []
        num_pixels = len(valid_pixels)
        
        for i in tqdm(range(0, num_pixels, batch_size), desc="处理像素批次"):
            end_idx = min(i + batch_size, num_pixels)
            batch_pixels = valid_pixels[i:end_idx]
            batch_data = pixel_data[i:end_idx]
            
            # 检查每个像素的数据有效性
            for j, (pixel_coord, pixel_time_series) in enumerate(zip(batch_pixels, batch_data)):
                # 检查是否有足够的非NaN数据
                non_nan_ratio = np.mean(~np.isnan(pixel_time_series))
                if non_nan_ratio > 0.1:  # 至少10%的数据非NaN
                    pixel_results.append((pixel_coord, pixel_time_series))
        
        logger.info(f"批量处理完成，有效像素: {len(pixel_results)}/{num_pixels}")
        return pixel_results
    
    def _process_pixels_parallel_optimized(self, valid_pixels, all_dates, preloaded_data):
        """优化的并行像素处理方法"""
        logger.info("开始优化的像素处理...")
        
        # 步骤1: 重组预加载数据为向量化格式
        organized_data = self._reorganize_preloaded_data(preloaded_data, all_dates)
        
        # 步骤2: 使用向量化操作提取所有像素
        pixel_data = self._extract_pixels_vectorized(organized_data, valid_pixels)
        
        # 步骤3: 批量处理像素数据
        pixel_results = self._process_pixels_batch_vectorized(valid_pixels, pixel_data)
        
        # 清理内存
        del organized_data, pixel_data
        gc.collect()
        
        return pixel_results

    def _process_pixels_parallel(self, valid_pixels, all_dates, preloaded_data):
        """并行处理所有像素 - 使用优化版本"""
        return self._process_pixels_parallel_optimized(valid_pixels, all_dates, preloaded_data)
    
    def generate_dataset_for_year(self, target_year):
        """为指定年份生成完整数据集"""
        logger.info(f"开始生成年份 {target_year} 的完整数据集...")
        
        # 步骤1: 构建全局日期映射（一次性，所有年份共用）
        if not hasattr(self, '_global_date_mapping'):
            self._global_date_mapping = self._build_global_date_mapping()
        all_date_to_file = self._global_date_mapping
        
        # 步骤2: 获取该年份需要的日期
        all_dates = self._get_all_required_dates(target_year)
        
        # 步骤3: 获取有效像素位置
        valid_pixels, geo_reference = self._get_valid_pixels(all_date_to_file, target_year)
        
        if not valid_pixels:
            logger.error(f"年份 {target_year} 没有找到有效像素")
            return
        
        # 步骤4: 并行预加载该年份的数据
        preloaded_data = self._preload_all_data_parallel(all_dates, all_date_to_file)
        
        # 步骤5: 创建输出文件
        output_file = os.path.join(self.output_dir, f'{target_year}_year_dataset.h5')
        
        if os.path.exists(output_file):
            logger.info(f"输出文件已存在，跳过: {output_file}")
            return
        
        logger.info(f"创建输出文件: {output_file}")
        
        # 步骤6: 并行处理像素
        pixel_results = self._process_pixels_parallel(valid_pixels, all_dates, preloaded_data)
        
        # 步骤7: 保存结果到HDF5文件
        with h5py.File(output_file, 'w') as h5_file:
            # 写入全局属性
            h5_file.attrs['year'] = str(target_year)
            h5_file.attrs['total_time_steps'] = len(all_dates)
            h5_file.attrs['total_channels'] = self.total_channels
            h5_file.attrs['data_format'] = 'channels_x_time'
            h5_file.attrs['data_type'] = 'year_pixel_time_series'
            h5_file.attrs['driver_names'] = self.driver_order
            h5_file.attrs['total_pixels'] = len(valid_pixels)
            h5_file.attrs['processed_pixels'] = len(pixel_results)
            h5_file.attrs['start_date'] = all_dates[0].strftime('%Y-%m-%d')
            h5_file.attrs['end_date'] = all_dates[-1].strftime('%Y-%m-%d')
            
            # 保存通道映射信息
            for driver_name, channel_info in self.channel_mapping.items():
                h5_file.attrs[f'channel_mapping_{driver_name}'] = f"{channel_info['start_idx']}-{channel_info['end_idx']-1}"
            
            # 保存地理参考信息
            if geo_reference:
                h5_file.attrs['geotransform'] = geo_reference['geotransform']
                h5_file.attrs['projection'] = geo_reference['projection']
                h5_file.attrs['raster_size'] = geo_reference['raster_size']
                h5_file.attrs['raster_count'] = geo_reference['raster_count']
            
            # 保存所有像素数据
            for pixel_coord, pixel_time_series in tqdm(pixel_results, desc="保存数据"):
                try:
                    dataset_name = f"{pixel_coord[0]}_{pixel_coord[1]}"
                    dataset = h5_file.create_dataset(
                        dataset_name,
                        data=pixel_time_series,
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=6,
                        shuffle=True
                    )
                    
                    # 添加数据集属性
                    dataset.attrs['pixel_coord'] = pixel_coord
                    dataset.attrs['data_shape'] = f"{pixel_time_series.shape[0]}x{pixel_time_series.shape[1]}"
                    
                except Exception as e:
                    logger.error(f"保存像素 {pixel_coord} 数据时出错: {e}")
                    continue
        
        logger.info(f"年份 {target_year} 完整数据集生成完成: {output_file}")
        logger.info(f"成功处理 {len(pixel_results)}/{len(valid_pixels)} 个像素")
        
        # 输出数据集信息
        self._print_dataset_info(output_file)
        
        # 清理内存
        del preloaded_data
        gc.collect()
    
    def generate_all_datasets(self):
        """生成所有年份的完整数据集"""
        logger.info(f"开始生成 {len(self.target_years)} 个年份的完整数据集...")
        
        for year_idx, target_year in enumerate(self.target_years):
            logger.info(f"\n{'='*60}")
            logger.info(f"处理年份 {target_year} ({year_idx + 1}/{len(self.target_years)})")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            self.generate_dataset_for_year(target_year)
            end_time = time.time()
            
            logger.info(f"年份 {target_year} 完成，耗时: {end_time - start_time:.2f} 秒")
        
        logger.info(f"\n所有年份数据集生成完成！")
    
    def _print_dataset_info(self, output_file):
        """打印数据集信息"""
        try:
            with h5py.File(output_file, 'r') as f:
                print(f"\n数据集信息:")
                print(f"="*50)
                print(f"文件: {output_file}")
                print(f"年份: {f.attrs.get('year', 'N/A')}")
                print(f"日期范围: {f.attrs.get('start_date', 'N/A')} ~ {f.attrs.get('end_date', 'N/A')}")
                print(f"数据集数量: {len(f.keys())}")
                print(f"总通道数: {f.attrs.get('total_channels', 'N/A')}")
                print(f"总时间步数: {f.attrs.get('total_time_steps', 'N/A')}")
                print(f"数据格式: {f.attrs.get('data_format', 'N/A')}")
                
                # 显示通道映射
                print(f"\n通道映射:")
                for key in f.attrs.keys():
                    if key.startswith('channel_mapping_'):
                        driver_name = key.replace('channel_mapping_', '')
                        mapping = f.attrs[key]
                        print(f"  {driver_name}: {mapping}")
                
                # 检查第一个数据集的形状
                if len(f.keys()) > 0:
                    first_key = list(f.keys())[0]
                    first_dataset = f[first_key]
                    print(f"\n样本数据集 ({first_key}):")
                    print(f"  形状: {first_dataset.shape}")
                    print(f"  数据类型: {first_dataset.dtype}")
                    print(f"  数据范围: {np.nanmin(first_dataset[:])} ~ {np.nanmax(first_dataset[:])}") 
                
                print(f"="*50)
        except Exception as e:
            logger.error(f"打印数据集信息失败: {e}")


def parse_year_list(year_string):
    """解析年份列表字符串"""
    try:
        # 尝试解析为Python列表格式
        return ast.literal_eval(year_string)
    except:
        try:
            # 尝试解析为逗号分隔的数字
            return [int(x.strip()) for x in year_string.split(',')]
        except:
            # 单个年份
            return [int(year_string)]


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成年份范围数据集 - 并行优化版本')
    parser.add_argument('--data_dir', 
                       # default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked',  # 10 time downsampled result
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked',
                       help='数据根目录')
    parser.add_argument('--output_dir',
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/all_data_masked_withoutdownsampling',
                       help='输出目录')
    parser.add_argument('--years', type=str, default='[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]',
                       help='目标年份列表，格式：[2000,2021,2022,2023,2024] 或 2000,2021,2022,2023,2024')
    parser.add_argument('--max_workers', type=int, default=24,
                       help='最大并行工作进程数（主要用于数据加载）')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='像素批处理大小')
    
    args = parser.parse_args()
    
    # 解析年份列表
    target_years = parse_year_list(args.years)
    
    # 创建生成器
    generator = YearOnlyDatasetGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_years=target_years,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
    
    # 生成所有年份的数据集
    generator.generate_all_datasets()


if __name__ == "__main__":
    main() 