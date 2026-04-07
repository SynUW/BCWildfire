#!/usr/bin/env python3
"""
完整数据集生成器 - 优化版本
支持并行数据加载、并行像素处理和多年份生成
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

class OptimizedFullDatasetGenerator:
    def __init__(self, data_dir, output_dir, target_years=[2024], 
                 past_days=365, future_days=30, current_year_days=365,
                 max_workers=12, batch_size=100):
        """
        初始化优化的完整数据集生成器
        
        Args:
            data_dir: 数据根目录
            output_dir: 输出目录
            target_years: 目标年份列表
            past_days: 过去天数
            future_days: 未来天数
            current_year_days: 当前年份天数
            max_workers: 最大并行工作进程数
            batch_size: 批处理大小
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.target_years = target_years if isinstance(target_years, list) else [target_years]
        self.past_days = past_days
        self.future_days = future_days
        self.current_year_days = current_year_days
        self.total_time_steps = past_days + current_year_days + future_days
        self.max_workers = min(max_workers, cpu_count())
        self.batch_size = batch_size
        
        # 固定的驱动因素顺序（与merge_pixel_samples.py一致）
        self.driver_order = [
            'Firms_Detection_resampled',
            'ERA5_multi_bands',
            'LULC_BCBoundingbox_resampled',
            'Topo_Distance_WGS84_resize_resampled',
            'NDVI_EVI',
            'Reflection_500_merge_TerraAquaWGS84_clip',
            'MODIS_Terra_Aqua_B20_21_merged_resampled',
            'MOD21A1DN_multibands_filtered_resampled',
            'LAI_BCBoundingbox_resampled'
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
        logger.info(f"时间范围: 过去{past_days}天 + 当前{current_year_days}天 + 未来{future_days}天 = {self.total_time_steps}天")
        logger.info(f"找到 {len(self.driver_dirs)} 个驱动因素")
        logger.info(f"总通道数: {self.total_channels}")
        logger.info(f"并行工作进程: {self.max_workers}")
        logger.info(f"批处理大小: {batch_size}")
    
    def _get_driver_directories(self):
        """获取所有驱动因素目录"""
        driver_dirs = {}
        
        # 扫描数据目录下的所有子目录
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                # 检查是否包含tif文件
                tif_files = glob.glob(os.path.join(item_path, '*.tif'))
                if tif_files:
                    driver_dirs[item] = item_path
        
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
        """获取指定年份的所有需要的日期"""
        # 目标年份的起始和结束日期
        year_start = datetime(target_year, 1, 1)
        year_end = datetime(target_year, 12, 31)
        
        # 计算总的日期范围
        start_date = year_start - timedelta(days=self.past_days)
        end_date = year_end + timedelta(days=self.future_days)
        
        # 生成日期列表
        all_dates = []
        current_date = start_date
        while current_date <= end_date:
            all_dates.append(current_date)
            current_date += timedelta(days=1)
        
        logger.info(f"年份 {target_year} 日期范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"总天数: {len(all_dates)}")
        
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
        geo_reference = None
        
        if 'Firms_Detection_resampled' in self.driver_dirs:
            # 查找最接近的日期
            closest_date = None
            min_diff = float('inf')
            
            for date in all_date_to_file.keys():
                if abs((date - target_date).days) < min_diff:
                    min_diff = abs((date - target_date).days)
                    closest_date = date
            
            if closest_date and 'Firms_Detection_resampled' in all_date_to_file[closest_date]:
                firms_file = all_date_to_file[closest_date]['Firms_Detection_resampled']
                firms_data, geo_reference = self._load_single_file_with_gdal(firms_file)
                
                if firms_data is not None:
                    if len(firms_data.shape) == 3:
                        firms_data = firms_data[0]  # 取第一个波段
        
        if firms_data is None:
            logger.error("无法获取FIRMS数据来确定有效像素")
            return [], None
        
        # 定义有效值（排除NoData）
        nodata_values = [255]
        valid_pixels = []
        
        height, width = firms_data.shape
        for row in range(height):
            for col in range(width):
                firms_value = firms_data[row, col]
                if firms_value not in nodata_values and not np.isnan(firms_value):
                    valid_pixels.append((row, col))
        
        logger.info(f"找到 {len(valid_pixels)} 个有效像素")
        return valid_pixels, geo_reference
    
    def _load_date_data_worker(self, args):
        """并行数据加载的工作函数"""
        date, all_date_to_file, driver_order = args
        
        date_data = {}
        
        if date in all_date_to_file:
            for driver_name in driver_order:
                if driver_name in all_date_to_file[date]:
                    file_path = all_date_to_file[date][driver_name]
                    data, _ = self._load_single_file_with_gdal(file_path)
                    if data is not None:
                        date_data[driver_name] = data
        
        return date, date_data
    
    def _preload_all_data_parallel(self, all_dates, all_date_to_file):
        """并行预加载所有需要的数据到内存"""
        logger.info(f"并行预加载所有数据到内存（使用 {self.max_workers} 个进程）...")
        
        preloaded_data = {}
        
        # 准备并行任务参数
        tasks = [(date, all_date_to_file, self.driver_order) for date in all_dates]
        
        # 使用进程池并行加载数据
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_date = {executor.submit(self._load_date_data_worker, task): task[0] 
                             for task in tasks}
            
            # 收集结果
            with tqdm(total=len(all_dates), desc="并行加载数据") as pbar:
                for future in as_completed(future_to_date):
                    date = future_to_date[future]
                    try:
                        loaded_date, date_data = future.result()
                        preloaded_data[loaded_date] = date_data
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"加载日期 {date} 数据失败: {e}")
                        preloaded_data[date] = {}
                        pbar.update(1)
        
        logger.info(f"并行预加载完成，总共 {len(preloaded_data)} 个日期的数据")
        return preloaded_data
    
    def _extract_pixel_time_series_from_preloaded(self, pixel_coord, all_dates, preloaded_data):
        """从预加载数据中提取单个像素的完整时间序列"""
        row, col = pixel_coord
        
        # 初始化时间序列数组
        pixel_time_series = np.full((self.total_channels, len(all_dates)), np.nan, dtype=np.float32)
        
        for time_idx, date in enumerate(all_dates):
            if date in preloaded_data:
                date_data = preloaded_data[date]
                
                # 按照固定顺序处理驱动因素
                for driver_name in self.driver_order:
                    if driver_name in self.channel_mapping and driver_name in date_data:
                        data = date_data[driver_name]
                        channel_info = self.channel_mapping[driver_name]
                        start_idx = channel_info['start_idx']
                        channels = channel_info['channels']
                        
                        if len(data.shape) == 3:  # 多波段
                            for band_idx in range(channels):
                                pixel_value = data[band_idx, row, col]
                                pixel_time_series[start_idx + band_idx, time_idx] = pixel_value
                        else:  # 单波段
                            pixel_value = data[row, col]
                            pixel_time_series[start_idx, time_idx] = pixel_value
        
        return pixel_time_series
    
    def _process_pixel_worker(self, args):
        """并行像素处理的工作函数"""
        pixel_coord, all_dates, preloaded_data, total_channels = args
        
        try:
            # 重建self属性（因为多进程）
            row, col = pixel_coord
            pixel_time_series = np.full((total_channels, len(all_dates)), np.nan, dtype=np.float32)
            
            # 重建channel_mapping和driver_order
            driver_order = [
                'Firms_Detection_resampled',
                'ERA5_multi_bands',
                'LULC_BCBoundingbox_resampled',
                'Topo_Distance_WGS84_resize_resampled',
                'NDVI_EVI',
                'Reflection_500_merge_TerraAquaWGS84_clip',
                'MODIS_Terra_Aqua_B20_21_merged_resampled',
                'MOD21A1DN_multibands_filtered_resampled',
                'LAI_BCBoundingbox_resampled'
            ]
            
            # 重建channel_mapping
            channel_mapping = {}
            total_ch = 0
            sample_channels = [1, 12, 1, 6, 2, 4, 4, 8, 1]  # 各驱动因素的通道数
            
            for i, driver_name in enumerate(driver_order):
                channels = sample_channels[i]
                channel_mapping[driver_name] = {
                    'start_idx': total_ch,
                    'channels': channels,
                    'end_idx': total_ch + channels
                }
                total_ch += channels
            
            for time_idx, date in enumerate(all_dates):
                if date in preloaded_data:
                    date_data = preloaded_data[date]
                    
                    for driver_name in driver_order:
                        if driver_name in channel_mapping and driver_name in date_data:
                            data = date_data[driver_name]
                            channel_info = channel_mapping[driver_name]
                            start_idx = channel_info['start_idx']
                            channels = channel_info['channels']
                            
                            if len(data.shape) == 3:  # 多波段
                                for band_idx in range(channels):
                                    pixel_value = data[band_idx, row, col]
                                    pixel_time_series[start_idx + band_idx, time_idx] = pixel_value
                            else:  # 单波段
                                pixel_value = data[row, col]
                                pixel_time_series[start_idx, time_idx] = pixel_value
            
            return pixel_coord, pixel_time_series
            
        except Exception as e:
            logger.error(f"处理像素 {pixel_coord} 时出错: {e}")
            return pixel_coord, None
    
    def _process_pixels_parallel(self, valid_pixels, all_dates, preloaded_data):
        """并行处理所有像素"""
        logger.info(f"并行处理像素（使用 {self.max_workers} 个进程）...")
        
        # 准备并行任务参数
        tasks = [(pixel_coord, all_dates, preloaded_data, self.total_channels) 
                for pixel_coord in valid_pixels]
        
        results = []
        
        # 使用线程池（因为数据已经在内存中）
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_pixel = {executor.submit(self._process_pixel_worker, task): task[0] 
                              for task in tasks}
            
            # 收集结果
            with tqdm(total=len(valid_pixels), desc="并行处理像素") as pbar:
                for future in as_completed(future_to_pixel):
                    pixel_coord = future_to_pixel[future]
                    try:
                        result_pixel, pixel_time_series = future.result()
                        if pixel_time_series is not None:
                            results.append((result_pixel, pixel_time_series))
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"处理像素 {pixel_coord} 失败: {e}")
                        pbar.update(1)
        
        logger.info(f"并行处理完成，成功处理 {len(results)}/{len(valid_pixels)} 个像素")
        return results
    
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
        output_file = os.path.join(self.output_dir, f'{target_year}_full_dataset.h5')
        
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
            h5_file.attrs['past_days'] = self.past_days
            h5_file.attrs['future_days'] = self.future_days
            h5_file.attrs['current_year_days'] = self.current_year_days
            h5_file.attrs['total_time_steps'] = len(all_dates)
            h5_file.attrs['total_channels'] = self.total_channels
            h5_file.attrs['data_format'] = 'channels_x_time'
            h5_file.attrs['data_type'] = 'full_pixel_time_series'
            h5_file.attrs['driver_names'] = self.driver_order
            h5_file.attrs['total_pixels'] = len(valid_pixels)
            h5_file.attrs['processed_pixels'] = len(pixel_results)
            
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
    
    parser = argparse.ArgumentParser(description='生成完整数据集 - 并行优化版本')
    parser.add_argument('--data_dir', 
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked_normalized',
                       help='数据根目录')
    parser.add_argument('--output_dir',
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/full_datasets',
                       help='输出目录')
    parser.add_argument('--years', type=str, default='[2021,2022,2023,2024]',
                       help='目标年份列表，格式：[2000,2021,2022,2023,2024] 或 2000,2021,2022,2023,2024')
    parser.add_argument('--past_days', type=int, default=365,
                       help='过去天数')
    parser.add_argument('--future_days', type=int, default=30,
                       help='未来天数')
    parser.add_argument('--current_year_days', type=int, default=365,
                       help='当前年份天数')
    parser.add_argument('--max_workers', type=int, default=12,
                       help='最大并行工作进程数')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='批处理大小')
    
    args = parser.parse_args()
    
    # 解析年份列表
    target_years = parse_year_list(args.years)
    
    # 创建生成器
    generator = OptimizedFullDatasetGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_years=target_years,
        past_days=args.past_days,
        future_days=args.future_days,
        current_year_days=args.current_year_days,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
    
    # 生成所有年份的数据集
    generator.generate_all_datasets()


if __name__ == "__main__":
    main() 