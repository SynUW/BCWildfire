#!/usr/bin/env python3
"""
像素级采样工具
- 从Firms_Detection_resampled选择值>0的像元作为正样本
- 选择4倍数量的负样本（值=0的像元）
- 获取指定时间序列长度内的所有驱动因素数据
- 生成H5文件，数据保存格式与原始代码相同
- 不使用10x10窗口，直接采样单个像素点
"""

import os
import logging
import numpy as np
import h5py
from osgeo import gdal
import glob
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import gc
from functools import lru_cache
import psutil
import time
import random
from osgeo import gdalconst

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 改回INFO级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('PixelSampling')

# 设置GDAL配置
gdal.SetConfigOption('GDAL_CACHEMAX', '2048')
gdal.SetConfigOption('GDAL_NUM_THREADS', '4')
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
gdal.SetConfigOption('VSI_CACHE', 'FALSE')

class DataCache:
    """高效的数据缓存管理器"""
    def __init__(self, max_memory_gb=20):
        self.cache = {}
        self.access_count = defaultdict(int)
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.current_memory = 0
        
    def get(self, key, loader_func):
        if key not in self.cache:
            # 检查内存使用
            self._check_memory_limit()
            
            data = loader_func()
            if data is not None:
                data_size = data.nbytes if hasattr(data, 'nbytes') else 0
                self.cache[key] = data
                self.current_memory += data_size
                
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def _check_memory_limit(self):
        """检查内存限制并清理缓存"""
        while self.current_memory > self.max_memory_bytes and self.cache:
            # 移除最少访问的项
            least_used = min(self.cache.keys(), 
                           key=lambda k: self.access_count[k])
            data = self.cache[least_used]
            data_size = data.nbytes if hasattr(data, 'nbytes') else 0
            
            del self.cache[least_used]
            del self.access_count[least_used]
            self.current_memory -= data_size
            
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
        self.current_memory = 0
        gc.collect()

class PixelSampler:
    def __init__(self, data_dir, output_dir, past_days=365, future_days=30, negative_ratio=1.0):
        """初始化像素采样器"""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.past_days = past_days
        self.future_days = future_days
        self.negative_ratio = negative_ratio
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据缓存
        self.data_cache = DataCache(max_memory_gb=20)
        
        # 获取所有驱动因素目录
        self.driver_dirs = self._get_driver_directories()
        logger.info(f"找到 {len(self.driver_dirs)} 个driver目录")
        
        logger.info("开始像素级采样...")
        
    def _get_driver_directories(self):
        """获取所有驱动因素目录"""
        driver_dirs = {}
        
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                # 检查目录中是否有tif文件
                tif_files = glob.glob(os.path.join(item_path, '*.tif'))
                if tif_files:
                    driver_dirs[item] = item_path
        
        return driver_dirs

    def _load_driver_data_cached(self, driver_dir, date):
        """使用缓存加载驱动因素数据"""
        cache_key = (driver_dir, date.strftime('%Y%m%d'))
        
        def loader():
            try:
                return self._load_driver_data(driver_dir, date)
            except Exception as e:
                logger.error(f"加载driver数据出错: {driver_dir}, {date}, 错误: {str(e)}")
                return None
                
        return self.data_cache.get(cache_key, loader)
    
    def _convert_to_float(self, data):
        """直接返回原始数据，不进行任何转换"""
        return data
    
    def _get_data_dtype_and_scale(self, driver_dir):
        """返回原始数据类型和缩放因子"""
        # 所有数据都保持原始格式
        return None, 1.0

    @lru_cache(maxsize=1000)
    def _get_date_from_filename(self, filename):
        """从文件名提取日期"""
        try:
            # 假设文件名格式为 *_YYYY_MM_DD.tif
            parts = filename.split('_')
            for i in range(len(parts) - 2):
                if (len(parts[i]) == 4 and parts[i].isdigit() and
                    len(parts[i+1]) == 2 and parts[i+1].isdigit() and
                    len(parts[i+2]) == 2 and parts[i+2].isdigit()):
                    return datetime(int(parts[i]), int(parts[i+1]), int(parts[i+2]))
            return None
        except:
            return None

    def _get_past_future_dates(self, date):
        """获取过去和未来的日期列表"""
        try:
            # 生成过去的日期
            past_dates = []
            for i in range(self.past_days, 0, -1):
                past_date = date - timedelta(days=i)
                past_dates.append(past_date)
            
            # 生成未来的日期
            future_dates = []
            for i in range(1, self.future_days + 1):
                future_date = date + timedelta(days=i)
                future_dates.append(future_date)
            
            # 检查数据长度是否足够
            if len(past_dates) < self.past_days or len(future_dates) < self.future_days:
                return None, None
                
            return past_dates, future_dates
            
        except Exception as e:
            logger.error(f"生成日期列表时出错: {str(e)}")
            return None, None

    def _extract_pixel_time_series(self, pixel_coord, driver_data, dates, event_date):
        """为单个像素提取时间序列数据
        
        Args:
            pixel_coord: 像素坐标 (row, col)
            driver_data: 单个driver的数据字典 {date: data}
            dates: 日期列表
            event_date: 事件日期
            
        Returns:
            numpy array: shape为 (time_steps, channels)
            压缩格式的数据将保持int16类型，原始数据保持原类型
            
        Raises:
            ValueError: 当没有有效数据时抛出异常
        """
        row, col = pixel_coord
        time_series_data = []
        data_dtype = None
        
        for date in dates:
            if date not in driver_data or driver_data[date] is None:
                raise ValueError(f"日期 {date} 没有有效数据")
                
            data = driver_data[date]
            
            # 确定数据类型（第一次迭代时）
            if data_dtype is None:
                data_dtype = data.dtype
            
            # 获取数据尺寸
            if len(data.shape) == 3:
                channels, height, width = data.shape
            else:
                height, width = data.shape
                channels = 1
                data = data.reshape(1, height, width)
            
            # 检查像素坐标是否有效
            if row < 0 or row >= height or col < 0 or col >= width:
                raise ValueError(f"像素坐标超出边界: ({row}, {col}), 数据尺寸: {height}x{width}")
            
            # 提取像素值
            pixel_values = data[:, row, col]  # shape: (channels,)
            time_series_data.append(pixel_values)
        
        if not time_series_data:
            raise ValueError(f"像素 ({row}, {col}) 在日期范围 {dates[0]} 到 {dates[-1]} 内没有有效数据")
            
        # 堆叠时间序列数据，保持原始数据类型
        return np.stack(time_series_data, axis=0)  # shape: (time_steps, channels)

    def _sample_pixels_from_firms(self, firms_data):
        """从FIRMS数据中采样像素并获取其具体值
        
        Args:
            firms_data: FIRMS数据数组
            
        Returns:
            list: 像素事件列表 [(row, col, firms_value), ...]
                其中firms_value是该像素的具体FIRMS值（如0, 1, 3, 20等）
        """
        if len(firms_data.shape) == 3:
            # 多波段数据，取第一个波段
            firms_data = firms_data[0]
        
        # 获取所有唯一的FIRMS值
        unique_values = np.unique(firms_data)
        logger.debug(f"FIRMS数据中的唯一值: {unique_values}")
        
        pixel_events = []
        
        # 为每个唯一值采样像素
        for firms_value in unique_values:
            # 找到该值的所有像素位置
            value_mask = firms_data == firms_value
            coords = np.where(value_mask)
            pixels = list(zip(coords[0], coords[1]))
            
            # 如果是0值（负样本），按比例采样
            if firms_value == 0:
                # 计算需要的负样本数量（基于所有非零像素的总数）
                non_zero_count = np.sum(firms_data > 0)
                target_negative_count = int(non_zero_count * self.negative_ratio)
                
                if len(pixels) > target_negative_count:
                    pixels = random.sample(pixels, target_negative_count)
                    
                # logger.info(f"FIRMS值 {firms_value}: 从 {len(coords[0])} 个像素中采样 {len(pixels)} 个")
            # else:
                # 非零值（正样本），全部采样
                # logger.info(f"FIRMS值 {firms_value}: 采样 {len(pixels)} 个像素")
            
            # 添加到事件列表，包含具体的FIRMS值
            for pixel in pixels:
                pixel_events.append((pixel, firms_value))
        
        return pixel_events

    def _process_year_pixels_with_adjacent(self, current_year_files, all_available_files, target_year):
        """处理一年的像素采样数据，包含相邻年份的数据
        
        Args:
            current_year_files: 当前年份的文件列表
            all_available_files: 所有可用的文件列表（包含相邻年份）
            target_year: 目标年份
        """
        logger.info(f"处理年份 {target_year}，当前年文件 {len(current_year_files)} 个，总可用文件 {len(all_available_files)} 个")
        
        # 构建全局日期到文件的映射（包含相邻年份）
        all_date_to_file = {}
        all_dates = []
        
        for file_path in all_available_files:
            filename = os.path.basename(file_path)
            date = self._extract_date_from_filename(filename)
            if date:
                all_dates.append(date)
                all_date_to_file[date] = file_path
        
        all_dates.sort()
        logger.info(f"全局可用日期范围: {all_dates[0]} 到 {all_dates[-1]}，共 {len(all_dates)} 个日期")
        
        # 获取当前年份的日期（用于采样事件）
        current_year_dates = []
        for file_path in current_year_files:
            filename = os.path.basename(file_path)
            date = self._extract_date_from_filename(filename)
            if date:
                current_year_dates.append(date)
        
        current_year_dates.sort()
        
        # 简单检查数据完整性
        valid_dates = []
        logger.info(f"检查 {len(current_year_dates)} 个日期的数据完整性...")
        
        for date in tqdm(current_year_dates, desc=f"检查数据完整性 {target_year}"):
            # 检查是否有足够的历史和未来数据
            past_dates, future_dates = self._get_past_future_dates(date)
            if past_dates is None or future_dates is None:
                continue
            
            # 检查所有需要的日期是否都有数据（使用全局date_to_file）
            required_dates = past_dates + [date] + future_dates
            missing_dates = [req_date for req_date in required_dates if req_date not in all_date_to_file]
            
            if not missing_dates:
                valid_dates.append(date)
        
        logger.info(f"找到 {len(valid_dates)} 个数据完整的日期")
        
        if not valid_dates:
            logger.warning(f"年份 {target_year}: 没有有效的日期")
            return
        
        # 收集所有像素事件（只从有效日期采样）
        pixel_events = []
        
        logger.info(f"从 {len(valid_dates)} 个有效日期中采样像素事件...")
        
        for date in tqdm(valid_dates, desc=f"采样像素事件 {target_year}"):
            # 加载FIRMS数据
            firms_data = self._load_driver_data_cached('Firms_Detection_resampled', date)
            if firms_data is None:
                continue
            
            # 采样像素并获取FIRMS值
            date_pixel_events = self._sample_pixels_from_firms(firms_data)
            
            # 添加到事件列表，格式：(pixel_coord, date, firms_value)
            for pixel_coord, firms_value in date_pixel_events:
                pixel_events.append((pixel_coord, date, firms_value))
        
        logger.info(f"年份 {target_year}: 从 {len(valid_dates)} 个有效日期收集到 {len(pixel_events)} 个像素事件")
        
        if not pixel_events:
            logger.warning(f"年份 {target_year}: 没有有效的像素事件")
            return
        
        # 处理每个驱动因素（使用全局日期范围）
        self._process_driver_data_batch(self.driver_dirs, all_dates, target_year, pixel_events)

    def _process_driver_data_batch(self, driver_dirs, all_dates, year, pixel_events):
        """批量处理所有驱动因素数据，支持跨年份查询"""
        logger.info(f"为年份 {year} 创建 {len(pixel_events)} 个像素样本的时间序列")
        
        # 预先收集所有需要的日期
        required_dates = set()
        for pixel_coord, event_date, firms_value in pixel_events:
            past_dates, future_dates = self._get_past_future_dates(event_date)
            if past_dates and future_dates:
                required_dates.update(past_dates + [event_date] + future_dates)
        
        required_dates = sorted(required_dates)
        logger.info(f"需要加载 {len(required_dates)} 个日期的数据")
        
        # 批量加载所有驱动因素的所有需要日期的数据
        all_driver_data, geo_reference = self._batch_load_all_driver_data(driver_dirs, required_dates)
        
        # 为每个像素事件创建时间序列
        pixel_samples = []
        
        logger.info("开始创建时间序列...")
        for pixel_coord, event_date, firms_value in tqdm(pixel_events, desc=f"创建时间序列"):
            # 获取该事件的时间序列日期
            past_dates, future_dates = self._get_past_future_dates(event_date)
            if past_dates is None or future_dates is None:
                continue
            
            all_event_dates = past_dates + [event_date] + future_dates
            
            # 为每个驱动因素提取时间序列
            sample_data = {}
            valid_sample = True
            
            for driver_name in driver_dirs.keys():
                if driver_name not in all_driver_data:
                    valid_sample = False
                    break
                
                # 检查所有日期是否都有数据
                missing_dates = [d for d in all_event_dates if d not in all_driver_data[driver_name]]
                if missing_dates:
                    valid_sample = False
                    break
                
                # 分别提取过去和未来的时间序列
                past_series = []
                current_value = None
                future_series = []
                
                # 过去的数据
                for date in past_dates:
                    data = all_driver_data[driver_name][date]
                    if data is None:
                        valid_sample = False
                        break
                    
                    # 获取指定像素的值，使用(row, col)坐标
                    row, col = pixel_coord
                    if len(data.shape) == 3:
                        # 多波段数据，提取所有波段的像素值
                        pixel_value = data[:, row, col]  # shape: (bands,)
                    else:
                        # 单波段数据
                        pixel_value = data[row, col]  # 标量值
                    past_series.append(pixel_value)
                
                # 当前的数据
                if valid_sample:
                    data = all_driver_data[driver_name][event_date]
                    if data is None:
                        valid_sample = False
                    else:
                        row, col = pixel_coord
                        if len(data.shape) == 3:
                            # 多波段数据，提取所有波段的像素值
                            current_value = data[:, row, col]  # shape: (bands,)
                        else:
                            # 单波段数据
                            current_value = data[row, col]  # 标量值
                
                # 未来的数据
                if valid_sample:
                    for date in future_dates:
                        data = all_driver_data[driver_name][date]
                        if data is None:
                            valid_sample = False
                            break
                        
                        # 获取指定像素的值，使用(row, col)坐标
                        row, col = pixel_coord
                        if len(data.shape) == 3:
                            # 多波段数据，提取所有波段的像素值
                            pixel_value = data[:, row, col]  # shape: (bands,)
                        else:
                            # 单波段数据
                            pixel_value = data[row, col]  # 标量值
                        future_series.append(pixel_value)
                
                if not valid_sample:
                    break
                
                # 确保数据形状一致并保存
                # 将past_series转换为numpy数组，处理混合的标量和数组
                if past_series:
                    # 检查第一个元素的类型来确定数据结构
                    first_past = past_series[0]
                    if np.isscalar(first_past):
                        # 所有元素都是标量，直接转换
                        past_array = np.array(past_series, dtype=np.float32)
                    else:
                        # 包含数组元素，需要堆叠
                        past_array = np.array(past_series, dtype=np.float32)
                        if past_array.ndim == 2:
                            # 转置为 (bands, time_steps)
                            past_array = past_array.T
                else:
                    past_array = np.array([], dtype=np.float32)
                
                # 处理future_series
                if future_series:
                    first_future = future_series[0]
                    if np.isscalar(first_future):
                        future_array = np.array(future_series, dtype=np.float32)
                    else:
                        future_array = np.array(future_series, dtype=np.float32)
                        if future_array.ndim == 2:
                            # 转置为 (bands, time_steps)
                            future_array = future_array.T
                else:
                    future_array = np.array([], dtype=np.float32)
                
                # 处理current_value
                if np.isscalar(current_value):
                    current_array = np.array([current_value], dtype=np.float32)
                else:
                    current_array = np.array(current_value, dtype=np.float32)
                
                sample_data[driver_name] = {
                    'past': past_array,
                    'current': current_array,
                    'future': future_array
                }
            
            if valid_sample:
                pixel_samples.append({
                    'pixel_coord': pixel_coord,
                    'date': event_date,
                    'firms_value': firms_value,  # 使用具体的FIRMS值
                    'data': sample_data
                })
        
        logger.info(f"年份 {year}: 成功创建 {len(pixel_samples)} 个有效样本")
        
        if pixel_samples:
            self._save_pixel_samples(pixel_samples, year, geo_reference)

    def _batch_load_all_driver_data(self, driver_dirs, required_dates):
        """批量加载所有驱动因素数据，使用单进程和统一进度条"""
        logger.info(f"开始加载所有驱动因素数据")
        
        # 构建所有需要加载的任务
        load_tasks = []
        for driver_name, driver_dir in driver_dirs.items():
            driver_files = glob.glob(os.path.join(driver_dir, '*.tif'))
            driver_date_to_file = {}
            
            for file_path in driver_files:
                filename = os.path.basename(file_path)
                date = self._extract_date_from_filename(filename)
                if date and date in required_dates:
                    driver_date_to_file[date] = file_path
            
            # 为每个需要的日期创建加载任务
            for date in required_dates:
                if date in driver_date_to_file:
                    load_tasks.append((driver_name, date, driver_date_to_file[date]))
        
        logger.info(f"总共需要加载 {len(load_tasks)} 个文件")
        
        # 使用单进程顺序加载
        all_driver_data = {}
        geo_reference = None  # 保存地理参考信息（所有驱动因素相同）
        
        # 使用统一进度条显示所有数据加载进度
        for driver_name, date, file_path in tqdm(load_tasks, desc="加载所有驱动因素数据"):
            try:
                data, geo_info = self._load_single_file_with_gdal(file_path)
                if data is not None:
                    if driver_name not in all_driver_data:
                        all_driver_data[driver_name] = {}
                    all_driver_data[driver_name][date] = data
                    
                    # 保存第一个成功加载的地理参考信息
                    if geo_reference is None and geo_info is not None:
                        geo_reference = geo_info
                        
            except Exception as e:
                logger.warning(f"加载文件失败: {file_path}, 错误: {str(e)}")
        
        logger.info(f"数据加载完成，共加载 {len(all_driver_data)} 个驱动因素的数据")
        return all_driver_data, geo_reference

    def _load_single_file_with_gdal(self, file_path):
        """使用GDAL加载单个文件，同时获取地理参考信息"""
        try:
            from osgeo import gdal
            import numpy as np
            
            # 打开数据集
            dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
            if dataset is None:
                return None, None
            
            # 读取所有波段数据
            bands = dataset.RasterCount
            if bands == 1:
                # 单波段
                band = dataset.GetRasterBand(1)
                data = band.ReadAsArray()
            else:
                # 多波段
                data = []
                for i in range(1, bands + 1):
                    band = dataset.GetRasterBand(i)
                    band_data = band.ReadAsArray()
                    data.append(band_data)
                data = np.array(data)
            
            # 获取地理参考信息
            geo_info = {
                'geotransform': dataset.GetGeoTransform(),
                'projection': dataset.GetProjection(),
                'raster_size': (dataset.RasterXSize, dataset.RasterYSize),
                'raster_count': bands
            }
            
            # 关闭数据集
            dataset = None
            
            return data, geo_info
            
        except Exception as e:
            return None, None

    def _save_geo_reference(self, h5_file, geo_info):
        """保存地理参考信息到H5文件"""
        if geo_info:
            h5_file.attrs['geotransform'] = geo_info['geotransform']
            h5_file.attrs['projection'] = geo_info['projection']
            h5_file.attrs['raster_size'] = geo_info['raster_size']
            h5_file.attrs['raster_count'] = geo_info['raster_count']

    def process_all_years(self):
        """处理所有年份的数据"""
        # 使用新的年份获取方法
        available_years = self._get_available_years()
        
        if not available_years:
            logger.error("没有找到任何可用的年份数据")
            return
        
        # 获取FIRMS文件列表并按年份分组
        firms_dir = os.path.join(self.data_dir, 'Firms_Detection_resampled')
        if not os.path.exists(firms_dir):
            logger.error(f"FIRMS目录不存在: {firms_dir}")
            return
        
        firms_files = glob.glob(os.path.join(firms_dir, '*.tif'))
        if not firms_files:
            logger.error(f"FIRMS目录中没有找到tif文件: {firms_dir}")
            return
        
        # 按年份分组FIRMS文件
        year_groups = defaultdict(list)
        for file_path in firms_files:
            filename = os.path.basename(file_path)
            date = self._extract_date_from_filename(filename)  # 使用新的日期提取方法
            if date:
                year_groups[date.year].append(file_path)
        
        # 只处理在available_years中的年份
        valid_years = [year for year in available_years if year in year_groups]
        logger.info(f"开始处理 {len(valid_years)} 个年份")
        
        # 处理每个年份
        for year in sorted(valid_years):
            logger.info(f"=" * 50)
            logger.info(f"开始处理年份: {year}")
            
            # 检查该年份的样本数据是否已经全部存在
            sample_files_exist = self._check_sample_files_exist(year)
            full_files_exist = self._check_full_files_exist(year) if 2020 <= year <= 2024 else True
            
            if sample_files_exist and full_files_exist:
                logger.info(f"年份 {year} 的所有输出文件已存在，跳过处理")
                continue
            
            # 计算需要的年份范围（包含相邻年份）
            years_needed = []
            if year - 1 in year_groups:
                years_needed.append(year - 1)  # 前一年
            years_needed.append(year)  # 当前年
            if year + 1 in year_groups:
                years_needed.append(year + 1)  # 后一年
            
            # 收集所有需要的文件
            all_year_files = []
            for y in years_needed:
                if y in year_groups:
                    all_year_files.extend(year_groups[y])
            
            # 常规采样处理
            if not sample_files_exist:
                logger.info(f"处理年份 {year} 的样本数据...")
                self._process_year_pixels_with_adjacent(sorted(year_groups[year]), sorted(all_year_files), year)
            else:
                logger.info(f"年份 {year} 的样本数据已存在，跳过")
            
            # 为2020-2024年额外创建完整数据文件
            if 2020 <= year <= 2024 and not full_files_exist:
                logger.info(f"为年份 {year} 创建完整数据文件...")
                self._process_full_year_data(sorted(year_groups[year]), sorted(all_year_files), year)
            elif 2020 <= year <= 2024:
                logger.info(f"年份 {year} 的完整数据已存在，跳过")
            
            # 清理缓存
            self.data_cache.clear()
            logger.info(f"年份 {year} 处理完成")
            logger.info(f"=" * 50)

    def _get_available_years(self):
        """获取所有可用的年份"""
        firms_dir = os.path.join(self.data_dir, 'Firms_Detection_resampled')
        if not os.path.exists(firms_dir):
            logger.error(f"FIRMS目录不存在: {firms_dir}")
            return []
        
        firms_files = glob.glob(os.path.join(firms_dir, '*.tif'))
        if not firms_files:
            logger.error(f"FIRMS目录中没有找到tif文件")
            return []
        
        years = set()
        for file_path in firms_files:
            filename = os.path.basename(file_path)
            date = self._extract_date_from_filename(filename)
            if date:
                years.add(date.year)
        
        available_years = sorted(years)
        logger.info(f"找到 {len(available_years)} 个年份: {min(available_years)}-{max(available_years)}")
        return available_years
    
    def _check_sample_files_exist(self, year):
        """检查指定年份的样本数据文件是否已存在"""
        driver_names = list(self.driver_dirs.keys())
        for driver_name in driver_names:
            h5_path = os.path.join(self.output_dir, f'{year}_{driver_name}.h5')
            if not os.path.exists(h5_path):
                return False
        return True
    
    def _check_full_files_exist(self, year):
        """检查指定年份的完整数据文件是否已存在"""
        driver_names = list(self.driver_dirs.keys())
        for driver_name in driver_names:
            h5_path = os.path.join(self.output_dir, f'{year}_{driver_name}_full.h5')
            if not os.path.exists(h5_path):
                return False
        return True

    def _extract_date_from_filename(self, filename):
        """从文件名中提取日期信息"""
        try:
            # 新的文件名格式: DriverName_yyyy_mm_dd.tif
            parts = filename.split('_')
            if len(parts) >= 4:
                year = int(parts[1])    # 年份在第二个位置
                month = int(parts[2])   # 月份在第三个位置
                day_part = parts[3].replace('.tif', '')  # 移除.tif后缀
                day = int(day_part)     # 日期在第四个位置
                return datetime(year, month, day)
        except (ValueError, IndexError) as e:
            logger.warning(f"无法从文件名解析日期: {filename}, 错误: {e}")
        return None

    def _load_driver_data(self, driver_dir, date):
        """加载指定日期的驱动因素数据"""
        # 构建文件路径
        if isinstance(driver_dir, str):
            # 如果是字符串，说明是目录名，需要构建完整路径
            if driver_dir in self.driver_dirs:
                driver_path = self.driver_dirs[driver_dir]
            else:
                driver_path = os.path.join(self.data_dir, driver_dir)
        else:
            # 如果已经是完整路径
            driver_path = driver_dir
        
        # 查找匹配的文件
        date_str = date.strftime('%Y_%m_%d')
        pattern = os.path.join(driver_path, f'*{date_str}.tif')
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            return None
        
        file_path = matching_files[0]
        
        try:
            data, geo_info = self._load_single_file_with_gdal(file_path)
            return data
            
        except Exception as e:
            logger.error(f"使用GDAL加载文件失败: {file_path}, 错误: {str(e)}")
            return None

    def _process_full_year_data(self, current_year_files, all_available_files, target_year):
        """处理完整年份数据，保存所有像素（用于2020-2024年）
        
        Args:
            current_year_files: 当前年份的文件列表
            all_available_files: 所有可用的文件列表（包含相邻年份）
            target_year: 目标年份
        """
        logger.info(f"处理年份 {target_year} 的完整数据")
        
        # 构建全局日期到文件的映射（包含相邻年份）
        all_date_to_file = {}
        all_dates = []
        
        for file_path in all_available_files:
            filename = os.path.basename(file_path)
            date = self._extract_date_from_filename(filename)
            if date:
                all_dates.append(date)
                all_date_to_file[date] = file_path
        
        all_dates.sort()
        
        # 获取当前年份的日期
        current_year_dates = []
        for file_path in current_year_files:
            filename = os.path.basename(file_path)
            date = self._extract_date_from_filename(filename)
            if date:
                current_year_dates.append(date)
        
        current_year_dates.sort()
        
        # 找到有完整时间序列数据的日期
        valid_dates = []
        
        logger.info(f"检查 {len(current_year_dates)} 个日期的完整性...")
        
        for date in tqdm(current_year_dates, desc=f"检查完整数据 {target_year}"):
            # 检查是否有足够的历史和未来数据
            past_dates, future_dates = self._get_past_future_dates(date)
            if past_dates is None or future_dates is None:
                continue
            
            # 检查所有需要的日期是否都有数据
            required_dates = past_dates + [date] + future_dates
            missing_dates = [req_date for req_date in required_dates if req_date not in all_date_to_file]
            
            if not missing_dates:
                valid_dates.append(date)
        
        logger.info(f"年份 {target_year}: 找到 {len(valid_dates)} 个有完整数据的日期")
        
        if not valid_dates:
            logger.warning(f"年份 {target_year}: 没有找到有完整数据的日期")
            return
        
        # 获取第一个有效日期的FIRMS数据来确定数据维度
        first_date = valid_dates[0]
        firms_data = self._load_driver_data_cached('Firms_Detection_resampled', first_date)
        if firms_data is None:
            logger.error(f"无法加载FIRMS数据: {first_date}")
            return
        
        if len(firms_data.shape) == 3:
            firms_data = firms_data[0]
        
        total_pixels = firms_data.shape[0] * firms_data.shape[1]
        logger.info(f"数据维度: {firms_data.shape}, 总像素数: {total_pixels}")
        
        # 为每个驱动因素创建完整数据文件
        self._create_full_data_files(self.driver_dirs, all_dates, target_year, valid_dates, total_pixels, firms_data.shape)

    def _create_full_data_files(self, driver_dirs, all_dates, year, valid_dates, total_pixels, data_shape):
        """为所有驱动因素创建完整数据文件 - 超级优化版本"""
        logger.info(f"为年份 {year} 创建完整数据文件，共 {len(valid_dates)} 个有效日期")
        
        # 预先收集所有需要的日期
        required_dates = set()
        for date in valid_dates:
            past_dates, future_dates = self._get_past_future_dates(date)
            if past_dates and future_dates:
                required_dates.update(past_dates + [date] + future_dates)
        
        required_dates = sorted(required_dates)
        logger.info(f"完整数据需要加载 {len(required_dates)} 个日期的数据")
        
        # 批量加载所有驱动因素的所有需要日期的数据
        all_driver_data, geo_reference = self._batch_load_all_driver_data(driver_dirs, required_dates)
        
        # 为每个驱动因素创建完整数据文件
        for driver_name in driver_dirs.keys():
            h5_path = os.path.join(self.output_dir, f'{year}_{driver_name}_full.h5')
            
            # 检查文件是否已存在
            if os.path.exists(h5_path):
                logger.info(f"完整数据文件已存在，跳过: {h5_path}")
                continue
            
            logger.info(f"创建完整数据文件: {driver_name}")
            
            if driver_name not in all_driver_data:
                logger.warning(f"驱动因素 {driver_name} 没有加载到数据")
                continue
            
            # 使用超级优化策略
            self._create_full_data_ultra_fast(h5_path, driver_name, all_driver_data, 
                                            valid_dates, data_shape, geo_reference, year)

    def _create_full_data_ultra_fast(self, h5_path, driver_name, all_driver_data, 
                                   valid_dates, data_shape, geo_reference, year):
        """超级优化的完整数据创建方法"""
        driver_data = all_driver_data[driver_name]
        height, width = data_shape
        
        with h5py.File(h5_path, 'w') as h5_file:
            # 写入全局属性
            h5_file.attrs['year'] = str(year)
            h5_file.attrs['driver'] = driver_name
            h5_file.attrs['past_days'] = self.past_days
            h5_file.attrs['future_days'] = self.future_days
            h5_file.attrs['data_type'] = 'full_pixels_optimized'
            h5_file.attrs['total_pixels'] = height * width
            h5_file.attrs['data_shape'] = data_shape
            h5_file.attrs['valid_dates'] = len(valid_dates)
            
            # 保存地理参考信息
            if geo_reference:
                self._save_geo_reference(h5_file, geo_reference)
            
            dataset_count = 0
            
            # 分块处理以减少内存使用和提高效率
            chunk_size = 1000  # 每次处理1000个像素
            total_processed = 0
            
            logger.info(f"开始批量处理 {height}×{width} = {height*width} 个像素")
            
            # 预计算所有有效日期的数据结构
            valid_date_data = {}
            for date in valid_dates:
                past_dates, future_dates = self._get_past_future_dates(date)
                if past_dates is None or future_dates is None:
                    continue
                
                all_event_dates = past_dates + [date] + future_dates
                missing_dates = [d for d in all_event_dates if d not in driver_data]
                if missing_dates:
                    continue
                
                # 获取FIRMS数据
                current_firms_data = None
                if 'Firms_Detection_resampled' in all_driver_data and date in all_driver_data['Firms_Detection_resampled']:
                    current_firms_data = all_driver_data['Firms_Detection_resampled'][date]
                    if len(current_firms_data.shape) == 3:
                        current_firms_data = current_firms_data[0]
                
                valid_date_data[date] = {
                    'past_dates': past_dates,
                    'future_dates': future_dates,
                    'firms_data': current_firms_data,
                    'date_str': date.strftime('%Y%m%d')
                }
            
            logger.info(f"有效处理日期数: {len(valid_date_data)}")
            
            # 批量处理像素
            pixel_batch = []
            for row in range(height):
                for col in range(width):
                    pixel_batch.append((row, col))
                    
                    if len(pixel_batch) >= chunk_size:
                        # 处理这一批像素
                        count = self._process_pixel_batch_ultra_fast(
                            h5_file, pixel_batch, valid_date_data, 
                            driver_data, driver_name
                        )
                        dataset_count += count
                        total_processed += len(pixel_batch)
                        
                        # 清空批次
                        pixel_batch = []
                        
                        # 进度报告
                        if total_processed % 50000 == 0:
                            progress = total_processed / (height * width) * 100
                            logger.info(f"处理进度: {progress:.1f}% ({total_processed}/{height*width})")
                        
                        # 强制垃圾回收
                        if total_processed % 100000 == 0:
                            gc.collect()
            
            # 处理剩余的像素
            if pixel_batch:
                count = self._process_pixel_batch_ultra_fast(
                    h5_file, pixel_batch, valid_date_data, 
                    driver_data, driver_name
                )
                dataset_count += count
                total_processed += len(pixel_batch)
            
            h5_file.attrs['total_datasets'] = dataset_count
            logger.info(f"完整数据文件创建完成: {driver_name}, 共 {dataset_count} 个数据集")

    def _process_pixel_batch_ultra_fast(self, h5_file, pixel_batch, valid_date_data, 
                                      driver_data, driver_name):
        """超高速批量处理像素"""
        dataset_count = 0
        
        for date, date_info in valid_date_data.items():
            past_dates = date_info['past_dates']
            future_dates = date_info['future_dates'] 
            firms_data = date_info['firms_data']
            date_str = date_info['date_str']
            
            # 批量提取所有像素的数据
            batch_past_data = []
            batch_future_data = []
            batch_coords = []
            batch_firms_values = []
            
            for row, col in pixel_batch:
                # 获取FIRMS值
                if firms_data is not None:
                    firms_value = int(firms_data[row, col])
                else:
                    firms_value = 0
                
                # 提取过去的时间序列
                past_series = []
                valid_past = True
                for ts_date in past_dates:
                    if ts_date not in driver_data or driver_data[ts_date] is None:
                        valid_past = False
                        break
                    
                    data = driver_data[ts_date]
                    if len(data.shape) == 3:
                        pixel_value = data[:, row, col]
                    else:
                        pixel_value = data[row, col]
                    past_series.append(pixel_value)
                
                # 提取未来的时间序列
                future_series = []
                valid_future = True
                for ts_date in future_dates:
                    if ts_date not in driver_data or driver_data[ts_date] is None:
                        valid_future = False
                        break
                    
                    data = driver_data[ts_date]
                    if len(data.shape) == 3:
                        pixel_value = data[:, row, col]
                    else:
                        pixel_value = data[row, col]
                    future_series.append(pixel_value)
                
                # 只保存有效的像素数据
                if valid_past and valid_future:
                    # 处理数据格式
                    if past_series:
                        first_past = past_series[0]
                        if np.isscalar(first_past):
                            past_array = np.array(past_series, dtype=np.float32)
                        else:
                            past_array = np.array(past_series, dtype=np.float32)
                            if past_array.ndim == 2:
                                past_array = past_array.T
                    else:
                        past_array = np.array([], dtype=np.float32)
                    
                    if future_series:
                        first_future = future_series[0]
                        if np.isscalar(first_future):
                            future_array = np.array(future_series, dtype=np.float32)
                        else:
                            future_array = np.array(future_series, dtype=np.float32)
                            if future_array.ndim == 2:
                                future_array = future_array.T
                    else:
                        future_array = np.array([], dtype=np.float32)
                    
                    batch_past_data.append(past_array)
                    batch_future_data.append(future_array)
                    batch_coords.append((row, col))
                    batch_firms_values.append(firms_value)
            
            # 批量写入数据集 - 使用最快的设置
            for i, (past_array, future_array, (row, col), firms_value) in enumerate(
                zip(batch_past_data, batch_future_data, batch_coords, batch_firms_values)):
                
                # 过去的时间序列
                past_dataset_name = f"{date_str}_past_{firms_value}_{row}_{col}"
                past_dataset = h5_file.create_dataset(
                    past_dataset_name,
                    data=past_array,
                    dtype=np.float32,
                    compression=None,  # 完全关闭压缩
                    shuffle=False,
                    chunks=False
                )
                dataset_count += 1
                
                # 未来的时间序列
                future_dataset_name = f"{date_str}_future_{firms_value}_{row}_{col}"
                future_dataset = h5_file.create_dataset(
                    future_dataset_name,
                    data=future_array,
                    dtype=np.float32,
                    compression=None,  # 完全关闭压缩
                    shuffle=False,
                    chunks=False
                )
                dataset_count += 1
        
        return dataset_count

    def _save_pixel_samples(self, pixel_samples, year, geo_reference=None):
        """保存像素样本数据到H5文件"""
        if not pixel_samples:
            return
        
        # 按驱动因素分组保存
        driver_names = list(pixel_samples[0]['data'].keys())
        
        for driver_name in driver_names:
            h5_path = os.path.join(self.output_dir, f'{year}_{driver_name}.h5')
            
            # 检查文件是否已存在
            if os.path.exists(h5_path):
                logger.info(f"样本文件已存在，跳过: {h5_path}")
                continue
            
            logger.info(f"保存样本数据: {driver_name}")
            
            with h5py.File(h5_path, 'w') as h5_file:
                # 写入全局属性（减少冗余）
                h5_file.attrs['year'] = str(year)
                h5_file.attrs['driver'] = driver_name
                h5_file.attrs['past_days'] = self.past_days
                h5_file.attrs['future_days'] = self.future_days
                h5_file.attrs['negative_ratio'] = self.negative_ratio
                h5_file.attrs['sampling_type'] = 'pixel_level'
                h5_file.attrs['data_type'] = 'sampled_pixels'
                
                # 保存地理参考信息
                if geo_reference:
                    self._save_geo_reference(h5_file, geo_reference)
                
                dataset_count = 0
                
                # 保存每个样本的过去和未来数据
                for sample in pixel_samples:
                    pixel_coord = sample['pixel_coord']
                    date = sample['date']
                    firms_value = sample['firms_value']  # 使用具体的FIRMS值
                    driver_data = sample['data'][driver_name]
                    
                    date_str = date.strftime('%Y%m%d')
                    
                    # 保存过去的时间序列，使用具体的FIRMS值作为标签
                    past_dataset_name = f"{date_str}_past_{firms_value}_{pixel_coord[0]}_{pixel_coord[1]}"
                    past_dataset = h5_file.create_dataset(
                        past_dataset_name,
                        data=np.array(driver_data['past']),
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=6,
                        shuffle=True
                    )
                    dataset_count += 1
                    
                    # 保存未来的时间序列，使用具体的FIRMS值作为标签
                    future_dataset_name = f"{date_str}_future_{firms_value}_{pixel_coord[0]}_{pixel_coord[1]}"
                    future_dataset = h5_file.create_dataset(
                        future_dataset_name,
                        data=np.array(driver_data['future']),
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=6,
                        shuffle=True
                    )
                    dataset_count += 1
                
                h5_file.attrs['total_datasets'] = dataset_count
                logger.info(f"样本数据保存完成: {driver_name}, 共 {dataset_count} 个数据集")

def main():
    """主函数"""
    # 配置路径
    data_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked_normalized'
    output_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples'
    
    # 创建采样器
    sampler = PixelSampler(
        data_dir=data_dir,
        output_dir=output_dir,
        past_days=365,
        future_days=30,
        negative_ratio=4,
    )
    
    # 开始处理
    logger.info("开始像素级采样...")
    sampler.process_all_years()
    logger.info("像素级采样完成！")

if __name__ == "__main__":
    main() 