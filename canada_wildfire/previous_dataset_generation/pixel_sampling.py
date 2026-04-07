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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from multiprocessing import Pool, Manager, Lock
from tqdm import tqdm
from collections import defaultdict
import gc
from functools import lru_cache
import psutil
import time
import random
from osgeo import gdalconst
import pickle
import tempfile

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

# 流式内存缓存类
class StreamingCache:
    """流式内存缓存 - 智能管理内存使用"""
    def __init__(self, max_size_gb=20):
        self.cache = {}
        self.access_times = {}
        self.data_sizes = {}
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.current_size = 0
        
    def get(self, key, loader_func):
        """获取缓存数据"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        
        # 检查内存限制
        self._check_memory_limit()
        
        # 加载数据
        data = loader_func()
        if data is not None:
            data_size = data.nbytes if hasattr(data, 'nbytes') else 0
            
            self.cache[key] = data
            self.access_times[key] = time.time()
            self.data_sizes[key] = data_size
            self.current_size += data_size
            
        return data
    
    def _check_memory_limit(self):
        """检查内存限制并清理缓存"""
        while self.current_size > self.max_size_bytes and self.cache:
            # 移除最久未访问的项
            oldest_key = min(self.cache.keys(), key=lambda k: self.access_times[k])
            self._remove_item(oldest_key)
    
    def _remove_item(self, key):
        """移除缓存项"""
        if key in self.cache:
            self.current_size -= self.data_sizes[key]
            del self.cache[key]
            del self.access_times[key]
            del self.data_sizes[key]
    
    def cleanup_old_data(self, current_date, window_days):
        """清理超出时间窗口的数据"""
        cutoff_time = time.time() - window_days * 24 * 3600  # 转换为秒
        
        old_keys = [k for k, t in self.access_times.items() if t < cutoff_time]
        for key in old_keys:
            self._remove_item(key)
        
        if old_keys:
            gc.collect()
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
        self.data_sizes.clear()
        self.current_size = 0
        gc.collect()

def _process_pixel_chunk_worker(work_args):
    """工作进程函数：处理一个像素块"""
    try:
        pixel_chunk = work_args['pixel_chunk']
        valid_date_data = work_args['valid_date_data']
        driver_data = work_args['driver_data']
        driver_name = work_args['driver_name']
        temp_file = work_args['temp_file']
        chunk_idx = work_args['chunk_idx']
        
        dataset_count = 0
        processed_pixels = 0
        
        with h5py.File(temp_file, 'w') as h5_file:
            # 处理这个块中的每个像素
            for row, col in pixel_chunk:
                processed_pixels += 1
                
                # 为每个有效日期处理当前像素
                for date, date_info in valid_date_data.items():
                    past_dates = date_info['past_dates']
                    future_dates = date_info['future_dates'] 
                    firms_data = date_info['firms_data']
                    date_str = date_info['date_str']
                    
                    # 获取FIRMS值
                    firms_value = 0
                    if firms_data is not None:
                        firms_value = int(firms_data[row, col])
                    
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
                        
                        # 写入数据集
                        past_dataset_name = f"{date_str}_past_{firms_value}_{row}_{col}"
                        past_dataset = h5_file.create_dataset(
                            past_dataset_name,
                            data=past_array,
                            dtype=np.float32,
                            compression=None,  # 关闭压缩以提高速度
                            shuffle=False
                        )
                        dataset_count += 1
                        
                        future_dataset_name = f"{date_str}_future_{firms_value}_{row}_{col}"
                        future_dataset = h5_file.create_dataset(
                            future_dataset_name,
                            data=future_array,
                            dtype=np.float32,
                            compression=None,  # 关闭压缩以提高速度
                            shuffle=False
                        )
                        dataset_count += 1
        
        return {
            'success': True,
            'chunk_idx': chunk_idx,
            'processed_pixels': processed_pixels,
            'dataset_count': dataset_count
        }
        
    except Exception as e:
                 return {
             'success': False,
             'chunk_idx': chunk_idx,
             'processed_pixels': 0,
             'dataset_count': 0,
             'error': str(e)
         }

def _process_driver_worker(work_args):
    """驱动因素级工作进程函数：处理一个完整的驱动因素"""
    import os
    import sys
    
    # 为每个进程设置独立的标识
    process_id = os.getpid()
    
    try:
        h5_path = work_args['h5_path']
        driver_name = work_args['driver_name']
        all_driver_data = work_args['all_driver_data']
        valid_dates = work_args['valid_dates']
        data_shape = work_args['data_shape']
        geo_reference = work_args['geo_reference']
        year = work_args['year']
        past_days = work_args['past_days']
        future_days = work_args['future_days']
        
        # 调试信息
        print(f"[进程 {process_id}] 开始处理驱动因素: {driver_name}", flush=True)
        print(f"[进程 {process_id}] 输出路径: {h5_path}", flush=True)
        print(f"[进程 {process_id}] 可用驱动因素: {list(all_driver_data.keys())}", flush=True)
        print(f"[进程 {process_id}] 数据形状: {data_shape}", flush=True)
        print(f"[进程 {process_id}] 有效日期数: {len(valid_dates)}", flush=True)
        
        # 检查驱动因素是否存在
        if driver_name not in all_driver_data:
            return {
                'success': False,
                'driver_name': driver_name,
                'error': f'驱动因素 {driver_name} 不在加载的数据中'
            }
        
        # 获取有效像素位置
        height, width = data_shape[:2]
        valid_pixels = _get_valid_pixel_positions_standalone(all_driver_data, height, width)
        print(f"[进程 {process_id}] {driver_name}: 有效像素数 = {len(valid_pixels)}", flush=True)
        
        # 准备有效日期数据
        valid_date_data = {}
        for date in valid_dates:
            past_dates, future_dates = _get_past_future_dates_standalone(date, past_days, future_days)
            if past_dates is None or future_dates is None:
                continue
            
            # 获取FIRMS数据
            firms_data = None
            if 'Firms_Detection_resampled' in all_driver_data:
                firms_data = all_driver_data['Firms_Detection_resampled'].get(date)
            
            valid_date_data[date] = {
                'past_dates': past_dates,
                'future_dates': future_dates,
                'firms_data': firms_data,
                'date_str': date.strftime('%Y%m%d')
            }
        
        print(f"[进程 {process_id}] {driver_name}: 有效日期数 = {len(valid_date_data)}", flush=True)
        
        # 获取当前驱动因素的数据
        driver_data = all_driver_data[driver_name]
        print(f"[进程 {process_id}] {driver_name}: 驱动因素数据日期数 = {len(driver_data)}", flush=True)
        
        # 始终使用单进程处理，并显示独立进度条
        print(f"[进程 {process_id}] {driver_name}: 开始处理数据", flush=True)
        
        # 使用原有的高效版本，但添加进度报告
        result = _create_full_data_ultra_fast_standalone(
            h5_path, driver_name, driver_data, valid_pixels, valid_date_data, 
            data_shape, geo_reference, year, past_days, future_days
        )
        
        print(f"[进程 {process_id}] {driver_name}: 处理完成, 结果={result.get('success', False)}", flush=True)
        return result
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        driver_name = work_args.get('driver_name', 'unknown')
        print(f"[进程 {process_id}] 处理驱动因素 {driver_name} 时出错:", flush=True)
        print(error_detail, flush=True)
        return {
            'success': False,
            'driver_name': driver_name,
            'error': f"{str(e)}\n{error_detail}"
        }

def _get_past_future_dates_standalone(date, past_days, future_days):
    """独立函数：获取过去和未来的日期列表"""
    try:
        # 生成过去的日期
        past_dates = []
        for i in range(past_days, 0, -1):
            past_date = date - timedelta(days=i)
            past_dates.append(past_date)
        
        # 生成未来的日期
        future_dates = []
        for i in range(1, future_days + 1):
            future_date = date + timedelta(days=i)
            future_dates.append(future_date)
        
        # 检查数据长度是否足够
        if len(past_dates) < past_days or len(future_dates) < future_days:
            return None, None
        
        return past_dates, future_dates
    except Exception:
        return None, None

def _get_valid_pixel_positions_standalone(all_driver_data, height, width):
    """独立函数：获取所有有效像素位置"""
    valid_pixels = []
    
    # 获取FIRMS数据的第一个可用日期来确定真正的NoData像素
    firms_data = None
    if 'Firms_Detection_resampled' in all_driver_data:
        for date, data in all_driver_data['Firms_Detection_resampled'].items():
            if data is not None:
                firms_data = data
                if len(firms_data.shape) == 3:
                    firms_data = firms_data[0]  # 取第一个波段
                break
    
    if firms_data is None:
        # 如果没有FIRMS数据，返回所有像素位置
        for row in range(height):
            for col in range(width):
                valid_pixels.append((row, col))
        return valid_pixels
    
    # 筛选有效像素
    nodata_values = [255]  # 只有255是真正的NoData
    
    for row in range(height):
        for col in range(width):
            pixel_value = firms_data[row, col]
            
            # 检查是否为NoData值
            if pixel_value in nodata_values or np.isnan(pixel_value):
                continue
            
            # 这是有效像素（包括0值）
            valid_pixels.append((row, col))
    
    return valid_pixels

def _save_geo_reference_standalone(h5_file, geo_info):
    """独立函数：保存地理参考信息"""
    if geo_info:
        geo_group = h5_file.create_group('geo_reference')
        geo_group.attrs['geotransform'] = geo_info['geotransform']
        geo_group.attrs['projection'] = geo_info['projection']

def _create_full_data_single_standalone(h5_path, driver_name, driver_data, valid_pixels, valid_date_data, 
                                       data_shape, geo_reference, year, past_days, future_days):
    """独立函数：单进程处理驱动因素数据"""
    try:
        dataset_count = 0
        
        with h5py.File(h5_path, 'w') as h5_file:
            # 写入全局属性
            h5_file.attrs['year'] = str(year)
            h5_file.attrs['driver'] = driver_name
            h5_file.attrs['past_days'] = past_days
            h5_file.attrs['future_days'] = future_days
            h5_file.attrs['sampling_type'] = 'pixel_level'
            h5_file.attrs['data_type'] = 'full_year_data_driver_parallel'
            
            if geo_reference:
                _save_geo_reference_standalone(h5_file, geo_reference)
            
            # 处理所有像素
            for row, col in valid_pixels:
                # 为每个有效日期处理当前像素
                for date, date_info in valid_date_data.items():
                    past_dates = date_info['past_dates']
                    future_dates = date_info['future_dates'] 
                    firms_data = date_info['firms_data']
                    date_str = date_info['date_str']
                    
                    # 获取FIRMS值
                    firms_value = 0
                    if firms_data is not None:
                        firms_value = int(firms_data[row, col])
                    
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
                        
                        # 写入数据集
                        past_dataset_name = f"{date_str}_past_{firms_value}_{row}_{col}"
                        h5_file.create_dataset(
                            past_dataset_name,
                            data=past_array,
                            dtype=np.float32,
                            compression=None,
                            shuffle=False
                        )
                        dataset_count += 1
                        
                        future_dataset_name = f"{date_str}_future_{firms_value}_{row}_{col}"
                        h5_file.create_dataset(
                            future_dataset_name,
                            data=future_array,
                            dtype=np.float32,
                            compression=None,
                            shuffle=False
                        )
                        dataset_count += 1
            
            h5_file.attrs['total_datasets'] = dataset_count
            h5_file.attrs['valid_pixels'] = len(valid_pixels)
            h5_file.attrs['total_pixels_processed'] = len(valid_pixels)
        
        return {
            'success': True,
            'driver_name': driver_name,
            'dataset_count': dataset_count,
            'processed_pixels': len(valid_pixels)
        }
        
    except Exception as e:
        return {
            'success': False,
            'driver_name': driver_name,
            'error': str(e)
        }

def _create_full_data_single_standalone_with_progress(h5_path, driver_name, driver_data, valid_pixels, valid_date_data, 
                                                    data_shape, geo_reference, year, past_days, future_days):
    """独立函数：单进程处理驱动因素数据，带独立进度条 - 高效版本"""
    import os
    import sys
    import gc
    
    process_id = os.getpid()
    
    try:
        from tqdm import tqdm
        
        dataset_count = 0
        
        print(f"[进程 {process_id}] {driver_name}: 开始创建H5文件", flush=True)
        
        with h5py.File(h5_path, 'w') as h5_file:
            # 写入全局属性
            h5_file.attrs['year'] = str(year)
            h5_file.attrs['driver'] = driver_name
            h5_file.attrs['past_days'] = past_days
            h5_file.attrs['future_days'] = future_days
            h5_file.attrs['sampling_type'] = 'pixel_level'
            h5_file.attrs['data_type'] = 'full_year_data_driver_parallel'
            
            if geo_reference:
                _save_geo_reference_standalone(h5_file, geo_reference)
            
            # 创建进度条 - 按像素计数
            total_pixels = len(valid_pixels)
            
            print(f"[进程 {process_id}] {driver_name}: 像素数 = {total_pixels}, 日期数 = {len(valid_date_data)}", flush=True)
            
            # 使用进程ID创建唯一的进度条位置
            position = process_id % 10
            
            # 使用像素级进度条，但处理逻辑优化
            pbar = tqdm(total=total_pixels, 
                       desc=f"{driver_name}", 
                       position=position,
                       leave=True,
                       ncols=80,
                       unit="像素",
                       file=sys.stdout)
            
            try:
                print(f"[进程 {process_id}] {driver_name}: 开始高效处理", flush=True)
                
                # 使用类似ultra_fast的高效逻辑
                for pixel_idx, (row, col) in enumerate(valid_pixels):
                    
                    # 为每个有效日期处理当前像素
                    for date, date_info in valid_date_data.items():
                        past_dates = date_info['past_dates']
                        future_dates = date_info['future_dates'] 
                        firms_data = date_info['firms_data']
                        date_str = date_info['date_str']
                        
                        # 获取FIRMS值
                        firms_value = 0
                        if firms_data is not None:
                            firms_value = int(firms_data[row, col])
                        
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
                            
                            # 写入数据集 - 关闭压缩以提高速度
                            past_dataset_name = f"{date_str}_past_{firms_value}_{row}_{col}"
                            h5_file.create_dataset(
                                past_dataset_name,
                                data=past_array,
                                dtype=np.float32,
                                compression=None,  # 关闭压缩
                                shuffle=False
                            )
                            dataset_count += 1
                            
                            future_dataset_name = f"{date_str}_future_{firms_value}_{row}_{col}"
                            h5_file.create_dataset(
                                future_dataset_name,
                                data=future_array,
                                dtype=np.float32,
                                compression=None,  # 关闭压缩
                                shuffle=False
                            )
                            dataset_count += 1
                    
                    # 更新进度条
                    pbar.update(1)
                    
                    # 定期垃圾回收和状态报告
                    if (pixel_idx + 1) % 1000 == 0:
                        progress = (pixel_idx + 1) / len(valid_pixels) * 100
                        print(f"[进程 {process_id}] {driver_name}: 进度 {progress:.1f}% ({pixel_idx + 1}/{len(valid_pixels)}), 数据集数: {dataset_count}", flush=True)
                        
                    if (pixel_idx + 1) % 5000 == 0:
                        gc.collect()
                
            finally:
                pbar.close()
            
            h5_file.attrs['total_datasets'] = dataset_count
            h5_file.attrs['valid_pixels'] = len(valid_pixels)
            h5_file.attrs['total_pixels_processed'] = len(valid_pixels)
        
        print(f"[进程 {process_id}] {driver_name}: 成功完成，数据集数={dataset_count}", flush=True)
        
        return {
            'success': True,
            'driver_name': driver_name,
            'dataset_count': dataset_count,
            'processed_pixels': len(valid_pixels)
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[进程 {process_id}] {driver_name}: 处理失败", flush=True)
        print(error_detail, flush=True)
        return {
            'success': False,
            'driver_name': driver_name,
            'error': f"{str(e)}\n{error_detail}"
        }

def _create_full_data_parallel_standalone(h5_path, driver_name, driver_data, valid_pixels, valid_date_data, 
                                        data_shape, geo_reference, year, past_days, future_days, n_processes):
    """独立函数：并行处理驱动因素数据"""
    try:
        # 将像素分割给多个进程
        pixels_per_process = len(valid_pixels) // n_processes
        if len(valid_pixels) % n_processes != 0:
            pixels_per_process += 1
        
        pixel_chunks = []
        for i in range(0, len(valid_pixels), pixels_per_process):
            chunk = valid_pixels[i:i + pixels_per_process]
            pixel_chunks.append(chunk)
        
        # 创建临时文件来存储每个进程的结果
        temp_dir = tempfile.mkdtemp()
        temp_files = []
        
        try:
            # 准备工作参数
            work_args = []
            for idx, pixel_chunk in enumerate(pixel_chunks):
                temp_file = os.path.join(temp_dir, f"chunk_{idx}.h5")
                temp_files.append(temp_file)
                
                work_args.append({
                    'pixel_chunk': pixel_chunk,
                    'valid_date_data': valid_date_data,
                    'driver_data': driver_data,
                    'driver_name': driver_name,
                    'temp_file': temp_file,
                    'chunk_idx': idx
                })
            
            # 并行处理
            with Pool(processes=n_processes) as pool:
                results = list(pool.imap(_process_pixel_chunk_worker, work_args))
            
            # 合并结果到最终文件
            total_datasets = 0
            
            with h5py.File(h5_path, 'w') as final_h5:
                # 写入全局属性
                final_h5.attrs['year'] = str(year)
                final_h5.attrs['driver'] = driver_name
                final_h5.attrs['past_days'] = past_days
                final_h5.attrs['future_days'] = future_days
                final_h5.attrs['sampling_type'] = 'pixel_level'
                final_h5.attrs['data_type'] = 'full_year_data_driver_parallel'
                
                if geo_reference:
                    _save_geo_reference_standalone(final_h5, geo_reference)
                
                # 合并所有临时文件
                for temp_file, result in zip(temp_files, results):
                    if os.path.exists(temp_file) and result['success']:
                        with h5py.File(temp_file, 'r') as temp_h5:
                            # 复制所有数据集
                            for dataset_name in temp_h5.keys():
                                temp_h5.copy(dataset_name, final_h5)
                                total_datasets += 1
                
                final_h5.attrs['total_datasets'] = total_datasets
                final_h5.attrs['valid_pixels'] = len(valid_pixels)
                final_h5.attrs['total_pixels_processed'] = len(valid_pixels)
                
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            os.rmdir(temp_dir)
        
        # 统计结果
        successful_chunks = sum(1 for r in results if r['success'])
        total_processed_pixels = sum(r['processed_pixels'] for r in results)
        
        return {
            'success': True,
            'driver_name': driver_name,
            'dataset_count': total_datasets,
            'processed_pixels': total_processed_pixels,
            'successful_chunks': successful_chunks
        }
        
    except Exception as e:
        return {
            'success': False,
            'driver_name': driver_name,
            'error': str(e)
        }

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
    def __init__(self, data_dir, output_dir, sample_past_days=365, full_past_days=30, future_days=30, negative_ratio=1.0, 
                 sample_years=None, 
                 use_parallel=True, n_processes=32, time_windows_mode=False, 
                 window_start_date=None, window_end_date=None, window_shift_days=1):
        """
        初始化像素采样器
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            sample_past_days: 抽样数据的历史天数
            full_past_days: 完整数据的历史天数
            future_days: 未来天数
            negative_ratio: 负样本比例

                          sample_years: 抽样数据年份列表
            use_parallel: 是否使用并行处理
            n_processes: 并行进程数
            time_windows_mode: 是否启用时间窗口模式
            window_start_date: 窗口开始日期 (datetime对象)
            window_end_date: 窗口结束日期 (datetime对象)
            window_shift_days: 窗口滑动天数
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.sample_past_days = sample_past_days
        self.full_past_days = full_past_days
        self.future_days = future_days
        self.negative_ratio = negative_ratio
        self.sample_years = sample_years
        self.use_parallel = use_parallel
        self.n_processes = n_processes
        
        # 新增时间窗口模式参数
        self.time_windows_mode = time_windows_mode
        self.window_start_date = window_start_date
        self.window_end_date = window_end_date
        self.window_shift_days = window_shift_days
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取驱动因素目录
        self.driver_dirs = self._get_driver_directories()
        if not self.driver_dirs:
            raise ValueError("没有找到任何驱动因素目录")
        
        logger.info(f"找到 {len(self.driver_dirs)} 个驱动因素目录")
        for name, path in self.driver_dirs.items():
            logger.info(f"  {name}: {path}")
        
        # 初始化缓存
        self.data_cache = DataCache(max_memory_gb=20)
        
        # 时间窗口模式日志
        if self.time_windows_mode:
            logger.info(f"启用时间窗口模式:")
            logger.info(f"  窗口开始日期: {window_start_date}")
            logger.info(f"  窗口结束日期: {window_end_date}")
            logger.info(f"  窗口滑动天数: {window_shift_days}")
        
        logger.info(f"抽样数据配置 - 历史天数: {self.sample_past_days}, 未来天数: {self.future_days}, 负样本比例: {self.negative_ratio}")
        logger.info(f"完整数据配置 - 历史天数: {self.full_past_days}, 未来天数: {self.future_days}")
        
        # 并行处理配置检查
        if self.use_parallel:
            cpu_count = multiprocessing.cpu_count()
            if self.n_processes > cpu_count:
                logger.warning(f"请求的进程数 {self.n_processes} 超过了CPU核心数 {cpu_count}，建议调整")
            else:
                logger.info(f"并行处理配置: {self.n_processes} 进程 (可用CPU: {cpu_count})")
        else:
            logger.info("使用单进程处理")
            
        if self.sample_years:
            logger.info(f"指定样本数据年份: {self.sample_years}")
        if self.full_years:
            logger.info(f"指定完整数据年份: {self.full_years}")
        
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
    


    @lru_cache(maxsize=1000)
    def _get_date_from_filename(self, filename):
        """从文件名提取日期"""
        try:
            # 移除文件扩展名
            filename_no_ext = filename.replace('.tif', '').replace('.TIF', '')
            # 假设文件名格式为 *_YYYY_MM_DD
            parts = filename_no_ext.split('_')
            for i in range(len(parts) - 2):
                if (len(parts[i]) == 4 and parts[i].isdigit() and
                    len(parts[i+1]) == 2 and parts[i+1].isdigit() and
                    len(parts[i+2]) == 2 and parts[i+2].isdigit()):
                    return datetime(int(parts[i]), int(parts[i+1]), int(parts[i+2]))
            return None
        except:
            return None

    def _get_past_future_dates(self, date, available_dates=None, past_days=None):
        """获取过去和未来的日期列表，并验证数据可用性
        
        Args:
            date: 中心日期
            available_dates: 可用日期集合，用于验证数据存在性
            past_days: 历史天数，如果为None则使用sample_past_days
        
        Returns:
            tuple: (past_dates, future_dates) 或 (None, None) 如果数据不完整
        """
        if past_days is None:
            past_days = self.sample_past_days  # 默认使用抽样数据的天数
            
        try:
            # 生成过去的日期
            past_dates = []
            for i in range(past_days, 0, -1):
                past_date = date - timedelta(days=i)
                past_dates.append(past_date)
            
            # 生成未来的日期
            future_dates = []
            for i in range(1, self.future_days + 1):
                future_date = date + timedelta(days=i)
                future_dates.append(future_date)
            
            # 检查数据长度是否足够
            if len(past_dates) < past_days or len(future_dates) < self.future_days:
                return None, None
            
            # 如果提供了available_dates，检查所有日期是否都可用
            if available_dates is not None:
                all_required_dates = past_dates + [date] + future_dates
                missing_dates = [d for d in all_required_dates if d not in available_dates]
                if missing_dates:
                    return None, None
                
            return past_dates, future_dates
            
        except Exception as e:
            logger.error(f"生成日期列表时出错: {str(e)}")
            return None, None




    def _sample_pixels_from_firms(self, firms_data):
        """从FIRMS数据中采样像素并获取其具体值
        
        采样策略：
        - 正样本 (FIRMS > 0): 全部采样，保留所有正样本
        - 负样本 (FIRMS = 0): 按negative_ratio比例采样
        - 背景值 (FIRMS = 255): 不参与采样
        
        Args:
            firms_data: FIRMS数据数组
            
        Returns:
            list: 像素事件列表 [(pixel_coord, firms_value), ...]
                其中firms_value是该像素的具体FIRMS值（如0, 1, 3, 20等）
        """
        if len(firms_data.shape) == 3:
            # 多波段数据，取第一个波段
            firms_data = firms_data[0]
        
        # 获取所有唯一的FIRMS值
        unique_values = np.unique(firms_data)
        logger.debug(f"FIRMS数据中的唯一值: {unique_values}")
        
        # 定义NoData值，不应参与样本生成
        nodata_values = [255]  # FIRMS的NoData值
        
        # 过滤掉NoData值
        valid_unique_values = [val for val in unique_values if val not in nodata_values and not np.isnan(val)]
        logger.debug(f"FIRMS有效唯一值（排除NoData）: {valid_unique_values}")
        
        pixel_events = []
        
        # 分别处理正样本和负样本
        positive_pixels = []  # 正样本 (FIRMS > 0)
        negative_pixels = []  # 负样本 (FIRMS = 0)
        
        for firms_value in valid_unique_values:
            # 找到该值的所有像素位置
            value_mask = firms_data == firms_value
            coords = np.where(value_mask)
            pixels = list(zip(coords[0], coords[1]))
            
            if firms_value == 0:
                # 负样本：暂存，稍后按比例采样
                for pixel in pixels:
                    negative_pixels.append((pixel, firms_value))
            else:
                # 正样本：全部保留
                for pixel in pixels:
                    positive_pixels.append((pixel, firms_value))
        
        # 计算负样本的采样数量
        total_positive_count = len(positive_pixels)
        target_negative_count = int(total_positive_count * self.negative_ratio)
        
        # 对负样本进行采样
        if len(negative_pixels) > target_negative_count:
            sampled_negative_pixels = random.sample(negative_pixels, target_negative_count)
        else:
            sampled_negative_pixels = negative_pixels
        
        # 合并正样本和采样后的负样本
        pixel_events = positive_pixels + sampled_negative_pixels
        
        # logger.info(f"采样结果 - 正样本: {len(positive_pixels)}, 负样本: {len(sampled_negative_pixels)}/{len(negative_pixels)} (目标:{target_negative_count}), 总计: {len(pixel_events)}")
        
        return pixel_events

    def _process_year_pixels_with_adjacent(self, current_year_files, all_available_files, target_year):
        """处理一年的像素采样数据，包含相邻年份的数据
        
        优化后的流程：
        1. 先收集当前年份所有正负样本像素位置
        2. 再检查这些样本的时间序列完整性
        3. 最后只加载需要的数据
        
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
            date = self._get_date_from_filename(filename)
            if date:
                all_dates.append(date)
                all_date_to_file[date] = file_path
        
        all_dates.sort()
        logger.info(f"全局可用日期范围: {all_dates[0]} 到 {all_dates[-1]}，共 {len(all_dates)} 个日期")
        
        # 获取当前年份的日期
        current_year_dates = []
        for file_path in current_year_files:
            filename = os.path.basename(file_path)
            date = self._get_date_from_filename(filename)
            if date:
                current_year_dates.append(date)
        
        current_year_dates.sort()
        
        # ========== 步骤1：先收集当前年份的所有正负样本 ========== #
        logger.info(f"步骤1：从 {len(current_year_dates)} 个日期收集所有正负样本...")
        all_pixel_candidates = []
        
        for date in tqdm(current_year_dates, desc=f"收集样本 {target_year}"):
            # 加载FIRMS数据
            firms_data = self._load_driver_data_cached('Firms_Detection_resampled', date)
            if firms_data is None:
                continue
            
            # 采样像素并获取FIRMS值
            date_pixel_events = self._sample_pixels_from_firms(firms_data)
            
            # 添加到候选列表，格式：(pixel_coord, date, firms_value)
            for pixel_coord, firms_value in date_pixel_events:
                all_pixel_candidates.append((pixel_coord, date, firms_value))
        
        logger.info(f"步骤1完成：收集到 {len(all_pixel_candidates)} 个候选样本")
        
        if not all_pixel_candidates:
            logger.warning(f"年份 {target_year}: 没有找到任何正负样本")
            return
        
        # ========== 步骤2：检查样本的时间序列完整性 ========== #
        logger.info(f"步骤2：检查 {len(all_pixel_candidates)} 个样本的时间序列完整性...")
        valid_pixel_events = []
        
        # 将all_date_to_file转换为集合以提高查找效率
        available_dates_set = set(all_date_to_file.keys())
        
        for pixel_coord, event_date, firms_value in tqdm(all_pixel_candidates, desc=f"检查完整性 {target_year}"):
            # 获取该事件需要的时间序列日期，同时验证数据可用性
            past_dates, future_dates = self._get_past_future_dates(event_date, available_dates_set, past_days=self.sample_past_days)
            
            if past_dates is None or future_dates is None:
                continue
            
            # 如果通过了可用性检查，直接添加到有效样本
            valid_pixel_events.append((pixel_coord, event_date, firms_value))
        
        logger.info(f"步骤2完成：找到 {len(valid_pixel_events)} 个时间序列完整的样本")
        
        if not valid_pixel_events:
            logger.warning(f"年份 {target_year}: 没有时间序列完整的样本")
            return
        
        # ========== 步骤3：只加载需要的数据并生成时间序列 ========== #
        logger.info(f"步骤3：为 {len(valid_pixel_events)} 个有效样本生成时间序列...")
        self._process_driver_data_batch(self.driver_dirs, all_dates, target_year, valid_pixel_events)

    def _process_driver_data_batch(self, driver_dirs, all_dates, year, pixel_events):
        """批量处理所有驱动因素数据，支持跨年份查询"""
        logger.info(f"为年份 {year} 创建 {len(pixel_events)} 个像素样本的时间序列")
        
        # 预先收集所有需要的日期
        required_dates = set()
        for pixel_coord, event_date, firms_value in pixel_events:
            past_dates, future_dates = self._get_past_future_dates(event_date, past_days=self.sample_past_days)
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
            past_dates, future_dates = self._get_past_future_dates(event_date, past_days=self.sample_past_days)
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
                date = self._get_date_from_filename(filename)
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
        """处理所有年份的数据，根据配置决定处理什么"""
        # 检查是否是时间窗口模式
        if self.time_windows_mode:
            return self._process_time_windows_data()
        
        # 原有的按年份处理逻辑
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
            date = self._get_date_from_filename(filename)  # 使用新的日期提取方法
            if date:
                year_groups[date.year].append(file_path)
        
        # 确定要处理的年份
        if self.sample_years:
            # 用户指定了样本数据年份
            sample_years_to_process = [year for year in self.sample_years if year in available_years and year in year_groups]
        else:
            # 默认处理所有年份的样本数据
            sample_years_to_process = [year for year in available_years if year in year_groups]
        
        # 合并所有需要处理的年份
        all_years_to_process = set(sample_years_to_process)
        
        logger.info(f"样本数据年份: {sorted(sample_years_to_process)}")
        logger.info(f"完整数据年份: {sorted(all_years_to_process)}")
        logger.info(f"开始处理 {len(all_years_to_process)} 个年份")
        
        # 处理每个年份
        for year in sorted(all_years_to_process):
            logger.info(f"=" * 50)
            logger.info(f"开始处理年份: {year}")
            
            # 检查是否需要处理样本数据
            need_sample_data = year in sample_years_to_process
            
            # 检查文件是否已存在
            sample_files_exist = self._check_sample_files_exist(year) if need_sample_data else True
            
            if sample_files_exist:
                logger.info(f"年份 {year} 的所有需要的输出文件已存在，跳过处理")
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
            
            # 处理样本数据
            if need_sample_data and not sample_files_exist:
                logger.info(f"处理年份 {year} 的样本数据...")
                self._process_year_pixels_with_adjacent(sorted(year_groups[year]), sorted(all_year_files), year)
            elif need_sample_data:
                logger.info(f"年份 {year} 的样本数据已存在，跳过")
            
            # 清理缓存
            self.data_cache.clear()
            logger.info(f"年份 {year} 处理完成")
            logger.info(f"=" * 50)

    def _process_time_windows_data(self):
        """处理时间窗口模式的数据生成"""
        logger.info("开始处理时间窗口模式的数据...")
        
        if not self.window_start_date or not self.window_end_date:
            logger.error("时间窗口模式需要指定window_start_date和window_end_date")
            return
        
        # 计算所有时间窗口
        time_windows = self._generate_time_windows()
        logger.info(f"生成 {len(time_windows)} 个时间窗口")
        
        # 构建全局日期映射（包含所需的所有年份数据）
        all_date_to_file = self._build_global_date_mapping(time_windows)
        
        # 获取数据形状和地理参考信息
        geo_reference, data_shape = self._get_data_info_fast(all_date_to_file)
        if geo_reference is None or data_shape is None:
            logger.error("无法获取数据信息")
            return
        
        logger.info(f"数据形状: {data_shape}")
        
        # 处理每个时间窗口
        successful_windows = 0
        for i, (window_start, window_end) in enumerate(time_windows):
            window_name = f"{window_start.strftime('%Y_%m%d')}_{window_end.strftime('%m%d')}"
            logger.info(f"处理时间窗口 {i+1}/{len(time_windows)}: {window_name}")
            
            # 检查是否已存在
            if self._check_time_window_files_exist(window_name):
                logger.info(f"时间窗口 {window_name} 的文件已存在，跳过")
                continue
            
            try:
                # 处理这个时间窗口
                result = self._process_single_time_window(
                    window_start, window_end, window_name, 
                    all_date_to_file, data_shape, geo_reference
                )
                
                if result:
                    successful_windows += 1
                    logger.info(f"时间窗口 {window_name} 处理成功")
                else:
                    logger.error(f"时间窗口 {window_name} 处理失败")
                    
            except Exception as e:
                logger.error(f"处理时间窗口 {window_name} 时出错: {str(e)}")
            
            # 清理缓存
            self.data_cache.clear()
        
        logger.info(f"时间窗口处理完成: {successful_windows}/{len(time_windows)} 个成功")

    def _generate_time_windows(self):
        """生成所有时间窗口"""
        windows = []
        current_start = self.window_start_date
        
        while current_start <= self.window_end_date:
            # 计算当前窗口的结束日期（固定7天窗口）
            window_end = current_start + timedelta(days=6)  # 包括首尾共7天
            windows.append((current_start, window_end))
            
            # 移动到下一个窗口
            current_start += timedelta(days=self.window_shift_days)
        
        logger.info(f"生成的时间窗口:")
        for i, (start, end) in enumerate(windows):
            logger.info(f"  {i+1}: {start.strftime('%Y-%m-%d')} 到 {end.strftime('%Y-%m-%d')}")
        
        return windows

    def _build_global_date_mapping(self, time_windows):
        """构建全局日期到文件的映射"""
        logger.info("构建全局日期到文件映射...")
        
        # 计算所有需要的日期
        all_required_dates = set()
        
        for window_start, window_end in time_windows:
            # 计算这个窗口需要的所有日期
            past_dates, future_dates = self._get_past_future_dates(
                window_start, past_days=self.full_past_days
            )
            
            if past_dates:
                all_required_dates.update(past_dates)
            
            # 添加未来日期（窗口内的所有日期）
            current_date = window_start
            while current_date <= window_end:
                all_required_dates.add(current_date)
                current_date += timedelta(days=1)
        
        logger.info(f"总共需要 {len(all_required_dates)} 个日期的数据")
        
        # 构建日期到文件的映射
        all_date_to_file = {}
        missing_dates = []
        
        for date in all_required_dates:
            # 为每个驱动因素查找对应的文件
            found_any = False
            for driver_name, driver_dir in self.driver_dirs.items():
                date_str = date.strftime('%Y_%m_%d')
                pattern = os.path.join(driver_dir, f'*{date_str}.tif')
                matching_files = glob.glob(pattern)
                
                if matching_files:
                    all_date_to_file[date] = matching_files[0]  # 使用第一个匹配的文件作为代表
                    found_any = True
                    break
            
            if not found_any:
                missing_dates.append(date)
        
        if missing_dates:
            logger.warning(f"缺少 {len(missing_dates)} 个日期的数据文件")
            for date in missing_dates[:10]:  # 只显示前10个
                logger.warning(f"  缺少日期: {date.strftime('%Y-%m-%d')}")
            if len(missing_dates) > 10:
                logger.warning(f"  ... 还有 {len(missing_dates) - 10} 个日期")
        
        logger.info(f"成功映射 {len(all_date_to_file)} 个日期")
        return all_date_to_file

    def _check_time_window_files_exist(self, window_name):
        """检查时间窗口的所有输出文件是否已存在"""
        for driver_name in self.driver_dirs.keys():
            h5_path = os.path.join(self.output_dir, f'{window_name}_{driver_name}_full.h5')
            if not os.path.exists(h5_path):
                return False
        return True

    def _process_single_time_window(self, window_start, window_end, window_name, 
                                   all_date_to_file, data_shape, geo_reference):
        """处理单个时间窗口"""
        logger.info(f"开始处理时间窗口: {window_start.strftime('%Y-%m-%d')} 到 {window_end.strftime('%Y-%m-%d')}")
        
        # 构建这个窗口的有效日期数据
        valid_date_data = self._build_window_date_data(window_start, window_end, all_date_to_file)
        
        if not valid_date_data:
            logger.error(f"时间窗口 {window_name} 没有有效的日期数据")
            return False
        
        # 为每个驱动因素生成数据
        successful_drivers = 0
        for driver_name in self.driver_dirs.keys():
            h5_path = os.path.join(self.output_dir, f'{window_name}_{driver_name}_full.h5')
            
            if os.path.exists(h5_path):
                logger.info(f"文件已存在，跳过: {h5_path}")
                successful_drivers += 1
                continue
            
            try:
                # 处理这个驱动因素的数据
                result = self._create_window_driver_data(
                    h5_path, driver_name, valid_date_data, 
                    data_shape, geo_reference, window_name
                )
                
                if result:
                    successful_drivers += 1
                    logger.info(f"驱动因素 {driver_name} 处理成功")
                else:
                    logger.error(f"驱动因素 {driver_name} 处理失败")
                    
            except Exception as e:
                logger.error(f"处理驱动因素 {driver_name} 时出错: {str(e)}")
        
        success = successful_drivers == len(self.driver_dirs)
        logger.info(f"时间窗口 {window_name} 处理完成: {successful_drivers}/{len(self.driver_dirs)} 个驱动因素成功")
        return success

    def _build_window_date_data(self, window_start, window_end, all_date_to_file):
        """构建时间窗口的日期数据结构"""
        # 计算过去日期
        past_dates, _ = self._get_past_future_dates(window_start, past_days=self.full_past_days)
        
        # 构建未来日期（窗口内的所有日期）
        future_dates = []
        current_date = window_start
        while current_date <= window_end:
            future_dates.append(current_date)
            current_date += timedelta(days=1)
        
        # 构建有效日期数据结构（与原有格式兼容）
        valid_date_data = {}
        
        # 我们将整个窗口看作一个"事件"，起点是window_start
        date_str = window_start.strftime('%Y_%m_%d')
        valid_date_data[window_start] = {
            'past_dates': past_dates,
            'future_dates': future_dates,
            'firms_data': None,  # 将在处理时加载
            'date_str': date_str
        }
        
        logger.info(f"窗口日期数据: past_days={len(past_dates)}, future_days={len(future_dates)}")
        return valid_date_data

    def _create_window_driver_data(self, h5_path, driver_name, valid_date_data, 
                                  data_shape, geo_reference, window_name):
        """为时间窗口创建单个驱动因素的数据"""
        logger.info(f"为驱动因素 {driver_name} 创建窗口数据: {window_name}")
        
        # 加载这个驱动因素的所有需要数据
        driver_data = self._load_window_driver_data(driver_name, valid_date_data)
        
        if not driver_data:
            logger.error(f"无法加载驱动因素 {driver_name} 的数据")
            return False
        
        # 获取有效像素位置（使用FIRMS数据）
        valid_pixels = self._get_window_valid_pixels(valid_date_data, data_shape)
        
        if not valid_pixels:
            logger.error(f"没有找到有效像素")
            return False
        
        # 使用现有的数据创建函数
        try:
            result = _create_full_data_single_standalone(
                h5_path, driver_name, driver_data, valid_pixels, valid_date_data,
                data_shape, geo_reference, window_name, self.full_past_days, 
                len(valid_date_data[list(valid_date_data.keys())[0]]['future_dates'])
            )
            return result.get('success', False)
            
        except Exception as e:
            logger.error(f"创建数据时出错: {str(e)}")
            return False

    def _load_window_driver_data(self, driver_name, valid_date_data):
        """加载时间窗口的驱动因素数据"""
        driver_dir = self.driver_dirs[driver_name]
        driver_data = {}
        
        # 收集所有需要的日期
        all_dates = set()
        for date_info in valid_date_data.values():
            all_dates.update(date_info['past_dates'])
            all_dates.update(date_info['future_dates'])
        
        # 加载所有日期的数据
        for date in all_dates:
            try:
                data = self._load_driver_data(driver_dir, date)
                if data is not None:
                    driver_data[date] = data
            except Exception as e:
                logger.warning(f"加载数据失败: {driver_name}, {date}, 错误: {str(e)}")
        
        logger.info(f"驱动因素 {driver_name} 加载了 {len(driver_data)}/{len(all_dates)} 个日期的数据")
        return driver_data

    def _get_window_valid_pixels(self, valid_date_data, data_shape):
        """获取时间窗口的有效像素位置"""
        # 加载窗口起始日期的FIRMS数据来确定有效像素
        window_start = list(valid_date_data.keys())[0]
        
        try:
            # 构建FIRMS文件路径
            date_str = window_start.strftime('%Y_%m_%d')
            firms_dir = os.path.join(self.data_dir, 'Firms_Detection_resampled')
            firms_pattern = os.path.join(firms_dir, f'*{date_str}.tif')
            firms_files = glob.glob(firms_pattern)
            
            if firms_files:
                firms_data, _ = self._load_single_file_with_gdal(firms_files[0])
                if firms_data is not None:
                    # 收集所有有效像素（与完整数据处理一致）
                    pixel_events = self._collect_all_valid_pixels(firms_data)
                    valid_pixels = [event[0] for event in pixel_events]  # 只要坐标
                    
                    logger.info(f"找到 {len(valid_pixels)} 个有效像素")
                    return valid_pixels
            
            logger.error(f"无法加载FIRMS数据: {date_str}")
            return []
            
        except Exception as e:
            logger.error(f"获取有效像素时出错: {str(e)}")
            return []

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
            date = self._get_date_from_filename(filename)
            if date:
                years.add(date.year)
        
        available_years = sorted(years)
        if available_years:
            logger.info(f"找到 {len(available_years)} 个年份: {min(available_years)}-{max(available_years)}")
        else:
            logger.warning("没有找到任何有效年份")
        return available_years
    
    def _check_sample_files_exist(self, year):
        """检查指定年份的样本数据文件是否已存在"""
        driver_names = list(self.driver_dirs.keys())
        for driver_name in driver_names:
            h5_path = os.path.join(self.output_dir, f'{year}_{driver_name}.h5')
            if not os.path.exists(h5_path):
                return False
        return True
    
        



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

    def _process_full_year_data_REMOVED(self, current_year_files, all_available_files, target_year):
        """恢复一年一个文件的完整年份数据处理 - 按驱动因素分别生成
        
        处理方式：
        1. 为每个驱动因素生成一个H5文件：output_dir/YYYY_DriverName_full.h5
        2. 每个H5文件包含该驱动因素的所有有效像素时间序列
        3. 使用past_days=30天的历史数据
        
        Args:
            current_year_files: 当前年份的文件列表
            all_available_files: 所有可用的文件列表（包含相邻年份）
            target_year: 目标年份
        """
        logger.info(f"开始按驱动因素处理年份 {target_year} 的完整数据，past_days={self.full_past_days}")
        
        # 直接使用输出目录，不创建年份子目录
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建全局日期映射
        all_date_to_file = {}
        for file_path in all_available_files:
            filename = os.path.basename(file_path)
            date = self._get_date_from_filename(filename)
            if date:
                all_date_to_file[date] = file_path
        
        # 获取当前年份的有效日期
        current_year_dates = []
        for file_path in current_year_files:
            filename = os.path.basename(file_path)
            date = self._get_date_from_filename(filename)
            if date:
                current_year_dates.append(date)
        
        current_year_dates.sort()
        logger.info(f"当前年份有效日期: {len(current_year_dates)} 个")
        
        # 预处理：确定数据形状和地理参考信息
        geo_reference, data_shape = self._get_data_info_fast(all_date_to_file)
        if geo_reference is None or data_shape is None:
            logger.error(f"无法获取年份 {target_year} 的数据信息")
            return
        
        logger.info(f"数据形状: {data_shape}")
        
        # 使用传统的按驱动因素处理模式
        if self.use_parallel:
            self._process_full_data_parallel_by_driver(
                all_date_to_file, current_year_dates, target_year, data_shape, geo_reference
            )
        else:
            # 单进程版本，用于调试
            self._process_full_data_sequential_by_driver(
                all_date_to_file, current_year_dates, target_year, data_shape, geo_reference
            )

    def _collect_all_valid_pixels(self, firms_data):
        """收集所有非NoData像素（用于完整数据集，不进行任何采样）
        
        与抽样不同，这里收集所有有效像素：
        - 所有非NoData像素 (FIRMS ≠ 255) 都被收集
        - 包括正样本 (FIRMS > 0) 和负样本 (FIRMS = 0)
        - 不进行任何比例采样
        
        Args:
            firms_data: FIRMS数据数组
            
        Returns:
            list: 像素事件列表 [(pixel_coord, firms_value), ...]
        """
        if len(firms_data.shape) == 3:
            # 多波段数据，取第一个波段
            firms_data = firms_data[0]
        
        # 定义NoData值
        nodata_values = [255]  # FIRMS的NoData值
        
        pixel_events = []
        
        # 获取所有非NoData像素的位置和值
        valid_mask = np.isin(firms_data, nodata_values, invert=True) & ~np.isnan(firms_data)
        valid_coords = np.where(valid_mask)
        
        # 提取所有有效像素的坐标和FIRMS值
        for i in range(len(valid_coords[0])):
            row, col = valid_coords[0][i], valid_coords[1][i]
            firms_value = firms_data[row, col]
            pixel_events.append(((row, col), firms_value))
        
        logger.debug(f"收集到 {len(pixel_events)} 个非NoData像素")
        
        return pixel_events

    def _get_data_info_fast(self, all_date_to_file):
        """快速获取数据形状和地理参考信息"""
        try:
            # 找到第一个可用的FIRMS文件来获取基本信息
            first_date = min(all_date_to_file.keys())
            firms_file = None
            
            # 构建对应的FIRMS文件路径
            date_str = first_date.strftime('%Y_%m_%d')
            firms_dir = os.path.join(self.data_dir, 'Firms_Detection_resampled')
            firms_pattern = os.path.join(firms_dir, f'*{date_str}.tif')
            firms_files = glob.glob(firms_pattern)
            
            if firms_files:
                firms_file = firms_files[0]
                data, geo_info = self._load_single_file_with_gdal(firms_file)
                if data is not None and geo_info is not None:
                    data_shape = data.shape if len(data.shape) == 2 else data.shape[1:]
                    return geo_info, data_shape
            
            return None, None
            
        except Exception as e:
            logger.error(f"获取数据信息时出错: {str(e)}")
            return None, None

    def _process_full_data_parallel_by_date_preloaded(self, all_driver_data, valid_processing_dates, 
                                                    target_year, data_shape, geo_reference, year_output_dir):
        """使用预加载数据按日期并行处理完整数据"""
        logger.info(f"开始使用预加载数据按日期并行处理 {len(valid_processing_dates)} 个日期")
        
        # 准备工作参数
        work_args = []
        for date in valid_processing_dates:
            date_str = date.strftime('%Y_%m_%d')
            h5_path = os.path.join(year_output_dir, f'{date_str}_full.h5')
            
            work_args.append({
                'date': date,
                'h5_path': h5_path,
                'all_driver_data': all_driver_data,  # 传递预加载的数据
                'target_year': target_year,
                'data_shape': data_shape,
                'geo_reference': geo_reference,
                'past_days': self.full_past_days,
                'future_days': self.future_days,
                'data_dir': self.data_dir
            })
        
        logger.info(f"需要处理 {len(work_args)} 个日期")
        
        # 并行处理 - 每个进程处理一个日期，使用预加载的数据
        with ProcessPoolExecutor(max_workers=min(self.n_processes, len(work_args))) as executor:
            results = list(executor.map(_process_single_date_preloaded_worker, work_args))
        
        # 统计结果
        successful = sum(1 for r in results if r.get('success', False))
        logger.info(f"使用预加载数据按日期并行处理完成: {successful}/{len(work_args)} 个日期成功")
        
        for result in results:
            if result.get('success', False):
                date_str = result.get('date_str', 'unknown')
                logger.info(f"日期 {date_str}: 数据集 {result.get('dataset_count', 0)}, 像素 {result.get('processed_pixels', 0)}")
            else:
                date_str = result.get('date_str', 'unknown')
                logger.error(f"日期 {date_str} 处理失败: {result.get('error', 'unknown error')}")

    def _process_full_data_parallel_by_date(self, all_date_to_file, valid_processing_dates, 
                                          target_year, data_shape, geo_reference, year_output_dir):
        """按日期并行处理完整数据（备用方法）"""
        logger.info(f"开始按日期并行处理 {len(valid_processing_dates)} 个日期")
        
        # 准备工作参数
        work_args = []
        for date in valid_processing_dates:
            date_str = date.strftime('%Y_%m_%d')
            h5_path = os.path.join(year_output_dir, f'{date_str}_full.h5')
            
            work_args.append({
                'date': date,
                'h5_path': h5_path,
                'all_date_to_file': all_date_to_file,
                'driver_dirs': self.driver_dirs,
                'target_year': target_year,
                'data_shape': data_shape,
                'geo_reference': geo_reference,
                'past_days': self.full_past_days,
                'future_days': self.future_days,
                'data_dir': self.data_dir
            })
        
        logger.info(f"需要处理 {len(work_args)} 个日期")
        
        # 并行处理 - 每个进程处理一个日期
        with ProcessPoolExecutor(max_workers=min(self.n_processes, len(work_args))) as executor:
            results = list(executor.map(_process_single_date_full_data_worker, work_args))
        
        # 统计结果
        successful = sum(1 for r in results if r.get('success', False))
        logger.info(f"按日期并行处理完成: {successful}/{len(work_args)} 个日期成功")
        
        for result in results:
            if result.get('success', False):
                date_str = result.get('date_str', 'unknown')
                logger.info(f"日期 {date_str}: 数据集 {result.get('dataset_count', 0)}, 像素 {result.get('processed_pixels', 0)}")
            else:
                date_str = result.get('date_str', 'unknown')
                logger.error(f"日期 {date_str} 处理失败: {result.get('error', 'unknown error')}")

    def _process_full_data_parallel_by_driver(self, all_date_to_file, current_year_dates, 
                                            target_year, data_shape, geo_reference):
        """按驱动因素并行处理完整数据 - 大内存优化版本（保留作为备用方法）"""
        logger.info(f"开始大内存优化并行处理 {len(self.driver_dirs)} 个驱动因素")
        
        # 第一步：预加载所有驱动因素的所有需要的数据到内存
        logger.info("第一步：预加载所有驱动因素数据到内存（利用120GB大内存）")
        all_driver_data = self._preload_all_driver_data_optimized(all_date_to_file, current_year_dates)
        
        if not all_driver_data:
            logger.error("数据预加载失败")
            return
        
        logger.info(f"数据预加载完成，共加载 {len(all_driver_data)} 个驱动因素的数据")
        
        # 准备工作参数
        work_args = []
        for driver_name in self.driver_dirs.keys():
            h5_path = os.path.join(self.output_dir, f'{target_year}_{driver_name}_full.h5')
            
            # 检查文件是否已存在
            if os.path.exists(h5_path):
                logger.info(f"完整数据文件已存在，跳过: {h5_path}")
                continue
            
            work_args.append({
                'h5_path': h5_path,
                'driver_name': driver_name,
                'all_driver_data': all_driver_data,  # 传递预加载的数据
                'current_year_dates': current_year_dates,
                'target_year': target_year,
                'data_shape': data_shape,
                'geo_reference': geo_reference,
                'past_days': self.full_past_days,
                'future_days': self.future_days
            })
        
        if not work_args:
            logger.info("所有完整数据文件已存在，无需处理")
            return
        
        logger.info(f"需要处理 {len(work_args)} 个驱动因素")
        
        # 并行处理 - 使用预加载的数据
        with ProcessPoolExecutor(max_workers=min(self.n_processes, len(work_args))) as executor:
            results = list(executor.map(_process_driver_with_preloaded_data_worker, work_args))
        
        # 统计结果
        successful = sum(1 for r in results if r.get('success', False))
        logger.info(f"大内存优化并行处理完成: {successful}/{len(work_args)} 个驱动因素成功")
        
        for result in results:
            if result.get('success', False):
                logger.info(f"驱动因素 {result['driver_name']}: 数据集 {result.get('dataset_count', 0)}, 像素 {result.get('processed_pixels', 0)}")
            else:
                logger.error(f"驱动因素 {result.get('driver_name', 'unknown')} 处理失败: {result.get('error', 'unknown error')}")

    def _process_full_data_sequential_by_date_preloaded(self, all_driver_data, valid_processing_dates, 
                                                      target_year, data_shape, geo_reference, year_output_dir):
        """使用预加载数据顺序处理每个日期的完整数据（调试版本）"""
        logger.info(f"开始使用预加载数据顺序处理 {len(valid_processing_dates)} 个日期（调试模式）")
        
        for date in valid_processing_dates:
            date_str = date.strftime('%Y_%m_%d')
            h5_path = os.path.join(year_output_dir, f'{date_str}_full.h5')
            
            logger.info(f"处理日期: {date_str}")
            
            # 直接调用使用预加载数据的单日期处理函数
            result = _process_single_date_preloaded_standalone(
                date, h5_path, all_driver_data,
                target_year, data_shape, geo_reference, 
                self.full_past_days, self.future_days, self.data_dir
            )
            
            if result.get('success', False):
                logger.info(f"日期 {date_str} 处理成功: 数据集 {result.get('dataset_count', 0)}, 像素 {result.get('processed_pixels', 0)}")
            else:
                logger.error(f"日期 {date_str} 处理失败: {result.get('error', 'unknown error')}")

    def _process_full_data_sequential_by_date(self, all_date_to_file, valid_processing_dates, 
                                            target_year, data_shape, geo_reference, year_output_dir):
        """顺序处理每个日期的完整数据（调试版本，备用方法）"""
        logger.info(f"开始顺序处理 {len(valid_processing_dates)} 个日期（调试模式）")
        
        for date in valid_processing_dates:
            date_str = date.strftime('%Y_%m_%d')
            h5_path = os.path.join(year_output_dir, f'{date_str}_full.h5')
            
            logger.info(f"处理日期: {date_str}")
            
            # 直接调用单日期处理函数
            result = _process_single_date_full_data_standalone(
                date, h5_path, all_date_to_file, self.driver_dirs,
                target_year, data_shape, geo_reference, 
                self.full_past_days, self.future_days, self.data_dir
            )
            
            if result.get('success', False):
                logger.info(f"日期 {date_str} 处理成功: 数据集 {result.get('dataset_count', 0)}, 像素 {result.get('processed_pixels', 0)}")
            else:
                logger.error(f"日期 {date_str} 处理失败: {result.get('error', 'unknown error')}")

    def _process_full_data_sequential_by_driver(self, all_date_to_file, current_year_dates, 
                                              target_year, data_shape, geo_reference):
        """顺序处理每个驱动因素的完整数据（调试版本，保留作为备用方法）"""
        logger.info(f"开始顺序处理 {len(self.driver_dirs)} 个驱动因素（调试模式）")
        
        for driver_name in self.driver_dirs.keys():
            h5_path = os.path.join(self.output_dir, f'{target_year}_{driver_name}_full.h5')
            
            # 检查文件是否已存在
            if os.path.exists(h5_path):
                logger.info(f"完整数据文件已存在，跳过: {h5_path}")
                continue
            
            logger.info(f"处理驱动因素: {driver_name}")
            
            # 直接调用流式处理函数
            result = _create_full_data_streaming_worker(
                h5_path, driver_name, self.driver_dirs[driver_name],
                all_date_to_file, current_year_dates, data_shape, geo_reference, target_year,
                self.full_past_days, self.future_days, self.data_dir
            )
            
            if result.get('success', False):
                logger.info(f"驱动因素 {driver_name} 处理成功: 数据集 {result.get('dataset_count', 0)}, 像素 {result.get('processed_pixels', 0)}")
            else:
                logger.error(f"驱动因素 {driver_name} 处理失败: {result.get('error', 'unknown error')}")

    def _preload_all_driver_data_optimized(self, all_date_to_file, valid_processing_dates):
        """预加载所有驱动因素的所有需要数据到内存 - 大内存优化
        
        Args:
            all_date_to_file: 全局日期到文件的映射
            valid_processing_dates: 需要处理的有效日期列表
        """
        logger.info("开始预加载所有驱动因素数据...")
        
        # 计算所有需要的日期（包括时间序列的过去和未来日期）
        all_required_dates = set()
        for date in valid_processing_dates:
            past_dates, future_dates = self._get_past_future_dates(date, past_days=self.full_past_days)
            if past_dates and future_dates:
                all_required_dates.update(past_dates + [date] + future_dates)
        
        all_required_dates = sorted(all_required_dates)
        logger.info(f"总共需要加载 {len(all_required_dates)} 个日期的数据")
        
        # 预加载所有驱动因素的所有日期数据
        all_driver_data = {}
        total_files = len(self.driver_dirs) * len(all_required_dates)
        loaded_files = 0
        
        for driver_name, driver_dir in self.driver_dirs.items():
            logger.info(f"预加载驱动因素: {driver_name}")
            driver_data = {}
            
            for date in all_required_dates:
                try:
                    data = self._load_driver_data(driver_dir, date)
                    if data is not None:
                        driver_data[date] = data
                        loaded_files += 1
                except Exception as e:
                    logger.warning(f"加载数据失败: {driver_name}, {date}, 错误: {str(e)}")
                
                # 报告进度
                if loaded_files % 1000 == 0:
                    progress = loaded_files / total_files * 100
                    logger.info(f"预加载进度: {progress:.1f}% ({loaded_files}/{total_files})")
            
            all_driver_data[driver_name] = driver_data
            logger.info(f"驱动因素 {driver_name} 预加载完成: {len(driver_data)}/{len(all_required_dates)} 个日期")
        
        # 估算内存使用
        total_memory_mb = 0
        for driver_name, driver_data in all_driver_data.items():
            for date, data in driver_data.items():
                if data is not None and hasattr(data, 'nbytes'):
                    total_memory_mb += data.nbytes / (1024 * 1024)
        
        logger.info(f"预加载完成！总内存使用: {total_memory_mb:.2f} MB ({total_memory_mb/1024:.2f} GB)")
        
        # 验证预加载数据的完整性
        missing_data_count = 0
        for driver_name, driver_data in all_driver_data.items():
            for date in all_required_dates:
                if date not in driver_data or driver_data[date] is None:
                    missing_data_count += 1
        
        if missing_data_count > 0:
            logger.warning(f"预加载数据中有 {missing_data_count} 个日期/驱动因素组合的数据缺失")
        else:
            logger.info("预加载数据完整性验证通过，所有需要的数据都已加载")
        
        return all_driver_data

    def _get_valid_pixel_positions_2(self, all_driver_data, height, width):
        """获取所有有效像素位置用于完整数据集生成（备用方法）
        
        对于完整数据集：
        - 所有非背景像素 (FIRMS ≠ 255) 都参与数据集生成
        - 包括正样本 (FIRMS > 0) 和负样本 (FIRMS = 0)
        - 排除背景值 (FIRMS = 255, NoData)
        """
        valid_pixels = []
        
        # 获取FIRMS数据的第一个可用日期来确定真正的NoData像素
        firms_data = None
        if 'Firms_Detection_resampled' in all_driver_data:
            for date, data in all_driver_data['Firms_Detection_resampled'].items():
                if data is not None:
                    firms_data = data
                    break
        
        if firms_data is None:
            logger.warning("没有找到FIRMS数据用于确定有效像素位置")
            # 如果没有FIRMS数据，则包含所有像素位置
            for i in range(height):
                for j in range(width):
                    valid_pixels.append((i, j))
            return valid_pixels
        
        # 处理多波段数据
        if len(firms_data.shape) == 3:
            firms_data = firms_data[0]  # 取第一个波段
        
        # 定义NoData值
        nodata_values = [255]
        
        # 找到所有非NoData的像素位置
        for i in range(height):
            for j in range(width):
                firms_value = firms_data[i, j]
                
                # 排除NoData值和NaN值
                if firms_value not in nodata_values and not np.isnan(firms_value):
                    valid_pixels.append((i, j))
        
        logger.info(f"完整数据集：找到 {len(valid_pixels)}/{height*width} 个有效像素")
        return valid_pixels

    def _create_full_data_ultra_fast(self, h5_path, driver_name, all_driver_data, 
                                   valid_dates, data_shape, geo_reference, year):
        """超高速创建完整数据文件 - 取消分批处理，直接遍历所有像素"""
        logger.info(f"开始创建完整数据文件: {driver_name}")
        
        # 获取有效像素位置
        height, width = data_shape[:2]
        valid_pixels = self._get_valid_pixel_positions_2(all_driver_data, height, width)
        
        logger.info(f"有效像素数: {len(valid_pixels)}/{height*width} ({len(valid_pixels)/(height*width)*100:.1f}%)")
        
        with h5py.File(h5_path, 'w') as h5_file:
            # 写入全局属性
            h5_file.attrs['year'] = str(year)
            h5_file.attrs['driver'] = driver_name
            h5_file.attrs['past_days'] = self.full_past_days
            h5_file.attrs['future_days'] = self.future_days
            h5_file.attrs['sampling_type'] = 'pixel_level'
            h5_file.attrs['data_type'] = 'full_year_data'
            
            if geo_reference:
                self._save_geo_reference(h5_file, geo_reference)
            
            # 准备有效日期数据
            valid_date_data = {}
            for date in valid_dates:
                past_dates, future_dates = self._get_past_future_dates(date, past_days=self.full_past_days)
                if past_dates is None or future_dates is None:
                    continue
                
                # 获取FIRMS数据
                firms_data = None
                if 'Firms_Detection_resampled' in all_driver_data:
                    firms_data = all_driver_data['Firms_Detection_resampled'].get(date)
                
                valid_date_data[date] = {
                    'past_dates': past_dates,
                    'future_dates': future_dates,
                    'firms_data': firms_data,
                    'date_str': date.strftime('%Y%m%d')
                }
            
            # 获取当前驱动因素的数据
            driver_data = all_driver_data[driver_name]
            
            dataset_count = 0
            
            # 直接处理所有像素，不分批
            logger.info(f"开始处理 {len(valid_pixels)} 个像素的数据（无分批，直接遍历）")
            
            # 使用进度条处理所有像素
            for pixel_idx, (row, col) in enumerate(tqdm(valid_pixels, desc=f"处理{driver_name}像素")):
                
                # 为每个有效日期处理当前像素
                for date, date_info in valid_date_data.items():
                    past_dates = date_info['past_dates']
                    future_dates = date_info['future_dates'] 
                    firms_data = date_info['firms_data']
                    date_str = date_info['date_str']
                    
                    # 获取FIRMS值
                    firms_value = 0
                    if firms_data is not None:
                        firms_value = int(firms_data[row, col])
                    
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
                        
                        # 写入数据集
                        past_dataset_name = f"{date_str}_past_{firms_value}_{row}_{col}"
                        past_dataset = h5_file.create_dataset(
                            past_dataset_name,
                            data=past_array,
                            dtype=np.float32,
                            compression=None,  # 关闭压缩以提高速度
                            shuffle=False
                        )
                        dataset_count += 1
                        
                        future_dataset_name = f"{date_str}_future_{firms_value}_{row}_{col}"
                        future_dataset = h5_file.create_dataset(
                            future_dataset_name,
                            data=future_array,
                            dtype=np.float32,
                            compression=None,  # 关闭压缩以提高速度
                            shuffle=False
                        )
                        dataset_count += 1
                
                # 每处理1000个像素报告一次进度
                if (pixel_idx + 1) % 1000 == 0 or pixel_idx == len(valid_pixels) - 1:
                    progress = (pixel_idx + 1) / len(valid_pixels) * 100
                    logger.info(f"像素处理进度: {progress:.1f}% ({pixel_idx + 1}/{len(valid_pixels)})")
                
                # 定期垃圾回收
                if (pixel_idx + 1) % 5000 == 0:
                    gc.collect()
            
            h5_file.attrs['total_datasets'] = dataset_count
            h5_file.attrs['valid_pixels'] = len(valid_pixels)
            h5_file.attrs['total_pixels_processed'] = len(valid_pixels)
            logger.info(f"完整数据文件创建完成: {driver_name}, 共 {dataset_count} 个数据集，处理了 {len(valid_pixels)} 个有效像素")

    def _create_full_data_parallel(self, h5_path, driver_name, all_driver_data, 
                                 valid_dates, data_shape, geo_reference, year, n_processes=32):
        """并行化创建完整数据文件 - 使用多进程处理像素"""
        logger.info(f"开始并行创建完整数据文件: {driver_name} (使用 {n_processes} 个进程)")
        
        # 获取有效像素位置
        height, width = data_shape[:2]
        valid_pixels = self._get_valid_pixel_positions_2(all_driver_data, height, width)
        
        logger.info(f"有效像素数: {len(valid_pixels)}/{height*width} ({len(valid_pixels)/(height*width)*100:.1f}%)")
        
        # 准备有效日期数据
        valid_date_data = {}
        for date in valid_dates:
            past_dates, future_dates = self._get_past_future_dates(date, past_days=self.full_past_days)
            if past_dates is None or future_dates is None:
                continue
            
            # 获取FIRMS数据
            firms_data = None
            if 'Firms_Detection_resampled' in all_driver_data:
                firms_data = all_driver_data['Firms_Detection_resampled'].get(date)
            
            valid_date_data[date] = {
                'past_dates': past_dates,
                'future_dates': future_dates,
                'firms_data': firms_data,
                'date_str': date.strftime('%Y%m%d')
            }
        
        # 获取当前驱动因素的数据
        driver_data = all_driver_data[driver_name]
        
        logger.info(f"有效处理日期数: {len(valid_date_data)}")
        
        # 将像素分割给多个进程
        pixels_per_process = len(valid_pixels) // n_processes
        if len(valid_pixels) % n_processes != 0:
            pixels_per_process += 1
        
        pixel_chunks = []
        for i in range(0, len(valid_pixels), pixels_per_process):
            chunk = valid_pixels[i:i + pixels_per_process]
            pixel_chunks.append(chunk)
        
        logger.info(f"将 {len(valid_pixels)} 个像素分配给 {len(pixel_chunks)} 个进程，每个进程处理约 {pixels_per_process} 个像素")
        
        # 创建临时文件来存储每个进程的结果
        temp_dir = tempfile.mkdtemp()
        temp_files = []
        
        try:
            # 准备工作参数
            work_args = []
            for idx, pixel_chunk in enumerate(pixel_chunks):
                temp_file = os.path.join(temp_dir, f"chunk_{idx}.h5")
                temp_files.append(temp_file)
                
                work_args.append({
                    'pixel_chunk': pixel_chunk,
                    'valid_date_data': valid_date_data,
                    'driver_data': driver_data,
                    'driver_name': driver_name,
                    'temp_file': temp_file,
                    'chunk_idx': idx
                })
            
            # 并行处理
            logger.info("开始并行处理像素数据...")
            with Pool(processes=n_processes) as pool:
                results = list(tqdm(
                    pool.imap(_process_pixel_chunk_worker, work_args),
                    total=len(work_args),
                    desc=f"并行处理{driver_name}像素"
                ))
            
            # 合并结果到最终文件
            logger.info("合并并行处理结果...")
            total_datasets = 0
            
            with h5py.File(h5_path, 'w') as final_h5:
                # 写入全局属性
                final_h5.attrs['year'] = str(year)
                final_h5.attrs['driver'] = driver_name
                final_h5.attrs['past_days'] = self.past_days
                final_h5.attrs['future_days'] = self.future_days
                final_h5.attrs['sampling_type'] = 'pixel_level'
                final_h5.attrs['data_type'] = 'full_year_data_parallel'
                
                if geo_reference:
                    self._save_geo_reference(final_h5, geo_reference)
                
                # 合并所有临时文件
                for temp_file, result in zip(temp_files, results):
                    if os.path.exists(temp_file) and result['success']:
                        with h5py.File(temp_file, 'r') as temp_h5:
                            # 复制所有数据集
                            for dataset_name in temp_h5.keys():
                                temp_h5.copy(dataset_name, final_h5)
                                total_datasets += 1
                
                final_h5.attrs['total_datasets'] = total_datasets
                final_h5.attrs['valid_pixels'] = len(valid_pixels)
                final_h5.attrs['total_pixels_processed'] = len(valid_pixels)
                
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            os.rmdir(temp_dir)
        
        # 统计结果
        successful_chunks = sum(1 for r in results if r['success'])
        total_processed_pixels = sum(r['processed_pixels'] for r in results)
        
        logger.info(f"并行处理完成: {driver_name}")
        logger.info(f"成功处理块数: {successful_chunks}/{len(pixel_chunks)}")
        logger.info(f"总数据集数: {total_datasets}")
        logger.info(f"处理像素数: {total_processed_pixels}")



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
                h5_file.attrs['past_days'] = self.sample_past_days
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

    def _create_full_data_vectorized_with_progress(self, h5_path, driver_name, driver_data, valid_pixels, valid_date_data, 
                                              data_shape, geo_reference, year, past_days, future_days):
        """超高效向量化处理版本 - 大幅减少循环嵌套"""
        import os
        import sys
        import gc
        import numpy as np
        
        process_id = os.getpid()
        
        try:
            from tqdm import tqdm
            
            dataset_count = 0
            
            print(f"[进程 {process_id}] {driver_name}: 开始向量化处理", flush=True)
            
            with h5py.File(h5_path, 'w') as h5_file:
                # 写入全局属性
                h5_file.attrs['year'] = str(year)
                h5_file.attrs['driver'] = driver_name
                h5_file.attrs['past_days'] = past_days
                h5_file.attrs['future_days'] = future_days
                h5_file.attrs['sampling_type'] = 'pixel_level'
                h5_file.attrs['data_type'] = 'full_year_data_vectorized'
                
                if geo_reference:
                    _save_geo_reference_standalone(h5_file, geo_reference)
                
                # 创建进度条
                total_pixels = len(valid_pixels)
                position = process_id % 10
                
                print(f"[进程 {process_id}] {driver_name}: 像素数 = {total_pixels}, 日期数 = {len(valid_date_data)}", flush=True)
                
                pbar = tqdm(total=total_pixels, 
                           desc=f"{driver_name}", 
                           position=position,
                           leave=True,
                           ncols=80,
                           unit="像素",
                           file=sys.stdout)
                
                try:
                    # 预处理：将driver_data转换为numpy数组以提高访问速度
                    print(f"[进程 {process_id}] {driver_name}: 预处理数据结构", flush=True)
                    
                    # 创建日期到索引的映射
                    date_to_idx = {}
                    sorted_dates = sorted(driver_data.keys())
                    date_arrays = {}
                    
                    # 预加载所有数据到内存（如果内存允许）
                    print(f"[进程 {process_id}] {driver_name}: 预加载数据到内存", flush=True)
                    for i, date in enumerate(sorted_dates):
                        date_to_idx[date] = i
                        data = driver_data[date]
                        if data is not None:
                            if len(data.shape) == 3:
                                date_arrays[date] = data
                            else:
                                date_arrays[date] = data
                    
                    print(f"[进程 {process_id}] {driver_name}: 数据预处理完成，开始批量处理", flush=True)
                    
                    # 批量处理像素 - 减少循环嵌套
                    batch_size = 100  # 每批处理100个像素
                    
                    for batch_start in range(0, len(valid_pixels), batch_size):
                        batch_end = min(batch_start + batch_size, len(valid_pixels))
                        batch_pixels = valid_pixels[batch_start:batch_end]
                        
                        # 批量处理当前批次的像素
                        for pixel_idx, (row, col) in enumerate(batch_pixels):
                            actual_pixel_idx = batch_start + pixel_idx
                            
                            # 为每个有效日期处理当前像素
                            for date, date_info in valid_date_data.items():
                                past_dates = date_info['past_dates']
                                future_dates = date_info['future_dates'] 
                                firms_data = date_info['firms_data']
                                date_str = date_info['date_str']
                                
                                # 获取FIRMS值
                                firms_value = 0
                                if firms_data is not None:
                                    firms_value = int(firms_data[row, col])
                                
                                # 向量化提取时间序列 - 减少循环
                                past_series = []
                                valid_past = True
                                
                                # 批量检查过去日期的数据可用性
                                for ts_date in past_dates:
                                    if ts_date not in date_arrays:
                                        valid_past = False
                                        break
                                        
                                    data = date_arrays[ts_date]
                                    if len(data.shape) == 3:
                                        pixel_value = data[:, row, col]
                                    else:
                                        pixel_value = data[row, col]
                                    past_series.append(pixel_value)
                                
                                # 批量检查未来日期的数据
                                future_series = []
                                valid_future = True
                                
                                for ts_date in future_dates:
                                    if ts_date not in date_arrays:
                                        valid_future = False
                                        break
                                        
                                    data = date_arrays[ts_date]
                                    if len(data.shape) == 3:
                                        pixel_value = data[:, row, col]
                                    else:
                                        pixel_value = data[row, col]
                                    future_series.append(pixel_value)
                                
                                # 只保存有效的像素数据
                                if valid_past and valid_future and past_series and future_series:
                                    # 向量化数据处理
                                    past_array = np.array(past_series, dtype=np.float32)
                                    future_array = np.array(future_series, dtype=np.float32)
                                    
                                    if past_array.ndim == 2 and past_array.shape[0] > past_array.shape[1]:
                                        past_array = past_array.T
                                    if future_array.ndim == 2 and future_array.shape[0] > future_array.shape[1]:
                                        future_array = future_array.T
                                    
                                    # 快速写入数据集
                                    past_dataset_name = f"{date_str}_past_{firms_value}_{row}_{col}"
                                    h5_file.create_dataset(
                                        past_dataset_name,
                                        data=past_array,
                                        dtype=np.float32,
                                        compression=None,
                                        shuffle=False
                                    )
                                    dataset_count += 1
                                    
                                    future_dataset_name = f"{date_str}_future_{firms_value}_{row}_{col}"
                                    h5_file.create_dataset(
                                        future_dataset_name,
                                        data=future_array,
                                        dtype=np.float32,
                                        compression=None,
                                        shuffle=False
                                    )
                                    dataset_count += 1
                            
                            # 更新进度条
                            pbar.update(1)
                            
                            # 定期状态报告
                            if (actual_pixel_idx + 1) % 1000 == 0:
                                progress = (actual_pixel_idx + 1) / len(valid_pixels) * 100
                                print(f"[进程 {process_id}] {driver_name}: 进度 {progress:.1f}%, 数据集: {dataset_count}", flush=True)
                        
                        # 批次完成后垃圾回收
                        if (batch_start // batch_size) % 10 == 0:
                            gc.collect()
                    
                finally:
                    pbar.close()
                
                h5_file.attrs['total_datasets'] = dataset_count
                h5_file.attrs['valid_pixels'] = len(valid_pixels)
                h5_file.attrs['total_pixels_processed'] = len(valid_pixels)
            
            print(f"[进程 {process_id}] {driver_name}: 向量化处理完成，数据集数={dataset_count}", flush=True)
            
            return {
                'success': True,
                'driver_name': driver_name,
                'dataset_count': dataset_count,
                'processed_pixels': len(valid_pixels)
            }
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[进程 {process_id}] {driver_name}: 向量化处理失败", flush=True)
            print(error_detail, flush=True)
            return {
                'success': False,
                'driver_name': driver_name,
                'error': f"{str(e)}\n{error_detail}"
            }

    def _create_full_data_vectorized_standalone(self, h5_path, driver_name, driver_data, valid_pixels, valid_date_data, 
                                            data_shape, geo_reference, year, past_days, future_days):
        """独立函数：超高效向量化处理版本"""
        import os
        import sys
        import gc
        
        process_id = os.getpid()
        
        try:
            from tqdm import tqdm
            
            dataset_count = 0
            
            print(f"[进程 {process_id}] {driver_name}: 开始向量化处理", flush=True)
            
            with h5py.File(h5_path, 'w') as h5_file:
                # 写入全局属性
                h5_file.attrs['year'] = str(year)
                h5_file.attrs['driver'] = driver_name
                h5_file.attrs['past_days'] = past_days
                h5_file.attrs['future_days'] = future_days
                h5_file.attrs['sampling_type'] = 'pixel_level'
                h5_file.attrs['data_type'] = 'full_year_data_vectorized'
                
                if geo_reference:
                    _save_geo_reference_standalone(h5_file, geo_reference)
                
                # 创建进度条
                total_pixels = len(valid_pixels)
                position = process_id % 10
                
                print(f"[进程 {process_id}] {driver_name}: 像素数 = {total_pixels}, 日期数 = {len(valid_date_data)}", flush=True)
                
                pbar = tqdm(total=total_pixels, 
                           desc=f"{driver_name}", 
                           position=position,
                           leave=True,
                           ncols=80,
                           unit="像素",
                           file=sys.stdout)
                
                try:
                    # 预处理：预加载数据以减少字典查找
                    print(f"[进程 {process_id}] {driver_name}: 预加载数据到内存", flush=True)
                    date_arrays = {}
                    
                    for date, data in driver_data.items():
                        if data is not None:
                            date_arrays[date] = data
                    
                    print(f"[进程 {process_id}] {driver_name}: 预加载完成，开始处理", flush=True)
                    
                    # 高效处理所有像素
                    for pixel_idx, (row, col) in enumerate(valid_pixels):
                        
                        # 为每个有效日期处理当前像素
                        for date, date_info in valid_date_data.items():
                            past_dates = date_info['past_dates']
                            future_dates = date_info['future_dates'] 
                            firms_data = date_info['firms_data']
                            date_str = date_info['date_str']
                            
                            # 获取FIRMS值
                            firms_value = 0
                            if firms_data is not None:
                                firms_value = int(firms_data[row, col])
                            
                            # 快速提取过去时间序列
                            past_series = []
                            valid_past = True
                            
                            for ts_date in past_dates:
                                if ts_date not in date_arrays:
                                    valid_past = False
                                    break
                                    
                                data = date_arrays[ts_date]
                                if len(data.shape) == 3:
                                    pixel_value = data[:, row, col]
                                else:
                                    pixel_value = data[row, col]
                                past_series.append(pixel_value)
                            
                            # 快速提取未来时间序列
                            future_series = []
                            valid_future = True
                            
                            for ts_date in future_dates:
                                if ts_date not in date_arrays:
                                    valid_future = False
                                    break
                                    
                                data = date_arrays[ts_date]
                                if len(data.shape) == 3:
                                    pixel_value = data[:, row, col]
                                else:
                                    pixel_value = data[row, col]
                                future_series.append(pixel_value)
                            
                            # 只保存有效的像素数据
                            if valid_past and valid_future and past_series and future_series:
                                # 向量化数据处理
                                past_array = np.array(past_series, dtype=np.float32)
                                future_array = np.array(future_series, dtype=np.float32)
                                
                                if past_array.ndim == 2 and past_array.shape[0] > past_array.shape[1]:
                                    past_array = past_array.T
                                if future_array.ndim == 2 and future_array.shape[0] > future_array.shape[1]:
                                    future_array = future_array.T
                                
                                # 快速写入数据集
                                past_dataset_name = f"{date_str}_past_{firms_value}_{row}_{col}"
                                h5_file.create_dataset(
                                    past_dataset_name,
                                    data=past_array,
                                    dtype=np.float32,
                                    compression=None,
                                    shuffle=False
                                )
                                dataset_count += 1
                                
                                future_dataset_name = f"{date_str}_future_{firms_value}_{row}_{col}"
                                h5_file.create_dataset(
                                    future_dataset_name,
                                    data=future_array,
                                    dtype=np.float32,
                                    compression=None,
                                    shuffle=False
                                )
                                dataset_count += 1
                        
                        # 更新进度条
                        pbar.update(1)
                        
                        # 定期状态报告和垃圾回收
                        if (pixel_idx + 1) % 1000 == 0:
                            progress = (pixel_idx + 1) / len(valid_pixels) * 100
                            print(f"[进程 {process_id}] {driver_name}: 进度 {progress:.1f}%, 数据集: {dataset_count}", flush=True)
                            
                        if (pixel_idx + 1) % 5000 == 0:
                            gc.collect()
                    
                finally:
                    pbar.close()
                
                h5_file.attrs['total_datasets'] = dataset_count
                h5_file.attrs['valid_pixels'] = len(valid_pixels)
                h5_file.attrs['total_pixels_processed'] = len(valid_pixels)
            
            print(f"[进程 {process_id}] {driver_name}: 向量化处理完成，数据集数={dataset_count}", flush=True)
            
            return {
                'success': True,
                'driver_name': driver_name,
                'dataset_count': dataset_count,
                'processed_pixels': len(valid_pixels)
            }
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[进程 {process_id}] {driver_name}: 向量化处理失败", flush=True)
            print(error_detail, flush=True)
            return {
                'success': False,
                'driver_name': driver_name,
                'error': f"{str(e)}\n{error_detail}"
            }

def _create_full_data_ultra_fast_standalone(h5_path, driver_name, driver_data, valid_pixels, valid_date_data, 
                                           data_shape, geo_reference, year, past_days, future_days):
    """独立函数：基于ultra_fast的高效处理，添加进程级进度报告"""
    import os
    import gc
    
    process_id = os.getpid()
    
    try:
        dataset_count = 0
        
        print(f"[进程 {process_id}] {driver_name}: 开始高效处理", flush=True)
        
        with h5py.File(h5_path, 'w') as h5_file:
            # 写入全局属性
            h5_file.attrs['year'] = str(year)
            h5_file.attrs['driver'] = driver_name
            h5_file.attrs['past_days'] = past_days
            h5_file.attrs['future_days'] = future_days
            h5_file.attrs['sampling_type'] = 'pixel_level'
            h5_file.attrs['data_type'] = 'full_year_data_ultra_fast'
            
            if geo_reference:
                _save_geo_reference_standalone(h5_file, geo_reference)
            
            print(f"[进程 {process_id}] {driver_name}: 开始处理 {len(valid_pixels)} 个像素", flush=True)
            
            # 直接处理所有像素，使用ultra_fast逻辑
            for pixel_idx, (row, col) in enumerate(valid_pixels):
                
                # 为每个有效日期处理当前像素
                for date, date_info in valid_date_data.items():
                    past_dates = date_info['past_dates']
                    future_dates = date_info['future_dates'] 
                    firms_data = date_info['firms_data']
                    date_str = date_info['date_str']
                    
                    # 获取FIRMS值
                    firms_value = 0
                    if firms_data is not None:
                        firms_value = int(firms_data[row, col])
                    
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
                        
                        # 写入数据集 - 关闭压缩以提高速度
                        past_dataset_name = f"{date_str}_past_{firms_value}_{row}_{col}"
                        h5_file.create_dataset(
                            past_dataset_name,
                            data=past_array,
                            dtype=np.float32,
                            compression=None,
                            shuffle=False
                        )
                        dataset_count += 1
                        
                        future_dataset_name = f"{date_str}_future_{firms_value}_{row}_{col}"
                        h5_file.create_dataset(
                            future_dataset_name,
                            data=future_array,
                            dtype=np.float32,
                            compression=None,
                            shuffle=False
                        )
                        dataset_count += 1
                
                # 定期进度报告
                if (pixel_idx + 1) % 1000 == 0 or pixel_idx == len(valid_pixels) - 1:
                    progress = (pixel_idx + 1) / len(valid_pixels) * 100
                    print(f"[进程 {process_id}] {driver_name}: 进度 {progress:.1f}% ({pixel_idx + 1}/{len(valid_pixels)}), 数据集数: {dataset_count}", flush=True)
                
                # 定期垃圾回收
                if (pixel_idx + 1) % 5000 == 0:
                    gc.collect()
            
            h5_file.attrs['total_datasets'] = dataset_count
            h5_file.attrs['valid_pixels'] = len(valid_pixels)
            h5_file.attrs['total_pixels_processed'] = len(valid_pixels)
        
        print(f"[进程 {process_id}] {driver_name}: 高效处理完成，数据集数={dataset_count}", flush=True)
        
        return {
            'success': True,
            'driver_name': driver_name,
            'dataset_count': dataset_count,
            'processed_pixels': len(valid_pixels)
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[进程 {process_id}] {driver_name}: 高效处理失败", flush=True)
        print(error_detail, flush=True)
        return {
            'success': False,
            'driver_name': driver_name,
            'error': f"{str(e)}\n{error_detail}"
        }

def _process_single_driver_full_data_worker(work_args):
    """处理单个驱动因素的完整数据 - 工作进程函数"""
    import os
    import sys
    
    process_id = os.getpid()
    
    try:
        h5_path = work_args['h5_path']
        driver_name = work_args['driver_name']
        driver_dir = work_args['driver_dir']
        all_date_to_file = work_args['all_date_to_file']
        current_year_dates = work_args['current_year_dates']
        target_year = work_args['target_year']
        data_shape = work_args['data_shape']
        geo_reference = work_args['geo_reference']
        past_days = work_args['past_days']
        future_days = work_args['future_days']
        data_dir = work_args['data_dir']
        
        print(f"[进程 {process_id}] 开始处理驱动因素: {driver_name}", flush=True)
        
        # 使用流式处理
        result = _create_full_data_streaming_worker(
            h5_path, driver_name, driver_dir, all_date_to_file,
            current_year_dates, data_shape, geo_reference, target_year,
            past_days, future_days, data_dir
        )
        
        print(f"[进程 {process_id}] {driver_name} 处理完成: {result.get('success', False)}", flush=True)
        return result
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        driver_name = work_args.get('driver_name', 'unknown')
        print(f"[进程 {process_id}] 处理驱动因素 {driver_name} 时出错:", flush=True)
        print(error_detail, flush=True)
        return {
            'success': False,
            'driver_name': driver_name,
            'error': f"{str(e)}\n{error_detail}"
        }

def _create_full_data_streaming_worker(h5_path, driver_name, driver_dir, all_date_to_file,
                                     current_year_dates, data_shape, geo_reference, target_year,
                                     past_days, future_days, data_dir):
    """工作进程的流式处理函数 - 优化版本：避免大量任务预收集"""
    import os
    import gc
    import time
    from tqdm import tqdm
    
    process_id = os.getpid()
    
    try:
        height, width = data_shape[:2]
        dataset_count = 0
        processed_pixels = 0
        
        # 创建独立的缓存 - 大内存优化
        memory_cache = StreamingCache(max_size_gb=20)  # 每个进程20GB缓存
        
        print(f"[进程 {process_id}] {driver_name}: 开始流式处理（优化版本）", flush=True)
        
        with h5py.File(h5_path, 'w') as h5_file:
            # 写入全局属性
            h5_file.attrs['year'] = str(target_year)
            h5_file.attrs['driver'] = driver_name
            h5_file.attrs['past_days'] = past_days
            h5_file.attrs['future_days'] = future_days
            h5_file.attrs['sampling_type'] = 'pixel_level'
            h5_file.attrs['data_type'] = 'full_year_data_streaming_optimized'
            
            if geo_reference:
                _save_geo_reference_standalone(h5_file, geo_reference)
            
            # 优化策略：不预收集所有任务，而是按日期逐个处理
            print(f"[进程 {process_id}] {driver_name}: 开始按日期逐个处理 {len(current_year_dates)} 个日期", flush=True)
            
            available_dates_set = set(all_date_to_file.keys())
            
            for date_idx, date in enumerate(current_year_dates):
                # 检查时间序列完整性
                past_dates, future_dates = _get_past_future_dates_standalone(date, past_days, future_days)
                if past_dates is None or future_dates is None:
                    if date_idx < 5:
                        print(f"[进程 {process_id}] {driver_name}: 日期 {date} 时间序列长度不足", flush=True)
                    continue
                
                # 检查所有需要的日期是否都存在
                all_required_dates = past_dates + [date] + future_dates
                missing_dates = [d for d in all_required_dates if d not in available_dates_set]
                if missing_dates:
                    if date_idx < 5:
                        print(f"[进程 {process_id}] {driver_name}: 日期 {date} 缺少 {len(missing_dates)} 个必需日期", flush=True)
                    continue
                
                # 加载FIRMS数据
                firms_data = _load_firms_data_worker(date, data_dir)
                if firms_data is None:
                    if date_idx < 5:
                        print(f"[进程 {process_id}] {driver_name}: 日期 {date} 无法加载FIRMS数据", flush=True)
                    continue
                
                # 获取有效像素（当日处理，不预收集）
                valid_pixels = _get_valid_pixels_from_firms_worker(firms_data)
                
                if date_idx < 5:
                    print(f"[进程 {process_id}] {driver_name}: 日期 {date} 找到 {len(valid_pixels)} 个有效像素", flush=True)
                
                # 立即处理当前日期的所有像素
                successful_pixels = 0
                for pixel_idx, (pixel_coord, firms_value) in enumerate(valid_pixels):
                    # 获取时间序列数据
                    past_data, future_data = _get_time_series_streaming_worker(
                        driver_dir, date, pixel_coord, all_date_to_file, 
                        memory_cache, past_days, future_days
                    )
                    
                    if past_data is not None and future_data is not None:
                        # 保存数据
                        date_str = date.strftime('%Y%m%d')
                        row, col = pixel_coord
                        
                        past_dataset_name = f"{date_str}_past_{firms_value}_{row}_{col}"
                        h5_file.create_dataset(
                            past_dataset_name,
                            data=past_data,
                            dtype=np.float32,
                            compression=None,
                            shuffle=False
                        )
                        dataset_count += 1
                        
                        future_dataset_name = f"{date_str}_future_{firms_value}_{row}_{col}"
                        h5_file.create_dataset(
                            future_dataset_name,
                            data=future_data,
                            dtype=np.float32,
                            compression=None,
                            shuffle=False
                        )
                        dataset_count += 1
                        processed_pixels += 1
                        successful_pixels += 1
                    
                    # 对前几个像素提供详细调试信息
                    if date_idx < 2 and pixel_idx < 3:
                        print(f"[进程 {process_id}] {driver_name}: 日期{date} 像素{pixel_idx} ({pixel_coord[0]},{pixel_coord[1]}) 成功:{past_data is not None and future_data is not None}", flush=True)
                    
                    # 每100个像素清理一次内存
                    if (pixel_idx + 1) % 100 == 0:
                        gc.collect()
                
                if date_idx < 5:
                    print(f"[进程 {process_id}] {driver_name}: 日期 {date} 完成, 成功像素: {successful_pixels}/{len(valid_pixels)}", flush=True)
                
                # 定期清理和报告进度
                if (date_idx + 1) % 5 == 0:
                    progress = (date_idx + 1) / len(current_year_dates) * 100
                    print(f"[进程 {process_id}] {driver_name}: 进度 {progress:.1f}%, 数据集: {dataset_count}", flush=True)
                    memory_cache.cleanup_old_data(date, past_days + future_days + 10)
                    gc.collect()
                    
                # 每处理一定数量日期后强制刷新文件
                if (date_idx + 1) % 10 == 0:
                    h5_file.flush()
                    print(f"[进程 {process_id}] {driver_name}: 文件已刷新，当前数据集数: {dataset_count}", flush=True)
            
            h5_file.attrs['total_datasets'] = dataset_count
            h5_file.attrs['processed_pixels'] = processed_pixels
        
        # 清理缓存
        memory_cache.clear()
        
        print(f"[进程 {process_id}] {driver_name}: 流式处理完成, 数据集: {dataset_count}, 像素: {processed_pixels}", flush=True)
        
        return {
            'success': True,
            'driver_name': driver_name,
            'dataset_count': dataset_count,
            'processed_pixels': processed_pixels
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[进程 {process_id}] {driver_name}: 流式处理失败", flush=True)
        print(error_detail, flush=True)
        return {
            'success': False,
            'driver_name': driver_name,
            'error': f"{str(e)}\n{error_detail}"
        }

def _collect_processing_tasks_worker(current_year_dates, all_date_to_file, height, width, 
                                   past_days, future_days, data_dir):
    """工作进程中收集处理任务"""
    import os
    process_id = os.getpid()
    
    processing_tasks = []
    available_dates_set = set(all_date_to_file.keys())
    
    print(f"[进程 {process_id}] 开始收集任务: 当前年份日期数={len(current_year_dates)}, 可用日期数={len(available_dates_set)}", flush=True)
    
    dates_with_complete_series = 0
    dates_with_firms_data = 0
    total_valid_pixels = 0
    
    for date_idx, date in enumerate(current_year_dates):
        # 检查时间序列完整性
        past_dates, future_dates = _get_past_future_dates_standalone(date, past_days, future_days)
        if past_dates is None or future_dates is None:
            if date_idx < 5:  # 只打印前5个失败的日期
                print(f"[进程 {process_id}] 日期 {date} 时间序列长度不足", flush=True)
            continue
        
        # 检查所有需要的日期是否都存在
        all_required_dates = past_dates + [date] + future_dates
        missing_dates = [d for d in all_required_dates if d not in available_dates_set]
        if missing_dates:
            if date_idx < 5:  # 只打印前5个失败的日期
                print(f"[进程 {process_id}] 日期 {date} 缺少 {len(missing_dates)} 个必需日期", flush=True)
            continue
        
        dates_with_complete_series += 1
        
        # 加载FIRMS数据
        firms_data = _load_firms_data_worker(date, data_dir)
        if firms_data is None:
            if date_idx < 5:
                print(f"[进程 {process_id}] 日期 {date} 无法加载FIRMS数据", flush=True)
            continue
        
        dates_with_firms_data += 1
        
        # 获取有效像素
        valid_pixels = _get_valid_pixels_from_firms_worker(firms_data)
        
        if date_idx < 5:  # 打印前5个成功日期的像素数
            print(f"[进程 {process_id}] 日期 {date} 找到 {len(valid_pixels)} 个有效像素", flush=True)
        
        # 创建任务
        for pixel_coord, firms_value in valid_pixels:
            processing_tasks.append({
                'pixel_coord': pixel_coord,
                'date': date,
                'firms_value': firms_value
            })
        
        total_valid_pixels += len(valid_pixels)
        
        # 每50个日期报告一次进度
        if (date_idx + 1) % 50 == 0:
            print(f"[进程 {process_id}] 进度: {date_idx + 1}/{len(current_year_dates)}, 完整时间序列: {dates_with_complete_series}, FIRMS数据: {dates_with_firms_data}, 总像素: {total_valid_pixels}", flush=True)
    
    print(f"[进程 {process_id}] 任务收集完成: 处理日期 {len(current_year_dates)}, 完整时间序列 {dates_with_complete_series}, 有FIRMS数据 {dates_with_firms_data}, 总任务数 {len(processing_tasks)}", flush=True)
    
    return processing_tasks

def _load_firms_data_worker(date, data_dir):
    """工作进程中加载FIRMS数据"""
    import os
    import glob
    process_id = os.getpid()
    
    try:
        # 构建FIRMS文件路径
        date_str = date.strftime('%Y_%m_%d')
        firms_dir = os.path.join(data_dir, 'Firms_Detection_resampled')
        firms_pattern = os.path.join(firms_dir, f'*{date_str}.tif')
        firms_files = glob.glob(firms_pattern)
        
        # 调试信息：打印第一个日期的详细信息
        if date.strftime('%Y%m%d') == '20200101':
            print(f"[进程 {process_id}] 调试 - FIRMS目录: {firms_dir}", flush=True)
            print(f"[进程 {process_id}] 调试 - 搜索模式: {firms_pattern}", flush=True)
            print(f"[进程 {process_id}] 调试 - 找到文件: {firms_files}", flush=True)
            print(f"[进程 {process_id}] 调试 - 目录存在: {os.path.exists(firms_dir)}", flush=True)
            if os.path.exists(firms_dir):
                all_files = os.listdir(firms_dir)[:10]  # 只显示前10个文件
                print(f"[进程 {process_id}] 调试 - 目录中的文件样例: {all_files}", flush=True)
        
        if not firms_files:
            return None
        
        from osgeo import gdal
        dataset = gdal.Open(firms_files[0], gdal.GA_ReadOnly)
        if dataset is None:
            return None
            
        if dataset.RasterCount == 1:
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray()
        else:
            data = []
            for i in range(1, dataset.RasterCount + 1):
                band = dataset.GetRasterBand(i)
                band_data = band.ReadAsArray()
                data.append(band_data)
            data = np.array(data)
        
        dataset = None
        return data
    except Exception as e:
        if date.strftime('%Y%m%d') == '20200101':
            print(f"[进程 {process_id}] 调试 - 加载FIRMS数据异常: {str(e)}", flush=True)
        return None

def _get_valid_pixels_from_firms_worker(firms_data):
    """工作进程中从FIRMS数据获取有效像素"""
    if len(firms_data.shape) == 3:
        firms_data = firms_data[0]
    
    valid_pixels = []
    nodata_values = [255]
    
    # 使用numpy向量化操作
    valid_mask = ~np.isin(firms_data, nodata_values) & ~np.isnan(firms_data)
    valid_coords = np.where(valid_mask)
    
    for i in range(len(valid_coords[0])):
        row, col = valid_coords[0][i], valid_coords[1][i]
        firms_value = int(firms_data[row, col])
        valid_pixels.append(((row, col), firms_value))
    
    return valid_pixels

def _get_time_series_streaming_worker(driver_dir, event_date, pixel_coord, all_date_to_file, 
                                    memory_cache, past_days, future_days):
    """工作进程中获取时间序列数据"""
    import os
    process_id = os.getpid()
    
    # 调试计数器 - 只对特定像素提供详细调试
    debug_this_pixel = (pixel_coord[0] < 5 and pixel_coord[1] < 5)
    
    try:
        # 获取需要的日期
        past_dates, future_dates = _get_past_future_dates_standalone(event_date, past_days, future_days)
        if past_dates is None or future_dates is None:
            if debug_this_pixel:
                print(f"[进程 {process_id}] 调试 - 像素({pixel_coord[0]},{pixel_coord[1]}) 日期{event_date} 时间序列长度不足", flush=True)
            return None, None
        
        row, col = pixel_coord
        
        # 处理过去的数据
        past_series = []
        failed_past_dates = 0
        for date in past_dates:
            data = _load_driver_data_streaming_worker(driver_dir, date, memory_cache)
            if data is None:
                failed_past_dates += 1
                if debug_this_pixel and failed_past_dates <= 3:
                    print(f"[进程 {process_id}] 调试 - 像素({row},{col}) 无法加载过去日期数据: {date}", flush=True)
                if failed_past_dates > 10:  # 如果失败太多，直接返回
                    return None, None
                continue
            
            pixel_value = data[:, row, col] if len(data.shape) == 3 else data[row, col]
            past_series.append(pixel_value)
        
        if len(past_series) < past_days * 0.9:  # 如果缺失数据超过10%，认为无效
            if debug_this_pixel:
                print(f"[进程 {process_id}] 调试 - 像素({row},{col}) 过去数据不足: {len(past_series)}/{past_days}", flush=True)
            return None, None
        
        # 处理未来的数据
        future_series = []
        failed_future_dates = 0
        for date in future_dates:
            data = _load_driver_data_streaming_worker(driver_dir, date, memory_cache)
            if data is None:
                failed_future_dates += 1
                if debug_this_pixel and failed_future_dates <= 3:
                    print(f"[进程 {process_id}] 调试 - 像素({row},{col}) 无法加载未来日期数据: {date}", flush=True)
                if failed_future_dates > 5:  # 未来数据要求更严格
                    return None, None
                continue
            
            pixel_value = data[:, row, col] if len(data.shape) == 3 else data[row, col]
            future_series.append(pixel_value)
        
        if len(future_series) < future_days * 0.9:  # 如果缺失数据超过10%，认为无效
            if debug_this_pixel:
                print(f"[进程 {process_id}] 调试 - 像素({row},{col}) 未来数据不足: {len(future_series)}/{future_days}", flush=True)
            return None, None
        
        # 转换为numpy数组
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
        
        if debug_this_pixel:
            print(f"[进程 {process_id}] 调试 - 像素({row},{col}) 时间序列成功: past shape={past_array.shape}, future shape={future_array.shape}", flush=True)
        
        return past_array, future_array
        
    except Exception as e:
        if debug_this_pixel:
            print(f"[进程 {process_id}] 调试 - 像素({pixel_coord[0]},{pixel_coord[1]}) 时间序列异常: {str(e)}", flush=True)
        return None, None

def _load_driver_data_streaming_worker(driver_dir, date, memory_cache):
    """工作进程中流式加载驱动因素数据"""
    cache_key = f"{os.path.basename(driver_dir)}_{date.strftime('%Y%m%d')}"
    
    def loader():
        date_str = date.strftime('%Y_%m_%d')
        pattern = os.path.join(driver_dir, f'*{date_str}.tif')
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            return None
        
        try:
            from osgeo import gdal
            dataset = gdal.Open(matching_files[0], gdal.GA_ReadOnly)
            if dataset is None:
                return None
            
            if dataset.RasterCount == 1:
                band = dataset.GetRasterBand(1)
                data = band.ReadAsArray()
            else:
                data = []
                for i in range(1, dataset.RasterCount + 1):
                    band = dataset.GetRasterBand(i)
                    band_data = band.ReadAsArray()
                    data.append(band_data)
                data = np.array(data)
            
            dataset = None
            return data
        except:
            return None
    
    return memory_cache.get(cache_key, loader)

def main():
    """主函数"""
    
    # ================== 配置参数 ==================
    # 基础路径配置
    data_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked_normalized'
    output_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples'
    
    # ================== 时间窗口模式配置 ==================
    # 启用时间窗口模式以生成指定时间段的数据集
    time_windows_mode = True
    
    if time_windows_mode:
        # 时间窗口配置：2024年7月13日到7月19日，滑动窗口
        from datetime import datetime
        window_start_date = datetime(2024, 7, 13)  # 第一个窗口的起点
        window_end_date = datetime(2024, 7, 19)    # 最后一个窗口的起点
        window_shift_days = 1                      # 窗口滑动天数
        
        # 时间序列参数（时间窗口模式）
        past_days = 365      # 过去天数（每个窗口起点前365天）
        future_days = 7      # 未来天数（固定7天窗口）
        
        logger.info("=" * 60)
        logger.info("时间窗口模式配置:")
        logger.info(f"窗口起始日期: {window_start_date.strftime('%Y-%m-%d')}")
        logger.info(f"窗口结束日期: {window_end_date.strftime('%Y-%m-%d')}")
        logger.info(f"窗口滑动天数: {window_shift_days}")
        logger.info(f"过去天数: {past_days}")
        logger.info(f"未来天数: {future_days} (每个窗口包括首尾)")
        
        # 计算将生成的数据集数量
        from datetime import timedelta
        num_windows = (window_end_date - window_start_date).days + 1
        logger.info(f"将生成 {num_windows} 个时间窗口数据集")
        
        # 显示所有窗口
        current_start = window_start_date
        window_idx = 1
        while current_start <= window_end_date:
            window_end = current_start + timedelta(days=6)  # 7天窗口
            logger.info(f"  窗口 {window_idx}: {current_start.strftime('%Y-%m-%d')} 到 {window_end.strftime('%Y-%m-%d')}")
            current_start += timedelta(days=window_shift_days)
            window_idx += 1
            
        logger.info("=" * 60)
        
        # 时间窗口模式不需要年份配置
        sample_years = None
        full_years = None
        
    else:
        # 原有的年份模式配置
        window_start_date = None
        window_end_date = None
        window_shift_days = None
        
        # 时间序列参数（年份模式）
        past_days = 30       # 历史天数
        future_days = 30     # 未来天数
        
        # 指定年份（None表示使用默认规则）
        sample_years = None        # 样本数据年份，如 [2001, 2002, 2003] 或 None（所有年份）
        full_years = None          # 完整数据年份，如 [2020, 2021, 2022] 或 None（默认2020-2024）
    
    # 共同配置
    negative_ratio = 4.0 # 负样本比例
    
    # 数据生成控制

    
    # 并行处理配置  
    use_parallel_processing = True  # 是否启用并行处理
    n_processes = 9                # 并行进程数
    
    if not time_windows_mode:
        logger.info("=" * 60)
        logger.info("像素级采样配置（年份模式）:")
        logger.info(f"数据目录: {data_dir}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"抽样数据历史天数: 365")
        logger.info(f"完整数据历史天数: {past_days}")
        logger.info(f"未来天数: {future_days}")
        logger.info(f"负样本比例: {negative_ratio}")

        logger.info(f"并行处理: {use_parallel_processing}")
        if use_parallel_processing:
            logger.info(f"并行进程数: {n_processes}")
        logger.info(f"样本数据年份: {sample_years if sample_years else '所有可用年份'}")
        logger.info(f"完整数据年份: {full_years if full_years else '默认2020-2024年'}")
        logger.info("=" * 60)
    
    # 创建采样器
    sampler = PixelSampler(
        data_dir=data_dir,
        output_dir=output_dir,
        sample_past_days=365,                    # 抽样数据使用365天
        full_past_days=past_days,               # 完整数据历史天数
        future_days=future_days,                # 未来天数
        negative_ratio=negative_ratio,

        sample_years=sample_years,
        use_parallel=use_parallel_processing,
        n_processes=n_processes,
        time_windows_mode=time_windows_mode,    # 新增：时间窗口模式
        window_start_date=window_start_date,    # 新增：窗口起始日期
        window_end_date=window_end_date,        # 新增：窗口结束日期
        window_shift_days=window_shift_days     # 新增：窗口滑动天数
    )
    
    # 开始处理
    logger.info("开始像素级采样...")
    sampler.process_all_years()
    logger.info("像素级采样完成！")

def _preload_all_driver_data_optimized_standalone(driver_dirs, all_date_to_file, current_year_dates, past_days, future_days):
    """独立函数：预加载所有驱动因素的所有需要数据到内存 - 大内存优化"""
    import logging
    logger = logging.getLogger('PixelSampling')
    
    logger.info("开始预加载所有驱动因素数据...")
    
    # 计算所有需要的日期
    all_required_dates = set()
    for date in current_year_dates:
        past_dates, future_dates = _get_past_future_dates_standalone(date, past_days, future_days)
        if past_dates and future_dates:
            all_required_dates.update(past_dates + [date] + future_dates)
    
    all_required_dates = sorted(all_required_dates)
    logger.info(f"总共需要加载 {len(all_required_dates)} 个日期的数据")
    
    # 预加载所有驱动因素的所有日期数据
    all_driver_data = {}
    total_files = len(driver_dirs) * len(all_required_dates)
    loaded_files = 0
    
    for driver_name, driver_dir in driver_dirs.items():
        logger.info(f"预加载驱动因素: {driver_name}")
        driver_data = {}
        
        for date in all_required_dates:
            try:
                # 使用已有的数据加载函数
                date_str = date.strftime('%Y_%m_%d')
                pattern = os.path.join(driver_dir, f'*{date_str}.tif')
                matching_files = glob.glob(pattern)
                
                if matching_files:
                    data, geo_info = _load_single_file_with_gdal_standalone(matching_files[0])
                    if data is not None:
                        driver_data[date] = data
                        loaded_files += 1
            except Exception as e:
                logger.warning(f"加载数据失败: {driver_name}, {date}, 错误: {str(e)}")
            
            # 报告进度
            if loaded_files % 1000 == 0:
                progress = loaded_files / total_files * 100
                logger.info(f"预加载进度: {progress:.1f}% ({loaded_files}/{total_files})")
        
        all_driver_data[driver_name] = driver_data
        logger.info(f"驱动因素 {driver_name} 预加载完成: {len(driver_data)}/{len(all_required_dates)} 个日期")
    
    # 估算内存使用
    total_memory_mb = 0
    for driver_name, driver_data in all_driver_data.items():
        for date, data in driver_data.items():
            if data is not None and hasattr(data, 'nbytes'):
                total_memory_mb += data.nbytes / (1024 * 1024)
    
    logger.info(f"预加载完成！总内存使用: {total_memory_mb:.2f} MB ({total_memory_mb/1024:.2f} GB)")
    return all_driver_data

def _load_single_file_with_gdal_standalone(file_path):
    """独立函数：使用GDAL加载单个文件"""
    try:
        from osgeo import gdal
        import numpy as np
        
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
        return None, None

def _process_single_date_preloaded_worker(work_args):
    """工作进程函数：使用预加载数据处理单个日期的完整数据"""
    import os
    import gc
    
    process_id = os.getpid()
    
    try:
        date = work_args['date']
        h5_path = work_args['h5_path']
        all_driver_data = work_args['all_driver_data']
        target_year = work_args['target_year']
        data_shape = work_args['data_shape']
        geo_reference = work_args['geo_reference']
        past_days = work_args['past_days']
        future_days = work_args['future_days']
        data_dir = work_args['data_dir']
        
        date_str = date.strftime('%Y_%m_%d')
        print(f"[进程 {process_id}] 开始处理日期: {date_str} (使用预加载数据)", flush=True)
        
        # 调用使用预加载数据的独立处理函数
        result = _process_single_date_preloaded_standalone(
            date, h5_path, all_driver_data,
            target_year, data_shape, geo_reference, 
            past_days, future_days, data_dir
        )
        
        result['date_str'] = date_str
        print(f"[进程 {process_id}] 日期 {date_str} 处理完成: {result.get('success', False)}", flush=True)
        return result
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        date_str = work_args['date'].strftime('%Y_%m_%d') if 'date' in work_args else 'unknown'
        print(f"[进程 {process_id}] 处理日期 {date_str} 时出错:", flush=True)
        print(error_detail, flush=True)
        return {
            'success': False,
            'date_str': date_str,
            'error': f"{str(e)}\n{error_detail}"
        }

def _process_single_date_preloaded_standalone(date, h5_path, all_driver_data,
                                            target_year, data_shape, geo_reference, 
                                            past_days, future_days, data_dir):
    """独立函数：使用预加载数据处理单个日期的完整数据，包含所有驱动因素"""
    import os
    import gc
    from tqdm import tqdm
    
    process_id = os.getpid()
    date_str = date.strftime('%Y_%m_%d')
    
    try:
        print(f"[进程 {process_id}] {date_str}: 开始使用预加载数据处理单日完整数据", flush=True)
        
        # 获取该日期需要的时间序列日期
        past_dates, future_dates = _get_past_future_dates_standalone(date, past_days, future_days)
        if past_dates is None or future_dates is None:
            return {
                'success': False,
                'error': f'日期 {date_str} 时间序列长度不足'
            }
        
        # 检查预加载数据中是否包含所有需要的日期
        all_required_dates = past_dates + [date] + future_dates
        
        # 检查每个驱动因素是否有所有需要的日期数据
        missing_data = False
        for driver_name, driver_data in all_driver_data.items():
            for req_date in all_required_dates:
                if req_date not in driver_data or driver_data[req_date] is None:
                    print(f"[进程 {process_id}] {date_str}: 驱动因素 {driver_name} 缺少日期 {req_date} 的数据", flush=True)
                    missing_data = True
                    break
            if missing_data:
                break
        
        if missing_data:
            return {
                'success': False,
                'error': f'日期 {date_str} 预加载数据中缺少必需的时间序列数据'
            }
        
        # 加载FIRMS数据以确定有效像素
        firms_data = _load_firms_data_worker(date, data_dir)
        if firms_data is None:
            return {
                'success': False,
                'error': f'日期 {date_str} 无法加载FIRMS数据'
            }
        
        # 获取有效像素位置
        valid_pixels = _get_valid_pixels_from_firms_worker(firms_data)
        print(f"[进程 {process_id}] {date_str}: 找到 {len(valid_pixels)} 个有效像素", flush=True)
        
        if not valid_pixels:
            return {
                'success': False,
                'error': f'日期 {date_str} 没有有效像素'
            }
        
        print(f"[进程 {process_id}] {date_str}: 预加载数据准备完成，开始生成H5文件", flush=True)
        
        dataset_count = 0
        processed_pixels = 0
        
        with h5py.File(h5_path, 'w') as h5_file:
            # 写入全局属性
            h5_file.attrs['date'] = date_str
            h5_file.attrs['year'] = str(target_year)
            h5_file.attrs['past_days'] = past_days
            h5_file.attrs['future_days'] = future_days
            h5_file.attrs['sampling_type'] = 'pixel_level'
            h5_file.attrs['data_type'] = 'single_date_full_data_preloaded'
            h5_file.attrs['num_drivers'] = len(all_driver_data)
            h5_file.attrs['driver_names'] = list(all_driver_data.keys())
            
            if geo_reference:
                _save_geo_reference_standalone(h5_file, geo_reference)
            
            # 处理每个有效像素
            for pixel_idx, (pixel_coord, firms_value) in enumerate(tqdm(valid_pixels, desc=f"处理{date_str}像素", disable=False)):
                row, col = pixel_coord
                
                # 为每个驱动因素提取该像素的时间序列
                merged_past_data = []
                merged_future_data = []
                valid_pixel = True
                
                for driver_name in all_driver_data.keys():
                    driver_data = all_driver_data[driver_name]
                    
                    # 提取过去时间序列
                    past_series = []
                    for ts_date in past_dates:
                        data = driver_data[ts_date]
                        if len(data.shape) == 3:
                            pixel_value = data[:, row, col]  # 多波段
                        else:
                            pixel_value = data[row, col]     # 单波段
                        past_series.append(pixel_value)
                    
                    # 提取未来时间序列
                    future_series = []
                    for ts_date in future_dates:
                        data = driver_data[ts_date]
                        if len(data.shape) == 3:
                            pixel_value = data[:, row, col]  # 多波段
                        else:
                            pixel_value = data[row, col]     # 单波段
                        future_series.append(pixel_value)
                    
                    # 合并当前驱动因素的数据
                    if past_series and future_series:
                        # 处理过去数据
                        if np.isscalar(past_series[0]):
                            past_array = np.array(past_series, dtype=np.float32)
                        else:
                            past_array = np.array(past_series, dtype=np.float32)
                            if past_array.ndim == 2:
                                past_array = past_array.T  # 转置为 (bands, time_steps)
                        
                        # 处理未来数据
                        if np.isscalar(future_series[0]):
                            future_array = np.array(future_series, dtype=np.float32)
                        else:
                            future_array = np.array(future_series, dtype=np.float32)
                            if future_array.ndim == 2:
                                future_array = future_array.T  # 转置为 (bands, time_steps)
                        
                        # 扩展维度以便合并
                        if past_array.ndim == 1:
                            past_array = past_array.reshape(1, -1)  # (1, time_steps)
                        if future_array.ndim == 1:
                            future_array = future_array.reshape(1, -1)  # (1, time_steps)
                        
                        merged_past_data.append(past_array)
                        merged_future_data.append(future_array)
                
                # 只保存有效的像素数据
                if valid_pixel and merged_past_data and merged_future_data:
                    # 合并所有驱动因素的数据
                    final_past_data = np.concatenate(merged_past_data, axis=0)  # (total_bands, past_time_steps)
                    final_future_data = np.concatenate(merged_future_data, axis=0)  # (total_bands, future_time_steps)
                    
                    # 保存数据集
                    past_dataset_name = f"{date_str}_past_{firms_value}_{row}_{col}"
                    h5_file.create_dataset(
                        past_dataset_name,
                        data=final_past_data,
                        dtype=np.float32,
                        compression=None,
                        shuffle=False
                    )
                    dataset_count += 1
                    
                    future_dataset_name = f"{date_str}_future_{firms_value}_{row}_{col}"
                    h5_file.create_dataset(
                        future_dataset_name,
                        data=final_future_data,
                        dtype=np.float32,
                        compression=None,
                        shuffle=False
                    )
                    dataset_count += 1
                    processed_pixels += 1
                
                # 定期垃圾回收
                if (pixel_idx + 1) % 1000 == 0:
                    gc.collect()
            
            h5_file.attrs['total_datasets'] = dataset_count
            h5_file.attrs['processed_pixels'] = processed_pixels
            h5_file.attrs['valid_pixels'] = len(valid_pixels)
        
        print(f"[进程 {process_id}] {date_str}: 使用预加载数据处理完成，数据集={dataset_count}, 像素={processed_pixels}", flush=True)
        
        return {
            'success': True,
            'dataset_count': dataset_count,
            'processed_pixels': processed_pixels
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[进程 {process_id}] {date_str}: 使用预加载数据处理失败", flush=True)
        print(error_detail, flush=True)
        return {
            'success': False,
            'error': f"{str(e)}\n{error_detail}"
        }

def _process_single_date_full_data_worker(work_args):
    """工作进程函数：处理单个日期的完整数据"""
    import os
    import gc
    
    process_id = os.getpid()
    
    try:
        date = work_args['date']
        h5_path = work_args['h5_path']
        all_date_to_file = work_args['all_date_to_file']
        driver_dirs = work_args['driver_dirs']
        target_year = work_args['target_year']
        data_shape = work_args['data_shape']
        geo_reference = work_args['geo_reference']
        past_days = work_args['past_days']
        future_days = work_args['future_days']
        data_dir = work_args['data_dir']
        
        date_str = date.strftime('%Y_%m_%d')
        print(f"[进程 {process_id}] 开始处理日期: {date_str}", flush=True)
        
        # 调用独立处理函数
        result = _process_single_date_full_data_standalone(
            date, h5_path, all_date_to_file, driver_dirs,
            target_year, data_shape, geo_reference, 
            past_days, future_days, data_dir
        )
        
        result['date_str'] = date_str
        print(f"[进程 {process_id}] 日期 {date_str} 处理完成: {result.get('success', False)}", flush=True)
        return result
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        date_str = work_args['date'].strftime('%Y_%m_%d') if 'date' in work_args else 'unknown'
        print(f"[进程 {process_id}] 处理日期 {date_str} 时出错:", flush=True)
        print(error_detail, flush=True)
        return {
            'success': False,
            'date_str': date_str,
            'error': f"{str(e)}\n{error_detail}"
        }

def _process_single_date_full_data_standalone(date, h5_path, all_date_to_file, driver_dirs,
                                            target_year, data_shape, geo_reference, 
                                            past_days, future_days, data_dir):
    """独立函数：处理单个日期的完整数据，包含所有驱动因素"""
    import os
    import gc
    from tqdm import tqdm
    
    process_id = os.getpid()
    date_str = date.strftime('%Y_%m_%d')
    
    try:
        print(f"[进程 {process_id}] {date_str}: 开始处理单日完整数据", flush=True)
        
        # 获取该日期需要的时间序列日期
        past_dates, future_dates = _get_past_future_dates_standalone(date, past_days, future_days)
        if past_dates is None or future_dates is None:
            return {
                'success': False,
                'error': f'日期 {date_str} 时间序列长度不足'
            }
        
        # 检查所有需要的日期数据是否存在
        all_required_dates = past_dates + [date] + future_dates
        available_dates_set = set(all_date_to_file.keys())
        missing_dates = [d for d in all_required_dates if d not in available_dates_set]
        if missing_dates:
            return {
                'success': False,
                'error': f'日期 {date_str} 缺少 {len(missing_dates)} 个必需日期'
            }
        
        # 加载FIRMS数据以确定有效像素
        firms_data = _load_firms_data_worker(date, data_dir)
        if firms_data is None:
            return {
                'success': False,
                'error': f'日期 {date_str} 无法加载FIRMS数据'
            }
        
        # 获取有效像素位置
        valid_pixels = _get_valid_pixels_from_firms_worker(firms_data)
        print(f"[进程 {process_id}] {date_str}: 找到 {len(valid_pixels)} 个有效像素", flush=True)
        
        if not valid_pixels:
            return {
                'success': False,
                'error': f'日期 {date_str} 没有有效像素'
            }
        
        # 预加载该日期所有驱动因素的时间序列数据
        all_driver_data = {}
        for driver_name, driver_dir in driver_dirs.items():
            driver_data = {}
            for ts_date in all_required_dates:
                data = _load_driver_data_streaming_worker(driver_dir, ts_date, StreamingCache(max_size_gb=5))
                if data is not None:
                    driver_data[ts_date] = data
            all_driver_data[driver_name] = driver_data
        
        print(f"[进程 {process_id}] {date_str}: 预加载完成，开始生成H5文件", flush=True)
        
        dataset_count = 0
        processed_pixels = 0
        
        with h5py.File(h5_path, 'w') as h5_file:
            # 写入全局属性
            h5_file.attrs['date'] = date_str
            h5_file.attrs['year'] = str(target_year)
            h5_file.attrs['past_days'] = past_days
            h5_file.attrs['future_days'] = future_days
            h5_file.attrs['sampling_type'] = 'pixel_level'
            h5_file.attrs['data_type'] = 'single_date_full_data'
            h5_file.attrs['num_drivers'] = len(driver_dirs)
            h5_file.attrs['driver_names'] = list(driver_dirs.keys())
            
            if geo_reference:
                _save_geo_reference_standalone(h5_file, geo_reference)
            
            # 处理每个有效像素
            for pixel_idx, (pixel_coord, firms_value) in enumerate(tqdm(valid_pixels, desc=f"处理{date_str}像素")):
                row, col = pixel_coord
                
                # 为每个驱动因素提取该像素的时间序列
                merged_past_data = []
                merged_future_data = []
                valid_pixel = True
                
                for driver_name in driver_dirs.keys():
                    driver_data = all_driver_data[driver_name]
                    
                    # 提取过去时间序列
                    past_series = []
                    for ts_date in past_dates:
                        if ts_date not in driver_data or driver_data[ts_date] is None:
                            valid_pixel = False
                            break
                        
                        data = driver_data[ts_date]
                        if len(data.shape) == 3:
                            pixel_value = data[:, row, col]  # 多波段
                        else:
                            pixel_value = data[row, col]     # 单波段
                        past_series.append(pixel_value)
                    
                    if not valid_pixel:
                        break
                    
                    # 提取未来时间序列
                    future_series = []
                    for ts_date in future_dates:
                        if ts_date not in driver_data or driver_data[ts_date] is None:
                            valid_pixel = False
                            break
                        
                        data = driver_data[ts_date]
                        if len(data.shape) == 3:
                            pixel_value = data[:, row, col]  # 多波段
                        else:
                            pixel_value = data[row, col]     # 单波段
                        future_series.append(pixel_value)
                    
                    if not valid_pixel:
                        break
                    
                    # 合并当前驱动因素的数据
                    if past_series and future_series:
                        # 处理过去数据
                        if np.isscalar(past_series[0]):
                            past_array = np.array(past_series, dtype=np.float32)
                        else:
                            past_array = np.array(past_series, dtype=np.float32)
                            if past_array.ndim == 2:
                                past_array = past_array.T  # 转置为 (bands, time_steps)
                        
                        # 处理未来数据
                        if np.isscalar(future_series[0]):
                            future_array = np.array(future_series, dtype=np.float32)
                        else:
                            future_array = np.array(future_series, dtype=np.float32)
                            if future_array.ndim == 2:
                                future_array = future_array.T  # 转置为 (bands, time_steps)
                        
                        # 扩展维度以便合并
                        if past_array.ndim == 1:
                            past_array = past_array.reshape(1, -1)  # (1, time_steps)
                        if future_array.ndim == 1:
                            future_array = future_array.reshape(1, -1)  # (1, time_steps)
                        
                        merged_past_data.append(past_array)
                        merged_future_data.append(future_array)
                
                # 只保存有效的像素数据
                if valid_pixel and merged_past_data and merged_future_data:
                    # 合并所有驱动因素的数据
                    final_past_data = np.concatenate(merged_past_data, axis=0)  # (total_bands, past_time_steps)
                    final_future_data = np.concatenate(merged_future_data, axis=0)  # (total_bands, future_time_steps)
                    
                    # 保存数据集
                    past_dataset_name = f"{date_str}_past_{firms_value}_{row}_{col}"
                    h5_file.create_dataset(
                        past_dataset_name,
                        data=final_past_data,
                        dtype=np.float32,
                        compression=None,
                        shuffle=False
                    )
                    dataset_count += 1
                    
                    future_dataset_name = f"{date_str}_future_{firms_value}_{row}_{col}"
                    h5_file.create_dataset(
                        future_dataset_name,
                        data=final_future_data,
                        dtype=np.float32,
                        compression=None,
                        shuffle=False
                    )
                    dataset_count += 1
                    processed_pixels += 1
                
                # 定期垃圾回收
                if (pixel_idx + 1) % 1000 == 0:
                    gc.collect()
            
            h5_file.attrs['total_datasets'] = dataset_count
            h5_file.attrs['processed_pixels'] = processed_pixels
            h5_file.attrs['valid_pixels'] = len(valid_pixels)
        
        print(f"[进程 {process_id}] {date_str}: 处理完成，数据集={dataset_count}, 像素={processed_pixels}", flush=True)
        
        return {
            'success': True,
            'dataset_count': dataset_count,
            'processed_pixels': processed_pixels
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[进程 {process_id}] {date_str}: 处理失败", flush=True)
        print(error_detail, flush=True)
        return {
            'success': False,
            'error': f"{str(e)}\n{error_detail}"
        }

def _process_driver_with_preloaded_data_worker(work_args):
    """工作进程函数：使用预加载数据处理驱动因素"""
    import os
    import gc
    
    process_id = os.getpid()
    
    try:
        h5_path = work_args['h5_path']
        driver_name = work_args['driver_name']
        all_driver_data = work_args['all_driver_data']
        current_year_dates = work_args['current_year_dates']
        target_year = work_args['target_year']
        data_shape = work_args['data_shape']
        geo_reference = work_args['geo_reference']
        past_days = work_args['past_days']
        future_days = work_args['future_days']
        
        print(f"[进程 {process_id}] {driver_name}: 开始使用预加载数据处理", flush=True)
        
        height, width = data_shape[:2]
        
        # 获取有效像素位置
        valid_pixels = _get_valid_pixel_positions_standalone(all_driver_data, height, width)
        print(f"[进程 {process_id}] {driver_name}: 有效像素数 = {len(valid_pixels)}", flush=True)
        
        # 准备有效日期数据
        valid_date_data = {}
        for date in current_year_dates:
            past_dates, future_dates = _get_past_future_dates_standalone(date, past_days, future_days)
            if past_dates is None or future_dates is None:
                continue
            
            # 获取FIRMS数据
            firms_data = None
            if 'Firms_Detection_resampled' in all_driver_data:
                firms_data = all_driver_data['Firms_Detection_resampled'].get(date)
            
            valid_date_data[date] = {
                'past_dates': past_dates,
                'future_dates': future_dates,
                'firms_data': firms_data,
                'date_str': date.strftime('%Y%m%d')
            }
        
        # 获取当前驱动因素的数据
        driver_data = all_driver_data[driver_name]
        
        # 直接调用高效处理逻辑（该函数内部会处理H5文件的创建和写入）
        result = _create_full_data_ultra_fast_standalone(
            h5_path, driver_name, driver_data, valid_pixels, valid_date_data, 
            data_shape, geo_reference, target_year, past_days, future_days
        )
        
        return result
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        driver_name = work_args.get('driver_name', 'unknown')
        print(f"[进程 {process_id}] 处理驱动因素 {driver_name} 时出错:", flush=True)
        print(error_detail, flush=True)
        return {
            'success': False,
            'driver_name': driver_name,
            'error': f"{str(e)}\n{error_detail}"
        }

if __name__ == "__main__":
    main() 