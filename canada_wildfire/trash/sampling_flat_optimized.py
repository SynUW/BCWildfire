import os
import logging
import numpy as np
import h5py
from osgeo import gdal
import glob
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import concurrent.futures
from collections import defaultdict
import gc
from functools import lru_cache
import psutil
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('WildfireSampling')

class DataCache:
    """高效的数据缓存管理器"""
    def __init__(self, max_memory_gb=40):
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

class OptimizedWindowSampler:
    def __init__(self, 
                 driver_factors_dir,
                 output_dir,
                 window_size=64,
                 stride=32,
                 min_fire_pixels=5,
                 past_days=14,
                 future_days=7,
                 negative_ratio=2,
                 max_cache_memory_gb=8,
                 batch_size=100):
        """
        优化版采样器，支持数据压缩和-9999值处理
        
        参数:
            driver_factors_dir: 驱动因素数据目录
            output_dir: 输出目录
            window_size: 窗口大小
            stride: 滑动步长
            min_fire_pixels: 最小火灾像素数
            past_days: 历史数据天数
            future_days: 预测天数
            negative_ratio: 负样本比例
            max_cache_memory_gb: 最大缓存内存(GB)
            batch_size: 批处理大小
        """
        self.driver_factors_dir = driver_factors_dir
        self.output_dir = output_dir
        self.window_size = window_size
        self.stride = stride
        self.min_fire_pixels = min_fire_pixels
        self.past_days = past_days
        self.future_days = future_days
        self.negative_ratio = negative_ratio
        self.batch_size = batch_size
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化缓存
        self.data_cache = DataCache(max_cache_memory_gb)
        
        # 获取driver目录列表
        self.driver_dirs = [d for d in os.listdir(self.driver_factors_dir) 
                           if os.path.isdir(os.path.join(self.driver_factors_dir, d))]
        logger.info(f"找到 {len(self.driver_dirs)} 个driver目录: {self.driver_dirs}")
        
    def _process_invalid_values(self, data):
        """处理无效值：将-9999替换为0，处理NaN和无穷大值"""
        if data is None:
            return None
            
        # 将-9999替换为0
        data = np.where(data == -9999, 0, data)
        
        # 处理其他异常值
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        return data
        
    def _load_driver_data_cached(self, driver_dir, date):
        """带缓存的driver数据加载，支持数据压缩和无效值处理"""
        cache_key = f"{driver_dir}_{date.strftime('%Y_%m_%d')}"
        
        def loader():
            try:
                date_str = date.strftime('%Y_%m_%d')
                file_pattern = os.path.join(self.driver_factors_dir, driver_dir, f'*_{date_str}.tif')
                files = glob.glob(file_pattern)
                
                if not files:
                    return None
                
                # 优化GDAL设置
                gdal.SetConfigOption('GDAL_CACHEMAX', '2048')  # 减少到2GB GDAL缓存
                gdal.SetConfigOption('GDAL_NUM_THREADS', '4')   # 减少到4线程避免过载
                gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
                
                ds = gdal.Open(files[0])
                if ds is None:
                    return None
                    
                # 一次性读取所有波段
                data = ds.ReadAsArray()
                ds = None
                
                if data is not None:
                    # 处理无效值
                    data = self._process_invalid_values(data)
                    
                    # 对于FIRMS数据，保持原始格式
                    if 'Firms_Detection_resampled' in driver_dir:
                        return data.astype(np.uint8)
                    else:
                        # 使用更高效的数据类型转换
                        if data.dtype != np.float32:
                            data = data.astype(np.float32)
                        
                        # 批量处理数据压缩
                        scaled_data = data * 10000.0
                        scaled_data = np.clip(scaled_data, -32767, 32767)
                        return scaled_data.astype(np.int16)
                    
                return None
                
            except Exception as e:
                logger.error(f"加载driver数据出错: {driver_dir}, {date_str}, 错误: {str(e)}")
                return None
                
        return self.data_cache.get(cache_key, loader)
    
    def _convert_to_float(self, data):
        """将压缩的int16数据转换回float32"""
        if data is None:
            return None
        return data.astype(np.float32) / 10000.0
    
    def _get_data_dtype_and_scale(self, driver_dir):
        """获取数据类型和缩放因子"""
        # 对于FIRMS数据，保持原始格式（0/1值）
        if 'Firms_Detection_resampled' in driver_dir:
            return np.uint8, 1.0
        else:
            # 其他驱动因素使用压缩格式
            return np.int16, 10000.0

    def _is_valid_window(self, window, shape):
        """检查窗口是否有效"""
        x1, y1, x2, y2 = window
        height, width = shape
        return (x1 >= 0 and y1 >= 0 and 
                x2 <= width and y2 <= height and 
                x2 > x1 and y2 > y1)
                
    def _calculate_overlap_ratio(self, window1, window2):
        """计算两个窗口的重叠率"""
        x1, y1, x2, y2 = window1
        ex1, ey1, ex2, ey2 = window2
        
        overlap_x1 = max(x1, ex1)
        overlap_y1 = max(y1, ey1)
        overlap_x2 = min(x2, ex2)
        overlap_y2 = min(y2, ey2)
        
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return 0.0
            
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        window_area = (x2 - x1) * (y2 - y1)
        
        return overlap_area / window_area if window_area > 0 else 0.0
        
    def _sample_positive_windows_efficient(self, fire_data):
        """高效的正样本窗口采样"""
        fire_y, fire_x = np.where(fire_data == 1)
        if len(fire_x) == 0:
            return []
            
        # 聚类火点，减少重叠
        fire_points = list(zip(fire_x, fire_y))
        selected_windows = []
        
        # 使用网格化方法减少重叠检查
        grid_size = self.window_size // 2
        grid = {}
        
        for x, y in fire_points:
            grid_key = (x // grid_size, y // grid_size)
            if grid_key not in grid:
                grid[grid_key] = []
            grid[grid_key].append((x, y))
            
        # 从每个网格单元选择代表点
        for cell_points in grid.values():
            if not cell_points:
                continue
                
            # 选择网格中心最近的点
            center_x = sum(p[0] for p in cell_points) / len(cell_points)
            center_y = sum(p[1] for p in cell_points) / len(cell_points)
            
            best_point = min(cell_points, 
                           key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
            
            x, y = best_point
            window = (x - self.window_size//2, y - self.window_size//2,
                     x + self.window_size//2, y + self.window_size//2)
                     
            if self._is_valid_window(window, fire_data.shape):
                # 检查与已选窗口的重叠
                valid = True
                for existing_window in selected_windows:
                    if self._calculate_overlap_ratio(window, existing_window) > 0.5:
                        valid = False
                        break
                        
                if valid:
                    selected_windows.append(window)
                    
        return selected_windows
        
    def _sample_negative_windows_efficient(self, fire_data, positive_windows, target_count):
        """高效的负样本窗口采样"""
        height, width = fire_data.shape
        half_width = width // 2
        
        # 创建右半边的候选网格点
        step = max(1, self.stride)
        y_coords = np.arange(self.window_size//2, height - self.window_size//2, step)
        x_coords = np.arange(half_width + self.window_size//2, width - self.window_size//2, step)
        
        if len(y_coords) == 0 or len(x_coords) == 0:
            raise ValueError("无法创建有效的网格点")
            
        # 创建网格
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
        candidate_points = list(zip(Y.flatten(), X.flatten()))
        
        # 批量检查火灾像素
        valid_candidates = []
        for y, x in candidate_points:
            window = (x - self.window_size//2, y - self.window_size//2,
                     x + self.window_size//2, y + self.window_size//2)
            
            if self._is_valid_window(window, fire_data.shape):
                x1, y1, x2, y2 = window
                window_data = fire_data[y1:y2, x1:x2]
                if not np.any(window_data == 1):  # 无火灾像素
                    valid_candidates.append(window)
                    
        # 去除与正样本重叠的窗口
        final_windows = []
        for window in valid_candidates:
            valid = True
            for pos_window in positive_windows:
                if self._calculate_overlap_ratio(window, pos_window) > 0.1:
                    valid = False
                    break
            if valid:
                final_windows.append(window)
                
        # 检查负样本数量是否足够
        if len(final_windows) < target_count:
            raise ValueError(f"负样本数量不足: 需要 {target_count} 个，但只找到 {len(final_windows)} 个")
            
        # 使用固定的种子进行随机采样，确保结果一致
        np.random.seed(42)
        indices = np.random.choice(len(final_windows), target_count, replace=False)
        selected_windows = [final_windows[i] for i in indices]
        np.random.seed(None)  # 重置随机种子
        
        return selected_windows 

    @lru_cache(maxsize=1000)
    def _get_date_from_filename(self, filename):
        """从文件名中提取日期（带缓存）"""
        try:
            basename = os.path.basename(filename)
            parts = basename.split('_')
            if len(parts) >= 3:
                year = int(parts[-3])
                month = int(parts[-2])
                day = int(parts[-1].split('.')[0])
                return datetime(year, month, day)
            return None
        except Exception as e:
            logger.error(f"从文件名提取日期出错: {str(e)}")
            return None
            
    def _get_past_future_dates(self, date):
        """获取过去和未来的日期列表
        
        Returns:
            tuple: (past_dates, future_dates) 如果数据长度不足则返回 (None, None)
        """
        past_dates = []
        future_dates = []
        
        # 过去日期（按时间顺序排列）
        for i in range(self.past_days, 0, -1):
            past_dates.append(date - timedelta(days=i))
        
        # 未来日期（不包括当前日期）
        for i in range(1, self.future_days):
            future_date = date + timedelta(days=i)
            # 只添加2024年及之前的日期
            if future_date.year <= 2024:
                future_dates.append(future_date)
            else:
                # 如果遇到2025年的日期，直接返回None
                return None, None
            
        # 检查数据长度是否足够
        if len(past_dates) < self.past_days or len(future_dates) < self.future_days - 1:
            return None, None
            
        return past_dates, future_dates
        
    def _extract_time_series_datacube(self, window, driver_data, dates, event_date, firms_window=None):
        """为单个窗口提取时间序列数据立方体
        
        Args:
            window: 窗口坐标 (x1, y1, x2, y2)
            driver_data: 单个driver的数据字典 {date: data}
            dates: 日期列表
            event_date: 事件日期
            firms_window: FIRMS数据的窗口信息，用于判断窗口是否超限
            
        Returns:
            numpy array: shape为 (time_steps, channels, height, width)
            压缩格式的数据将保持int16类型，原始数据保持原类型
            
        Raises:
            ValueError: 当没有有效数据时抛出异常
        """
        x1, y1, x2, y2 = window
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
                height, width = data.shape[1:]
            else:
                height, width = data.shape
            
            # 检查窗口是否超出边界
            if firms_window is not None:
                # 如果提供了FIRMS窗口信息，使用它来判断是否超限
                fx1, fy1, fx2, fy2 = firms_window
                if (x1 < 0 or y1 < 0 or x2 > width or y2 > height) and \
                   (fx1 >= 0 and fy1 >= 0 and fx2 <= width and fy2 <= height):
                    raise ValueError(f"窗口在FIRMS数据中不超限，但在当前数据中超限: {window}, 数据尺寸: {width}x{height}")
                
                # 确保窗口位置与FIRMS完全一致
                if x1 != fx1 or y1 != fy1 or x2 != fx2 or y2 != fy2:
                    raise ValueError(f"窗口位置与FIRMS不一致: 当前窗口 {window}, FIRMS窗口 {firms_window}")
            
            # 创建全零数组作为基础（保持原始数据类型）
            if len(data.shape) == 3:
                window_data = np.zeros((data.shape[0], self.window_size, self.window_size), dtype=data_dtype)
            else:
                window_data = np.zeros((1, self.window_size, self.window_size), dtype=data_dtype)
            
            # 计算有效区域
            x1_valid = max(0, x1)
            y1_valid = max(0, y1)
            x2_valid = min(width, x2)
            y2_valid = min(height, y2)
            
            # 计算在window_data中的对应位置
            x1_offset = max(0, -x1)
            y1_offset = max(0, -y1)
            
            # 提取有效数据
            if len(data.shape) == 3:
                # 多波段数据
                valid_data = data[:, y1_valid:y2_valid, x1_valid:x2_valid]
                window_data[:, y1_offset:y1_offset+valid_data.shape[1], 
                          x1_offset:x1_offset+valid_data.shape[2]] = valid_data
            else:
                # 单波段数据
                valid_data = data[y1_valid:y2_valid, x1_valid:x2_valid]
                window_data[0, y1_offset:y1_offset+valid_data.shape[0], 
                          x1_offset:x1_offset+valid_data.shape[1]] = valid_data
            
            time_series_data.append(window_data)
        
        if not time_series_data:
            raise ValueError(f"窗口 {window} 在日期范围 {dates[0]} 到 {dates[-1]} 内没有有效数据")
            
        # 堆叠时间序列数据，保持原始数据类型
        return np.stack(time_series_data, axis=0) 

    def _preload_driver_data_batch(self, driver_dirs, all_dates, year, window_events):
        """批量预加载driver数据并创建时间序列数据立方体，支持内存优化和-9999值处理"""
        logger.info(f"开始处理 {len(window_events)} 个窗口事件")
        
        # 预先计算所有需要的日期，避免加载不必要的数据
        required_dates = set()
        for window, event_date, label in window_events:
            past_dates, future_dates = self._get_past_future_dates(event_date)
            if past_dates and future_dates:
                required_dates.update(past_dates)
                required_dates.update(future_dates)
                required_dates.add(event_date)
        
        # 过滤all_dates，只保留实际需要的日期
        filtered_dates = [date for date in all_dates if date in required_dates]
        logger.info(f"原始日期数: {len(all_dates)}, 实际需要的日期数: {len(filtered_dates)}")
        
        # 记录初始内存使用
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
        logger.info(f"开始处理前内存使用: {initial_memory:.2f} GB")
        
        # 首先处理FIRMS数据
        firms_dir = 'Firms_Detection_resampled'
        if firms_dir in driver_dirs:
            logger.info(f"首先处理FIRMS数据...")
            firms_data = {}
            
            # 加载FIRMS数据前的内存
            memory_before_firms = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
            
            # 并行加载FIRMS数据
            firms_args_list = [(self.driver_factors_dir, firms_dir, date) for date in filtered_dates]
            
            logger.info(f"开始并行加载FIRMS数据，共 {len(firms_args_list)} 个文件...")
            with ThreadPoolExecutor(max_workers=8) as executor:  # 减少线程数
                firms_results = list(tqdm(
                    executor.map(load_driver_data_global, firms_args_list),
                    total=len(firms_args_list),
                    desc="加载FIRMS数据",
                    ncols=100,  # 固定进度条宽度
                    unit="文件"
                ))
                
                # 处理FIRMS结果
                for date, data in firms_results:
                    if data is not None:
                        firms_data[date] = data
            
            logger.info(f"FIRMS数据加载完成，成功加载 {len(firms_data)} 个文件")
            
            # 处理FIRMS数据
            h5_path = os.path.join(self.output_dir, f'wildfire_samples_{year}_{firms_dir}.h5')
            if not os.path.exists(h5_path):
                with h5py.File(h5_path, 'w') as h5_file:
                    # 写入属性
                    h5_file.attrs['year'] = str(year)
                    h5_file.attrs['driver'] = firms_dir
                    h5_file.attrs['window_size'] = self.window_size
                    h5_file.attrs['past_days'] = self.past_days
                    h5_file.attrs['future_days'] = self.future_days
                    h5_file.attrs['invalid_value_handling'] = '-9999 replaced with 0'
                    
                    dataset_count = 0
                    
                    # 以第一个past和future的 shape[1:] 为基准
                    first_past_shape = None
                    first_future_shape = None
                    
                    # 处理每个窗口事件
                    for i, (window, event_date, label) in enumerate(tqdm(window_events, 
                                                                       desc=f"处理 {firms_dir} 窗口",
                                                                       leave=False)):
                        x1, y1, x2, y2 = window
                        event_date_str = event_date.strftime('%Y%m%d')
                        
                        # 获取过去和未来的日期
                        past_dates, future_dates = self._get_past_future_dates(event_date)
                        
                        # 如果数据长度不足，跳过该窗口
                        if past_dates is None or future_dates is None:
                            logger.warning(f"窗口 {window} 在日期 {event_date} 的数据长度不足，已跳过")
                            continue
                        
                        # 创建过去数据的时间序列立方体
                        if past_dates:
                            try:
                                past_datacube = self._extract_time_series_datacube(
                                    window, {date: firms_data.get(date) for date in past_dates}, 
                                    past_dates, event_date)
                                
                                if past_datacube is not None:
                                    if first_past_shape is None:
                                        first_past_shape = past_datacube.shape[1:]
                                    else:
                                        if past_datacube.shape[1:] != first_past_shape:
                                            logger.error(
                                                f"窗口 {window} 的 past shape {past_datacube.shape} 与第一个窗口 shape[1:] {first_past_shape} 不一致，已跳过！"
                                            )
                                            continue
                                    
                                    dataset_name = f"{event_date_str}_past_{label}_{x1}_{y1}_{x2}_{y2}"
                                    # 获取数据类型和缩放因子
                                    save_dtype, scale_factor = self._get_data_dtype_and_scale(firms_dir)
                                    
                                    dataset = h5_file.create_dataset(
                                        dataset_name,
                                        data=past_datacube,
                                        dtype=save_dtype,
                                        compression='gzip',
                                        compression_opts=6,
                                        shuffle=True
                                    )
                                    # 添加缩放因子属性
                                    dataset.attrs['scale_factor'] = scale_factor
                                    dataset.attrs['data_type'] = 'compressed' if scale_factor > 1 else 'original'
                                    dataset.attrs['invalid_value_handling'] = '-9999 replaced with 0'
                                    dataset_count += 1
                            except Exception as e:
                                logger.error(f"处理past数据时出错: {dataset_name}, 错误: {str(e)}")
                        
                        # 创建未来数据的时间序列立方体（包含当前日期）
                        future_dates_with_current = [event_date] + future_dates
                        if future_dates_with_current:
                            try:
                                future_datacube = self._extract_time_series_datacube(
                                    window, {date: firms_data.get(date) for date in future_dates_with_current}, 
                                    future_dates_with_current, event_date)
                                
                                if future_datacube is not None:
                                    if first_future_shape is None:
                                        first_future_shape = future_datacube.shape[1:]
                                    else:
                                        if future_datacube.shape[1:] != first_future_shape:
                                            logger.error(
                                                f"窗口 {window} 的 future shape {future_datacube.shape} 与第一个窗口 shape[1:] {first_future_shape} 不一致，已跳过！"
                                            )
                                            continue
                                    
                                    dataset_name = f"{event_date_str}_future_{label}_{x1}_{y1}_{x2}_{y2}"
                                    # 获取数据类型和缩放因子
                                    save_dtype, scale_factor = self._get_data_dtype_and_scale(firms_dir)
                                    
                                    dataset = h5_file.create_dataset(
                                        dataset_name,
                                        data=future_datacube,
                                        dtype=save_dtype,
                                        compression='gzip',
                                        compression_opts=6,
                                        shuffle=True
                                    )
                                    # 添加缩放因子属性
                                    dataset.attrs['scale_factor'] = scale_factor
                                    dataset.attrs['data_type'] = 'compressed' if scale_factor > 1 else 'original'
                                    dataset.attrs['invalid_value_handling'] = '-9999 replaced with 0'
                                    dataset_count += 1
                            except Exception as e:
                                logger.error(f"处理future数据时出错: {dataset_name}, 错误: {str(e)}")
                    
                    # 记录统计信息
                    h5_file.attrs['total_datasets'] = dataset_count
                    logger.info(f"成功保存 {year} 年 {firms_dir} 数据，共 {dataset_count} 个数据集")
            
            # 清理FIRMS数据
            firms_data.clear()
            gc.collect()
        
        # 处理其他驱动因素
        other_drivers = [d for d in driver_dirs if d != firms_dir]
        logger.info(f"开始处理其他 {len(other_drivers)} 个驱动因素...")
        
        for i, driver_dir in enumerate(other_drivers):
            logger.info(f"正在处理driver ({i+1}/{len(other_drivers)}): {driver_dir}")
            
            # 检查输出文件是否已存在
            h5_path = os.path.join(self.output_dir, f'wildfire_samples_{year}_{driver_dir}.h5')
            if os.path.exists(h5_path):
                logger.info(f"文件已存在，跳过处理: {h5_path}")
                continue
            
            # 为当前driver加载所有日期的数据
            driver_data = {}
            
            # 使用线程池进行并行加载 - I/O密集型任务使用线程更高效
            max_workers = min(8, len(filtered_dates))  # 大幅减少线程数
            
            # 准备参数
            args_list = [(self.driver_factors_dir, driver_dir, date) for date in filtered_dates]
            
            logger.info(f"开始并行加载 {driver_dir} 数据，共 {len(args_list)} 个文件...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(
                    executor.map(load_driver_data_global, args_list),
                    total=len(args_list),
                    desc=f"加载 {driver_dir} 数据",
                    ncols=100,
                    unit="文件"
                ))
                
                # 处理结果
                for date, data in results:
                    if data is not None:
                        driver_data[date] = data
            
            logger.info(f"完成 {driver_dir} 数据加载，共加载 {len(driver_data)} 个文件")
            
            # 为每个driver创建单独的h5文件
            with h5py.File(h5_path, 'w') as h5_file:
                # 写入属性
                h5_file.attrs['year'] = str(year)
                h5_file.attrs['driver'] = driver_dir
                h5_file.attrs['window_size'] = self.window_size
                h5_file.attrs['past_days'] = self.past_days
                h5_file.attrs['future_days'] = self.future_days
                h5_file.attrs['invalid_value_handling'] = '-9999 replaced with 0'
                
                dataset_count = 0
                
                # 以第一个past和future的 shape[1:] 为基准
                first_past_shape = None
                first_future_shape = None
                
                # 处理每个窗口事件
                for i, (window, event_date, label) in enumerate(tqdm(window_events, 
                                                                   desc=f"处理 {driver_dir} 窗口",
                                                                   leave=False)):
                    x1, y1, x2, y2 = window
                    event_date_str = event_date.strftime('%Y%m%d')
                    
                    # 获取过去和未来的日期
                    past_dates, future_dates = self._get_past_future_dates(event_date)
                    
                    # 如果数据长度不足，跳过该窗口
                    if past_dates is None or future_dates is None:
                        logger.warning(f"窗口 {window} 在日期 {event_date} 的数据长度不足，已跳过")
                        continue
                    
                    # 创建过去数据的时间序列立方体
                    if past_dates:
                        try:
                            past_datacube = self._extract_time_series_datacube(
                                window, driver_data, past_dates, event_date,
                                firms_window=window)
                            
                            if past_datacube is not None:
                                if first_past_shape is None:
                                    first_past_shape = past_datacube.shape[1:]
                                else:
                                    if past_datacube.shape[1:] != first_past_shape:
                                        logger.error(
                                            f"窗口 {window} 的 past shape {past_datacube.shape} 与第一个窗口 shape[1:] {first_past_shape} 不一致，已跳过！"
                                        )
                                        continue
                                
                                dataset_name = f"{event_date_str}_past_{label}_{x1}_{y1}_{x2}_{y2}"
                                # 获取数据类型和缩放因子
                                save_dtype, scale_factor = self._get_data_dtype_and_scale(driver_dir)
                                
                                dataset = h5_file.create_dataset(
                                    dataset_name,
                                    data=past_datacube,
                                    dtype=save_dtype,
                                    compression='gzip',
                                    compression_opts=6,
                                    shuffle=True
                                )
                                # 添加缩放因子属性
                                dataset.attrs['scale_factor'] = scale_factor
                                dataset.attrs['data_type'] = 'compressed' if scale_factor > 1 else 'original'
                                dataset.attrs['invalid_value_handling'] = '-9999 replaced with 0'
                                dataset_count += 1
                        except Exception as e:
                            logger.error(f"处理past数据时出错: {dataset_name}, 错误: {str(e)}")
                    
                    # 创建未来数据的时间序列立方体（包含当前日期）
                    future_dates_with_current = [event_date] + future_dates
                    if future_dates_with_current:
                        try:
                            future_datacube = self._extract_time_series_datacube(
                                window, driver_data, future_dates_with_current, event_date,
                                firms_window=window)
                            
                            if future_datacube is not None:
                                if first_future_shape is None:
                                    first_future_shape = future_datacube.shape[1:]
                                else:
                                    if future_datacube.shape[1:] != first_future_shape:
                                        logger.error(
                                            f"窗口 {window} 的 future shape {future_datacube.shape} 与第一个窗口 shape[1:] {first_future_shape} 不一致，已跳过！"
                                        )
                                        continue
                                
                                dataset_name = f"{event_date_str}_future_{label}_{x1}_{y1}_{x2}_{y2}"
                                # 获取数据类型和缩放因子
                                save_dtype, scale_factor = self._get_data_dtype_and_scale(driver_dir)
                                
                                dataset = h5_file.create_dataset(
                                    dataset_name,
                                    data=future_datacube,
                                    dtype=save_dtype,
                                    compression='gzip',
                                    compression_opts=6,
                                    shuffle=True
                                )
                                # 添加缩放因子属性
                                dataset.attrs['scale_factor'] = scale_factor
                                dataset.attrs['data_type'] = 'compressed' if scale_factor > 1 else 'original'
                                dataset.attrs['invalid_value_handling'] = '-9999 replaced with 0'
                                dataset_count += 1
                        except Exception as e:
                            logger.error(f"处理future数据时出错: {dataset_name}, 错误: {str(e)}")
                
                # 记录统计信息
                h5_file.attrs['total_datasets'] = dataset_count
                logger.info(f"成功保存 {year} 年 {driver_dir} 数据，共 {dataset_count} 个数据集")
            
            # 清理当前driver的数据
            driver_data.clear()
            self.data_cache.clear()
            gc.collect()
            
            # 记录内存使用
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
            logger.info(f"处理完 {driver_dir} 后内存使用: {memory_usage:.2f} GB")

    def find_windows_optimized(self, fire_file):
        """优化版窗口查找"""
        try:
            # 读取火灾数据
            ds = gdal.Open(fire_file)
            if ds is None:
                return None, None, None, None
                
            fire_data = ds.ReadAsArray()
            ds = None  # 立即释放
            
            if fire_data is None:
                return None, None, None, None
                
            if len(fire_data.shape) == 3:
                fire_data = fire_data[0]
            
            # 处理无效值（包括-9999）
            fire_data = self._process_invalid_values(fire_data)
                
            # 获取日期
            date = self._get_date_from_filename(fire_file)
            if date is None:
                return None, None, None, None
                
            # 检查是否有火灾像素
            if not np.any(fire_data == 1):
                return None, None, None, None
                
            # 高效采样正样本窗口
            positive_windows = self._sample_positive_windows_efficient(fire_data)
            if not positive_windows:
                return None, None, None, None
                
            # 高效采样负样本窗口
            target_negative_count = len(positive_windows) * self.negative_ratio
            negative_windows = self._sample_negative_windows_efficient(
                fire_data, positive_windows, target_negative_count)
                
            return positive_windows, negative_windows, None, date
            
        except Exception as e:
            logger.error(f"处理文件时出错: {fire_file}, 错误: {str(e)}")
            return None, None, None, None 

    def process_year_optimized(self, year_files):
        """优化版年度数据处理，支持内存优化和-9999值处理"""
        try:
            # 从FIRMS文件名中获取年份
            if not year_files:
                return
            
            # 找到FIRMS文件
            firms_file = None
            for file in year_files:
                if 'Firms_Detection_resampled' in file and 'FIRMS_' in file:
                    firms_file = file
                    break
                
            if not firms_file:
                logger.error("未找到FIRMS文件")
                return
            
            # 从FIRMS文件名中提取年份
            try:
                year = self._get_date_from_filename(firms_file).year
            except Exception as e:
                logger.error(f"从FIRMS文件名中提取年份失败: {str(e)}")
                return
            
            # 检查该年份是否所有驱动因素都有对应的h5文件
            all_drivers_exist = True
            for driver_dir in self.driver_dirs:
                h5_path = os.path.join(self.output_dir, f'wildfire_samples_{year}_{driver_dir}.h5')
                if not os.path.exists(h5_path):
                    all_drivers_exist = False
                    break
            
            if all_drivers_exist:
                logger.info(f"{year} 年的所有驱动因素文件已存在，跳过处理")
                return
            
            # 如果文件不完整，继续处理
            year = None
            
            # 收集所有窗口事件
            window_events = []  # 每个元素: (window, event_date, label)
            all_dates = set()
            
            # 使用进程池并行查找窗口
            with ProcessPoolExecutor(max_workers=4) as executor:
                future_to_file = {
                    executor.submit(self.find_windows_optimized, fire_file): fire_file 
                    for fire_file in year_files
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                                 total=len(year_files),
                                 desc="查找窗口"):
                    fire_file = future_to_file[future]
                    try:
                        pos_windows, neg_windows, _, date = future.result()
                        
                        if pos_windows is not None:
                            if year is None:
                                year = date.year
                            
                            # 获取过去和未来的日期
                            past_dates, future_dates = self._get_past_future_dates(date)
                            
                            # 如果未来日期列表为空（说明需要2025年数据），跳过这个窗口
                            if not future_dates:
                                continue
                            
                            # 添加正样本窗口事件
                            for window in pos_windows:
                                window_events.append((window, date, 'positive'))
                            
                            # 添加负样本窗口事件
                            for window in neg_windows:
                                window_events.append((window, date, 'negative'))
                            
                            # 收集所有需要的日期
                            all_dates.update(past_dates)
                            all_dates.update(future_dates)
                            all_dates.add(date)  # 添加当前日期
                            
                    except Exception as e:
                        logger.error(f"处理文件时出错: {fire_file}, 错误: {str(e)}")
            
            if not window_events:
                logger.warning(f"年份 {year} 没有找到有效的窗口事件")
                return
            
            logger.info(f"年份 {year} 找到 {len(window_events)} 个窗口事件")
            
            # 批量预加载数据并创建时间序列数据立方体
            all_dates = list(all_dates)
            self._preload_driver_data_batch(self.driver_dirs, all_dates, year, window_events)
                
        except Exception as e:
            logger.error(f"处理年份数据时出错: {str(e)}")
            raise

def load_and_decompress_data(h5_file_path, dataset_name):
    """
    从HDF5文件中加载数据并根据缩放因子转换回浮点型，处理-9999值
    
    Args:
        h5_file_path: HDF5文件路径
        dataset_name: 数据集名称
        
    Returns:
        numpy array: 转换后的浮点型数据
    """
    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            if dataset_name not in h5_file:
                raise ValueError(f"数据集 {dataset_name} 不存在于文件 {h5_file_path}")
            
            dataset = h5_file[dataset_name]
            data = dataset[:]
            
            # 获取缩放因子和数据类型
            scale_factor = dataset.attrs.get('scale_factor', 1.0)
            data_type = dataset.attrs.get('data_type', 'original')
            invalid_handling = dataset.attrs.get('invalid_value_handling', 'none')
            
            # 如果是压缩数据，转换回浮点型
            if data_type == 'compressed' and scale_factor > 1:
                result = data.astype(np.float32) / scale_factor
            else:
                result = data.astype(np.float32)
            
            logger.info(f"加载数据集 {dataset_name}:")
            logger.info(f"  - 原始数据类型: {data.dtype}")
            logger.info(f"  - 缩放因子: {scale_factor}")
            logger.info(f"  - 数据处理: {data_type}")
            logger.info(f"  - 无效值处理: {invalid_handling}")
            logger.info(f"  - 转换后数据范围: [{result.min():.4f}, {result.max():.4f}]")
            
            return result
                
    except Exception as e:
        logger.error(f"加载数据时出错: {h5_file_path}, {dataset_name}, 错误: {str(e)}")
        return None

def get_dataset_info(h5_file_path):
    """
    获取HDF5文件中所有数据集的信息
    
    Args:
        h5_file_path: HDF5文件路径
        
    Returns:
        dict: 包含数据集信息的字典
    """
    try:
        info = {}
        with h5py.File(h5_file_path, 'r') as h5_file:
            # 获取文件属性
            info['file_attrs'] = dict(h5_file.attrs)
            
            # 获取所有数据集信息
            info['datasets'] = {}
            for dataset_name in h5_file.keys():
                dataset = h5_file[dataset_name]
                info['datasets'][dataset_name] = {
                    'shape': dataset.shape,
                    'dtype': str(dataset.dtype),
                    'scale_factor': dataset.attrs.get('scale_factor', 1.0),
                    'data_type': dataset.attrs.get('data_type', 'original'),
                    'compression': dataset.compression,
                    'invalid_value_handling': dataset.attrs.get('invalid_value_handling', 'none')
                }
        
        return info
        
    except Exception as e:
        logger.error(f"获取文件信息时出错: {h5_file_path}, 错误: {str(e)}")
        return None

def example_usage():
    """
    使用示例：展示如何读取和使用压缩后的数据
    """
    # 示例：读取某个h5文件并检查数据
    h5_file_path = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/samples_365/wildfire_samples_2024_Temperature.h5'
    
    # 获取文件信息
    info = get_dataset_info(h5_file_path)
    if info:
        print("文件属性:")
        for key, value in info['file_attrs'].items():
            print(f"  {key}: {value}")
        
        print(f"\n数据集数量: {len(info['datasets'])}")
        
        # 显示前几个数据集的信息
        for i, (dataset_name, dataset_info) in enumerate(list(info['datasets'].items())[:3]):
            print(f"\n数据集 {i+1}: {dataset_name}")
            print(f"  形状: {dataset_info['shape']}")
            print(f"  数据类型: {dataset_info['dtype']}")
            print(f"  缩放因子: {dataset_info['scale_factor']}")
            print(f"  数据格式: {dataset_info['data_type']}")
            print(f"  压缩方式: {dataset_info['compression']}")
            print(f"  无效值处理: {dataset_info['invalid_value_handling']}")
            
            # 读取数据并显示统计信息
            data = load_and_decompress_data(h5_file_path, dataset_name)
            if data is not None:
                print(f"  数据范围: [{data.min():.4f}, {data.max():.4f}]")
                print(f"  数据均值: {data.mean():.4f}")
                print(f"  数据形状: {data.shape}")
                
                # 计算内存节省
                original_size = data.size * 4  # float32
                if dataset_info['data_type'] == 'compressed':
                    compressed_size = data.size * 2  # int16
                    print(f"  内存节省: {(1 - compressed_size/original_size)*100:.1f}%")
                elif dataset_info['dtype'] == 'uint8':
                    compressed_size = data.size * 1  # uint8
                    print(f"  内存节省: {(1 - compressed_size/original_size)*100:.1f}%")

def run_optimized_sampling():
    """运行优化版采样程序，按年份从2024到2000顺序依次处理"""
    try:
        # 配置参数 - 高性能优化版本
        config = {
            'driver_factors_dir': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data',
            'output_dir': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/samples_365',
            'window_size': 10,
            'stride': 5,
            'min_fire_pixels': 1,
            'past_days': 365,
            'future_days': 7,
            'negative_ratio': 2,
            'max_cache_memory_gb': 120,  # 增加缓存到120GB
            'batch_size': 100            # 增加批处理大小到100
        }
        
        # 设置全局GDAL优化
        gdal.SetConfigOption('GDAL_CACHEMAX', '2048')  # 减少到2GB GDAL缓存
        gdal.SetConfigOption('GDAL_NUM_THREADS', '4')   # 减少到4线程避免过载
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        gdal.SetConfigOption('VSI_CACHE', 'TRUE')
        gdal.SetConfigOption('VSI_CACHE_SIZE', '1000000000')  # 减少到1GB VSI缓存
        
        logger.info("开始运行高性能优化版采样程序...")
        logger.info(f"配置参数: {config}")
        
        fire_dir = os.path.join(config['driver_factors_dir'], 'Firms_Detection_resampled')
        all_files = glob.glob(os.path.join(fire_dir, '*.tif'))
        
        # 按年份分组
        year_files = {}
        for file in all_files:
            basename = os.path.basename(file)
            parts = basename.split('_')
            if len(parts) >= 3:
                year = parts[-3]
                if year not in year_files:
                    year_files[year] = []
                year_files[year].append(file)
        
        # 按年份从2024到2000排序
        sorted_years = sorted([y for y in year_files.keys() if y.isdigit()], reverse=True)
        sorted_years = [y for y in sorted_years if 2000 <= int(y) <= 2024]
        
        sampler = OptimizedWindowSampler(
            driver_factors_dir=config['driver_factors_dir'],
            output_dir=config['output_dir'],
            window_size=config['window_size'],
            stride=config['stride'],
            min_fire_pixels=config['min_fire_pixels'],
            past_days=config['past_days'],
            future_days=config['future_days'],
            negative_ratio=config['negative_ratio'],
            max_cache_memory_gb=config['max_cache_memory_gb'],
            batch_size=config['batch_size']
        )
        
        for year in sorted_years:
            logger.info(f"开始处理年份: {year}")
            sampler.process_year_optimized(year_files[year])
            logger.info(f"完成年份: {year}")
            
    except Exception as e:
        logger.error(f"运行优化版采样程序出错: {str(e)}")

def _process_invalid_values(data):
    """处理无效值：将-9999替换为0，处理NaN和无穷大值"""
    if data is None:
        return None
        
    # 将-9999替换为0
    data = np.where(data == -9999, 0, data)
    
    # 处理其他异常值
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    return data

def load_driver_data_global(args):
    """全局函数用于进程池加载数据，避免序列化开销"""
    driver_factors_dir, driver_dir, date = args
    
    try:
        date_str = date.strftime('%Y_%m_%d')
        file_pattern = os.path.join(driver_factors_dir, driver_dir, f'*_{date_str}.tif')
        files = glob.glob(file_pattern)
        
        if not files:
            return date, None
        
        # 检查文件是否存在且可读
        if not os.path.exists(files[0]) or not os.access(files[0], os.R_OK):
            return date, None
        
        # 优化GDAL设置
        gdal.SetConfigOption('GDAL_CACHEMAX', '512')  # 进一步减少缓存
        gdal.SetConfigOption('GDAL_NUM_THREADS', '1')   # 单线程避免竞争
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        
        ds = gdal.Open(files[0])
        if ds is None:
            return date, None
            
        # 一次性读取所有波段
        data = ds.ReadAsArray()
        ds = None
        
        if data is not None:
            # 处理无效值
            data = _process_invalid_values(data)
            
            # 对于FIRMS数据，保持原始格式
            if 'Firms_Detection_resampled' in driver_dir:
                return date, data.astype(np.uint8)
            else:
                # 使用更高效的数据类型转换
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                
                # 批量处理数据压缩
                scaled_data = data * 10000.0
                scaled_data = np.clip(scaled_data, -32767, 32767)
                return date, scaled_data.astype(np.int16)
            
        return date, None
        
    except Exception as e:
        logger.error(f"加载driver数据出错: {driver_dir}, {date_str}, 错误: {str(e)}")
        return date, None

if __name__ == '__main__':
    try:
        logger.info("优化版程序开始运行（支持-9999值处理和内存优化）...")
        
        # 可以选择运行采样程序或示例
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == 'example':
            example_usage()
        else:
            run_optimized_sampling()
            
        logger.info("优化版程序运行完成")
    except Exception as e:
        logger.error(f"程序异常退出: {str(e)}", exc_info=True) 