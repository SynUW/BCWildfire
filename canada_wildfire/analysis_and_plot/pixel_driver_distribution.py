#!/usr/bin/env python3
"""
高效并行化像元驱动因素整体分布分析器
- 统计燃烧vs未燃烧像元的驱动因素整体分布差异
- 使用流水线处理优化内存使用和处理速度
- 支持静态和动态驱动因素的分布对比分析
- 重点关注整体分布特征而非单个像元特征
已经更新成了run_temporal_sampling.py。出的是核密度估计图，与Wildfire Risk Prediction and Understanding的图类似。
"""

import os
import glob
import numpy as np
import pandas as pd
from osgeo import gdal
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict
import gc
import time
import re
import psutil
from multiprocessing import Pool, cpu_count
from functools import partial
import threading
from queue import Queue
import concurrent.futures
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('PixelDriverDistributionParallel')

# 设置GDAL配置
gdal.SetConfigOption('GDAL_CACHEMAX', '2048')
gdal.SetConfigOption('GDAL_NUM_THREADS', '4')

class EfficientPixelDriverDistributionAnalyzer:
    def __init__(self, data_dir, output_dir, analyze_full_timeseries=True, n_processes=None):
        """
        初始化像元驱动因素分布分析器
        
        Args:
            data_dir: 驱动因素数据根目录
            output_dir: 输出根目录
            analyze_full_timeseries: 是否分析完整时间序列
            n_processes: 并行进程数
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.analyze_full_timeseries = analyze_full_timeseries
        self.n_processes = n_processes or max(1, int(cpu_count()*0.6))
        
        # 创建专门的子文件夹用于存放pixel_driver_distribution的结果
        self.output_subdir = os.path.join(output_dir, "pixel_driver_distribution_results")
        os.makedirs(self.output_subdir, exist_ok=True)
        
        # 获取驱动因素目录
        self.driver_dirs = self._get_driver_directories()
        self.static_drivers, self.dynamic_drivers = self._identify_static_dynamic_drivers()
        
        # 存储像元分类结果
        self.burned_pixels = set()
        self.unburned_pixels = set()
        self.pixel_burn_times = {}  # 存储每个像元的燃烧时间
        
        # 存储统计结果
        self.distribution_stats = []
        
        logger.info(f"找到 {len(self.static_drivers)} 个静态驱动因素")
        logger.info(f"找到 {len(self.dynamic_drivers)} 个动态驱动因素")
        logger.info(f"输出目录: {self.output_subdir}")
        
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
    
    def _identify_static_dynamic_drivers(self):
        """识别静态和动态驱动因素"""
        static_drivers = {}
        dynamic_drivers = {}
        
        # 明确定义哪些是静态驱动因素（只有地形距离是真正静态的）
        known_static_drivers = {
            'Topo_Distance_WGS84_resize_resampled'  # 地形距离数据在时间序列中不变
        }
        
        for driver_name, driver_dir in self.driver_dirs.items():
            # 跳过FIRMS数据
            if driver_name == 'Firms_Detection_resampled':
                continue
                
            # 根据已知的静态驱动因素列表进行分类
            if driver_name in known_static_drivers:
                static_drivers[driver_name] = driver_dir
                logger.info(f"{driver_name} 识别为静态驱动因素 (地形数据不随时间变化)")
            else:
                dynamic_drivers[driver_name] = driver_dir
                logger.info(f"{driver_name} 识别为动态驱动因素")
        
        return static_drivers, dynamic_drivers
    
    def _load_file_data(self, file_path):
        """加载文件数据"""
        try:
            ds = gdal.Open(file_path, gdal.GA_ReadOnly)
            if ds is None:
                return None, None
            
            # 获取文件信息
            bands = ds.RasterCount
            height = ds.RasterYSize
            width = ds.RasterXSize
            
            # 读取所有波段数据
            data = ds.ReadAsArray()
            
            ds = None
            
            # 确保数据是3D格式 (bands, height, width)
            if data.ndim == 2:
                data = data[np.newaxis, :, :]  # 添加波段维度
            elif data.ndim == 3 and data.shape[0] != bands:
                # 可能需要转置
                if data.shape[2] == bands:
                    data = np.transpose(data, (2, 0, 1))
            
            return data, {'bands': bands, 'height': height, 'width': width}
            
        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {str(e)}")
            return None, None
    
    def identify_pixel_burn_status(self):
        """识别像元的燃烧状态"""
        logger.info("开始识别像元燃烧状态...")
        
        # 获取FIRMS数据目录
        firms_dir = self.driver_dirs.get('Firms_Detection_resampled')
        if not firms_dir:
            raise ValueError("未找到FIRMS数据目录")
        
        # 获取所有FIRMS文件
        firms_files = glob.glob(os.path.join(firms_dir, '*.tif'))
        firms_files.sort()
        
        logger.info(f"找到 {len(firms_files)} 个FIRMS文件")
        
        # 并行处理FIRMS文件
        effective_processes = min(self.n_processes, 32)
        chunk_size = max(1, len(firms_files) // (effective_processes * 4))
        logger.info(f"使用 {effective_processes} 个进程并行处理FIRMS文件 (chunk_size={chunk_size})")
        
        with Pool(effective_processes) as pool:
            results = []
            total_processed = 0
            
            # 使用imap_unordered减少进程间通信开销
            for result in pool.imap_unordered(self._process_firms_file, firms_files, chunksize=chunk_size):
                results.append(result)
                total_processed += 1
                
                # 每处理500个文件输出一次进度
                if total_processed % 500 == 0:
                    logger.info(f"FIRMS处理进度: {total_processed}/{len(firms_files)} "
                              f"({total_processed/len(firms_files)*100:.1f}%)")
            
        logger.info(f"FIRMS文件处理完成：{len(firms_files)} 个文件")
        
        # 合并结果
        pixel_burn_history = defaultdict(bool)
        all_valid_pixels = set()
        
        for valid_pixels, burned_pixels in results:
            if valid_pixels is not None:
                all_valid_pixels.update(valid_pixels)
                for pixel_coord in burned_pixels:
                    pixel_burn_history[pixel_coord] = True
        
        # 分类像元
        self.all_pixels = all_valid_pixels
        
        for pixel_coord in all_valid_pixels:
            if pixel_burn_history[pixel_coord]:
                self.burned_pixels.add(pixel_coord)
            else:
                self.unburned_pixels.add(pixel_coord)
        
        logger.info(f"总有效像元数: {len(self.all_pixels)}")
        logger.info(f"燃烧过的像元数: {len(self.burned_pixels)}")
        logger.info(f"从未燃烧的像元数: {len(self.unburned_pixels)}")
        
        # 保存像元分类结果
        self._save_pixel_classification()
        
        # 为时间窗口分析构建燃烧时间索引
        if self.analyze_full_timeseries:
            logger.info("构建燃烧时间索引...")
            self.pixel_burn_times = self._build_burn_time_index(firms_files)
            logger.info(f"完成燃烧时间索引，覆盖 {len(self.pixel_burn_times)} 个燃烧像元")
    
    def _build_burn_time_index(self, firms_files):
        """构建像元燃烧时间索引"""
        from datetime import datetime
        import re
        
        def extract_date_from_filename(filename):
            """从文件名提取日期"""
            match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
            if match:
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day))
            return None
        
        # 使用多进程构建时间索引
        effective_processes = min(self.n_processes, 16)
        
        with Pool(effective_processes) as pool:
            time_results = list(tqdm(
                pool.imap(self._process_firms_file_for_time, firms_files, chunksize=50),
                total=len(firms_files),
                desc="构建燃烧时间索引",
                unit="文件"
            ))
        
        # 合并时间索引
        pixel_burn_times = defaultdict(list)
        for result in time_results:
            if result:
                file_date, burned_pixels = result
                for pixel_coord in burned_pixels:
                    pixel_burn_times[pixel_coord].append(file_date)
        
        return dict(pixel_burn_times)
    
    def _process_firms_file_for_time(self, firms_file):
        """处理单个FIRMS文件以获取燃烧时间"""
        try:
            from datetime import datetime
            import re
            
            def extract_date_from_filename(filename):
                """从文件名提取日期"""
                match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
                if match:
                    year, month, day = match.groups()
                    return datetime(int(year), int(month), int(day))
                return None
            
            file_date = extract_date_from_filename(os.path.basename(firms_file))
            if not file_date:
                return None
            
            data, file_info = self._load_file_data(firms_file)
            if data is None:
                return None
            
            # 如果是多波段，取第一个波段
            if data.ndim == 3:
                data = data[0]
            
            # 找到燃烧像元（值为1）
            burned_pixels = set()
            rows, cols = np.where(data == 1)
            for row, col in zip(rows, cols):
                burned_pixels.add((int(row), int(col)))
            
            return file_date, burned_pixels
            
        except Exception as e:
            logger.error(f"处理FIRMS时间文件失败: {firms_file}, 错误: {str(e)}")
            return None
    
    def _process_firms_file(self, firms_file):
        """处理单个FIRMS文件"""
        try:
            data, file_info = self._load_file_data(firms_file)
            if data is None:
                return None, None
            
            # 如果是多波段，取第一个波段
            if data.ndim == 3:
                data = data[0]
            
            # 找到所有有效像元（非NaN、非NoData值、非背景值、非0值）
            valid_mask = (~np.isnan(data)) & (data != 255) & (data != -9999) & (data != 0)
            valid_pixels = set()
            rows, cols = np.where(valid_mask)
            for row, col in zip(rows, cols):
                valid_pixels.add((int(row), int(col)))
            
            # 找到燃烧像元（值为1）
            burned_pixels = set()
            rows, cols = np.where(data == 1)
            for row, col in zip(rows, cols):
                burned_pixels.add((int(row), int(col)))
            
            return valid_pixels, burned_pixels
            
        except Exception as e:
            logger.error(f"处理FIRMS文件失败: {firms_file}, 错误: {str(e)}")
            return None, None
    
    def _save_pixel_classification(self):
        """保存像元分类结果"""
        # 保存CSV格式
        pixel_data = []
        
        for row, col in self.burned_pixels:
            pixel_data.append({
                'row': row, 'col': col, 'pixel_type': 'burned'
            })
        
        for row, col in self.unburned_pixels:
            pixel_data.append({
                'row': row, 'col': col, 'pixel_type': 'unburned'
            })
        
        df = pd.DataFrame(pixel_data)
        csv_file = os.path.join(self.output_subdir, 'pixel_classification.csv')
        df.to_csv(csv_file, index=False)
        logger.info(f"像元分类结果保存到: {csv_file}")
        
        # 保存统计汇总
        summary_file = os.path.join(self.output_subdir, 'pixel_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"像元分类统计汇总\n")
            f.write(f"==================\n")
            f.write(f"总有效像元数: {len(self.all_pixels)}\n")
            f.write(f"燃烧过的像元数: {len(self.burned_pixels)}\n")
            f.write(f"从未燃烧的像元数: {len(self.unburned_pixels)}\n")
            f.write(f"燃烧率: {len(self.burned_pixels)/len(self.all_pixels)*100:.2f}%\n")
        
        logger.info(f"像元统计汇总保存到: {summary_file}")
    
    def analyze_driver_distributions(self):
        """分析驱动因素在燃烧vs未燃烧像元中的分布"""
        logger.info("开始分析驱动因素分布...")
        
        all_stats = []
        
        # 处理静态驱动因素
        for driver_name, driver_dir in self.static_drivers.items():
            logger.info(f"处理静态驱动因素: {driver_name}")
            stats = self._process_static_driver(driver_name, driver_dir)
            if stats:
                all_stats.extend(stats)
        
        # 处理动态驱动因素  
        for driver_name, driver_dir in self.dynamic_drivers.items():
            logger.info(f"处理动态驱动因素: {driver_name}")
            stats = self._process_dynamic_driver(driver_name, driver_dir)
            if stats:
                all_stats.extend(stats)
        
        # 保存结果
        if all_stats:
            df = pd.DataFrame(all_stats)
            csv_file = os.path.join(self.output_subdir, 'driver_distribution_analysis.csv')
            df.to_csv(csv_file, index=False)
            logger.info(f"驱动因素分布分析结果保存到: {csv_file}")
            
            # 创建汇总报告
            self._create_summary_report(df)
            
            # 创建分布可视化分析
            self._create_distribution_analysis(df)
        else:
            logger.warning("没有生成任何统计数据")
    
    def _calculate_statistics(self, driver_name, band_idx, burned_values, unburned_values):
        """计算整体分布统计量"""
        from scipy import stats
        
        # 转换为64位浮点数数组并过滤无效值
        burned_array = np.array(burned_values, dtype=np.float64)
        unburned_array = np.array(unburned_values, dtype=np.float64)
        
        # 过滤无穷大、NaN和极值
        burned_array = burned_array[np.isfinite(burned_array)]
        unburned_array = unburned_array[np.isfinite(unburned_array)]
        
        # 进一步过滤极值 (超过合理范围的值)
        burned_q99 = np.percentile(burned_array, 99) if len(burned_array) > 0 else 0
        burned_q01 = np.percentile(burned_array, 1) if len(burned_array) > 0 else 0
        unburned_q99 = np.percentile(unburned_array, 99) if len(unburned_array) > 0 else 0
        unburned_q01 = np.percentile(unburned_array, 1) if len(unburned_array) > 0 else 0
        
        # 设置合理的数据范围
        max_reasonable = max(abs(burned_q99), abs(unburned_q99), 1e6)
        min_reasonable = min(burned_q01, unburned_q01, -1e6)
        
        burned_array = burned_array[(burned_array >= min_reasonable) & (burned_array <= max_reasonable)]
        unburned_array = unburned_array[(unburned_array >= min_reasonable) & (unburned_array <= max_reasonable)]
        
        # 如果过滤后数据过少，返回默认值
        if len(burned_array) < 10 or len(unburned_array) < 10:
            return self._get_default_stats(driver_name, band_idx, len(burned_array), len(unburned_array))
        
        # 使用稳健的统计方法
        try:
            burned_mean = np.nanmean(burned_array)
            unburned_mean = np.nanmean(unburned_array)
            burned_std = np.nanstd(burned_array)
            unburned_std = np.nanstd(unburned_array)
            
            # 基础统计量
            stats_dict = {
                'feature_name': f"{driver_name}_band_{band_idx}",
                'burned_count': len(burned_array),
                'burned_mean': burned_mean,
                'burned_std': burned_std,
                'burned_median': np.nanmedian(burned_array),
                'burned_min': np.nanmin(burned_array),
                'burned_max': np.nanmax(burned_array),
                'burned_q25': np.nanpercentile(burned_array, 25),
                'burned_q75': np.nanpercentile(burned_array, 75),
                'unburned_count': len(unburned_array),
                'unburned_mean': unburned_mean,
                'unburned_std': unburned_std,
                'unburned_median': np.nanmedian(unburned_array),
                'unburned_min': np.nanmin(unburned_array),
                'unburned_max': np.nanmax(unburned_array),
                'unburned_q25': np.nanpercentile(unburned_array, 25),
                'unburned_q75': np.nanpercentile(unburned_array, 75),
                'mean_difference': burned_mean - unburned_mean,
                'median_difference': np.nanmedian(burned_array) - np.nanmedian(unburned_array),
                'std_ratio': burned_std / max(unburned_std, 1e-10)
            }
        except Exception as e:
            logger.warning(f"计算基础统计量时出错: {e}")
            return self._get_default_stats(driver_name, band_idx, len(burned_array), len(unburned_array))
        
        # 分布形状比较
        try:
            # 效应大小 (Cohen's d) - 使用稳健方法
            if burned_std > 1e-10 and unburned_std > 1e-10:
                pooled_var = ((len(burned_array) - 1) * burned_std**2 + 
                             (len(unburned_array) - 1) * unburned_std**2) / (len(burned_array) + len(unburned_array) - 2)
                pooled_std = np.sqrt(max(pooled_var, 1e-10))
                cohens_d = (burned_mean - unburned_mean) / pooled_std
                
                # 检查效应大小是否合理
                if np.isfinite(cohens_d) and abs(cohens_d) < 100:  # 合理的效应大小范围
                    stats_dict['effect_size_cohens_d'] = cohens_d
                else:
                    stats_dict['effect_size_cohens_d'] = 0.0
            else:
                stats_dict['effect_size_cohens_d'] = 0.0
            
            # 效应大小分类
            cohens_d_abs = abs(stats_dict['effect_size_cohens_d'])
            if cohens_d_abs < 0.2:
                effect_magnitude = "negligible"
            elif cohens_d_abs < 0.5:
                effect_magnitude = "small"
            elif cohens_d_abs < 0.8:
                effect_magnitude = "medium"
            else:
                effect_magnitude = "large"
            stats_dict['effect_magnitude'] = effect_magnitude
            
        except Exception as e:
            logger.warning(f"计算效应大小时出错: {e}")
            stats_dict['effect_size_cohens_d'] = 0.0
            stats_dict['effect_magnitude'] = "negligible"
        
        # 统计显著性检验
        try:
            # t检验 - 增强稳健性
            if len(burned_array) >= 10 and len(unburned_array) >= 10:
                t_stat, p_value = stats.ttest_ind(burned_array, unburned_array, equal_var=False, nan_policy='omit')
                if np.isfinite(t_stat) and np.isfinite(p_value):
                    stats_dict['t_statistic'] = t_stat
                    stats_dict['p_value'] = p_value
                    stats_dict['significant'] = p_value < 0.05
                else:
                    stats_dict['t_statistic'] = 0.0
                    stats_dict['p_value'] = 1.0
                    stats_dict['significant'] = False
            else:
                stats_dict['t_statistic'] = 0.0
                stats_dict['p_value'] = 1.0
                stats_dict['significant'] = False
                
        except Exception as e:
            logger.warning(f"t检验计算出错: {e}")
            stats_dict['t_statistic'] = 0.0
            stats_dict['p_value'] = 1.0
            stats_dict['significant'] = False
        
        try:
            # Mann-Whitney U检验 (非参数) - 增强稳健性
            if len(burned_array) >= 10 and len(unburned_array) >= 10:
                u_stat, u_p_value = stats.mannwhitneyu(burned_array, unburned_array, alternative='two-sided')
                if np.isfinite(u_stat) and np.isfinite(u_p_value):
                    stats_dict['mannwhitney_u_statistic'] = u_stat
                    stats_dict['mannwhitney_p_value'] = u_p_value
                    stats_dict['mannwhitney_significant'] = u_p_value < 0.05
                else:
                    stats_dict['mannwhitney_u_statistic'] = 0.0
                    stats_dict['mannwhitney_p_value'] = 1.0
                    stats_dict['mannwhitney_significant'] = False
            else:
                stats_dict['mannwhitney_u_statistic'] = 0.0
                stats_dict['mannwhitney_p_value'] = 1.0
                stats_dict['mannwhitney_significant'] = False
                
        except Exception as e:
            logger.warning(f"Mann-Whitney U检验计算出错: {e}")
            stats_dict['mannwhitney_u_statistic'] = 0.0
            stats_dict['mannwhitney_p_value'] = 1.0
            stats_dict['mannwhitney_significant'] = False
        
        try:
            # Kolmogorov-Smirnov检验 (分布形状差异) - 增强稳健性
            if len(burned_array) >= 10 and len(unburned_array) >= 10:
                ks_stat, ks_p_value = stats.ks_2samp(burned_array, unburned_array)
                if np.isfinite(ks_stat) and np.isfinite(ks_p_value):
                    stats_dict['ks_statistic'] = ks_stat
                    stats_dict['ks_p_value'] = ks_p_value
                    stats_dict['distributions_different'] = ks_p_value < 0.05
                else:
                    stats_dict['ks_statistic'] = 0.0
                    stats_dict['ks_p_value'] = 1.0
                    stats_dict['distributions_different'] = False
            else:
                stats_dict['ks_statistic'] = 0.0
                stats_dict['ks_p_value'] = 1.0
                stats_dict['distributions_different'] = False
                
        except Exception as e:
            logger.warning(f"KS检验计算出错: {e}")
            stats_dict['ks_statistic'] = 0.0
            stats_dict['ks_p_value'] = 1.0
            stats_dict['distributions_different'] = False
        
        return stats_dict
    
    def _get_default_stats(self, driver_name, band_idx, burned_count, unburned_count):
        """返回默认的统计值（用于数据不足的情况）"""
        return {
            'feature_name': f"{driver_name}_band_{band_idx}",
            'burned_count': burned_count,
            'burned_mean': 0.0,
            'burned_std': 0.0,
            'burned_median': 0.0,
            'burned_min': 0.0,
            'burned_max': 0.0,
            'burned_q25': 0.0,
            'burned_q75': 0.0,
            'unburned_count': unburned_count,
            'unburned_mean': 0.0,
            'unburned_std': 0.0,
            'unburned_median': 0.0,
            'unburned_min': 0.0,
            'unburned_max': 0.0,
            'unburned_q25': 0.0,
            'unburned_q75': 0.0,
            'mean_difference': 0.0,
            'median_difference': 0.0,
            'std_ratio': 1.0,
            'effect_size_cohens_d': 0.0,
            'effect_magnitude': 'negligible',
            't_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'mannwhitney_u_statistic': 0.0,
            'mannwhitney_p_value': 1.0,
            'mannwhitney_significant': False,
            'ks_statistic': 0.0,
            'ks_p_value': 1.0,
            'distributions_different': False
        }
    
    def _process_static_driver(self, driver_name, driver_dir):
        """处理静态驱动因素"""
        # 获取第一个文件（静态数据所有文件相同）
        driver_files = glob.glob(os.path.join(driver_dir, '*.tif'))
        if not driver_files:
            return None
        
        data, file_info = self._load_file_data(driver_files[0])
        if data is None:
            return None
            
            bands = file_info['bands']
        logger.info(f"{driver_name} 有 {bands} 个波段 (静态)")
            
        stats_list = []
        
        # 为每个波段计算统计量
        for band_idx in range(bands):
            if data.ndim == 3:
                band_data = data[band_idx]
            else:
                band_data = data  # 单波段情况
                
                burned_values = []
                unburned_values = []
                
            # 收集点燃过像元的值
            for row, col in self.burned_pixels:
                if row < band_data.shape[0] and col < band_data.shape[1]:
                    value = band_data[row, col]
                    if not np.isnan(value) and not np.isinf(value) and value != 0 and value != 255 and value != -9999:
                        burned_values.append(value)
            
            # 收集未点燃像元的值
            for row, col in self.unburned_pixels:
                if row < band_data.shape[0] and col < band_data.shape[1]:
                    value = band_data[row, col]
                    if not np.isnan(value) and not np.isinf(value) and value != 0 and value != 255 and value != -9999:
                        unburned_values.append(value)
            
            # 计算统计量
            if burned_values and unburned_values:
                stats = self._calculate_statistics(driver_name, band_idx + 1, burned_values, unburned_values)
                stats_list.append(stats)
                logger.info(f"{stats['feature_name']}: 燃烧像元{stats['burned_count']}个, "
                          f"未燃烧像元{stats['unburned_count']}个 (静态)")
        
        return stats_list
    
    def _process_dynamic_driver(self, driver_name, driver_dir):
        """处理动态驱动因素 - 高效实现：每文件只加载一次，内存循环波段"""
        # 获取该驱动因素的所有文件
        driver_files = glob.glob(os.path.join(driver_dir, '*.tif'))
        driver_files.sort()
                
        if not driver_files:
            return None
        
        # 获取波段信息
        sample_data, file_info = self._load_file_data(driver_files[0])
        if sample_data is None:
            return None
        
        bands = file_info['bands']
        total_files = len(driver_files)
                
        # 根据分析模式决定处理的文件
        if self.analyze_full_timeseries:
            logger.info(f"{driver_name} 有 {bands} 个波段，分析完整时间序列：{total_files} 个文件")
            files_to_process = driver_files
        else:
            # 采样模式
            max_files = 50
            if len(driver_files) > max_files:
                step = max(1, len(driver_files) // max_files)
                files_to_process = driver_files[::step]
                logger.info(f"{driver_name} 有 {bands} 个波段，采样模式：处理 {len(files_to_process)} 个文件（原{total_files}个）")
            else:
                files_to_process = driver_files
                logger.info(f"{driver_name} 有 {bands} 个波段，处理所有 {len(files_to_process)} 个文件")
        
        # 高效处理：每文件只加载一次，内存循环波段
        logger.info(f"开始高效处理驱动因素 {driver_name}，共 {len(files_to_process)} 个文件，{bands} 个波段")
        
        # 为每个波段初始化值列表
        all_burned_values = [[] for _ in range(bands)]
        all_unburned_values = [[] for _ in range(bands)]
        
        # 预计算未燃烧像元采样
        unburned_sample_size = min(len(self.burned_pixels) * 3, len(self.unburned_pixels))
        unburned_coords_sample = list(self.unburned_pixels)[:unburned_sample_size]
        
        # 逐文件处理，每文件只加载一次
        processed_files = 0
        for file_path in tqdm(files_to_process, desc=f"处理{driver_name}文件", unit="文件"):
            data, file_info = self._load_file_data(file_path)
            if data is None:
                continue
                    
            # 确保数据是3D格式 (bands, height, width)
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            
            # 对当前文件的所有波段进行统计
            for band_idx in range(bands):
                band_data = data[band_idx]
                    
                # 收集点燃过像元的值
                for row, col in self.burned_pixels:
                    if row < band_data.shape[0] and col < band_data.shape[1]:
                        value = band_data[row, col]
                    if not np.isnan(value) and not np.isinf(value) and value != 0 and value != 255 and value != -9999:
                        all_burned_values[band_idx].append(value)
                
                # 收集未点燃像元的值
                for row, col in unburned_coords_sample:
                    if row < band_data.shape[0] and col < band_data.shape[1]:
                        value = band_data[row, col]
                    if not np.isnan(value) and not np.isinf(value) and value != 0 and value != 255 and value != -9999:
                        all_unburned_values[band_idx].append(value)
                
                processed_files += 1
                    
            # 每处理100个文件输出一次状态
            if processed_files % 100 == 0:
                memory_usage = psutil.virtual_memory()
                logger.info(f"    已处理 {processed_files}/{len(files_to_process)} 个文件，"
                          f"内存使用率: {memory_usage.percent:.1f}%")
            
            # 立即释放当前文件的内存
            del data
        
        # 计算每个波段的统计量
        stats_list = []
        for band_idx in range(bands):
            burned_values = all_burned_values[band_idx]
            unburned_values = all_unburned_values[band_idx]
            
            if burned_values and unburned_values:
                stats = self._calculate_statistics(driver_name, band_idx + 1, burned_values, unburned_values)
                stats_list.append(stats)
                logger.info(f"  {stats['feature_name']}: 燃烧像元{stats['burned_count']}个, "
                          f"未燃烧像元{stats['unburned_count']}个")
        
        # 清理内存
        del all_burned_values, all_unburned_values
        gc.collect()
        
        logger.info(f"驱动因素 {driver_name} 高效处理完成")
        return stats_list
    
    def _pipeline_process_temporal_files(self, driver_files, event_window_data, driver_name, extract_date_from_filename):
        """流水线处理时间窗口分析文件：并行加载和统计"""
        
        # 创建数据队列
        data_queue = Queue(maxsize=2)
        processed_files = 0
        
        def temporal_data_loader():
            """时间窗口数据加载线程"""
            try:
                for file_path in driver_files:
                    # 提取文件日期
                    file_date = extract_date_from_filename(os.path.basename(file_path))
                    if not file_date:
                        continue
                    
                    # 检查是否有事件需要这个日期的数据
                    relevant_events = []
                    for result_idx, event_data in event_window_data.items():
                        if event_data['window_start'] <= file_date <= event_data['burn_date']:
                            relevant_events.append(result_idx)
                    
                    # 如果没有事件需要这个日期的数据，跳过
                    if not relevant_events:
                        continue
                    
                    # 加载当前文件
                    data, file_info = self._load_file_data(file_path)
                    if data is not None:
                        if data.ndim == 2:
                            data = data[np.newaxis, :, :]
                        data_queue.put((file_path, file_date, data, relevant_events))
                    
            except Exception as e:
                logger.error(f"时间窗口数据加载线程出错: {str(e)}")
            finally:
                # 发送结束信号
                data_queue.put(None)
        
        def temporal_data_processor():
            """时间窗口数据处理线程"""
            nonlocal processed_files
            
            try:
                while True:
                    item = data_queue.get()
                    if item is None:  # 结束信号
                        break
                    
                    file_path, file_date, data, relevant_events = item
                    
                    # 为需要这个日期数据的事件提取像元值
                    for result_idx in relevant_events:
                        event_data = event_window_data[result_idx]
                        pixel_coord = event_data['pixel_coord']
                        control_pixel = event_data['control_pixel']
                        
                        burned_values, control_values = self._extract_pixel_values_from_memory(
                            data, pixel_coord, control_pixel
                        )
                        
                        if burned_values and control_values:
                            for band_idx, (burned_val, control_val) in enumerate(zip(burned_values, control_values)):
                                event_data['window_values_burned'][band_idx].append(burned_val)
                                event_data['window_values_control'][band_idx].append(control_val)
                    
                    # 立即释放当前文件的内存
                    del data
                    processed_files += 1
                    
                    # 每处理50个文件输出一次进度
                    if processed_files % 50 == 0:
                        memory_usage = psutil.virtual_memory()
                        logger.info(f"    时间窗口流水线已处理 {processed_files} 个相关文件，"
                                  f"内存使用率: {memory_usage.percent:.1f}%")
                    
                    data_queue.task_done()
                    
            except Exception as e:
                logger.error(f"时间窗口数据处理线程出错: {str(e)}")
        
        # 启动时间窗口分析的加载和处理线程
        temporal_loader_thread = threading.Thread(
            target=temporal_data_loader, 
            name=f"TemporalLoader-{driver_name}"
        )
        temporal_processor_thread = threading.Thread(
            target=temporal_data_processor, 
            name=f"TemporalProcessor-{driver_name}"
        )
        
        temporal_loader_thread.start()
        temporal_processor_thread.start()
        
        # 显示进度
        with tqdm(total=len(driver_files), desc=f"时间窗口流水线{driver_name}", unit="文件") as pbar:
            last_processed = 0
            while temporal_processor_thread.is_alive() or not data_queue.empty():
                if processed_files > last_processed:
                    pbar.update(processed_files - last_processed)
                    last_processed = processed_files
                time.sleep(0.1)
            pbar.update(processed_files - last_processed)  # 确保进度条完成
        
        # 等待线程完成
        temporal_loader_thread.join()
        temporal_processor_thread.join()
        
        # 强制垃圾回收
        gc.collect()
                
    def _driver_by_driver_temporal_analysis(self, burn_events, unburned_list, window_days):
        """逐个驱动因素的内存优化时间窗口分析"""
        from datetime import datetime, timedelta
        import random
        import re
        
        def extract_date_from_filename(filename):
            """从文件名提取日期"""
            match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
            if match:
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day))
            return None
        
        logger.info(f"开始逐个驱动因素的时间窗口分析（{window_days}天）")
        
        # 初始化结果列表
        temporal_results = []
        
        # 为每个燃烧事件创建基础记录
        for pixel_coord, burn_date in burn_events:
            # 随机选择对照像元
            if unburned_list:
                control_pixel = random.choice(unburned_list)
            else:
                continue
            
            window_start = burn_date - timedelta(days=window_days)
            
            pixel_data = {
                'burned_pixel_row': pixel_coord[0],
                'burned_pixel_col': pixel_coord[1],
                'control_pixel_row': control_pixel[0],
                'control_pixel_col': control_pixel[1],
                'burn_date': burn_date.strftime('%Y-%m-%d'),
                'window_start': window_start.strftime('%Y-%m-%d'),
                'window_days': window_days
            }
            temporal_results.append(pixel_data)
        
        logger.info(f"创建了 {len(temporal_results)} 个分析任务")
        
        # 逐个处理每个驱动因素
        for driver_name, driver_dir in {**self.static_drivers, **self.dynamic_drivers}.items():
            if driver_name == 'Firms_Detection_resampled':
                continue
            
            logger.info(f"处理驱动因素: {driver_name}")
            
            if driver_name in self.static_drivers:
                # 静态驱动因素：只需加载一个文件
                driver_files = glob.glob(os.path.join(driver_dir, '*.tif'))
                if driver_files:
                    file_path = driver_files[0]
                    data, file_info = self._load_file_data(file_path)
                    if data is not None:
                        if data.ndim == 2:
                            data = data[np.newaxis, :, :]
                        
                        # 为所有分析任务添加静态驱动因素数据
                        for result_idx, result in enumerate(temporal_results):
                            pixel_coord = (result['burned_pixel_row'], result['burned_pixel_col'])
                            control_pixel = (result['control_pixel_row'], result['control_pixel_col'])
                            
                            burned_values, control_values = self._extract_pixel_values_from_memory(
                                data, pixel_coord, control_pixel
                            )
                            
                            if burned_values and control_values:
                                for band_idx, (burned_val, control_val) in enumerate(zip(burned_values, control_values)):
                                    temporal_results[result_idx][f'{driver_name}_band_{band_idx+1}_burned'] = burned_val
                                    temporal_results[result_idx][f'{driver_name}_band_{band_idx+1}_control'] = control_val
                                    temporal_results[result_idx][f'{driver_name}_band_{band_idx+1}_diff'] = burned_val - control_val
                        
                        logger.info(f"  静态驱动因素 {driver_name} 处理完成")
            else:
                # 动态驱动因素：流式处理时间窗口分析
                driver_files = glob.glob(os.path.join(driver_dir, '*.tif'))
                
                logger.info(f"  流式处理 {len(driver_files)} 个文件...")
                
                # 为每个事件创建时间窗口值的累积器
                event_window_data = {}
                for result_idx, result in enumerate(temporal_results):
                    event_window_data[result_idx] = {
                        'window_values_burned': defaultdict(list),
                        'window_values_control': defaultdict(list),
                        'burn_date': datetime.strptime(result['burn_date'], '%Y-%m-%d'),
                        'window_start': datetime.strptime(result['window_start'], '%Y-%m-%d'),
                        'pixel_coord': (result['burned_pixel_row'], result['burned_pixel_col']),
                        'control_pixel': (result['control_pixel_row'], result['control_pixel_col'])
                    }
                
                # 使用流水线处理时间窗口分析
                self._pipeline_process_temporal_files(
                    driver_files, event_window_data, driver_name, extract_date_from_filename
                )
                
                # 收集时间窗口内的所有数据用于整体分布分析
                temporal_distribution_data = {}
                
                for result_idx, event_data in event_window_data.items():
                    window_values_burned = event_data['window_values_burned']
                    window_values_control = event_data['window_values_control']
                    
                    for band_idx in window_values_burned:
                        if window_values_burned[band_idx] and window_values_control[band_idx]:
                            # 为每个事件保存单独的统计（保持原有格式用于兼容）
                            burned_mean = np.mean(window_values_burned[band_idx])
                            control_mean = np.mean(window_values_control[band_idx])
                            
                            temporal_results[result_idx][f'{driver_name}_band_{band_idx+1}_burned_mean'] = burned_mean
                            temporal_results[result_idx][f'{driver_name}_band_{band_idx+1}_control_mean'] = control_mean
                            temporal_results[result_idx][f'{driver_name}_band_{band_idx+1}_diff_mean'] = burned_mean - control_mean
                            temporal_results[result_idx][f'{driver_name}_band_{band_idx+1}_burned_std'] = np.std(window_values_burned[band_idx])
                            temporal_results[result_idx][f'{driver_name}_band_{band_idx+1}_control_std'] = np.std(window_values_control[band_idx])
                            temporal_results[result_idx][f'{driver_name}_band_{band_idx+1}_sample_count'] = len(window_values_burned[band_idx])
                            
                            # 动态初始化并收集所有值用于整体分布分析
                            if band_idx not in temporal_distribution_data:
                                temporal_distribution_data[band_idx] = {
                                    'burned_values': [],
                                    'control_values': []
                                }
                            temporal_distribution_data[band_idx]['burned_values'].extend(window_values_burned[band_idx])
                            temporal_distribution_data[band_idx]['control_values'].extend(window_values_control[band_idx])
                
                # 为当前驱动因素生成时间窗口整体分布统计
                temporal_stats = []
                for band_idx, band_data in temporal_distribution_data.items():
                    if band_data['burned_values'] and band_data['control_values']:
                        stats = self._calculate_statistics(
                            f"{driver_name}_temporal_{window_days}days", 
                            band_idx + 1, 
                            band_data['burned_values'], 
                            band_data['control_values']
                        )
                        temporal_stats.append(stats)
                        logger.info(f"  时间窗口{window_days}天 {stats['feature_name']}: "
                                  f"燃烧值{stats['burned_count']}个, 未燃烧值{stats['unburned_count']}个")
                
                # 将时间窗口分布统计添加到全局统计中
                if not hasattr(self, 'temporal_distribution_stats'):
                    self.temporal_distribution_stats = {}
                if window_days not in self.temporal_distribution_stats:
                    self.temporal_distribution_stats[window_days] = []
                self.temporal_distribution_stats[window_days].extend(temporal_stats)
                
                # 清理事件窗口数据
                del event_window_data
                gc.collect()
                
                logger.info(f"  动态驱动因素 {driver_name} 流水线处理完成")
        
        logger.info(f"逐个驱动因素时间窗口分析完成，生成 {len(temporal_results)} 条记录")
        return temporal_results
    
    def _extract_pixel_values_from_memory(self, data, burned_pixel, control_pixel):
        """从内存中的数据提取像元值"""
        try:
            burned_values = []
            control_values = []
            
            # 提取燃烧像元的值
            row, col = burned_pixel
            if row < data.shape[1] and col < data.shape[2]:
                for band_idx in range(data.shape[0]):
                    value = data[band_idx, row, col]
                    if not np.isnan(value) and not np.isinf(value) and value != 0 and value != 255 and value != -9999:
                        burned_values.append(float(value))
            
            # 提取对照像元的值
            row, col = control_pixel
            if row < data.shape[1] and col < data.shape[2]:
                for band_idx in range(data.shape[0]):
                    value = data[band_idx, row, col]
                    if not np.isnan(value) and not np.isinf(value) and value != 0 and value != 255 and value != -9999:
                        control_values.append(float(value))
            
            return burned_values, control_values
            
        except Exception as e:
            logger.error(f"从内存提取像元值失败: {str(e)}")
            return [], []
    
    def _create_summary_report(self, df):
        """创建整体分布分析汇总报告"""
        summary_file = os.path.join(self.output_subdir, 'driver_distribution_summary.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("驱动因素整体分布分析汇总报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析类型: 燃烧vs未燃烧像元的整体分布对比\n")
            f.write(f"总特征数量: {len(df)}\n")
            f.write(f"统计显著性特征数量: {len(df[df['significant']])}\n")
            f.write(f"非参数检验显著性特征数量: {len(df[df['mannwhitney_significant']])}\n")
            f.write(f"分布形状显著不同特征数量: {len(df[df['distributions_different']])}\n\n")
            
            # 效应大小分布
            if 'effect_magnitude' in df.columns:
                effect_counts = df['effect_magnitude'].value_counts()
                f.write("效应大小分布:\n")
                f.write("-" * 30 + "\n")
                for effect, count in effect_counts.items():
                    f.write(f"  {effect}: {count} 个特征\n")
                f.write("\n")
            
            # 按效应大小排序显示最重要的特征
            if 'effect_size_cohens_d' in df.columns:
                df_sorted = df.sort_values('effect_size_cohens_d', key=abs, ascending=False)
                
                f.write("效应大小最大的前15个特征 (Cohen's d):\n")
                f.write("-" * 50 + "\n")
                for i, (_, row) in enumerate(df_sorted.head(15).iterrows()):
                    f.write(f"{i+1:2d}. {row['feature_name']:<30} "
                           f"d={row.get('effect_size_cohens_d', 0):.3f} "
                           f"({row.get('effect_magnitude', 'unknown'):<8}) "
                           f"均值差={row['mean_difference']:.4f} "
                           f"显著={'✓' if row['significant'] else '✗'}\n")
                f.write("\n")
            
            # 按平均差异排序显示特征
            df_sorted_mean = df.sort_values('mean_difference', key=abs, ascending=False)
            
            f.write("平均差异最大的前15个特征:\n")
            f.write("-" * 50 + "\n")
            for i, (_, row) in enumerate(df_sorted_mean.head(15).iterrows()):
                f.write(f"{i+1:2d}. {row['feature_name']:<30} "
                       f"差异={row['mean_difference']:8.4f} "
                       f"燃烧={row['burned_mean']:8.4f} "
                       f"未燃烧={row['unburned_mean']:8.4f} "
                       f"显著={'✓' if row['significant'] else '✗'}\n")
            
            # 分布形状差异最大的特征
            if 'ks_statistic' in df.columns:
                df_sorted_ks = df.sort_values('ks_statistic', ascending=False)
                f.write("\n分布形状差异最大的前10个特征 (KS检验):\n")
                f.write("-" * 50 + "\n")
                for i, (_, row) in enumerate(df_sorted_ks.head(10).iterrows()):
                    f.write(f"{i+1:2d}. {row['feature_name']:<30} "
                           f"KS统计量={row.get('ks_statistic', 0):.4f} "
                           f"形状差异={'✓' if row.get('distributions_different', False) else '✗'}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("说明:\n")
            f.write("- Cohen's d: 效应大小，>0.8为大效应，0.5-0.8为中等，0.2-0.5为小效应\n")
            f.write("- 显著性: t检验p<0.05\n")
            f.write("- 形状差异: KS检验检测两个分布的形状是否不同\n")
        
        logger.info(f"整体分布分析汇总报告保存到: {summary_file}")
    
    def _create_distribution_analysis(self, df):
        """创建分布分析的详细信息"""
        analysis_file = os.path.join(self.output_subdir, 'distribution_detailed_analysis.txt')
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("驱动因素整体分布详细分析\n")
            f.write("=" * 70 + "\n\n")
            
            # 按不同标准分类特征
            f.write("1. 统计显著性分析\n")
            f.write("-" * 40 + "\n")
            
            significant_features = df[df['significant'] == True]
            non_significant_features = df[df['significant'] == False]
            
            f.write(f"统计显著的特征数量: {len(significant_features)}\n")
            f.write(f"统计不显著的特征数量: {len(non_significant_features)}\n\n")
            
            if len(significant_features) > 0:
                f.write("统计显著的特征 (按效应大小排序):\n")
                sig_sorted = significant_features.sort_values('effect_size_cohens_d', key=abs, ascending=False)
                for i, (_, row) in enumerate(sig_sorted.iterrows()):
                    f.write(f"  {i+1:2d}. {row['feature_name']:<35} "
                           f"Cohen's d={row.get('effect_size_cohens_d', 0):7.3f} "
                           f"p={row['p_value']:.3e}\n")
                f.write("\n")
            
            # 效应大小分析
            f.write("2. 效应大小分析\n")
            f.write("-" * 40 + "\n")
            
            if 'effect_magnitude' in df.columns:
                for magnitude in ['large', 'medium', 'small', 'negligible']:
                    magnitude_features = df[df['effect_magnitude'] == magnitude]
                    if len(magnitude_features) > 0:
                        f.write(f"\n{magnitude.upper()} 效应特征 ({len(magnitude_features)} 个):\n")
                        mag_sorted = magnitude_features.sort_values('effect_size_cohens_d', key=abs, ascending=False)
                        for i, (_, row) in enumerate(mag_sorted.iterrows()):
                            f.write(f"  {i+1:2d}. {row['feature_name']:<35} "
                                   f"d={row.get('effect_size_cohens_d', 0):7.3f} "
                                   f"差异={row['mean_difference']:8.4f}\n")
            
            # 分布形状差异分析
            f.write(f"\n3. 分布形状差异分析 (KS检验)\n")
            f.write("-" * 40 + "\n")
            
            if 'distributions_different' in df.columns:
                shape_different = df[df['distributions_different'] == True]
                f.write(f"分布形状显著不同的特征数量: {len(shape_different)}\n\n")
                
                if len(shape_different) > 0:
                    shape_sorted = shape_different.sort_values('ks_statistic', ascending=False)
                    for i, (_, row) in enumerate(shape_sorted.iterrows()):
                        f.write(f"  {i+1:2d}. {row['feature_name']:<35} "
                               f"KS={row.get('ks_statistic', 0):.4f} "
                               f"p={row.get('ks_p_value', 1):.3e}\n")
            
            # 总结性评估
            f.write(f"\n4. 整体评估\n")
            f.write("-" * 40 + "\n")
            
            large_effect = len(df[df.get('effect_magnitude', '') == 'large'])
            medium_effect = len(df[df.get('effect_magnitude', '') == 'medium'])
            significant_count = len(df[df['significant'] == True])
            
            f.write(f"- 具有大效应的驱动因素: {large_effect} 个\n")
            f.write(f"- 具有中等效应的驱动因素: {medium_effect} 个\n")
            f.write(f"- 统计显著的驱动因素: {significant_count} 个\n")
            f.write(f"- 总体显著率: {significant_count/len(df)*100:.1f}%\n\n")
            
            if large_effect > 0:
                f.write("重要发现: 存在大效应的驱动因素，表明燃烧和未燃烧区域在这些因素上有明显的整体分布差异。\n")
            elif medium_effect > 5:
                f.write("重要发现: 存在多个中等效应的驱动因素，表明燃烧和未燃烧区域有中等程度的分布差异。\n")
            else:
                f.write("发现: 大多数驱动因素的效应较小，燃烧和未燃烧区域的整体分布差异不明显。\n")
        
        logger.info(f"详细分布分析保存到: {analysis_file}")
    
    def analyze_temporal_driver_distributions(self, time_windows=[30, 365]):
        """
        高效时间窗口分析
        
        Args:
            time_windows: 时间窗口列表（天数），如[30, 365]表示30天和365天
        """
        from datetime import datetime, timedelta
        import random
        
        logger.info(f"开始高效时间窗口分析，时间窗口: {time_windows} 天")
        
        # 注意：时间窗口分析将逐个驱动因素处理，避免内存溢出
        logger.info("开始逐个驱动因素进行时间窗口分析...")
        
        # 获取所有燃烧事件
        all_burn_events = []
        
        for pixel_coord, burn_dates in self.pixel_burn_times.items():
            for burn_date in burn_dates:
                all_burn_events.append((pixel_coord, burn_date))
        
        logger.info(f"分析所有 {len(all_burn_events)} 个燃烧事件")
        
        # 获取未燃烧像元列表用于随机采样
        unburned_list = list(self.unburned_pixels)
        
        # 为每个时间窗口进行分析
        for window_days in time_windows:
            logger.info(f"分析 {window_days} 天时间窗口...")
            
            # 使用逐个驱动因素的内存优化分析
            temporal_results = self._driver_by_driver_temporal_analysis(
                all_burn_events, unburned_list, window_days
            )
            
            # 保存该时间窗口的结果
            if temporal_results:
                self._save_temporal_analysis_results(temporal_results, window_days)
                logger.info(f"  {window_days}天时间窗口分析完成，生成 {len(temporal_results)} 条记录")
                
                # 保存时间窗口的整体分布统计
                if hasattr(self, 'temporal_distribution_stats') and window_days in self.temporal_distribution_stats:
                    self._save_temporal_distribution_stats(window_days)
    
    def _save_temporal_analysis_results(self, temporal_results, window_days):
        """保存时间窗口分析结果"""
        df = pd.DataFrame(temporal_results)
        csv_file = os.path.join(self.output_subdir, f'temporal_analysis_{window_days}days.csv')
        df.to_csv(csv_file, index=False)
        logger.info(f"时间窗口分析结果保存到: {csv_file}")
    
    def _save_temporal_distribution_stats(self, window_days):
        """保存时间窗口的整体分布统计"""
        if not hasattr(self, 'temporal_distribution_stats') or window_days not in self.temporal_distribution_stats:
            return
        
        temporal_stats = self.temporal_distribution_stats[window_days]
        if not temporal_stats:
            return
        
        # 保存CSV格式的分布统计
        df = pd.DataFrame(temporal_stats)
        csv_file = os.path.join(self.output_subdir, f'temporal_distribution_analysis_{window_days}days.csv')
        df.to_csv(csv_file, index=False)
        logger.info(f"时间窗口{window_days}天整体分布统计保存到: {csv_file}")
        
        # 创建时间窗口的汇总报告
        self._create_temporal_summary_report(df, window_days)
        
        # 创建时间窗口的详细分析
        self._create_temporal_distribution_analysis(df, window_days)
    
    def _create_temporal_summary_report(self, df, window_days):
        """创建时间窗口整体分布分析汇总报告"""
        summary_file = os.path.join(self.output_subdir, f'temporal_distribution_summary_{window_days}days.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"时间窗口{window_days}天驱动因素整体分布分析汇总报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"时间窗口: 燃烧前{window_days}天\n")
            f.write(f"分析类型: 燃烧像元燃烧前{window_days}天 vs 对照未燃烧像元同期的整体分布对比\n")
            f.write(f"总特征数量: {len(df)}\n")
            f.write(f"统计显著性特征数量: {len(df[df['significant']])}\n")
            f.write(f"非参数检验显著性特征数量: {len(df[df['mannwhitney_significant']])}\n")
            f.write(f"分布形状显著不同特征数量: {len(df[df['distributions_different']])}\n\n")
            
            # 效应大小分布
            if 'effect_magnitude' in df.columns:
                effect_counts = df['effect_magnitude'].value_counts()
                f.write("效应大小分布:\n")
                f.write("-" * 30 + "\n")
                for effect, count in effect_counts.items():
                    f.write(f"  {effect}: {count} 个特征\n")
                f.write("\n")
            
            # 按效应大小排序显示最重要的特征
            if 'effect_size_cohens_d' in df.columns:
                df_sorted = df.sort_values('effect_size_cohens_d', key=abs, ascending=False)
                
                f.write(f"燃烧前{window_days}天效应大小最大的前15个特征 (Cohen's d):\n")
                f.write("-" * 60 + "\n")
                for i, (_, row) in enumerate(df_sorted.head(15).iterrows()):
                    f.write(f"{i+1:2d}. {row['feature_name']:<40} "
                           f"d={row.get('effect_size_cohens_d', 0):.3f} "
                           f"({row.get('effect_magnitude', 'unknown'):<8}) "
                           f"显著={'✓' if row['significant'] else '✗'}\n")
                f.write("\n")
            
            # 按平均差异排序显示特征
            df_sorted_mean = df.sort_values('mean_difference', key=abs, ascending=False)
            
            f.write(f"燃烧前{window_days}天平均差异最大的前15个特征:\n")
            f.write("-" * 60 + "\n")
            for i, (_, row) in enumerate(df_sorted_mean.head(15).iterrows()):
                f.write(f"{i+1:2d}. {row['feature_name']:<40} "
                       f"差异={row['mean_difference']:8.4f} "
                       f"显著={'✓' if row['significant'] else '✗'}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("说明:\n")
            f.write(f"- 本分析比较燃烧像元在燃烧前{window_days}天的特征分布与对照未燃烧像元同期的特征分布\n")
            f.write("- Cohen's d: 效应大小，>0.8为大效应，0.5-0.8为中等，0.2-0.5为小效应\n")
            f.write("- 显著性: t检验p<0.05\n")
        
        logger.info(f"时间窗口{window_days}天整体分布分析汇总报告保存到: {summary_file}")
    
    def _create_temporal_distribution_analysis(self, df, window_days):
        """创建时间窗口分布分析的详细信息"""
        analysis_file = os.path.join(self.output_subdir, f'temporal_distribution_detailed_analysis_{window_days}days.txt')
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(f"时间窗口{window_days}天驱动因素整体分布详细分析\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"分析说明: 比较燃烧像元在燃烧前{window_days}天与对照未燃烧像元同期的驱动因素整体分布差异\n\n")
            
            # 统计显著性分析
            f.write("1. 统计显著性分析\n")
            f.write("-" * 40 + "\n")
            
            significant_features = df[df['significant'] == True]
            f.write(f"统计显著的特征数量: {len(significant_features)}\n\n")
            
            if len(significant_features) > 0:
                f.write("统计显著的特征 (按效应大小排序):\n")
                sig_sorted = significant_features.sort_values('effect_size_cohens_d', key=abs, ascending=False)
                for i, (_, row) in enumerate(sig_sorted.iterrows()):
                    f.write(f"  {i+1:2d}. {row['feature_name']:<45} "
                           f"Cohen's d={row.get('effect_size_cohens_d', 0):7.3f} "
                           f"p={row['p_value']:.3e}\n")
                f.write("\n")
            
            # 效应大小分析
            f.write("2. 效应大小分析\n")
            f.write("-" * 40 + "\n")
            
            if 'effect_magnitude' in df.columns:
                for magnitude in ['large', 'medium', 'small', 'negligible']:
                    magnitude_features = df[df['effect_magnitude'] == magnitude]
                    if len(magnitude_features) > 0:
                        f.write(f"\n{magnitude.upper()} 效应特征 ({len(magnitude_features)} 个):\n")
                        mag_sorted = magnitude_features.sort_values('effect_size_cohens_d', key=abs, ascending=False)
                        for i, (_, row) in enumerate(mag_sorted.iterrows()):
                            f.write(f"  {i+1:2d}. {row['feature_name']:<45} "
                                   f"d={row.get('effect_size_cohens_d', 0):7.3f}\n")
            
            # 总结性评估
            f.write(f"\n3. 时间窗口{window_days}天整体评估\n")
            f.write("-" * 40 + "\n")
            
            large_effect = len(df[df.get('effect_magnitude', '') == 'large'])
            medium_effect = len(df[df.get('effect_magnitude', '') == 'medium'])
            significant_count = len(df[df['significant'] == True])
            
            f.write(f"- 燃烧前{window_days}天具有大效应的驱动因素: {large_effect} 个\n")
            f.write(f"- 燃烧前{window_days}天具有中等效应的驱动因素: {medium_effect} 个\n")
            f.write(f"- 燃烧前{window_days}天统计显著的驱动因素: {significant_count} 个\n")
            f.write(f"- 显著率: {significant_count/len(df)*100:.1f}%\n\n")
            
            if large_effect > 0:
                f.write(f"重要发现: 在燃烧前{window_days}天，存在大效应的驱动因素，表明即将燃烧的区域在这些因素上已显示出与未燃烧区域的明显差异。\n")
            elif medium_effect > 3:
                f.write(f"重要发现: 在燃烧前{window_days}天，存在多个中等效应的驱动因素，表明即将燃烧的区域在一些因素上已显示出差异。\n")
            else:
                f.write(f"发现: 在燃烧前{window_days}天，大多数驱动因素的效应较小，即将燃烧和未燃烧区域的差异不明显。\n")
        
        logger.info(f"时间窗口{window_days}天详细分布分析保存到: {analysis_file}")
    
    def run_analysis(self):
        """运行完整分析流程"""
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("开始运行像元驱动因素分布分析")
        logger.info("=" * 60)
        
        # 步骤1：识别像元燃烧状态
        self.identify_pixel_burn_status()
        
        # 步骤2：分析驱动因素分布
        self.analyze_driver_distributions()
        
        # 步骤3：时间窗口分析（如果启用完整时间序列分析）
        if self.analyze_full_timeseries and hasattr(self, 'pixel_burn_times'):
            self.analyze_temporal_driver_distributions([30, 365])

        total_time = time.time() - start_time
        logger.info(f"分析完成！总耗时: {total_time:.2f} 秒")
        logger.info(f"结果保存在: {self.output_subdir}")

def main():
    """主函数"""
    # 配置路径
    data_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked"
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials"
    
    # 创建分析器
    analyzer = EfficientPixelDriverDistributionAnalyzer(
        data_dir=data_dir,
        output_dir=output_dir,
        analyze_full_timeseries=True,  # 启用完整时间序列分析
        n_processes=8  # 使用8个进程
    )
    
    # 运行分析
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 