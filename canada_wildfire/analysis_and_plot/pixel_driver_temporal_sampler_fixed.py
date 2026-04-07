#!/usr/bin/env python3
"""
像元驱动因素时间对应抽样器 - 修正版本
修复数据处理问题，使用核密度估计替代bins
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import random
import logging
import argparse
import time
import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from tqdm import tqdm
from osgeo import gdal
import warnings
from scipy import stats
from sklearn.neighbors import KernelDensity

# 抑制警告
warnings.filterwarnings('ignore')
gdal.SetConfigOption('CPL_LOG', '/dev/null')
gdal.PushErrorHandler('CPLQuietErrorHandler')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PixelDriverTemporalSamplerFixed:
    def __init__(self, data_dir, output_dir, sampling_ratio=0.1, random_seed=42):
        """
        初始化时间对应抽样器 - 修正版本
        
        Args:
            data_dir: 驱动因素数据根目录
            output_dir: 输出根目录
            sampling_ratio: 抽样比例（默认10%）
            random_seed: 随机种子
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.sampling_ratio = sampling_ratio
        self.random_seed = random_seed
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 创建输出目录
        self.output_subdir = os.path.join(output_dir, "pixel_driver_temporal_samples_fixed")
        os.makedirs(self.output_subdir, exist_ok=True)
        
        # 获取驱动因素目录
        self.driver_dirs = self._get_driver_directories()
        self.static_drivers, self.dynamic_drivers = self._identify_static_dynamic_drivers()
        
        # 存储燃烧事件信息：{(row, col): burn_date}
        self.pixel_burn_events = {}
        # 存储所有有效像元
        self.all_valid_pixels = set()
        
        logger.info(f"找到 {len(self.static_drivers)} 个静态驱动因素")
        logger.info(f"找到 {len(self.dynamic_drivers)} 个动态驱动因素")
        logger.info(f"抽样比例: {sampling_ratio*100}%")
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
        
        static_keywords = ['topo', 'distance', 'elevation', 'slope', 'aspect']
        
        for driver_name, driver_dir in self.driver_dirs.items():
            # 检查是否是FIRMS数据（用于像元分类）
            if any(keyword in driver_name.lower() for keyword in ['firms', 'fire']):
                continue
                
            is_static = any(keyword in driver_name.lower() for keyword in static_keywords)
            
            if is_static:
                static_drivers[driver_name] = driver_dir
            else:
                dynamic_drivers[driver_name] = driver_dir
        
        return static_drivers, dynamic_drivers
    
    def _load_file_data(self, file_path):
        """加载TIFF文件数据"""
        try:
            ds = gdal.Open(file_path, gdal.GA_ReadOnly)
            if ds is None:
                return None, None
            
            bands = ds.RasterCount
            height = ds.RasterYSize
            width = ds.RasterXSize
            
            if bands == 1:
                data = ds.GetRasterBand(1).ReadAsArray()
            else:
                data = np.zeros((bands, height, width), dtype=np.float32)
                for i in range(bands):
                    data[i] = ds.GetRasterBand(i + 1).ReadAsArray()
            
            ds = None
            
            # 确保数据是3D格式 (bands, height, width)
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            elif data.ndim == 3 and data.shape[0] != bands:
                if data.shape[2] == bands:
                    data = np.transpose(data, (2, 0, 1))
            
            return data, {'bands': bands, 'height': height, 'width': width}
            
        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {str(e)}")
            return None, None
    
    def _clean_and_validate_value(self, value):
        """清理和验证数值"""
        try:
            # 转换为浮点数
            value = float(value)
            
            # 检查是否为有效数值
            if np.isnan(value) or np.isinf(value):
                return None
            
            # 检查是否为NoData值
            if value in [0, 255, -9999]:
                return None
            
            # 检查数值是否在合理范围内（避免异常大的值）
            if abs(value) > 1e10:
                return None
                
            return value
            
        except (ValueError, TypeError):
            return None
    
    def identify_burn_events(self):
        """识别燃烧事件及其发生日期"""
        logger.info("开始识别燃烧事件...")
        
        # 获取FIRMS数据目录
        firms_dir = None
        for driver_name, driver_dir in self.driver_dirs.items():
            if 'FIRMS' in driver_name or 'Firms' in driver_name or 'firms' in driver_name:
                firms_dir = driver_dir
                logger.info(f"使用FIRMS数据目录: {driver_name}")
                break
        
        if not firms_dir:
            raise ValueError("未找到FIRMS数据目录")
        
        # 获取所有FIRMS文件
        firms_files = glob.glob(os.path.join(firms_dir, '*.tif'))
        firms_files.sort()
        
        logger.info(f"找到 {len(firms_files)} 个FIRMS文件")
        
        # 处理每个FIRMS文件，记录燃烧事件
        all_valid_pixels = set()
        pixel_burn_events = {}
        
        for firms_file in tqdm(firms_files, desc="处理FIRMS文件"):
            file_date = self._extract_date_from_filename(os.path.basename(firms_file))
            
            try:
                # 转换为datetime对象以便计算
                burn_date = datetime.strptime(file_date, '%Y-%m-%d')
            except:
                logger.warning(f"无法解析日期: {file_date}, 跳过文件: {firms_file}")
                continue
            
            valid_pixels, burned_pixels = self._process_firms_file(firms_file)
            if valid_pixels is not None:
                all_valid_pixels.update(valid_pixels)
                
                # 记录燃烧事件
                for pixel_coord in burned_pixels:
                    # 只记录第一次燃烧事件（或者可以记录所有事件）
                    if pixel_coord not in pixel_burn_events:
                        pixel_burn_events[pixel_coord] = burn_date
        
        self.all_valid_pixels = all_valid_pixels
        self.pixel_burn_events = pixel_burn_events
        
        logger.info(f"识别完成:")
        logger.info(f"  总有效像元: {len(self.all_valid_pixels)}")
        logger.info(f"  燃烧事件数: {len(self.pixel_burn_events)}")
        
        # 统计燃烧日期分布
        if self.pixel_burn_events:
            burn_dates = list(self.pixel_burn_events.values())
            date_counts = Counter([d.strftime('%Y-%m-%d') for d in burn_dates])
            logger.info(f"  燃烧日期范围: {min(burn_dates).strftime('%Y-%m-%d')} 到 {max(burn_dates).strftime('%Y-%m-%d')}")
            logger.info(f"  涉及日期数: {len(date_counts)}")
    
    def _process_firms_file(self, firms_file):
        """处理单个FIRMS文件"""
        try:
            data, file_info = self._load_file_data(firms_file)
            if data is None:
                return None, None
            
            # FIRMS只有一个波段
            firms_data = data[0] if data.ndim == 3 else data
            
            # 获取所有有效像元位置
            valid_mask = (firms_data != 255) & (firms_data != -9999) & np.isfinite(firms_data)
            valid_pixels = set(zip(*np.where(valid_mask)))
            
            # 获取燃烧像元位置（值>0的位置）
            burned_mask = (firms_data > 0) & valid_mask
            burned_pixels = set(zip(*np.where(burned_mask)))
            
            return valid_pixels, burned_pixels
            
        except Exception as e:
            logger.error(f"处理FIRMS文件失败: {firms_file}, 错误: {str(e)}")
            return None, None
    
    def _extract_date_from_filename(self, filename):
        """从文件名提取日期"""
        patterns = [
            r'(\d{4})_(\d{2})_(\d{2})',
            r'(\d{4})-(\d{2})-(\d{2})',
            r'(\d{8})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                if len(match.groups()) == 3:
                    year, month, day = match.groups()
                    return f"{year}-{month}-{day}"
                else:
                    date_str = match.group(1)
                    if len(date_str) == 8:
                        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        return filename
    
    def sample_burn_events(self):
        """抽样燃烧事件"""
        logger.info("开始抽样燃烧事件...")
        
        # 计算抽样数量
        total_burn_events = len(self.pixel_burn_events)
        sample_size = max(1, int(total_burn_events * self.sampling_ratio))
        
        # 随机抽样燃烧事件
        sampled_burn_pixels = random.sample(list(self.pixel_burn_events.keys()), sample_size)
        self.sampled_burn_events = {pixel: self.pixel_burn_events[pixel] 
                                   for pixel in sampled_burn_pixels}
        
        logger.info(f"抽样完成:")
        logger.info(f"  抽样燃烧事件: {len(self.sampled_burn_events)} / {total_burn_events} "
                   f"({len(self.sampled_burn_events)/total_burn_events*100:.1f}%)")
        
        # 按日期统计抽样的燃烧事件
        sampled_dates = [d.strftime('%Y-%m-%d') for d in self.sampled_burn_events.values()]
        date_counts = Counter(sampled_dates)
        logger.info(f"  抽样涉及日期数: {len(date_counts)}")
        for date, count in sorted(date_counts.items())[:5]:  # 显示前5个日期
            logger.info(f"    {date}: {count} 个事件")
    
    def select_control_pixels(self):
        """为每个燃烧日期选择对照像元"""
        logger.info("开始选择对照像元...")
        
        # 按燃烧日期分组
        events_by_date = defaultdict(list)
        for pixel, burn_date in self.sampled_burn_events.items():
            events_by_date[burn_date].append(pixel)
        
        self.control_events = {}
        
        # 获取未燃烧的像元
        unburned_pixels = list(self.all_valid_pixels - set(self.pixel_burn_events.keys()))
        
        for burn_date, burned_pixels in events_by_date.items():
            # 为当前日期的燃烧像元选择相同数量的对照像元
            n_controls = len(burned_pixels)
            if len(unburned_pixels) >= n_controls:
                control_pixels = random.sample(unburned_pixels, n_controls)
                for control_pixel in control_pixels:
                    self.control_events[control_pixel] = burn_date
            else:
                logger.warning(f"日期 {burn_date.strftime('%Y-%m-%d')} 的未燃烧像元不足，"
                              f"需要 {n_controls} 个，可用 {len(unburned_pixels)} 个")
        
        logger.info(f"选择对照像元完成: {len(self.control_events)} 个")
    
    def extract_temporal_data(self):
        """提取时间对应的驱动因素数据"""
        logger.info("开始提取时间对应的驱动因素数据...")
        
        # 准备数据结构
        # data_matrix[feature_name][pixel_type][date] = [pixel_values]
        self.data_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # 收集所有需要处理的日期
        all_target_dates = set()
        
        # 燃烧像元：提取燃烧前一天的数据
        for pixel, burn_date in self.sampled_burn_events.items():
            target_date = burn_date - timedelta(days=1)
            all_target_dates.add(target_date)
        
        # 对照像元：提取相同日期的数据
        for pixel, burn_date in self.control_events.items():
            target_date = burn_date - timedelta(days=1)
            all_target_dates.add(target_date)
        
        logger.info(f"需要处理的日期数: {len(all_target_dates)}")
        
        # 处理静态驱动因素
        self._extract_static_temporal_data()
        
        # 处理动态驱动因素
        self._extract_dynamic_temporal_data(all_target_dates)
        
        logger.info("时间对应数据提取完成")
    
    def _extract_static_temporal_data(self):
        """提取静态驱动因素数据"""
        for driver_name, driver_dir in self.static_drivers.items():
            logger.info(f"处理静态驱动因素: {driver_name}")
            
            # 获取第一个文件（静态数据所有文件相同）
            driver_files = glob.glob(os.path.join(driver_dir, '*.tif'))
            if not driver_files:
                continue
            
            data, file_info = self._load_file_data(driver_files[0])
            if data is None:
                continue
                
            bands = file_info['bands']
            
            # 为每个波段提取数据
            for band_idx in range(bands):
                feature_name = f"{driver_name}_band_{band_idx+1}" if bands > 1 else driver_name
                band_data = data[band_idx]
                
                # 提取燃烧像元的值
                burned_values = []
                for pixel in self.sampled_burn_events.keys():
                    row, col = pixel
                    if row < band_data.shape[0] and col < band_data.shape[1]:
                        value = self._clean_and_validate_value(band_data[row, col])
                        if value is not None:
                            burned_values.append(value)
                
                # 提取对照像元的值
                control_values = []
                for pixel in self.control_events.keys():
                    row, col = pixel
                    if row < band_data.shape[0] and col < band_data.shape[1]:
                        value = self._clean_and_validate_value(band_data[row, col])
                        if value is not None:
                            control_values.append(value)
                
                # 对于静态数据，所有日期使用相同的值
                self.data_matrix[feature_name]['burned']['static'] = burned_values
                self.data_matrix[feature_name]['unburned']['static'] = control_values
    
    def _extract_dynamic_temporal_data(self, target_dates):
        """提取动态驱动因素数据"""
        for driver_name, driver_dir in self.dynamic_drivers.items():
            logger.info(f"处理动态驱动因素: {driver_name}")
            
            # 获取该驱动因素的所有文件
            driver_files = glob.glob(os.path.join(driver_dir, '*.tif'))
            
            # 建立日期到文件的映射
            date_to_file = {}
            for file_path in driver_files:
                file_date_str = self._extract_date_from_filename(os.path.basename(file_path))
                try:
                    file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                    date_to_file[file_date] = file_path
                except:
                    continue
            
            # 获取波段信息
            if not driver_files:
                continue
            sample_data, file_info = self._load_file_data(driver_files[0])
            if sample_data is None:
                continue
            bands = file_info['bands']
            
            # 为每个目标日期提取数据
            for target_date in tqdm(target_dates, desc=f"处理{driver_name}日期"):
                if target_date not in date_to_file:
                    continue
                
                file_path = date_to_file[target_date]
                data, _ = self._load_file_data(file_path)
                if data is None:
                    continue
                
                # 确保数据是3D格式
                if data.ndim == 2:
                    data = data[np.newaxis, :, :]
                
                date_str = target_date.strftime('%Y-%m-%d')
                
                # 为每个波段提取数据
                for band_idx in range(bands):
                    feature_name = f"{driver_name}_band_{band_idx+1}" if bands > 1 else driver_name
                    band_data = data[band_idx]
                    
                    # 提取燃烧像元的值
                    for pixel, burn_date in self.sampled_burn_events.items():
                        pixel_target_date = burn_date - timedelta(days=1)
                        if pixel_target_date == target_date:
                            row, col = pixel
                            if row < band_data.shape[0] and col < band_data.shape[1]:
                                value = self._clean_and_validate_value(band_data[row, col])
                                if value is not None:
                                    self.data_matrix[feature_name]['burned'][date_str].append(value)
                    
                    # 提取对照像元的值
                    for pixel, burn_date in self.control_events.items():
                        pixel_target_date = burn_date - timedelta(days=1)
                        if pixel_target_date == target_date:
                            row, col = pixel
                            if row < band_data.shape[0] and col < band_data.shape[1]:
                                value = self._clean_and_validate_value(band_data[row, col])
                                if value is not None:
                                    self.data_matrix[feature_name]['unburned'][date_str].append(value)
    
    def create_temporal_distribution_csv(self):
        """创建第一个CSV：时间分布表格（改进格式）"""
        logger.info("创建时间分布CSV...")
        
        # 收集所有日期
        all_dates = set()
        for feature_data in self.data_matrix.values():
            for pixel_type_data in feature_data.values():
                all_dates.update(pixel_type_data.keys())
        
        all_dates = sorted([d for d in all_dates if d != 'static'])
        if 'static' in [d for feature_data in self.data_matrix.values() 
                       for pixel_type_data in feature_data.values() 
                       for d in pixel_type_data.keys()]:
            all_dates = ['static'] + all_dates
        
        # 准备数据
        rows = []
        
        for feature_name in sorted(self.data_matrix.keys()):
            for pixel_type in ['burned', 'unburned']:
                row = {
                    'feature_name': feature_name,
                    'pixel_type': pixel_type
                }
                
                for date in all_dates:
                    values = self.data_matrix[feature_name][pixel_type].get(date, [])
                    # 将值列表转换为字符串，以分号分隔，保留合理精度
                    if values:
                        formatted_values = [f"{v:.6f}" for v in values]
                        row[date] = ';'.join(formatted_values)
                    else:
                        row[date] = ''
                
                rows.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(rows)
        csv_file = os.path.join(self.output_subdir, f'temporal_distribution_{self.sampling_ratio*100:.0f}pct.csv')
        df.to_csv(csv_file, index=False)
        
        logger.info(f"时间分布CSV已保存: {csv_file}")
        logger.info(f"包含 {len(df)} 行, {len(all_dates)} 个日期列")
        
        return csv_file
    
    def create_kde_density_csv(self):
        """创建第二个CSV：核密度估计（KDE）"""
        logger.info("创建核密度估计CSV...")
        
        rows = []
        
        for feature_name in sorted(self.data_matrix.keys()):
            for pixel_type in ['burned', 'unburned']:
                # 收集该特征和像元类型的所有值
                all_values = []
                for date_values in self.data_matrix[feature_name][pixel_type].values():
                    all_values.extend(date_values)
                
                if len(all_values) < 2:  # KDE需要至少2个数据点
                    continue
                
                all_values = np.array(all_values)
                
                # 计算KDE
                try:
                    # 使用高斯核密度估计
                    kde = stats.gaussian_kde(all_values)
                    
                    # 创建评估点（从最小值到最大值均匀分布）
                    min_val, max_val = np.min(all_values), np.max(all_values)
                    
                    # 扩展范围以获得更好的估计
                    range_ext = (max_val - min_val) * 0.1
                    eval_points = np.linspace(min_val - range_ext, max_val + range_ext, 200)
                    
                    # 计算每个点的密度
                    densities = kde(eval_points)
                    
                    # 为每个原始数据点计算密度值
                    original_densities = kde(all_values)
                    
                    # 创建数据记录
                    for i, (value, density) in enumerate(zip(all_values, original_densities)):
                        rows.append({
                            'feature_name': feature_name,
                            'pixel_type': pixel_type,
                            'pixel_value': float(value),
                            'kde_density': float(density),
                            'sample_index': i,
                            'total_samples': len(all_values),
                            'min_value': float(min_val),
                            'max_value': float(max_val),
                            'mean_value': float(np.mean(all_values)),
                            'std_value': float(np.std(all_values))
                        })
                    
                except Exception as e:
                    logger.warning(f"计算KDE失败 {feature_name} {pixel_type}: {str(e)}")
                    continue
        
        # 创建DataFrame并保存
        df = pd.DataFrame(rows)
        csv_file = os.path.join(self.output_subdir, f'kde_density_{self.sampling_ratio*100:.0f}pct.csv')
        df.to_csv(csv_file, index=False)
        
        logger.info(f"核密度估计CSV已保存: {csv_file}")
        logger.info(f"包含 {len(df)} 个数据点，{df['feature_name'].nunique()} 个特征")
        
        return csv_file
    
    def create_kde_curves_csv(self):
        """创建第三个CSV：KDE曲线数据（用于绘图）"""
        logger.info("创建KDE曲线数据CSV...")
        
        rows = []
        
        for feature_name in sorted(self.data_matrix.keys()):
            for pixel_type in ['burned', 'unburned']:
                # 收集该特征和像元类型的所有值
                all_values = []
                for date_values in self.data_matrix[feature_name][pixel_type].values():
                    all_values.extend(date_values)
                
                if len(all_values) < 2:
                    continue
                
                all_values = np.array(all_values)
                
                try:
                    # 计算KDE
                    kde = stats.gaussian_kde(all_values)
                    
                    # 创建评估点
                    min_val, max_val = np.min(all_values), np.max(all_values)
                    range_ext = (max_val - min_val) * 0.1
                    eval_points = np.linspace(min_val - range_ext, max_val + range_ext, 200)
                    
                    # 计算密度
                    densities = kde(eval_points)
                    
                    # 为每个评估点创建记录
                    for x, density in zip(eval_points, densities):
                        rows.append({
                            'feature_name': feature_name,
                            'pixel_type': pixel_type,
                            'x_value': float(x),
                            'kde_density': float(density),
                            'total_samples': len(all_values)
                        })
                    
                except Exception as e:
                    logger.warning(f"计算KDE曲线失败 {feature_name} {pixel_type}: {str(e)}")
                    continue
        
        # 创建DataFrame并保存
        df = pd.DataFrame(rows)
        csv_file = os.path.join(self.output_subdir, f'kde_curves_{self.sampling_ratio*100:.0f}pct.csv')
        df.to_csv(csv_file, index=False)
        
        logger.info(f"KDE曲线数据CSV已保存: {csv_file}")
        logger.info(f"包含 {len(df)} 个曲线点")
        
        return csv_file
    
    def create_summary_report(self):
        """创建汇总报告"""
        logger.info("创建汇总报告...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("像元驱动因素时间对应抽样分析报告 - 修正版本")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"抽样比例: {self.sampling_ratio*100:.1f}%")
        report_lines.append(f"随机种子: {self.random_seed}")
        report_lines.append("")
        
        # 燃烧事件统计
        report_lines.append("燃烧事件统计:")
        report_lines.append(f"  总燃烧事件: {len(self.pixel_burn_events)}")
        report_lines.append(f"  抽样燃烧事件: {len(self.sampled_burn_events)}")
        report_lines.append(f"  对照像元数: {len(self.control_events)}")
        report_lines.append("")
        
        # 时间范围统计
        if self.sampled_burn_events:
            burn_dates = list(self.sampled_burn_events.values())
            min_date = min(burn_dates).strftime('%Y-%m-%d')
            max_date = max(burn_dates).strftime('%Y-%m-%d')
            unique_dates = len(set(d.strftime('%Y-%m-%d') for d in burn_dates))
            
            report_lines.append("时间范围统计:")
            report_lines.append(f"  燃烧日期范围: {min_date} 到 {max_date}")
            report_lines.append(f"  涉及日期数: {unique_dates}")
            report_lines.append("")
        
        # 特征统计
        report_lines.append("特征统计:")
        report_lines.append(f"  总特征数: {len(self.data_matrix)}")
        report_lines.append(f"  静态驱动因素数: {len(self.static_drivers)}")
        report_lines.append(f"  动态驱动因素数: {len(self.dynamic_drivers)}")
        report_lines.append("")
        
        # 按特征统计样本数
        report_lines.append("按特征统计样本数:")
        for feature_name in sorted(self.data_matrix.keys()):
            burned_count = sum(len(values) for values in self.data_matrix[feature_name]['burned'].values())
            unburned_count = sum(len(values) for values in self.data_matrix[feature_name]['unburned'].values())
            total_count = burned_count + unburned_count
            report_lines.append(f"  {feature_name}: 总计{total_count} (燃烧{burned_count}, 未燃烧{unburned_count})")
        
        # 数据质量检查
        report_lines.append("")
        report_lines.append("数据质量检查:")
        for feature_name in sorted(self.data_matrix.keys()):
            for pixel_type in ['burned', 'unburned']:
                all_values = []
                for values in self.data_matrix[feature_name][pixel_type].values():
                    all_values.extend(values)
                if all_values:
                    min_val = min(all_values)
                    max_val = max(all_values)
                    mean_val = np.mean(all_values)
                    report_lines.append(f"  {feature_name} ({pixel_type}): 范围[{min_val:.6f}, {max_val:.6f}], 均值{mean_val:.6f}")
        
        report_lines.append("=" * 80)
        
        # 保存报告
        report_file = os.path.join(self.output_subdir, f'temporal_sampling_report_{self.sampling_ratio*100:.0f}pct.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"汇总报告已保存: {report_file}")
        
        # 同时输出到控制台
        for line in report_lines:
            print(line)
    
    def run_analysis(self):
        """运行完整分析"""
        logger.info("开始运行像元驱动因素时间对应抽样分析（修正版本）...")
        
        start_time = time.time()
        
        # 1. 识别燃烧事件及其日期
        self.identify_burn_events()
        
        # 2. 抽样燃烧事件
        self.sample_burn_events()
        
        # 3. 选择对照像元
        self.select_control_pixels()
        
        # 4. 提取时间对应的驱动因素数据
        self.extract_temporal_data()
        
        # 5. 创建CSV表格
        csv1 = self.create_temporal_distribution_csv()
        csv2 = self.create_kde_density_csv()
        csv3 = self.create_kde_curves_csv()
        
        # 6. 创建汇总报告
        self.create_summary_report()
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"分析完成，总耗时: {total_time:.2f} 秒")
        
        return csv1, csv2, csv3


def main():
    parser = argparse.ArgumentParser(description='像元驱动因素时间对应抽样器 - 修正版本')
    parser.add_argument('--data_dir', 
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked_normalized',
                       help='驱动因素数据根目录')
    parser.add_argument('--output_dir', 
                       default='./canada_wildfire',
                       help='输出目录')
    parser.add_argument('--sampling_ratio', 
                       type=float, 
                       default=0.1,
                       help='抽样比例 (默认0.1即10%%)')
    parser.add_argument('--random_seed', 
                       type=int, 
                       default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    logger.info("启动像元驱动因素时间对应抽样器 - 修正版本")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"抽样比例: {args.sampling_ratio*100}%")
    
    # 创建分析器并运行
    analyzer = PixelDriverTemporalSamplerFixed(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sampling_ratio=args.sampling_ratio,
        random_seed=args.random_seed
    )
    
    csv1, csv2, csv3 = analyzer.run_analysis()
    
    print("\n" + "="*60)
    print("✅ 分析完成！生成的文件:")
    print(f"📊 时间分布表格: {csv1}")
    print(f"📈 KDE密度数据: {csv2}")
    print(f"📈 KDE曲线数据: {csv3}")
    print(f"📄 汇总报告: {analyzer.output_subdir}/temporal_sampling_report_{args.sampling_ratio*100:.0f}pct.txt")


if __name__ == "__main__":
    main() 