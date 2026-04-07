#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
像元时间序列趋势分解分析器

功能：
- 按燃烧频次分层采样像元（1-2次、3-4次、5-6次、6-10次、10次以上各2个）
- 随机选择10个未燃烧像元作为对照
- 对每个像元的每个驱动因素进行时间序列趋势分解
- 使用LOESS方法分解为季节性、趋势和残差成分
- 计算趋势均值和Thiel-Sen斜率
"""

import os
import glob
import numpy as np
import pandas as pd
from osgeo import gdal
import logging
from datetime import datetime
import gc
import re
from tqdm import tqdm
import random
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 导入趋势分解相关库
try:
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from scipy import stats
    import matplotlib.pyplot as plt
    plt.style.use('default')
    DECOMPOSE_AVAILABLE = True
except ImportError as e:
    DECOMPOSE_AVAILABLE = False
    print(f"警告: 缺少必要的时间序列分析库: {e}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PixelTimeseriesTrendAnalysis')

class PixelTimeseriesTrendAnalyzer:
    def __init__(self, data_dir, output_dir, firms_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.firms_dir = firms_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.driver_dirs = self._get_driver_directories()
        self.dynamic_drivers = self._identify_dynamic_drivers()
        
        self.pixel_burn_counts = {}
        self.all_dates = []
        
        self.selected_pixels = {
            'burn_1_2': [], 'burn_3_4': [], 'burn_5_6': [],
            'burn_6_10': [], 'burn_10_plus': [], 'no_burn': []
        }
        
        logger.info(f"找到 {len(self.dynamic_drivers)} 个动态驱动因素")
        
        if not DECOMPOSE_AVAILABLE:
            raise ImportError("缺少必要的时间序列分析库，请先安装 statsmodels, scipy, matplotlib")
    
    def _get_driver_directories(self):
        driver_dirs = {}
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                tif_files = glob.glob(os.path.join(item_path, '*.tif'))
                if tif_files:
                    driver_dirs[item] = item_path
        return driver_dirs
    
    def _identify_dynamic_drivers(self):
        dynamic_drivers = {}
        static_drivers = {'Topo_Distance_WGS84_resize_resampled'}
        
        for driver_name, driver_dir in self.driver_dirs.items():
            if driver_name not in static_drivers and driver_name != 'Firms_Detection_resampled':
                dynamic_drivers[driver_name] = driver_dir
        
        return dynamic_drivers
    
    def _extract_date_from_filename(self, filename):
        patterns = [
            r'(\d{4})_(\d{2})_(\d{2})',
            r'(\d{4})-(\d{2})-(\d{2})', 
            r'(\d{4})(\d{2})(\d{2})',
        ]
        
        basename = os.path.basename(filename)
        for pattern in patterns:
            match = re.search(pattern, basename)
            if match:
                year, month, day = match.groups()
                try:
                    return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
        return None
    
    def _load_file_data(self, file_path):
        try:
            ds = gdal.Open(file_path, gdal.GA_ReadOnly)
            if ds is None:
                return None, None
            
            bands = ds.RasterCount
            data = ds.ReadAsArray()
            ds = None
            
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            elif data.ndim == 3 and data.shape[0] != bands:
                if data.shape[2] == bands:
                    data = np.transpose(data, (2, 0, 1))
            
            return data, {'bands': bands}
            
        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {str(e)}")
            return None, None
    
    def analyze_pixel_burn_frequency(self):
        logger.info("开始分析像元燃烧频次...")
        
        firms_files = glob.glob(os.path.join(self.firms_dir, '*.tif'))
        firms_files.sort()
        
        logger.info(f"找到 {len(firms_files)} 个FIRMS文件")
        
        for firms_file in firms_files:
            date = self._extract_date_from_filename(firms_file)
            if date:
                self.all_dates.append(date)
        
        self.all_dates.sort()
        logger.info(f"时间范围: {self.all_dates[0].strftime('%Y-%m-%d')} 至 {self.all_dates[-1].strftime('%Y-%m-%d')}")
        
        pixel_burns = defaultdict(int)
        all_pixels = set()
        
        for firms_file in tqdm(firms_files, desc="分析燃烧频次"):
            data, file_info = self._load_file_data(firms_file)
            if data is None:
                continue
            
            firms_data = data[0] if data.ndim == 3 else data
            
            for row in range(firms_data.shape[0]):
                for col in range(firms_data.shape[1]):
                    value = firms_data[row, col]
                    if not np.isnan(value) and value != 255:
                        all_pixels.add((row, col))
                        if value == 1:
                            pixel_burns[(row, col)] += 1
        
        burn_counts = Counter(pixel_burns.values())
        logger.info("燃烧频次分布:")
        for count in sorted(burn_counts.keys()):
            logger.info(f"  燃烧{count}次: {burn_counts[count]} 个像元")
        
        self.pixel_burn_counts = dict(pixel_burns)
        unburned_pixels = all_pixels - set(pixel_burns.keys())
        logger.info(f"未燃烧像元: {len(unburned_pixels)} 个")
        
        return len(all_pixels), len(pixel_burns), len(unburned_pixels)
    
    def select_representative_pixels(self):
        logger.info("开始选择代表性像元...")
        
        burn_groups = {
            'burn_1_2': [], 'burn_3_4': [], 'burn_5_6': [],
            'burn_6_10': [], 'burn_10_plus': []
        }
        
        for (row, col), count in self.pixel_burn_counts.items():
            if 1 <= count <= 2:
                burn_groups['burn_1_2'].append((row, col))
            elif 3 <= count <= 4:
                burn_groups['burn_3_4'].append((row, col))
            elif 5 <= count <= 6:
                burn_groups['burn_5_6'].append((row, col))
            elif 6 <= count <= 10:
                burn_groups['burn_6_10'].append((row, col))
            elif count > 10:
                burn_groups['burn_10_plus'].append((row, col))
        
        for group_name, pixels in burn_groups.items():
            if len(pixels) >= 2:
                selected = random.sample(pixels, 2)
                self.selected_pixels[group_name] = selected
                logger.info(f"{group_name}: 可选{len(pixels)}个，选中{len(selected)}个")
            else:
                self.selected_pixels[group_name] = pixels
                logger.info(f"{group_name}: 可选{len(pixels)}个，全部选中")
        
        # 选择未燃烧像元
        unburned_pixels = []
        firms_files = glob.glob(os.path.join(self.firms_dir, '*.tif'))
        if firms_files:
            data, _ = self._load_file_data(firms_files[0])
            if data is not None:
                firms_data = data[0] if data.ndim == 3 else data
                for row in range(firms_data.shape[0]):
                    for col in range(firms_data.shape[1]):
                        value = firms_data[row, col]
                        if (not np.isnan(value) and value != 255 and 
                            (row, col) not in self.pixel_burn_counts):
                            unburned_pixels.append((row, col))
        
        if len(unburned_pixels) >= 10:
            self.selected_pixels['no_burn'] = random.sample(unburned_pixels, 10)
        else:
            self.selected_pixels['no_burn'] = unburned_pixels
        
        logger.info(f"未燃烧像元: 可选{len(unburned_pixels)}个，选中{len(self.selected_pixels['no_burn'])}个")
        
        total_selected = sum(len(pixels) for pixels in self.selected_pixels.values())
        logger.info(f"总共选中 {total_selected} 个像元进行时间序列分析")
        
        return total_selected
    
    def extract_pixel_timeseries(self, pixel_pos, driver_name, driver_dir):
        row, col = pixel_pos
        timeseries_data = []
        
        driver_files = glob.glob(os.path.join(driver_dir, '*.tif'))
        
        dated_files = []
        for file_path in driver_files:
            date = self._extract_date_from_filename(file_path)
            if date:
                dated_files.append((date, file_path))
        
        dated_files.sort(key=lambda x: x[0])
        
        for date, file_path in dated_files:
            data, file_info = self._load_file_data(file_path)
            if data is None:
                continue
            
            bands = file_info['bands']
            
            for band_idx in range(bands):
                band_data = data[band_idx] if data.ndim == 3 else data
                
                if row < band_data.shape[0] and col < band_data.shape[1]:
                    value = band_data[row, col]
                    if np.isfinite(value) and value != 255:
                        timeseries_data.append({
                            'date': date,
                            'band': band_idx + 1,
                            'value': value,
                            'feature_name': f'{driver_name}_band_{band_idx + 1}'
                        })
        
        return timeseries_data
    
    def decompose_timeseries(self, timeseries_df, feature_name):
        if len(timeseries_df) < 24:
            return None
        
        try:
            ts_data = timeseries_df.set_index('date')['value'].sort_index()
            ts_data = ts_data.groupby(ts_data.index).mean()
            ts_weekly = ts_data.resample('W').interpolate(method='linear')
            ts_weekly = ts_weekly.dropna()
            
            if len(ts_weekly) < 52:
                return None
            
            try:
                stl = STL(ts_weekly, seasonal=13, period=52)
                result = stl.fit()
                
                trend = result.trend
                seasonal = result.seasonal
                resid = result.resid
                
                x_numeric = np.arange(len(trend))
                theil_sen_slope, theil_sen_intercept, _, _ = stats.theilslopes(trend.values, x_numeric)
                
                trend_mean = np.nanmean(trend)
                
                decomposition_stats = {
                    'feature_name': feature_name,
                    'trend_mean': trend_mean,
                    'theil_sen_slope': theil_sen_slope,
                    'theil_sen_intercept': theil_sen_intercept,
                    'trend_std': np.nanstd(trend),
                    'seasonal_amplitude': np.nanmax(seasonal) - np.nanmin(seasonal),
                    'residual_std': np.nanstd(resid),
                    'data_points': len(ts_weekly),
                    'time_span_days': (ts_weekly.index[-1] - ts_weekly.index[0]).days
                }
                
                return {'stats': decomposition_stats, 'decomposition': result}
                
            except Exception as e:
                logger.warning(f"STL分解失败 {feature_name}: {str(e)}")
                
                try:
                    decomposition = seasonal_decompose(ts_weekly, model='additive', period=52)
                    trend = decomposition.trend.dropna()
                    
                    if len(trend) > 0:
                        x_numeric = np.arange(len(trend))
                        theil_sen_slope, theil_sen_intercept, _, _ = stats.theilslopes(trend.values, x_numeric)
                        
                        return {
                            'stats': {
                                'feature_name': feature_name,
                                'trend_mean': np.nanmean(trend),
                                'theil_sen_slope': theil_sen_slope,
                                'theil_sen_intercept': theil_sen_intercept,
                                'trend_std': np.nanstd(trend),
                                'seasonal_amplitude': np.nan,
                                'residual_std': np.nan,
                                'data_points': len(trend),
                                'time_span_days': (ts_weekly.index[-1] - ts_weekly.index[0]).days
                            },
                            'decomposition': decomposition
                        }
                
                except Exception as e2:
                    logger.warning(f"备选分解也失败 {feature_name}: {str(e2)}")
                    return None
            
        except Exception as e:
            logger.error(f"时间序列分解失败 {feature_name}: {str(e)}")
            return None
    
    def analyze_single_pixel(self, pixel_pos, pixel_type, burn_count=0):
        logger.info(f"分析像元 {pixel_pos} (类型: {pixel_type}, 燃烧次数: {burn_count})")
        
        pixel_results = []
        
        for driver_name, driver_dir in self.dynamic_drivers.items():
            logger.debug(f"  处理驱动因素: {driver_name}")
            
            timeseries_data = self.extract_pixel_timeseries(pixel_pos, driver_name, driver_dir)
            
            if not timeseries_data:
                continue
            
            feature_groups = defaultdict(list)
            for item in timeseries_data:
                feature_groups[item['feature_name']].append(item)
            
            for feature_name, feature_data in feature_groups.items():
                df = pd.DataFrame(feature_data)
                
                if len(df) < 24:
                    continue
                
                decomp_result = self.decompose_timeseries(df, feature_name)
                
                if decomp_result and decomp_result['stats']:
                    stats = decomp_result['stats']
                    
                    result = {
                        'pixel_row': pixel_pos[0],
                        'pixel_col': pixel_pos[1],
                        'pixel_type': pixel_type,
                        'burn_count': burn_count,
                        'driver_name': driver_name,
                        **stats
                    }
                    
                    pixel_results.append(result)
        
        logger.info(f"  像元 {pixel_pos} 完成，生成 {len(pixel_results)} 条记录")
        return pixel_results
    
    def run_timeseries_analysis(self):
        logger.info("开始像元时间序列趋势分解分析")
        
        total_pixels, burned_pixels, unburned_pixels = self.analyze_pixel_burn_frequency()
        logger.info(f"像元统计: 总计{total_pixels}, 燃烧{burned_pixels}, 未燃烧{unburned_pixels}")
        
        selected_count = self.select_representative_pixels()
        if selected_count == 0:
            logger.error("未选中任何像元")
            return False
        
        all_results = []
        total_pixels_to_analyze = sum(len(pixels) for pixels in self.selected_pixels.values())
        
        with tqdm(total=total_pixels_to_analyze, desc="分析像元时间序列") as pbar:
            for pixel_type, pixels in self.selected_pixels.items():
                for pixel_pos in pixels:
                    burn_count = self.pixel_burn_counts.get(pixel_pos, 0)
                    pixel_results = self.analyze_single_pixel(pixel_pos, pixel_type, burn_count)
                    all_results.extend(pixel_results)
                    pbar.update(1)
        
        logger.info(f"完成时间序列分析，共生成 {len(all_results)} 条记录")
        
        if all_results:
            self._save_results(all_results)
            return True
        else:
            logger.error("未生成任何分析结果")
            return False
    
    def _save_results(self, results):
        df = pd.DataFrame(results)
        
        detail_file = os.path.join(self.output_dir, 'pixel_timeseries_trend_analysis.csv')
        df.to_csv(detail_file, index=False, encoding='utf-8-sig')
        logger.info(f"详细结果保存到: {detail_file}")
        
        self._create_summary_analysis(df)
        
        return detail_file
    
    def _create_summary_analysis(self, df):
        summary_file = os.path.join(self.output_dir, 'pixel_timeseries_trend_summary.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("像元时间序列趋势分解分析汇总报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总记录数: {len(df)}\n")
            f.write(f"分析像元数: {df[['pixel_row', 'pixel_col']].drop_duplicates().shape[0]}\n")
            f.write(f"驱动因素数: {df['driver_name'].nunique()}\n\n")
            
            f.write("按像元类型统计:\n")
            f.write("-" * 40 + "\n")
            for pixel_type in df['pixel_type'].unique():
                type_data = df[df['pixel_type'] == pixel_type]
                f.write(f"{pixel_type}:\n")
                f.write(f"  记录数: {len(type_data)}\n")
                f.write(f"  平均趋势均值: {type_data['trend_mean'].mean():.6f}\n")
                f.write(f"  平均Theil-Sen斜率: {type_data['theil_sen_slope'].mean():.6f}\n")
                f.write(f"  斜率标准差: {type_data['theil_sen_slope'].std():.6f}\n\n")
            
            f.write("按驱动因素统计:\n")
            f.write("-" * 40 + "\n")
            for driver in df['driver_name'].unique():
                driver_data = df[df['driver_name'] == driver]
                f.write(f"{driver}:\n")
                f.write(f"  记录数: {len(driver_data)}\n")
                f.write(f"  平均趋势均值: {driver_data['trend_mean'].mean():.6f}\n")
                f.write(f"  平均Theil-Sen斜率: {driver_data['theil_sen_slope'].mean():.6f}\n\n")
        
        logger.info(f"汇总分析保存到: {summary_file}")

def main():
    data_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked"
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials"
    firms_dir = os.path.join(data_dir, "Firms_Detection_resampled")
    
    random.seed(42)
    np.random.seed(42)
    
    analyzer = PixelTimeseriesTrendAnalyzer(
        data_dir=data_dir,
        output_dir=output_dir,
        firms_dir=firms_dir
    )
    
    success = analyzer.run_timeseries_analysis()
    
    if success:
        logger.info("时间序列趋势分解分析完成!")
    else:
        logger.error("分析失败!")

if __name__ == "__main__":
    main() 