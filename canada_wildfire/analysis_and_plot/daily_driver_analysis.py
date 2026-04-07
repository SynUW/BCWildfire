#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
野火点燃像素前10天逐日驱动因素分布分析器


功能：
- 统计所有燃烧像元在燃烧前1-10天每日的驱动因素分布
- 同时统计相同数量的对照未燃烧像元在相同日期的驱动因素分布
- 输出逐日、逐驱动因素的详细统计结果（均值、中值、最大值、最小值、标准差）
"""

import os
import glob
import numpy as np
import pandas as pd
from osgeo import gdal
import logging
from datetime import datetime, timedelta
import gc
import re
from tqdm import tqdm
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('DailyDriverAnalysis')

class DailyDriverAnalyzer:
    def __init__(self, data_dir, output_dir, firms_dir, days_before=10):
        """
        野火点燃像素前N天逐日驱动因素分布分析器
        
        Args:
            data_dir: 驱动因素数据根目录
            output_dir: 输出根目录
            firms_dir: FIRMS检测数据目录  
            days_before: 分析燃烧前多少天（默认10天）
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.firms_dir = firms_dir
        self.days_before = days_before
        
        # 创建专门的子文件夹用于存放daily_driver_analysis的结果
        self.output_subdir = os.path.join(output_dir, "daily_driver_analysis_results")
        os.makedirs(self.output_subdir, exist_ok=True)
        
        # 获取驱动因素目录
        self.driver_dirs = self._get_driver_directories()
        self.dynamic_drivers = self._identify_dynamic_drivers()
        
        # 存储燃烧事件和像元信息
        self.burn_events = []
        self.all_valid_pixels = set()
        
        logger.info(f"找到 {len(self.dynamic_drivers)} 个动态驱动因素")
        logger.info(f"输出目录: {self.output_subdir}")
    
    def _get_driver_directories(self):
        """获取所有驱动因素目录"""
        driver_dirs = {}
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                tif_files = glob.glob(os.path.join(item_path, '*.tif'))
                if tif_files:
                    driver_dirs[item] = item_path
        return driver_dirs
    
    def _identify_dynamic_drivers(self):
        """识别动态驱动因素"""
        dynamic_drivers = {}
        static_drivers = {'Topo_Distance_WGS84_resize_resampled'}
        
        for driver_name, driver_dir in self.driver_dirs.items():
            if driver_name not in static_drivers and driver_name != 'Firms_Detection_resampled':
                dynamic_drivers[driver_name] = driver_dir
        
        return dynamic_drivers
    
    def _extract_date_from_filename(self, filename):
        """从文件名提取日期"""
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
        """加载文件数据"""
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
    
    def identify_burn_events(self):
        """识别燃烧事件和有效像元"""
        logger.info("开始识别燃烧事件...")
        
        firms_files = glob.glob(os.path.join(self.firms_dir, '*.tif'))
        firms_files.sort()
        
        logger.info(f"找到 {len(firms_files)} 个FIRMS文件")
        
        for firms_file in tqdm(firms_files, desc="处理FIRMS文件"):
            date = self._extract_date_from_filename(firms_file)
            if date is None:
                continue
            
            data, file_info = self._load_file_data(firms_file)
            if data is None:
                continue
            
            firms_data = data[0] if data.ndim == 3 else data
            
            burned_pixels = []
            valid_pixels = []
            
            for row in range(firms_data.shape[0]):
                for col in range(firms_data.shape[1]):
                    value = firms_data[row, col]
                    if not np.isnan(value) and value != 255 and value != 0 and value != -9999:
                        valid_pixels.append((row, col))
                        if value == 1:
                            burned_pixels.append((row, col))
            
            if burned_pixels:
                self.burn_events.append({
                    'date': date,
                    'burned_pixels': burned_pixels,
                    'filename': os.path.basename(firms_file),
                    'total_burned': len(burned_pixels)
                })
                self.all_valid_pixels.update(valid_pixels)
        
        self.burn_events.sort(key=lambda x: x['date'])
        
        logger.info(f"识别到 {len(self.burn_events)} 个燃烧事件")
        logger.info(f"总有效像元数: {len(self.all_valid_pixels)}")
        
        return len(self.burn_events)
    
    def _select_control_pixels(self, burned_pixels, date):
        """选择对照未燃烧像元"""
        # 获取该日期及之后燃烧过的像元
        burned_before_or_on_date = set()
        for event in self.burn_events:
            if event['date'] <= date + timedelta(days=30):
                burned_before_or_on_date.update(event['burned_pixels'])
        
        available_control = list(self.all_valid_pixels - burned_before_or_on_date)
        
        if len(available_control) >= len(burned_pixels):
            return random.sample(available_control, len(burned_pixels))
        else:
            logger.warning(f"对照像元数量不足: 需要{len(burned_pixels)}, 可用{len(available_control)}")
            return available_control
    
    def _calculate_daily_statistics(self, values, prefix):
        """计算日统计量"""
        if not values:
            return {f'{prefix}_count': 0, f'{prefix}_mean': np.nan, f'{prefix}_median': np.nan,
                   f'{prefix}_std': np.nan, f'{prefix}_min': np.nan, f'{prefix}_max': np.nan,
                   f'{prefix}_q25': np.nan, f'{prefix}_q75': np.nan}
        
        values_array = np.array(values, dtype=np.float64)
        values_array = values_array[np.isfinite(values_array) & (values_array != 255) & (values_array != 0) & (values_array != -9999)]
        
        if len(values_array) == 0:
            return {f'{prefix}_count': 0, f'{prefix}_mean': np.nan, f'{prefix}_median': np.nan,
                   f'{prefix}_std': np.nan, f'{prefix}_min': np.nan, f'{prefix}_max': np.nan,
                   f'{prefix}_q25': np.nan, f'{prefix}_q75': np.nan}
        
        return {
            f'{prefix}_count': len(values_array),
            f'{prefix}_mean': np.nanmean(values_array),
            f'{prefix}_median': np.nanmedian(values_array),
            f'{prefix}_std': np.nanstd(values_array),
            f'{prefix}_min': np.nanmin(values_array),
            f'{prefix}_max': np.nanmax(values_array),
            f'{prefix}_q25': np.nanpercentile(values_array, 25),
            f'{prefix}_q75': np.nanpercentile(values_array, 75),
        }
    
    def analyze_daily_distributions(self):
        """分析燃烧前每日的驱动因素分布"""
        logger.info(f"开始分析燃烧前{self.days_before}天的逐日驱动因素分布...")
        
        all_results = []
        
        for event_idx, event in enumerate(tqdm(self.burn_events, desc="处理燃烧事件")):
            burn_date = event['date']
            burned_pixels = event['burned_pixels']
            control_pixels = self._select_control_pixels(burned_pixels, burn_date)
            
            for day_before in range(1, self.days_before + 1):
                target_date = burn_date - timedelta(days=day_before)
                
                for driver_name, driver_dir in self.dynamic_drivers.items():
                    driver_files = glob.glob(os.path.join(driver_dir, '*.tif'))
                    target_file = None
                    
                    for file_path in driver_files:
                        file_date = self._extract_date_from_filename(file_path)
                        if file_date and file_date.date() == target_date.date():
                            target_file = file_path
                            break
                    
                    if target_file is None:
                        continue
                    
                    data, file_info = self._load_file_data(target_file)
                    if data is None:
                        continue
                    
                    bands = file_info['bands']
                    
                    for band_idx in range(bands):
                        band_data = data[band_idx] if data.ndim == 3 else data
                        
                        # 提取燃烧像元的值
                        burned_values = []
                        for row, col in burned_pixels:
                            if row < band_data.shape[0] and col < band_data.shape[1]:
                                value = band_data[row, col]
                                if np.isfinite(value) and value != 255 and value != 0 and value != -9999:
                                    burned_values.append(value)
                        
                        # 提取对照像元的值
                        control_values = []
                        for row, col in control_pixels:
                            if row < band_data.shape[0] and col < band_data.shape[1]:
                                value = band_data[row, col]
                                if np.isfinite(value) and value != 255 and value != 0 and value != -9999:
                                    control_values.append(value)
                        
                        if burned_values and control_values:
                            result = {
                                'event_id': event_idx + 1,
                                'burn_date': burn_date.strftime('%Y-%m-%d'),
                                'target_date': target_date.strftime('%Y-%m-%d'),
                                'days_before': day_before,
                                'driver_name': driver_name,
                                'band_index': band_idx + 1,
                                'feature_name': f'{driver_name}_band_{band_idx + 1}',
                            }
                            
                            # 计算统计量
                            burned_stats = self._calculate_daily_statistics(burned_values, 'burned')
                            control_stats = self._calculate_daily_statistics(control_values, 'control')
                            
                            result.update(burned_stats)
                            result.update(control_stats)
                            
                            # 差异统计
                            if not np.isnan(burned_stats['burned_mean']) and not np.isnan(control_stats['control_mean']):
                                result['mean_difference'] = burned_stats['burned_mean'] - control_stats['control_mean']
                                result['median_difference'] = burned_stats['burned_median'] - control_stats['control_median']
                                result['std_ratio'] = burned_stats['burned_std'] / max(control_stats['control_std'], 1e-10)
                            else:
                                result['mean_difference'] = np.nan
                                result['median_difference'] = np.nan
                                result['std_ratio'] = np.nan
                            
                            all_results.append(result)
        
        logger.info(f"完成逐日分析，共生成 {len(all_results)} 条记录")
        return all_results
    
    def _save_results(self, results):
        """保存分析结果"""
        df = pd.DataFrame(results)
        
        # 保存详细结果
        detail_file = os.path.join(self.output_subdir, f'daily_driver_analysis_{self.days_before}days.csv')
        df.to_csv(detail_file, index=False, encoding='utf-8-sig')
        logger.info(f"详细结果保存到: {detail_file}")
        
        # 创建汇总分析
        self._create_summary_analysis(df)
        
        return detail_file
    
    def _create_summary_analysis(self, df):
        """创建汇总分析"""
        summary_file = os.path.join(self.output_subdir, f'daily_driver_summary_{self.days_before}days.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"野火点燃像素前{self.days_before}天逐日驱动因素分布分析汇总\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析期间: 燃烧前1-{self.days_before}天\n")
            f.write(f"燃烧事件数量: {df['event_id'].nunique()}\n")
            f.write(f"分析的驱动因素: {df['driver_name'].nunique()} 个\n")
            f.write(f"总记录数: {len(df)}\n\n")
            
            # 按天数统计
            f.write("按燃烧前天数统计:\n")
            f.write("-" * 40 + "\n")
            for day in range(1, self.days_before + 1):
                day_data = df[df['days_before'] == day]
                if len(day_data) > 0:
                    avg_mean_diff = day_data['mean_difference'].mean()
                    f.write(f"燃烧前第{day:2d}天: {len(day_data):4d} 条记录, "
                           f"平均差异: {avg_mean_diff:8.4f}\n")
            
            f.write("\n按驱动因素统计:\n")
            f.write("-" * 40 + "\n")
            for driver in df['driver_name'].unique():
                driver_data = df[df['driver_name'] == driver]
                avg_mean_diff = driver_data['mean_difference'].mean()
                f.write(f"{driver:<35}: {len(driver_data):4d} 条记录, "
                       f"平均差异: {avg_mean_diff:8.4f}\n")
        
        logger.info(f"汇总分析保存到: {summary_file}")
        
        # 创建包含所有统计信息的透视表
        self._create_comprehensive_pivot_tables(df)
    
    def _create_comprehensive_pivot_tables(self, df):
        """创建包含所有统计信息的透视表"""
        logger.info("创建包含燃烧像元和对照像元所有统计信息的透视表...")
        
        # 定义要创建透视表的统计量（燃烧像元的统计）
        burned_stats = [
            'burned_count', 'burned_mean', 'burned_median', 'burned_std',
            'burned_min', 'burned_max', 'burned_q25', 'burned_q75'
        ]
        
        # 定义要创建透视表的统计量（对照像元的统计）
        control_stats = [
            'control_count', 'control_mean', 'control_median', 'control_std',
            'control_min', 'control_max', 'control_q25', 'control_q75'
        ]
        
        all_stats = burned_stats + control_stats
        
        # 创建综合透视表，包含所有统计信息
        pivot_data = []
        
        for feature in df['feature_name'].unique():
            feature_data = df[df['feature_name'] == feature]
            
            for stat in all_stats:
                if stat in feature_data.columns:
                    # 对每个统计量按days_before创建透视
                    stat_pivot = feature_data.pivot_table(
                        values=stat, 
                        index=['feature_name'], 
                        columns='days_before', 
                        aggfunc='mean'
                    )
                    
                    # 重塑数据格式
                    for days_before in range(1, self.days_before + 1):
                        if days_before in stat_pivot.columns:
                            value = stat_pivot.loc[feature, days_before] if feature in stat_pivot.index else np.nan
                            pivot_data.append({
                                'feature_name': feature,
                                'statistic': stat,
                                'days_before': days_before,
                                'value': value
                            })
        
        # 转换为DataFrame并创建最终透视表
        pivot_df = pd.DataFrame(pivot_data)
        
        # 创建最终的透视表格式：特征名_统计量 vs 天数
        final_pivot = pivot_df.pivot_table(
            values='value',
            index=['feature_name', 'statistic'],
            columns='days_before',
            aggfunc='first'
        )
        
        # 重新索引，创建更清晰的行名
        final_pivot.index = [f"{feature}_{stat}" for feature, stat in final_pivot.index]
        
        # 保存综合透视表
        pivot_file = os.path.join(self.output_subdir, f'daily_driver_pivot_{self.days_before}days.csv')
        final_pivot.to_csv(pivot_file, encoding='utf-8-sig')
        logger.info(f"综合透视表保存到: {pivot_file}")
    

    def run_analysis(self):
        """运行完整的逐日分析"""
        logger.info("开始野火点燃像素前10天逐日驱动因素分布分析")
        
        # 识别燃烧事件
        num_events = self.identify_burn_events()
        if num_events == 0:
            logger.error("未找到任何燃烧事件")
            return False
        
        # 进行逐日分析
        results = self.analyze_daily_distributions()
        if not results:
            logger.error("未生成任何分析结果")
            return False
        
        # 保存结果
        self._save_results(results)
        
        logger.info("逐日驱动因素分布分析完成!")
        return True

def main():
    """主函数"""
    data_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked"
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials"
    firms_dir = os.path.join(data_dir, "Firms_Detection_resampled")
    
    analyzer = DailyDriverAnalyzer(
        data_dir=data_dir,
        output_dir=output_dir,
        firms_dir=firms_dir,
        days_before=10
    )
    
    success = analyzer.run_analysis()
    
    if success:
        logger.info("分析成功完成!")
    else:
        logger.error("分析失败!")

if __name__ == "__main__":
    main() 