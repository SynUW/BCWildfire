#!/usr/bin/env python3
"""
数据统计工具
- 统计除FIRMS外所有驱动因素数据的统计信息
- 计算最大值、最小值、99%/1%分位数、98%/2%分位数
- 对多波段图像，精确到每个波段
- 对每个波段的全部数据进行全局异常值过滤（移除最大最小的固定数量值）
- 结果保存为CSV文件
"""

import os
import glob
import numpy as np
import pandas as pd
from osgeo import gdal
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from collections import defaultdict
import gc
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('DataStatistics')

# 设置GDAL配置
gdal.SetConfigOption('GDAL_CACHEMAX', '2048')
gdal.SetConfigOption('GDAL_NUM_THREADS', '4')
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
gdal.SetConfigOption('VSI_CACHE', 'FALSE')

class DataStatisticsCalculator:
    def __init__(self, data_dir, output_csv, max_workers=8, sample_ratio=1.0):
        """
        数据统计计算器
        
        Args:
            data_dir: 数据根目录
            output_csv: 输出CSV文件路径
            max_workers: 最大线程数
            sample_ratio: 采样比例（1.0表示使用全部数据）
        """
        self.data_dir = data_dir
        self.output_csv = output_csv
        self.max_workers = max_workers
        self.sample_ratio = sample_ratio
        
        # 获取所有驱动因素目录（排除FIRMS）
        self.driver_dirs = [
            d for d in os.listdir(data_dir) 
            if (os.path.isdir(os.path.join(data_dir, d)) and 
                'Firms_Detection_resampled' not in d)
        ]
        
        logger.info(f"找到 {len(self.driver_dirs)} 个驱动因素目录: {self.driver_dirs}")
    
    def _apply_global_outlier_filter(self, all_values, remove_count_each_end=1000):
        """
        对全局数据应用异常值过滤，移除最大和最小的固定数量值
        
        Args:
            all_values: 所有像素值的数组
            remove_count_each_end: 每端移除的数量
            
        Returns:
            tuple: (过滤后的数据, 原始统计, 过滤后统计, 异常值信息)
        """
        if len(all_values) < remove_count_each_end * 4:  # 数据太少，不进行过滤
            logger.warning(f"数据量太少({len(all_values)})，跳过异常值过滤")
            original_stats = self._calculate_basic_stats(all_values)
            return all_values, original_stats, original_stats, {
                'method': 'no_filtering_insufficient_data',
                'outliers_removed': 0,
                'outlier_ratio': 0.0
            }
            
        all_values = np.array(all_values)
        data_size = len(all_values)
        
        # 计算原始数据统计
        original_stats = self._calculate_basic_stats(all_values)
        
        # 确保不会移除太多数据（最多移除1%）
        max_remove_each_end = max(100, int(data_size * 0.005))  # 最多移除0.5%
        actual_remove_count = min(remove_count_each_end, max_remove_each_end)
        
        logger.info(f"全局异常值过滤: 数据总量={data_size:,}, 计划移除每端{remove_count_each_end}个值, 实际移除每端{actual_remove_count}个值")
        
        # 对数据进行排序
        sorted_indices = np.argsort(all_values)
        
        # 创建掩膜，移除最小和最大的值
        mask = np.ones(data_size, dtype=bool)
        
        # 移除最小的值
        mask[sorted_indices[:actual_remove_count]] = False
        
        # 移除最大的值
        mask[sorted_indices[-actual_remove_count:]] = False
        
        filtered_data = all_values[mask]
        
        # 记录被移除的极值范围
        removed_min_values = all_values[sorted_indices[:actual_remove_count]]
        removed_max_values = all_values[sorted_indices[-actual_remove_count:]]
        
        # 计算过滤后数据统计
        filtered_stats = self._calculate_basic_stats(filtered_data)
        
        # 异常值信息
        outliers_removed = len(all_values) - len(filtered_data)
        outlier_ratio = outliers_removed / len(all_values) if len(all_values) > 0 else 0
        
        outlier_info = {
            'method': f'global_fixed_count_remove_{actual_remove_count}_each_end',
            'outliers_removed': outliers_removed,
            'outlier_ratio': outlier_ratio,
            'filter_criteria': {
                'remove_count_each_end': actual_remove_count,
                'removed_min_range': [float(np.min(removed_min_values)), float(np.max(removed_min_values))],
                'removed_max_range': [float(np.min(removed_max_values)), float(np.max(removed_max_values))],
                'kept_min': float(np.min(filtered_data)),
                'kept_max': float(np.max(filtered_data))
            }
        }
        
        # 记录过滤信息
        logger.info(f"全局异常值过滤完成: "
                   f"移除最小{actual_remove_count}个值 [{np.min(removed_min_values):.6f} ~ {np.max(removed_min_values):.6f}] "
                   f"和最大{actual_remove_count}个值 [{np.min(removed_max_values):.6f} ~ {np.max(removed_max_values):.6f}], "
                   f"保留范围: [{np.min(filtered_data):.6f} ~ {np.max(filtered_data):.6f}], "
                   f"数据量: {len(all_values):,} -> {len(filtered_data):,}")
        
        return filtered_data, original_stats, filtered_stats, outlier_info
    
    def _calculate_basic_stats(self, data):
        """计算基本统计信息"""
        if len(data) == 0:
            return {
                'min': np.nan, 'max': np.nan, 'mean': np.nan, 
                'median': np.nan, 'std': np.nan, 'count': 0
            }
        
        return {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'count': len(data)
        }
        
    def _load_and_process_file(self, file_path):
        """
        加载并处理单个文件，分别提取每个波段的原始数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            dict: 包含每个波段原始数据的字典
        """
        try:
            # 打开文件
            ds = gdal.Open(file_path)
            if ds is None:
                return None
            
            # 读取所有波段数据
            data = ds.ReadAsArray()
            num_bands = ds.RasterCount
            ds = None
            
            if data is None:
                return None
            
            # 处理无效值
            data = np.where(data == -9999, np.nan, data)
            data = data.astype(np.float64)  # 确保精度
            
            # 初始化结果字典
            file_data = {
                'file_path': file_path,
                'bands': {}
            }
            
            # 处理多波段或单波段数据
            if len(data.shape) == 3:  # 多波段
                for band_idx in range(num_bands):
                    band_data = data[band_idx].flatten()
                    valid_data = band_data[~np.isnan(band_data)]
                    
                    if len(valid_data) > 0:
                        file_data['bands'][f'band_{band_idx + 1}'] = {
                            'values': valid_data.tolist(),
                            'total_pixels': len(band_data),
                            'valid_pixels': len(valid_data)
                        }
            else:  # 单波段
                band_data = data.flatten()
                valid_data = band_data[~np.isnan(band_data)]
                
                if len(valid_data) > 0:
                    file_data['bands']['band_1'] = {
                        'values': valid_data.tolist(),
                        'total_pixels': len(band_data),
                        'valid_pixels': len(valid_data)
                    }
            
            return file_data if file_data['bands'] else None
            
        except Exception as e:
            logger.error(f"处理文件时出错 {file_path}: {e}")
            return None
    
    def _process_driver_folder(self, driver_dir):
        """
        处理单个驱动因素文件夹，收集所有数据并进行全局异常值过滤
        
        Args:
            driver_dir: 驱动因素目录名
            
        Returns:
            list: 每个波段的统计信息列表
        """
        logger.info(f"开始处理驱动因素: {driver_dir}")
        
        # 获取所有tif文件
        folder_path = os.path.join(self.data_dir, driver_dir)
        tif_files = glob.glob(os.path.join(folder_path, '*.tif'))
        
        if not tif_files:
            logger.warning(f"文件夹 {driver_dir} 中没有找到tif文件")
            return []
        
        # 根据采样比例选择文件
        if self.sample_ratio < 1.0:
            sample_size = max(50, int(len(tif_files) * self.sample_ratio))
            tif_files = np.random.choice(tif_files, sample_size, replace=False).tolist()
            logger.info(f"从 {len(glob.glob(os.path.join(folder_path, '*.tif')))} 个文件中采样 {len(tif_files)} 个")
        
        logger.info(f"处理 {len(tif_files)} 个文件，收集所有像素数据...")
        
        # 多线程处理文件，收集所有数据
        all_file_data = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tif_files))) as executor:
            results = list(tqdm(
                executor.map(self._load_and_process_file, tif_files),
                total=len(tif_files),
                desc=f"加载 {driver_dir} 数据",
                ncols=100
            ))
            
            # 收集有效结果
            for result in results:
                if result is not None:
                    all_file_data.append(result)
        
        if not all_file_data:
            logger.warning(f"驱动因素 {driver_dir} 没有有效的数据")
            return []
        
        logger.info(f"成功加载 {len(all_file_data)} 个文件的数据，开始按波段聚合...")
        
        # 按波段聚合所有数据
        band_data_collection = defaultdict(lambda: {
            'all_values': [],
            'total_pixels': 0,
            'valid_pixels': 0,
            'files_count': 0
        })
        
        # 收集每个波段的所有像素值
        for file_data in all_file_data:
            for band_name, band_info in file_data['bands'].items():
                band_key = f"{driver_dir}_{band_name}"
                
                band_data_collection[band_key]['all_values'].extend(band_info['values'])
                band_data_collection[band_key]['total_pixels'] += band_info['total_pixels']
                band_data_collection[band_key]['valid_pixels'] += band_info['valid_pixels']
                band_data_collection[band_key]['files_count'] += 1
        
        # 对每个波段进行全局异常值过滤和统计计算
        band_results = []
        for band_key, band_collection in band_data_collection.items():
            if not band_collection['all_values']:
                continue
            
            logger.info(f"处理波段 {band_key}: 总像素数={len(band_collection['all_values']):,}")
            
            # 应用全局异常值过滤
            filtered_data, original_stats, filtered_stats, outlier_info = self._apply_global_outlier_filter(
                band_collection['all_values'], 
                remove_count_each_end=1000  # 移除最大最小各1000个值
            )
            
            # 计算百分位数
            percentiles = np.percentile(filtered_data, [1, 2, 98, 99])
            
            # 构建最终统计结果
            final_stats = {
                'driver': driver_dir,
                'band': band_key.split('_band_')[1] if '_band_' in band_key else '1',
                # 过滤后的主要统计
                'min_value': filtered_stats['min'],
                'max_value': filtered_stats['max'],
                'mean': filtered_stats['mean'],
                'median': filtered_stats['median'],
                'std': filtered_stats['std'],
                'percentile_1': float(percentiles[0]),
                'percentile_2': float(percentiles[1]),
                'percentile_98': float(percentiles[2]),
                'percentile_99': float(percentiles[3]),
                # 原始数据统计（对比用）
                'original_min': original_stats['min'],
                'original_max': original_stats['max'],
                'original_mean': original_stats['mean'],
                'original_std': original_stats['std'],
                # 像素和文件信息
                'files_processed': band_collection['files_count'],
                'valid_pixels': filtered_stats['count'],
                'original_valid_pixels': original_stats['count'],
                'total_pixels': band_collection['total_pixels'],
                'valid_ratio': band_collection['valid_pixels'] / band_collection['total_pixels'] if band_collection['total_pixels'] > 0 else 0,
                # 异常值信息
                'outliers_removed': outlier_info['outliers_removed'],
                'outlier_ratio': outlier_info['outlier_ratio'],
                'outlier_method': outlier_info['method']
            }
            
            band_results.append(final_stats)
            
            logger.info(f"完成 {band_key} 全局统计: "
                       f"原始范围=[{original_stats['min']:.6f}, {original_stats['max']:.6f}], "
                       f"过滤后范围=[{filtered_stats['min']:.6f}, {filtered_stats['max']:.6f}], "
                       f"移除异常值={outlier_info['outliers_removed']:,}个")
        
        # 清理内存
        del all_file_data, band_data_collection
        gc.collect()
        
        return band_results
    
    def calculate_all_statistics(self):
        """
        计算所有驱动因素的统计信息
        """
        logger.info(f"开始计算 {len(self.driver_dirs)} 个驱动因素的统计信息")
        
        all_results = []
        
        for i, driver_dir in enumerate(self.driver_dirs):
            logger.info(f"处理进度: {i+1}/{len(self.driver_dirs)} - {driver_dir}")
            
            start_time = time.time()
            results = self._process_driver_folder(driver_dir)
            elapsed_time = time.time() - start_time
            
            if results:
                all_results.extend(results)
                logger.info(f"完成 {driver_dir}，获得 {len(results)} 个波段统计，耗时 {elapsed_time:.1f} 秒")
            else:
                logger.warning(f"跳过 {driver_dir}，没有有效数据")
        
        if not all_results:
            logger.error("没有有效的统计结果")
            return
        
        # 创建DataFrame并保存
        df = pd.DataFrame(all_results)
        
        # 重新排列列的顺序
        column_order = [
            'driver', 'band', 
            # 过滤后的主要统计
            'min_value', 'max_value', 'mean', 'median', 'std',
            'percentile_1', 'percentile_2', 'percentile_98', 'percentile_99',
            # 原始数据统计（对比用）
            'original_min', 'original_max', 'original_mean', 'original_std',
            # 像素和文件信息
            'files_processed', 'valid_pixels', 'original_valid_pixels', 'total_pixels', 'valid_ratio',
            # 异常值信息
            'outliers_removed', 'outlier_ratio', 'outlier_method'
        ]
        df = df[column_order]
        
        # 按驱动因素和波段排序
        df = df.sort_values(['driver', 'band'])
        
        # 保存为CSV
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df.to_csv(self.output_csv, index=False, float_format='%.6f')
        
        logger.info(f"统计结果已保存到: {self.output_csv}")
        logger.info(f"共处理 {len(set(df['driver']))} 个驱动因素，{len(df)} 个波段")
        
        # 显示摘要
        print("\n" + "="*100)
        print("数据统计摘要（按波段）")
        print("="*100)
        print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        print("="*100)

def main():
    """主函数"""
    # 配置路径
    data_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked'
    output_csv = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/data_statistics_by_band.csv'
    
    # 创建统计计算器
    calculator = DataStatisticsCalculator(
        data_dir=data_dir,
        output_csv=output_csv,
        max_workers=32,
        sample_ratio=1.0  # 使用全部数据进行全局统计
    )
    
    # 开始计算
    logger.info("开始数据统计分析（全局异常值过滤）...")
    calculator.calculate_all_statistics()
    logger.info("数据统计分析完成！")

if __name__ == "__main__":
    main() 