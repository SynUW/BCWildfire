#!/usr/bin/env python3
"""
像元驱动因素分布分析器
- 统计整个时间序列范围内所有点燃过和未点燃过像元的图像坐标
- 基于位置信息，统计所有点燃过和未点燃过驱动因素的分布
- 多波段图像的每个波段都作为独立特征进行统计
- 支持完整时间序列分析（所有文件）或采样分析（限制文件数量）
- 生成CSV表格保存统计结果

分析模式：
- analyze_full_timeseries=True: 分析所有文件，获得完整统计
- analyze_full_timeseries=False: 采样分析，节约内存和时间
"""

import os
import glob
import numpy as np
import pandas as pd
from osgeo import gdal
import logging
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import gc
import time
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('PixelDriverDistribution')

# 设置GDAL配置
gdal.SetConfigOption('GDAL_CACHEMAX', '2048')
gdal.SetConfigOption('GDAL_NUM_THREADS', '4')

class PixelDriverDistributionAnalyzer:
    def __init__(self, data_dir, output_dir, analyze_full_timeseries=True):
        """
        像元驱动因素分布分析器
        
        Args:
            data_dir: 驱动因素数据根目录
            output_dir: 输出目录
            analyze_full_timeseries: 是否分析完整时间序列（True）或采样分析（False）
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.analyze_full_timeseries = analyze_full_timeseries
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有驱动因素目录
        self.driver_dirs = self._get_driver_directories()
        logger.info(f"找到 {len(self.driver_dirs)} 个驱动因素目录: {list(self.driver_dirs.keys())}")
        
        # 存储像元状态信息
        self.burned_pixels = set()      # 点燃过的像元坐标
        self.unburned_pixels = set()    # 从未点燃的像元坐标
        self.all_pixels = set()         # 所有有效像元坐标
        
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
            
            # 获取地理参考信息
            geotransform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            
            ds = None
            
            # 确保数据是3D格式 (bands, height, width)
            if data.ndim == 2:
                data = data[np.newaxis, :, :]  # 添加波段维度
            elif data.ndim == 3 and data.shape[0] != bands:
                # 可能需要转置
                if data.shape[2] == bands:
                    data = np.transpose(data, (2, 0, 1))
            
            return data, {'bands': bands, 'height': height, 'width': width, 
                         'geotransform': geotransform, 'projection': projection}
            
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
        
        # 用于跟踪每个像元的燃烧历史
        pixel_burn_history = defaultdict(bool)  # True表示该像元曾经燃烧过
        all_valid_pixels = set()
        
        # 遍历所有FIRMS文件
        for firms_file in tqdm(firms_files, desc="处理FIRMS文件"):
            data, file_info = self._load_file_data(firms_file)
            if data is None:
                continue
            
            # FIRMS数据应该是单波段
            if data.ndim == 3:
                firms_data = data[0]  # 取第一个波段
            else:
                firms_data = data
            
            height, width = firms_data.shape
            
            # 遍历所有像元
            for row in range(height):
                for col in range(width):
                    pixel_value = firms_data[row, col]
                    
                    # 跳过NoData值（255）
                    if pixel_value == 255:
                        continue
                    
                    pixel_coord = (row, col)
                    all_valid_pixels.add(pixel_coord)
                    
                    # 如果像元值大于0，表示燃烧
                    if pixel_value > 0:
                        pixel_burn_history[pixel_coord] = True
        
        # 分类像元
        self.all_pixels = all_valid_pixels
        
        for pixel_coord in all_valid_pixels:
            if pixel_burn_history[pixel_coord]:
                self.burned_pixels.add(pixel_coord)
            else:
                self.unburned_pixels.add(pixel_coord)
        
        logger.info(f"总有效像元数: {len(self.all_pixels)}")
        logger.info(f"点燃过的像元数: {len(self.burned_pixels)}")
        logger.info(f"从未点燃的像元数: {len(self.unburned_pixels)}")
        
        # 保存像元分类结果
        self._save_pixel_classification()
    
    def _save_pixel_classification(self):
        """保存像元分类结果"""
        pixel_data = []
        
        # 添加点燃过的像元
        for row, col in self.burned_pixels:
            pixel_data.append({
                'row': row,
                'col': col,
                'burn_status': 'burned'
            })
        
        # 添加未点燃的像元
        for row, col in self.unburned_pixels:
            pixel_data.append({
                'row': row,
                'col': col,
                'burn_status': 'unburned'
            })
        
        df = pd.DataFrame(pixel_data)
        output_path = os.path.join(self.output_dir, 'pixel_burn_classification.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"像元分类结果已保存到: {output_path}")
    
    def analyze_driver_distributions(self):
        """分析驱动因素分布"""
        logger.info("开始分析驱动因素分布...")
        
        # 检查系统内存（如果使用完整时间序列分析）
        if self.analyze_full_timeseries:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            logger.info(f"系统总内存: {total_memory_gb:.1f} GB, 可用内存: {available_memory_gb:.1f} GB")
            if available_memory_gb < 8:
                logger.warning("可用内存较少，建议设置 analyze_full_timeseries=False 使用采样模式")
        
        # 存储统计结果
        distribution_stats = []
        
        # 分析每个驱动因素
        for driver_name, driver_dir in self.driver_dirs.items():
            logger.info(f"分析驱动因素: {driver_name}")
            
            # 获取该驱动因素的一个文件来了解波段信息
            sample_files = glob.glob(os.path.join(driver_dir, '*.tif'))
            if not sample_files:
                continue
            
            sample_data, file_info = self._load_file_data(sample_files[0])
            if sample_data is None:
                continue
            
            bands = file_info['bands']
            logger.info(f"{driver_name} 有 {bands} 个波段")
            
            # 为每个波段收集数据
            for band_idx in range(bands):
                logger.info(f"处理 {driver_name} 第 {band_idx+1}/{bands} 波段")
                
                burned_values = []
                unburned_values = []
                
                # 收集该驱动因素的所有文件
                driver_files = glob.glob(os.path.join(driver_dir, '*.tif'))
                driver_files.sort()
                
                total_files = len(driver_files)
                
                if self.analyze_full_timeseries:
                    logger.info(f"分析完整时间序列：{total_files} 个文件")
                else:
                    # 采样模式：限制文件数量以避免内存问题
                    max_files = 50
                    if len(driver_files) > max_files:
                        step = max(1, len(driver_files) // max_files)
                        driver_files = driver_files[::step]
                        logger.info(f"采样模式：采样了 {len(driver_files)} 个文件进行分析（原{total_files}个）")
                
                # 处理每个文件
                processed_files = 0
                for file_path in tqdm(driver_files, desc=f"处理{driver_name}波段{band_idx+1}"):
                    data, _ = self._load_file_data(file_path)
                    if data is None:
                        continue
                    
                    # 获取当前波段数据
                    if data.ndim == 3:
                        band_data = data[band_idx]
                    else:
                        band_data = data  # 单波段情况
                    
                    # 收集点燃过像元的值
                    for row, col in self.burned_pixels:
                        if row < band_data.shape[0] and col < band_data.shape[1]:
                            value = band_data[row, col]
                            # 跳过无效值
                            if not np.isnan(value) and not np.isinf(value):
                                burned_values.append(value)
                    
                    # 收集未点燃像元的值
                    # 为了平衡数据量，我们采样未燃烧像元，但保证有足够的样本
                    if self.analyze_full_timeseries:
                        # 完整分析模式：使用所有未燃烧像元，但限制每个文件的采样数量
                        unburned_sample_size = min(len(self.burned_pixels) * 3, len(self.unburned_pixels))
                    else:
                        # 采样模式：平衡采样
                        unburned_sample_size = min(len(burned_values) * 2, len(self.unburned_pixels))
                    
                    unburned_coords_sample = list(self.unburned_pixels)[:unburned_sample_size]
                    
                    for row, col in unburned_coords_sample:
                        if row < band_data.shape[0] and col < band_data.shape[1]:
                            value = band_data[row, col]
                            # 跳过无效值
                            if not np.isnan(value) and not np.isinf(value):
                                unburned_values.append(value)
                    
                    processed_files += 1
                    
                    # 完整分析模式下，每处理100个文件输出一次进度
                    if self.analyze_full_timeseries and processed_files % 100 == 0:
                        logger.info(f"已处理 {processed_files}/{len(driver_files)} 个文件，"
                                  f"燃烧像元样本数: {len(burned_values)}, 未燃烧像元样本数: {len(unburned_values)}")
                
                logger.info(f"完成{driver_name}波段{band_idx+1}处理：共处理 {processed_files} 个文件，"
                          f"燃烧像元样本数: {len(burned_values)}, 未燃烧像元样本数: {len(unburned_values)}")
                
                # 计算统计量
                if burned_values and unburned_values:
                    burned_array = np.array(burned_values)
                    unburned_array = np.array(unburned_values)
                    
                    stats = {
                        'driver_name': driver_name,
                        'band_index': band_idx + 1,
                        'feature_name': f"{driver_name}_band_{band_idx+1}",
                        
                        # 点燃过像元的统计
                        'burned_count': len(burned_array),
                        'burned_mean': np.mean(burned_array),
                        'burned_std': np.std(burned_array),
                        'burned_min': np.min(burned_array),
                        'burned_max': np.max(burned_array),
                        'burned_median': np.median(burned_array),
                        'burned_q25': np.percentile(burned_array, 25),
                        'burned_q75': np.percentile(burned_array, 75),
                        
                        # 未点燃像元的统计
                        'unburned_count': len(unburned_array),
                        'unburned_mean': np.mean(unburned_array),
                        'unburned_std': np.std(unburned_array),
                        'unburned_min': np.min(unburned_array),
                        'unburned_max': np.max(unburned_array),
                        'unburned_median': np.median(unburned_array),
                        'unburned_q25': np.percentile(unburned_array, 25),
                        'unburned_q75': np.percentile(unburned_array, 75),
                        
                        # 差异统计
                        'mean_difference': np.mean(burned_array) - np.mean(unburned_array),
                        'std_difference': np.std(burned_array) - np.std(unburned_array),
                    }
                    
                    distribution_stats.append(stats)
                    logger.info(f"{stats['feature_name']}: 点燃像元{stats['burned_count']}个, "
                              f"未点燃像元{stats['unburned_count']}个")
                
                # 清理内存
                del burned_values, unburned_values
                if hasattr(data, '__del__'):
                    del data
                gc.collect()
                
                # 完整分析模式下，每个波段处理完后强制垃圾回收
                if self.analyze_full_timeseries:
                    import psutil
                    memory_percent = psutil.virtual_memory().percent
                    logger.info(f"当前内存使用率: {memory_percent:.1f}%")
                    if memory_percent > 80:
                        logger.warning("内存使用率较高，建议考虑采样分析模式")
        
        # 保存结果
        if distribution_stats:
            df = pd.DataFrame(distribution_stats)
            output_path = os.path.join(self.output_dir, 'driver_distribution_analysis.csv')
            df.to_csv(output_path, index=False)
            logger.info(f"驱动因素分布分析结果已保存到: {output_path}")
            
            # 创建汇总报告
            self._create_summary_report(df)
        else:
            logger.warning("没有生成任何统计结果")
    
    def _create_summary_report(self, df):
        """创建汇总报告"""
        report_path = os.path.join(self.output_dir, 'distribution_analysis_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("驱动因素分布分析汇总报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据目录: {self.data_dir}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            
            f.write("像元统计:\n")
            f.write(f"- 总有效像元数: {len(self.all_pixels):,}\n")
            f.write(f"- 点燃过的像元数: {len(self.burned_pixels):,}\n")
            f.write(f"- 从未点燃的像元数: {len(self.unburned_pixels):,}\n")
            f.write(f"- 燃烧像元比例: {len(self.burned_pixels)/len(self.all_pixels)*100:.2f}%\n\n")
            
            f.write("驱动因素统计:\n")
            for driver_name in df['driver_name'].unique():
                driver_df = df[df['driver_name'] == driver_name]
                f.write(f"- {driver_name}: {len(driver_df)} 个波段/特征\n")
            
            f.write(f"\n总特征数: {len(df)}\n\n")
            
            f.write("显著差异特征（按均值差异排序）:\n")
            top_features = df.nlargest(10, 'mean_difference')[['feature_name', 'mean_difference', 'burned_mean', 'unburned_mean']]
            for _, row in top_features.iterrows():
                f.write(f"- {row['feature_name']}: 差异={row['mean_difference']:.4f} "
                       f"(燃烧={row['burned_mean']:.4f}, 未燃烧={row['unburned_mean']:.4f})\n")
        
        logger.info(f"汇总报告已保存到: {report_path}")
    
    def run_analysis(self):
        """运行完整分析"""
        logger.info("开始像元驱动因素分布分析")
        start_time = time.time()
        
        # 步骤1: 识别像元燃烧状态
        self.identify_pixel_burn_status()
        
        # 步骤2: 分析驱动因素分布
        self.analyze_driver_distributions()
        
        end_time = time.time()
        logger.info(f"分析完成，总耗时: {end_time - start_time:.2f} 秒")


def main():
    """主函数"""
    
    # ================== 配置参数 ==================
    data_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked_normalized'
    output_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_driver_distribution_analysis'
    analyze_full_timeseries = True  # 是否分析完整时间序列（True）或采样分析（False）
    
    logger.info("=" * 60)
    logger.info("像元驱动因素分布分析配置:")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"分析模式: {'完整时间序列' if analyze_full_timeseries else '采样分析'}")
    logger.info("=" * 60)
    
    # 创建分析器
    analyzer = PixelDriverDistributionAnalyzer(
        data_dir=data_dir,
        output_dir=output_dir,
        analyze_full_timeseries=analyze_full_timeseries
    )
    
    # 运行分析
    analyzer.run_analysis()
    logger.info("像素驱动因素分布分析完成！")


if __name__ == "__main__":
    main() 