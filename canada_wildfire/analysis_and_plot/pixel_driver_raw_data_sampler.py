#!/usr/bin/env python3
"""
像元驱动因素原始数据抽样器
- 基于pixel_driver_distribution.py的架构
- 抽样10%的正负样本
- 保存原始数值到CSV，而非统计量
- 支持静态和动态驱动因素的原始数据提取
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
import random
import argparse
import warnings

# 抑制警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('PixelDriverRawDataSampler')

# 设置GDAL配置
gdal.SetConfigOption('GDAL_CACHEMAX', '2048')
gdal.SetConfigOption('GDAL_NUM_THREADS', '4')
# 抑制GDAL警告和错误信息
gdal.SetConfigOption('CPL_LOG', '/dev/null')  # Linux/Mac
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
# 抑制GDAL错误输出到控制台
gdal.PushErrorHandler('CPLQuietErrorHandler')

class PixelDriverRawDataSampler:
    def __init__(self, data_dir, output_dir, sampling_ratio=0.1, random_seed=42, max_files_per_driver=100):
        """
        初始化像元驱动因素原始数据抽样器
        
        Args:
            data_dir: 驱动因素数据根目录
            output_dir: 输出根目录
            sampling_ratio: 抽样比例（默认10%）
            random_seed: 随机种子
            max_files_per_driver: 每个动态驱动因素处理的最大文件数
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.sampling_ratio = sampling_ratio
        self.random_seed = random_seed
        self.max_files_per_driver = max_files_per_driver
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 创建输出目录
        self.output_subdir = os.path.join(output_dir, "pixel_driver_raw_samples")
        os.makedirs(self.output_subdir, exist_ok=True)
        
        # 获取驱动因素目录
        self.driver_dirs = self._get_driver_directories()
        self.static_drivers, self.dynamic_drivers = self._identify_static_dynamic_drivers()
        
        # 存储像元分类结果
        self.burned_pixels = set()
        self.unburned_pixels = set()
        
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
        
        # 明确定义哪些是静态驱动因素
        known_static_drivers = {
            'Topo_Distance_WGS84_resize_resampled'
        }
        
        for driver_name, driver_dir in self.driver_dirs.items():
            # 跳过FIRMS数据
            if 'FIRMS' in driver_name or 'Firms' in driver_name:
                continue
                
            if driver_name in known_static_drivers:
                static_drivers[driver_name] = driver_dir
                logger.info(f"{driver_name} 识别为静态驱动因素")
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
                data = data[np.newaxis, :, :]
            elif data.ndim == 3 and data.shape[0] != bands:
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
        firms_dir = None
        for driver_name, driver_dir in self.driver_dirs.items():
            if 'FIRMS' in driver_name or 'Firms' in driver_name:
                firms_dir = driver_dir
                logger.info(f"使用FIRMS数据目录: {driver_name}")
                break
        
        if not firms_dir:
            raise ValueError("未找到FIRMS数据目录")
        
        # 获取所有FIRMS文件
        firms_files = glob.glob(os.path.join(firms_dir, '*.tif'))
        firms_files.sort()
        
        logger.info(f"找到 {len(firms_files)} 个FIRMS文件")
        
        # 处理FIRMS文件
        pixel_burn_history = defaultdict(bool)
        all_valid_pixels = set()
        
        for firms_file in tqdm(firms_files, desc="处理FIRMS文件"):
            valid_pixels, burned_pixels = self._process_firms_file(firms_file)
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
        
        logger.info(f"识别完成: 燃烧像元 {len(self.burned_pixels)} 个, 未燃烧像元 {len(self.unburned_pixels)} 个")
        logger.info(f"总有效像元: {len(self.all_pixels)} 个")
        
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
    
    def sample_pixels(self):
        """对正负样本进行抽样"""
        logger.info("开始抽样像元...")
        
        # 计算抽样数量
        burned_sample_size = max(1, int(len(self.burned_pixels) * self.sampling_ratio))
        unburned_sample_size = max(1, int(len(self.unburned_pixels) * self.sampling_ratio))
        
        # 随机抽样
        self.sampled_burned_pixels = set(random.sample(list(self.burned_pixels), burned_sample_size))
        self.sampled_unburned_pixels = set(random.sample(list(self.unburned_pixels), unburned_sample_size))
        
        logger.info(f"抽样完成:")
        logger.info(f"  燃烧像元: {len(self.sampled_burned_pixels)} / {len(self.burned_pixels)} "
                   f"({len(self.sampled_burned_pixels)/len(self.burned_pixels)*100:.1f}%)")
        logger.info(f"  未燃烧像元: {len(self.sampled_unburned_pixels)} / {len(self.unburned_pixels)} "
                   f"({len(self.sampled_unburned_pixels)/len(self.unburned_pixels)*100:.1f}%)")
    
    def extract_raw_data(self):
        """提取原始数据"""
        logger.info("开始提取原始数据...")
        
        all_data = []
        
        # 处理静态驱动因素
        for driver_name, driver_dir in self.static_drivers.items():
            logger.info(f"处理静态驱动因素: {driver_name}")
            data = self._extract_static_driver_data(driver_name, driver_dir)
            if data:
                all_data.extend(data)
        
        # 处理动态驱动因素  
        for driver_name, driver_dir in self.dynamic_drivers.items():
            logger.info(f"处理动态驱动因素: {driver_name}")
            data = self._extract_dynamic_driver_data(driver_name, driver_dir)
            if data:
                all_data.extend(data)
        
        # 保存到CSV
        if all_data:
            df = pd.DataFrame(all_data)
            csv_file = os.path.join(self.output_subdir, f'raw_driver_samples_{self.sampling_ratio*100:.0f}pct.csv')
            df.to_csv(csv_file, index=False)
            logger.info(f"原始数据样本保存到: {csv_file}")
            logger.info(f"总样本数: {len(df)}")
            
            # 生成统计报告
            self._create_summary_report(df)
        else:
            logger.warning("没有提取到任何数据")
    
    def _extract_static_driver_data(self, driver_name, driver_dir):
        """提取静态驱动因素的原始数据"""
        # 获取第一个文件（静态数据所有文件相同）
        driver_files = glob.glob(os.path.join(driver_dir, '*.tif'))
        if not driver_files:
            return []
        
        data, file_info = self._load_file_data(driver_files[0])
        if data is None:
            return []
            
        bands = file_info['bands']
        logger.info(f"{driver_name} 有 {bands} 个波段 (静态)")
        
        extracted_data = []
        
        # 为每个波段提取数据
        for band_idx in range(bands):
            if data.ndim == 3:
                band_data = data[band_idx]
            else:
                band_data = data
                
            # 提取燃烧像元的值
            for row, col in self.sampled_burned_pixels:
                if row < band_data.shape[0] and col < band_data.shape[1]:
                    value = band_data[row, col]
                    if not np.isnan(value) and not np.isinf(value) and value != 0 and value != 255 and value != -9999:
                        extracted_data.append({
                            'driver_name': driver_name,
                            'band_idx': band_idx + 1,
                            'pixel_row': row,
                            'pixel_col': col,
                            'pixel_type': 'burned',
                            'value': float(value),
                            'file_date': 'static',
                            'driver_type': 'static'
                        })
            
            # 提取未燃烧像元的值
            for row, col in self.sampled_unburned_pixels:
                if row < band_data.shape[0] and col < band_data.shape[1]:
                    value = band_data[row, col]
                    if not np.isnan(value) and not np.isinf(value) and value != 0 and value != 255 and value != -9999:
                        extracted_data.append({
                            'driver_name': driver_name,
                            'band_idx': band_idx + 1,
                            'pixel_row': row,
                            'pixel_col': col,
                            'pixel_type': 'unburned',
                            'value': float(value),
                            'file_date': 'static',
                            'driver_type': 'static'
                        })
        
        logger.info(f"{driver_name} 提取了 {len(extracted_data)} 个数据点")
        return extracted_data
    
    def _extract_dynamic_driver_data(self, driver_name, driver_dir):
        """提取动态驱动因素的原始数据"""
        # 获取该驱动因素的所有文件
        driver_files = glob.glob(os.path.join(driver_dir, '*.tif'))
        driver_files.sort()
        
        if not driver_files:
            return []
        
        # 限制文件数量以避免数据过多
        if len(driver_files) > self.max_files_per_driver:
            step = len(driver_files) // self.max_files_per_driver
            driver_files = driver_files[::step]
            logger.info(f"{driver_name} 文件采样: 处理 {len(driver_files)} 个文件")
        
        # 获取波段信息
        sample_data, file_info = self._load_file_data(driver_files[0])
        if sample_data is None:
            return []
        
        bands = file_info['bands']
        logger.info(f"{driver_name} 有 {bands} 个波段，处理 {len(driver_files)} 个文件")
        
        extracted_data = []
        
        # 逐文件处理
        for file_path in tqdm(driver_files, desc=f"处理{driver_name}文件", unit="文件"):
            # 提取文件日期
            file_date = self._extract_date_from_filename(os.path.basename(file_path))
            
            data, file_info = self._load_file_data(file_path)
            if data is None:
                continue
                
            # 确保数据是3D格式
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            
            # 对当前文件的所有波段进行数据提取
            for band_idx in range(bands):
                band_data = data[band_idx]
                
                # 提取燃烧像元的值
                for row, col in self.sampled_burned_pixels:
                    if row < band_data.shape[0] and col < band_data.shape[1]:
                        value = band_data[row, col]
                        if not np.isnan(value) and not np.isinf(value) and value != 0 and value != 255 and value != -9999:
                            extracted_data.append({
                                'driver_name': driver_name,
                                'band_idx': band_idx + 1,
                                'pixel_row': row,
                                'pixel_col': col,
                                'pixel_type': 'burned',
                                'value': float(value),
                                'file_date': file_date,
                                'driver_type': 'dynamic'
                            })
                
                # 提取未燃烧像元的值
                for row, col in self.sampled_unburned_pixels:
                    if row < band_data.shape[0] and col < band_data.shape[1]:
                        value = band_data[row, col]
                        if not np.isnan(value) and not np.isinf(value) and value != 0 and value != 255 and value != -9999:
                            extracted_data.append({
                                'driver_name': driver_name,
                                'band_idx': band_idx + 1,
                                'pixel_row': row,
                                'pixel_col': col,
                                'pixel_type': 'unburned',
                                'value': float(value),
                                'file_date': file_date,
                                'driver_type': 'dynamic'
                            })
            
            # 释放内存
            del data
        
        logger.info(f"{driver_name} 提取了 {len(extracted_data)} 个数据点")
        return extracted_data
    
    def _extract_date_from_filename(self, filename):
        """从文件名提取日期"""
        # 支持多种日期格式
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
        
        return filename  # 如果无法提取日期，返回文件名
    
    def _create_summary_report(self, df):
        """创建数据统计报告"""
        logger.info("生成统计报告...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("像元驱动因素原始数据抽样报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"抽样比例: {self.sampling_ratio*100:.1f}%")
        report_lines.append(f"随机种子: {self.random_seed}")
        report_lines.append("")
        
        # 像元统计
        report_lines.append("像元统计:")
        report_lines.append(f"  总燃烧像元: {len(self.burned_pixels)}")
        report_lines.append(f"  抽样燃烧像元: {len(self.sampled_burned_pixels)}")
        report_lines.append(f"  总未燃烧像元: {len(self.unburned_pixels)}")
        report_lines.append(f"  抽样未燃烧像元: {len(self.sampled_unburned_pixels)}")
        report_lines.append("")
        
        # 数据统计
        report_lines.append("数据统计:")
        report_lines.append(f"  总数据点: {len(df)}")
        
        # 按驱动因素统计
        driver_stats = df.groupby(['driver_name', 'pixel_type']).size().unstack(fill_value=0)
        report_lines.append("")
        report_lines.append("按驱动因素统计:")
        for driver_name in driver_stats.index:
            burned_count = driver_stats.loc[driver_name].get('burned', 0)
            unburned_count = driver_stats.loc[driver_name].get('unburned', 0)
            total_count = burned_count + unburned_count
            report_lines.append(f"  {driver_name}: 总计{total_count} (燃烧{burned_count}, 未燃烧{unburned_count})")
        
        # 按驱动因素类型统计
        type_stats = df.groupby(['driver_type', 'pixel_type']).size().unstack(fill_value=0)
        report_lines.append("")
        report_lines.append("按驱动因素类型统计:")
        for driver_type in type_stats.index:
            burned_count = type_stats.loc[driver_type].get('burned', 0)
            unburned_count = type_stats.loc[driver_type].get('unburned', 0)
            total_count = burned_count + unburned_count
            report_lines.append(f"  {driver_type}: 总计{total_count} (燃烧{burned_count}, 未燃烧{unburned_count})")
        
        # 数值范围统计
        report_lines.append("")
        report_lines.append("数值范围统计:")
        for driver_name in df['driver_name'].unique():
            driver_data = df[df['driver_name'] == driver_name]
            min_val = driver_data['value'].min()
            max_val = driver_data['value'].max()
            mean_val = driver_data['value'].mean()
            report_lines.append(f"  {driver_name}: 范围[{min_val:.3f}, {max_val:.3f}], 均值{mean_val:.3f}")
        
        report_lines.append("=" * 80)
        
        # 保存报告
        report_file = os.path.join(self.output_subdir, f'sampling_report_{self.sampling_ratio*100:.0f}pct.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"统计报告保存到: {report_file}")
        
        # 同时输出到控制台
        for line in report_lines:
            print(line)
    
    def run_analysis(self):
        """运行完整分析"""
        logger.info("开始运行像元驱动因素原始数据抽样分析...")
        
        start_time = time.time()
        
        # 1. 识别像元燃烧状态
        self.identify_pixel_burn_status()
        
        # 2. 抽样像元
        self.sample_pixels()
        
        # 3. 提取原始数据
        self.extract_raw_data()
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"分析完成，总耗时: {total_time:.2f} 秒")

def main():
    parser = argparse.ArgumentParser(description='像元驱动因素原始数据抽样器')
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
    parser.add_argument('--max_files_per_driver', 
                       type=int, 
                       default=10240,
                       help='每个动态驱动因素处理的最大文件数')
    
    args = parser.parse_args()
    
    logger.info("启动像元驱动因素原始数据抽样器")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"抽样比例: {args.sampling_ratio*100}%")
    
    # 创建分析器并运行
    analyzer = PixelDriverRawDataSampler(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sampling_ratio=args.sampling_ratio,
        random_seed=args.random_seed,
        max_files_per_driver=args.max_files_per_driver
    )
    
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 