#!/usr/bin/env python3
"""
从完整数据集h5文件中提取指定位置的像素样本数据
将每个位置的时间序列数据保存为CSV文件
"""

import os
import h5py
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PixelSampleExtractor:
    def __init__(self, h5_file_path, output_dir):
        """
        初始化像素样本提取器
        
        Args:
            h5_file_path: h5文件路径
            output_dir: CSV输出目录
        """
        self.h5_file_path = h5_file_path
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 目标像素位置 (column, row) - 基于实际h5文件中存在的坐标
        self.target_pixels = [
            # (5, 176),
            # (13, 268), 
            # (61, 224),
            # (62, 161)
            (80, 186)
        ]
        
        logger.info(f"H5文件: {h5_file_path}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"目标像素位置: {self.target_pixels}")
    
    def _parse_channel_mapping(self, h5_file):
        """解析通道映射信息"""
        channel_mapping = {}
        
        for attr_name in h5_file.attrs.keys():
            if attr_name.startswith('channel_mapping_'):
                driver_name = attr_name.replace('channel_mapping_', '')
                mapping_str = h5_file.attrs[attr_name]
                
                # 解析映射字符串 "start_idx-end_idx"
                if isinstance(mapping_str, bytes):
                    mapping_str = mapping_str.decode('utf-8')
                
                start_idx, end_idx = map(int, mapping_str.split('-'))
                
                channel_mapping[driver_name] = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'channels': end_idx - start_idx + 1
                }
        
        return channel_mapping
    
    def _create_column_names(self, channel_mapping):
        """创建CSV列名"""
        columns = []
        
        # 按驱动因素顺序创建列名
        driver_order = [
            'Firms_Detection_resampled',
            'ERA5_multi_bands', 
            'LULC_BCBoundingbox_resampled',
            'Topo_Distance_WGS84_resize_resampled',
            'NDVI_EVI',
            'Reflection_500_merge_TerraAquaWGS84_clip',
            'MODIS_Terra_Aqua_B20_21_merged_resampled',
            'MOD21A1DN_multibands_filtered_resampled',
            'LAI_BCBoundingbox_resampled'
        ]
        
        for driver_name in driver_order:
            if driver_name in channel_mapping:
                mapping = channel_mapping[driver_name]
                channels = mapping['channels']
                
                if channels == 1:
                    # 单通道，直接使用驱动因素名
                    columns.append(driver_name)
                else:
                    # 多通道，加上波段编号
                    for ch in range(channels):
                        columns.append(f"{driver_name}_band{ch+1}")
        
        return columns
    
    def _extract_pixel_data(self, h5_file, pixel_coord, channel_mapping):
        """提取单个像素的数据"""
        try:
            # 构建数据集名称
            dataset_name = f"{pixel_coord[0]}_{pixel_coord[1]}"
            
            if dataset_name not in h5_file:
                logger.warning(f"像素位置 {pixel_coord} 在h5文件中不存在")
                logger.info(f"可用的数据集: {list(h5_file.keys())[:5]}...")  # 显示前5个可用数据集
                return None
            
            # 读取数据 (channels, time_steps)
            pixel_data = h5_file[dataset_name][:]
            logger.info(f"像素 {pixel_coord} 原始数据形状: {pixel_data.shape}")
            
            # 转置为 (time_steps, channels) 以便保存为CSV
            pixel_data = pixel_data.T
            logger.info(f"像素 {pixel_coord} 转置后形状: {pixel_data.shape}")
            
            # 创建列名
            column_names = self._create_column_names(channel_mapping)
            
            # 确保列名数量与数据通道数匹配
            if len(column_names) != pixel_data.shape[1]:
                logger.warning(f"列名数量({len(column_names)})与数据通道数({pixel_data.shape[1]})不匹配")
                # 使用简单的列名
                column_names = [f"channel_{i+1}" for i in range(pixel_data.shape[1])]
            
            # 创建DataFrame，不使用日期索引
            df = pd.DataFrame(
                data=pixel_data,
                columns=column_names
            )
            
            return df
            
        except Exception as e:
            logger.error(f"提取像素 {pixel_coord} 数据失败: {e}")
            return None
    
    def _analyze_data_statistics(self, df, pixel_coord):
        """分析数据统计信息"""
        logger.info(f"\n像素 {pixel_coord} 数据统计:")
        logger.info(f"  数据形状: {df.shape}")
        logger.info(f"  时间步数: {df.shape[0]}")
        logger.info(f"  通道数: {df.shape[1]}")
        
        # 计算NaN统计
        total_values = df.size
        nan_count = df.isna().sum().sum()
        nan_percentage = (nan_count / total_values) * 100
        
        logger.info(f"  总数值: {total_values:,}")
        logger.info(f"  NaN数量: {nan_count:,} ({nan_percentage:.2f}%)")
        
        # 每个驱动因素的NaN比例
        nan_by_column = df.isna().sum()
        if nan_by_column.sum() > 0:
            logger.info(f"  各驱动因素NaN比例:")
            for col, nan_count in nan_by_column.items():
                if nan_count > 0:
                    nan_pct = (nan_count / len(df)) * 100
                    logger.info(f"    {col}: {nan_count} ({nan_pct:.1f}%)")
    
    def extract_samples(self):
        """提取样本数据"""
        if not os.path.exists(self.h5_file_path):
            logger.error(f"H5文件不存在: {self.h5_file_path}")
            return
        
        # 从文件名提取年份
        filename = os.path.basename(self.h5_file_path)
        year = filename.split('_')[0] if '_' in filename else 'unknown'
        
        try:
            with h5py.File(self.h5_file_path, 'r') as h5_file:
                logger.info(f"成功打开h5文件: {self.h5_file_path}")
                
                # 显示h5文件基本信息
                logger.info(f"h5文件中的数据集数量: {len(h5_file.keys())}")
                
                # 解析通道映射
                channel_mapping = self._parse_channel_mapping(h5_file)
                logger.info(f"找到 {len(channel_mapping)} 个驱动因素")
                
                for driver_name, mapping in channel_mapping.items():
                    logger.info(f"  {driver_name}: {mapping['channels']} 个通道")
                
                # 提取每个目标像素的数据
                success_count = 0
                for pixel_coord in tqdm(self.target_pixels, desc="提取像素数据"):
                    logger.info(f"提取像素 {pixel_coord} 的数据...")
                    
                    # 提取数据
                    df = self._extract_pixel_data(h5_file, pixel_coord, channel_mapping)
                    
                    if df is not None:
                        # 保存CSV文件
                        output_filename = f"pixel_{pixel_coord[0]}_{pixel_coord[1]}_year_{year}.csv"
                        output_path = os.path.join(self.output_dir, output_filename)
                        
                        df.to_csv(output_path, index=False)  # index=False 不保存行索引
                        logger.info(f"保存成功: {output_path}")
                        
                        # 分析数据统计
                        self._analyze_data_statistics(df, pixel_coord)
                        success_count += 1
                    else:
                        logger.warning(f"像素 {pixel_coord} 数据提取失败")
                
                logger.info(f"\n提取完成！成功提取 {success_count}/{len(self.target_pixels)} 个像素的数据")
                        
        except Exception as e:
            logger.error(f"处理文件 {self.h5_file_path} 失败: {e}")
    
    def create_summary_report(self):
        """创建汇总报告"""
        import glob
        
        csv_files = glob.glob(os.path.join(self.output_dir, "*.csv"))
        
        if not csv_files:
            logger.warning("没有找到CSV文件，无法创建汇总报告")
            return
        
        summary_data = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                filename = os.path.basename(csv_file)
                
                # 解析文件名获取像素坐标和年份
                parts = filename.replace('.csv', '').split('_')
                pixel_col = int(parts[1])
                pixel_row = int(parts[2])
                year = parts[4]
                
                # 计算统计信息
                total_values = df.size
                nan_count = df.isna().sum().sum()
                nan_percentage = (nan_count / total_values) * 100
                
                summary_data.append({
                    '像素坐标': f"({pixel_col}, {pixel_row})",
                    '年份': year,
                    '时间步数': df.shape[0],
                    '通道数': df.shape[1],
                    '总数值': total_values,
                    'NaN数量': nan_count,
                    'NaN比例(%)': round(nan_percentage, 2),
                    '文件名': filename
                })
                
            except Exception as e:
                logger.error(f"处理文件 {csv_file} 时出错: {e}")
        
        # 保存汇总报告
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.output_dir, "extraction_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"\n汇总报告保存至: {summary_file}")
        print("\n提取结果汇总:")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='从完整数据集h5文件中提取像素样本')
    # --h5_file /mnt/raid/zhengsen/wildfire_dataset/self_built_materials/full_datasets/2024_full_dataset.h5 
    parser.add_argument('--h5_file',
                       required=True,
                       help='h5文件路径')
    parser.add_argument('--output_dir',
                       default='./pixel_samples_output',
                       help='CSV文件输出目录')
    
    args = parser.parse_args()
    
    # 创建提取器
    extractor = PixelSampleExtractor(
        h5_file_path=args.h5_file,
        output_dir=args.output_dir
    )
    
    # 提取样本
    extractor.extract_samples()
    
    # 创建汇总报告
    extractor.create_summary_report()

if __name__ == "__main__":
    main() 