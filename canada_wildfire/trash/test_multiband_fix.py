#!/usr/bin/env python3
"""
测试多波段修复效果
"""

import os
import sys
import numpy as np
from osgeo import gdal

# 添加路径
sys.path.append('/home/zhengsen/wildfire/dataset_building/canada_wildfire')

def test_single_file_loading():
    """测试单个文件的多波段加载"""
    print("测试单个文件的多波段加载:")
    
    # 测试多波段文件
    test_files = [
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked_normalized/ERA5_multi_bands/ERA5_2006_01_12.tif',
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked_normalized/MOD21A1DN_multibands_filtered_resampled/Thermal_2015_11_23.tif',
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked_normalized/Firms_Detection_resampled/FIRMS_2010_07_21.tif'
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
            if dataset:
                bands = dataset.RasterCount
                print(f"  {os.path.basename(file_path)}: {bands} 波段")
                
                if bands == 1:
                    # 单波段
                    band = dataset.GetRasterBand(1)
                    data = band.ReadAsArray()
                    print(f"    单波段数据形状: {data.shape}")
                else:
                    # 多波段
                    data = []
                    for i in range(1, bands + 1):
                        band = dataset.GetRasterBand(i)
                        band_data = band.ReadAsArray()
                        data.append(band_data)
                    data = np.array(data)
                    print(f"    多波段数据形状: {data.shape}")
                    
                    # 测试像素提取
                    row, col = 50, 50  # 测试像素
                    if len(data.shape) == 3:
                        pixel_values = data[:, row, col]
                        print(f"    像素 ({row}, {col}) 值: {pixel_values.shape} = {pixel_values}")
                    else:
                        pixel_value = data[row, col]
                        print(f"    像素 ({row}, {col}) 值: {pixel_value}")
                
                dataset = None
        else:
            print(f"  文件不存在: {os.path.basename(file_path)}")

def simulate_pixel_extraction():
    """模拟像素提取过程"""
    print("\n模拟像素提取过程:")
    
    # 模拟不同波段数的数据
    single_band_data = np.random.rand(100, 100)  # 单波段
    multi_band_data = np.random.rand(4, 100, 100)  # 4波段
    
    row, col = 50, 50
    
    print(f"  单波段数据形状: {single_band_data.shape}")
    if len(single_band_data.shape) == 3:
        pixel_value = single_band_data[:, row, col]
        print(f"    提取的像素值形状: {pixel_value.shape}")
    else:
        pixel_value = single_band_data[row, col]
        print(f"    提取的像素值: {pixel_value} (标量)")
    
    print(f"  多波段数据形状: {multi_band_data.shape}")
    if len(multi_band_data.shape) == 3:
        pixel_values = multi_band_data[:, row, col]
        print(f"    提取的像素值形状: {pixel_values.shape}")
        print(f"    提取的像素值: {pixel_values}")
    else:
        pixel_value = multi_band_data[row, col]
        print(f"    提取的像素值: {pixel_value}")

def test_data_stacking():
    """测试数据堆叠"""
    print("\n测试数据堆叠:")
    
    # 模拟时间序列数据
    time_series_mixed = []
    
    # 添加一些标量值（单波段驱动因素）
    for i in range(5):
        time_series_mixed.append(np.random.rand())
    
    # 添加一些数组值（多波段驱动因素）
    for i in range(5):
        time_series_mixed.append(np.random.rand(4))  # 4波段
    
    print(f"  混合时间序列长度: {len(time_series_mixed)}")
    print(f"  前5个元素类型: {[type(x).__name__ for x in time_series_mixed[:5]]}")
    print(f"  后5个元素形状: {[x.shape if hasattr(x, 'shape') else 'scalar' for x in time_series_mixed[5:]]}")
    
    # 测试转换逻辑
    first_element = time_series_mixed[0]
    if np.isscalar(first_element):
        print("  检测到标量数据，直接转换")
        result = np.array(time_series_mixed[:5], dtype=np.float32)
        print(f"  结果形状: {result.shape}")
    else:
        print("  检测到数组数据，需要堆叠")
        result = np.array(time_series_mixed[5:], dtype=np.float32)
        print(f"  结果形状: {result.shape}")
        if result.ndim == 2:
            result_transposed = result.T
            print(f"  转置后形状: {result_transposed.shape}")

def main():
    """主函数"""
    print("=" * 60)
    print("多波段修复测试")
    print("=" * 60)
    
    test_single_file_loading()
    simulate_pixel_extraction()
    test_data_stacking()
    
    print("\n" + "=" * 60)
    print("测试完成")

if __name__ == "__main__":
    main() 