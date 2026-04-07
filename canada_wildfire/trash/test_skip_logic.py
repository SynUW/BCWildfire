#!/usr/bin/env python3
"""
测试pixel_sampling.py的跳过逻辑
"""

import os
import sys
sys.path.append('.')

from canada_wildfire.previous_dataset_generation.pixel_sampling import PixelSampler

def test_skip_logic():
    """测试跳过逻辑"""
    # 配置路径
    data_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked_normalized'
    output_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples'
    
    # 创建采样器
    sampler = PixelSampler(
        data_dir=data_dir,
        output_dir=output_dir,
        past_days=365,
        future_days=30,
        negative_ratio=4,
    )
    
    # 测试2001年的文件检查
    print("检查2001年样本文件是否存在:")
    sample_files_exist = sampler._check_sample_files_exist(2001)
    print(f"样本文件存在: {sample_files_exist}")
    
    print("\n检查2001年完整文件是否存在:")
    full_files_exist = sampler._check_full_files_exist(2001)
    print(f"完整文件存在: {full_files_exist}")
    
    # 列出驱动因素目录
    print(f"\n驱动因素目录: {list(sampler.driver_dirs.keys())}")
    
    # 检查每个驱动因素的文件
    print("\n检查每个驱动因素的样本文件:")
    for driver_name in sampler.driver_dirs.keys():
        h5_path = os.path.join(output_dir, f'2001_{driver_name}.h5')
        exists = os.path.exists(h5_path)
        print(f"  {driver_name}: {exists} ({h5_path})")
    
    print("\n检查每个驱动因素的完整文件:")
    for driver_name in sampler.driver_dirs.keys():
        h5_path = os.path.join(output_dir, f'2001_{driver_name}_full.h5')
        exists = os.path.exists(h5_path)
        print(f"  {driver_name}: {exists} ({h5_path})")

if __name__ == "__main__":
    test_skip_logic() 