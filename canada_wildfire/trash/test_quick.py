#!/usr/bin/env python3
from canada_wildfire.previous_dataset_generation.pixel_sampling import PixelSampler
import os

# 只测试一个驱动因素
data_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked_normalized"
output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples"

sampler = PixelSampler(
    data_dir=data_dir,
    output_dir=output_dir,
    past_days=365,
    future_days=30,
    negative_ratio=4.0,
    generate_full_data=True,
    sample_years=[],
    full_years=[2020],
    use_parallel=False,
    n_processes=1
)

# 只保留FIRMS驱动因素
original_dirs = sampler.driver_dirs
sampler.driver_dirs = {"Firms_Detection_resampled": original_dirs["Firms_Detection_resampled"]}

print("开始测试优化版本...")
sampler.process_all_years()
print("测试完成！")
