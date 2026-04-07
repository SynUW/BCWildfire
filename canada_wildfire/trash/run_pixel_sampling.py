#!/usr/bin/env python3
"""
运行像元驱动因素原始数据抽样器的示例脚本
"""

import os
import sys
from canada_wildfire.analysis_and_plot.pixel_driver_raw_data_sampler import PixelDriverRawDataSampler

def main():
    """运行抽样分析"""
    
    # 配置参数
    data_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials'
    output_dir = './canada_wildfire'
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录不存在: {data_dir}")
        print("请检查路径或根据实际情况修改data_dir参数")
        return
    
    print("=" * 60)
    print("像元驱动因素原始数据抽样器")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"抽样比例: 10%")
    print()
    
    # 创建分析器
    analyzer = PixelDriverRawDataSampler(
        data_dir=data_dir,
        output_dir=output_dir,
        sampling_ratio=0.1,  # 10%抽样
        random_seed=42,
        max_files_per_driver=100  # 限制每个动态驱动因素的文件数
    )
    
    # 运行分析
    try:
        analyzer.run_analysis()
        print("\n分析完成!")
        print(f"结果保存在: {analyzer.output_subdir}")
        print("生成的文件:")
        print("  - raw_driver_samples_10pct.csv: 原始数据样本")
        print("  - sampling_report_10pct.txt: 抽样统计报告")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 