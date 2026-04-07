#!/usr/bin/env python3
"""
运行像元驱动因素原始数据抽样器的优化脚本
- 抑制GDAL警告信息
- 更好的错误处理
- 清洁的输出显示
"""

import os
import sys
import warnings
from osgeo import gdal

# 抑制所有警告
warnings.filterwarnings('ignore')

# 抑制GDAL错误和警告输出
gdal.SetConfigOption('CPL_LOG', '/dev/null')
gdal.PushErrorHandler('CPLQuietErrorHandler')

from canada_wildfire.analysis_and_plot.pixel_driver_raw_data_sampler import PixelDriverRawDataSampler

def main():
    """运行抽样分析"""
    
    # 配置参数
    data_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked_normalized'
    output_dir = './canada_wildfire'
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请检查路径或根据实际情况修改data_dir参数")
        return
    
    print("🔥 像元驱动因素原始数据抽样器")
    print("=" * 60)
    print(f"📁 数据目录: {data_dir}")
    print(f"📂 输出目录: {output_dir}")
    print(f"📊 抽样比例: 10%")
    print(f"🎲 随机种子: 42")
    print(f"📄 最大文件数/驱动因素: 100")
    print("=" * 60)
    print()
    
    # 创建分析器
    try:
        analyzer = PixelDriverRawDataSampler(
            data_dir=data_dir,
            output_dir=output_dir,
            sampling_ratio=0.001,  # 10%抽样
            random_seed=42,
            max_files_per_driver=10240  # 限制每个动态驱动因素的文件数
        )
        
        print("✅ 分析器初始化成功")
        print()
        
    except Exception as e:
        print(f"❌ 分析器初始化失败: {str(e)}")
        return
    
    # 运行分析
    try:
        print("🚀 开始运行分析...")
        analyzer.run_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 分析完成!")
        print(f"📁 结果保存在: {analyzer.output_subdir}")
        print("\n📋 生成的文件:")
        print("   📊 raw_driver_samples_10pct.csv - 原始数据样本")
        print("   📄 sampling_report_10pct.txt - 抽样统计报告")
        print("=" * 60)
        
        # 检查生成的文件
        csv_file = os.path.join(analyzer.output_subdir, 'raw_driver_samples_10pct.csv')
        report_file = os.path.join(analyzer.output_subdir, 'sampling_report_10pct.txt')
        
        if os.path.exists(csv_file):
            file_size = os.path.getsize(csv_file) / (1024*1024)  # MB
            print(f"📊 CSV文件大小: {file_size:.1f} MB")
        
        if os.path.exists(report_file):
            print(f"📄 报告文件已生成")
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断了分析过程")
        print("已生成的部分结果可能保存在输出目录中")
        
    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {str(e)}")
        print("\n🔧 故障排除建议:")
        print("1. 检查数据目录是否包含正确的驱动因素文件")
        print("2. 确保有足够的磁盘空间存储结果")
        print("3. 检查FIRMS数据目录是否存在")
        print("4. 尝试减少抽样比例或最大文件数")

if __name__ == "__main__":
    main() 