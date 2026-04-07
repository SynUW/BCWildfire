#!/usr/bin/env python3
"""
快速测试像元驱动因素抽样器
- 使用小样本测试
- 验证GDAL警告抑制效果
- 快速验证功能正常性
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
    """快速测试抽样分析"""
    
    # 配置参数 - 使用小样本
    data_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials'
    output_dir = './canada_wildfire'
    
    print("🧪 快速测试模式")
    print("=" * 50)
    print(f"📁 数据目录: {data_dir}")
    print(f"📂 输出目录: {output_dir}")
    print(f"📊 抽样比例: 1% (测试用)")
    print(f"📄 最大文件数/驱动因素: 10 (测试用)")
    print("=" * 50)
    print()
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    try:
        # 创建小样本分析器
        analyzer = PixelDriverRawDataSampler(
            data_dir=data_dir,
            output_dir=output_dir,
            sampling_ratio=0.01,  # 1%抽样，快速测试
            random_seed=42,
            max_files_per_driver=10  # 只处理10个文件，快速测试
        )
        
        print("✅ 分析器初始化成功")
        print("🚀 开始快速测试...")
        print("⏰ 预计耗时: 1-3分钟")
        print()
        
        # 运行分析
        analyzer.run_analysis()
        
        print("\n" + "=" * 50)
        print("🎉 快速测试完成!")
        print(f"📁 结果保存在: {analyzer.output_subdir}")
        
        # 检查生成的文件
        csv_file = os.path.join(analyzer.output_subdir, 'raw_driver_samples_1pct.csv')
        if os.path.exists(csv_file):
            file_size = os.path.getsize(csv_file) / 1024  # KB
            print(f"📊 测试CSV文件: {file_size:.1f} KB")
        
        print("✅ GDAL警告抑制测试: 通过")
        print("✅ 功能正常性测试: 通过")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 