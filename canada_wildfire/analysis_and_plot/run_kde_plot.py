#!/usr/bin/env python3
"""
运行KDE绘图器
基于改进版本的KDE密度CSV文件绘制核密度估计图
"""

import os
import sys
import warnings

# 抑制警告
warnings.filterwarnings('ignore')

from canada_wildfire.analysis_and_plot.plot_kde_from_csv import KDEPlotter

def main():
    """运行KDE绘图"""
    
    # 自动查找KDE密度文件
    possible_dirs = [
        'canada_wildfire/pixel_driver_temporal_samples_fixed',
        'canada_wildfire/canada_wildfire/pixel_driver_temporal_samples_fixed'
    ]
    
    possible_files = []
    for base_dir in possible_dirs:
        possible_files.extend([
            f'{base_dir}/kde_density_10pct.csv',  # 完整分析
            f'{base_dir}/kde_density_1pct.csv',   # 快速测试
        ])
    
    csv_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            csv_file = file_path
            break
    
    # 检查CSV文件是否存在
    if not csv_file:
        print(f"❌ KDE密度CSV文件不存在，查找路径:")
        for path in possible_files:
            print(f"  • {path}")
        print()
        print("请先运行时间对应抽样器生成KDE密度数据:")
        print("  快速测试: python run_temporal_sampling.py --mode test")
        print("  完整分析: python run_temporal_sampling.py --mode full")
        return
    
    output_dir = os.path.join(os.path.dirname(csv_file), 'kde_plots')
    
    print("📊 核密度估计图绘制器")
    print("=" * 60)
    print(f"📄 KDE密度文件: {csv_file}")
    print(f"📂 输出目录: {output_dir}")
    print("=" * 60)
    print()
    print("🎨 绘制内容:")
    print("  • 重要特征的组合KDE图")
    print("  • 每个特征的详细KDE图")
    print("  • 燃烧 vs 未燃烧像元的密度对比")
    print("  • 连续的核密度估计曲线")
    print("=" * 60)
    print()
    
    try:
        # 创建绘图器
        plotter = KDEPlotter(csv_file, output_dir)
        
        print("✅ 绘图器初始化成功")
        print()
        
        # 绘制重要特征的组合KDE图
        print("🎨 正在绘制重要特征的组合KDE图...")
        plotter.plot_top_features(top_n=6, save_plot=True)
        
        # 获取可用特征并绘制单个特征的详细图
        print("🎨 正在绘制每个特征的详细KDE图...")
        
        # 绘制与组合图相同的指定特征
        selected_features = [
            ('ERA5_multi_bands_band_1', 'Temperature 2m'),
            ('MODIS_Terra_Aqua_B20_21_merged_resampled_band_4', 'Band 21 Night'),
            ('ERA5_multi_bands_band_3', 'V Wind 10m'),
            ('ERA5_multi_bands_band_5', 'Total Precipitation'),
            ('ERA5_multi_bands_band_12', 'Volumetric Soil Water L4'),
            ('LAI_BCBoundingbox_resampled', 'LAI'),
        ]
        
        for feature_name, display_name in selected_features:
            plotter.plot_single_feature(feature_name, display_name, save_plot=True)
        
        print("=" * 60)
        print("✅ 所有KDE图绘制完成!")
        print(f"📂 结果保存在: {output_dir}")
        print()
        print("生成的文件:")
        print("  📊 kde_top_6_features.png - 重要特征组合KDE图")
        print("  📁 kde_*.png - 每个特征的详细KDE图")
        print("  📄 各特征的KDE密度分布可视化结果")
        print()
        print("📋 KDE图说明:")
        print("  • 蓝色曲线: 燃烧像元的密度分布")
        print("  • 红色曲线: 未燃烧像元的密度分布")
        print("  • x轴: 特征值")
        print("  • y轴: 密度值")
        print("  • 曲线下方填充显示分布区域")
        
    except Exception as e:
        print(f"❌ 绘图过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 