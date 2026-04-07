#!/usr/bin/env python3
"""
像元驱动因素时间对应抽样器 - 统一运行脚本
支持快速测试和完整分析，与KDE绘图完美衔接
"""

import os
import sys
import warnings
import argparse
from osgeo import gdal

# 抑制警告
warnings.filterwarnings('ignore')
gdal.SetConfigOption('CPL_LOG', '/dev/null')
gdal.PushErrorHandler('CPLQuietErrorHandler')

from canada_wildfire.analysis_and_plot.pixel_driver_temporal_sampler_fixed import PixelDriverTemporalSamplerFixed

def main():
    """统一的时间对应抽样器入口"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='像元驱动因素时间对应抽样器')
    parser.add_argument('--mode', choices=['test', 'full'], default='full',
                       help='运行模式: test=快速测试(1%%), full=完整分析(10%%) (默认: full)')
    parser.add_argument('--sampling-ratio', type=float, default=None,
                       help='自定义抽样比例 (0.01-1.0)')
    parser.add_argument('--data-dir', type=str, 
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked',
                       help='数据目录路径')
    parser.add_argument('--output-dir', type=str, default='./canada_wildfire',
                       help='输出目录路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 确定抽样比例
    if args.sampling_ratio:
        sampling_ratio = args.sampling_ratio
        ratio_desc = f"{sampling_ratio*100}% (自定义)"
    elif args.mode == 'test':
        sampling_ratio = 0.01
        ratio_desc = "1% (快速测试)"
    else:
        sampling_ratio = 0.1
        ratio_desc = "10% (完整分析)"
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"❌ 数据目录不存在: {args.data_dir}")
        print("请检查路径或使用 --data-dir 参数指定正确路径")
        return
    
    # 显示运行信息
    mode_emoji = "🧪" if args.mode == 'test' else "🔥"
    print(f"{mode_emoji} 像元驱动因素时间对应抽样器 - {args.mode.upper()}模式")
    print("=" * 60)
    print(f"📁 数据目录: {args.data_dir}")
    print(f"📂 输出目录: {args.output_dir}")
    print(f"📊 抽样比例: {ratio_desc}")
    print(f"🎲 随机种子: {args.seed}")
    print("=" * 60)
    print()
    
    if args.mode == 'test':
        print("🧪 快速测试模式:")
        print("  • 使用1%抽样，快速验证流程")
        print("  • 验证数据格式和KDE计算")
        print("  • 生成测试级别的CSV文件")
    else:
        print("🔥 完整分析模式:")
        print("  • 使用10%抽样，生产级别分析")
        print("  • 高质量的KDE密度估计")
        print("  • 完整的可视化支持")
    
    print("=" * 60)
    print()
    print("🔧 修正版本特点:")
    print("  • ✅ 修复数据格式问题")
    print("  • ✅ 核密度估计(KDE)替代bins")
    print("  • ✅ 完美衔接KDE绘图工具")
    print("  • ✅ 燃烧前一天时间对应关系")
    print("=" * 60)
    print()
    
    try:
        # 创建分析器
        analyzer = PixelDriverTemporalSamplerFixed(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            sampling_ratio=sampling_ratio,
            random_seed=args.seed
        )
        
        print("✅ 分析器初始化成功")
        print()
        
        # 运行分析
        csv1, csv2, csv3 = analyzer.run_analysis()
        
        print("\n" + "="*60)
        print("🎉 分析完成！生成的文件:")
        print(f"📊 时间分布表格: {csv1}")
        print(f"📈 KDE密度数据: {csv2}")
        print(f"📈 KDE曲线数据: {csv3}")
        print(f"📄 汇总报告: {analyzer.output_subdir}/temporal_sampling_report_{int(sampling_ratio*100)}pct.txt")
        print()
        
        # 根据模式给出不同建议
        if args.mode == 'test':
            print("✅ 快速测试完成！")
            print("💡 如需完整分析，请运行: python run_temporal_sampling.py --mode full")
        else:
            print("✅ 完整分析完成！")
            print("🎨 接下来可以运行KDE绘图: python run_kde_plot.py")
        
        print()
        print("📋 文件说明:")
        print(f"  📊 {os.path.basename(csv1)}: 时间分布数据")
        print(f"  📈 {os.path.basename(csv2)}: KDE密度数据(用于绘图)")
        print(f"  📈 {os.path.basename(csv3)}: KDE曲线数据(连续分布)")
        print()
        print(f"🔧 输出目录: {analyzer.output_subdir}/")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 