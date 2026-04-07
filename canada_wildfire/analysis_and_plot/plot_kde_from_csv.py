#!/usr/bin/env python3
"""
基于KDE CSV数据绘制核密度估计图
直接使用temporal_sampler_v2生成的kde_density文件进行可视化
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
from scipy.stats import gaussian_kde # Added missing import

warnings.filterwarnings('ignore')

# 设置字体 - 完全匹配gaussian_plot.py
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

class KDEPlotter:
    def __init__(self, csv_file, output_dir=None):
        """
        初始化KDE绘图器
        
        Args:
            csv_file: KDE密度CSV文件路径
            output_dir: 输出目录，如果未指定则使用默认目录
        """
        self.csv_file = csv_file
        
        # 如果未指定输出目录，使用默认目录
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(csv_file), 'kde_plots')
        else:
            self.output_dir = output_dir
            
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 读取数据
        self.load_data()
        
        print(f"📊 KDE绘图器初始化完成")
        print(f"📄 数据文件: {self.csv_file}")
        print(f"📂 输出目录: {self.output_dir}")
        print(f"📋 数据摘要:")
    
    def load_data(self):
        """
        加载KDE密度CSV数据
        """
        self.df = pd.read_csv(self.csv_file)
        print(f"✅ 加载KDE数据: {len(self.df)} 个数据点")
        print(f"📊 特征数量: {self.df['feature_name'].nunique()}")
        print(f"📈 像元类型: {self.df['pixel_type'].unique()}")
    
    def plot_top_features(self, top_n=6, save_plot=True):
        """
        绘制指定的6个特征的KDE密度图
        
        Args:
            top_n: 保留此参数用于向后兼容，但将忽略并绘制指定的6个特征
            save_plot: 是否保存图片
        """
        # 指定要绘制的特征及其显示名称
        selected_features = [
            ('ERA5_multi_bands_band_1', 'Temperature 2m'),
            ('MODIS_Terra_Aqua_B20_21_merged_resampled_band_4', 'Band 21 Night'),
            ('ERA5_multi_bands_band_3', 'V Wind 10m'),
            ('ERA5_multi_bands_band_5', 'Total Precipitation'),
            ('ERA5_multi_bands_band_12', 'Soil Water L4'),
            ('LAI_BCBoundingbox_resampled', 'LAI'),
        ]
        
        print(f"📊 绘制指定的6个特征:")
        for i, (feature_name, display_name) in enumerate(selected_features, 1):
            print(f"  {i}. {display_name} ({feature_name})")
        
        # 一行6列布局 - 每个子图1.8x2.0英寸
        fig, axes = plt.subplots(1, 6, figsize=(10.8, 2.0))
        
        for idx, (feature_name, display_name) in enumerate(selected_features):
            ax = axes[idx]
            
            # 检查特征是否存在于数据中
            feature_data = self.df[self.df['feature_name'] == feature_name]
            if feature_data.empty:
                print(f"警告: 未找到特征 {feature_name}")
                ax.text(0.5, 0.5, f'{display_name}\n(No Data)', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # 按像元类型分组数据
            burned_data = feature_data[feature_data['pixel_type'] == 'burned']
            unburned_data = feature_data[feature_data['pixel_type'] == 'unburned']
            
            # 获取数据范围用于x轴
            all_values = np.concatenate([burned_data['pixel_value'].values, 
                                       unburned_data['pixel_value'].values])
            
            # 对Band 20 Day和Band 20 Night限定数值范围
            if display_name in ['Band 20 Day', 'Band 20 Night']:
                x_min, x_max = 225, 310
                print(f"  限定{display_name}数值范围: {x_min}-{x_max}")
            else:
                x_min, x_max = np.min(all_values), np.max(all_values)
            
            x_range = np.linspace(x_min, x_max, 200)
            
            # 颜色设置 - 完全匹配gaussian_plot.py
            colors = {'burned': 'red', 'unburned': 'blue'}
            labels = {'burned': 'Burned', 'unburned': 'Not Burned'}
            
            # 绘制KDE密度曲线
            for pixel_type, type_data in [('burned', burned_data), ('unburned', unburned_data)]:
                if not type_data.empty:
                    # 使用pixel_value重新计算KDE，获得平滑曲线
                    try:
                        pixel_values = type_data['pixel_value'].values
                        kde = gaussian_kde(pixel_values)
                        kde_values = kde(x_range)
                        
                        # 绘制KDE曲线 - 匹配gaussian_plot.py样式
                        ax.plot(x_range, kde_values, 
                               color=colors[pixel_type],
                               label=labels[pixel_type],
                               linewidth=1.5)
                        
                        # 填充区域 - 匹配gaussian_plot.py样式
                        ax.fill_between(x_range, kde_values, alpha=0.3, 
                                       color=colors[pixel_type])
                    except Exception as e:
                        print(f"警告: {pixel_type} 数据KDE计算失败: {e}")
                        # 备用方案：绘制直方图
                        ax.hist(type_data['pixel_value'], bins=50, alpha=0.3, 
                               density=True, color=colors[pixel_type], 
                               label=labels[pixel_type])
            
            # 设置样式 - 按用户要求修改
            ax.set_xlabel('')  # 无x轴标签
            ax.set_ylabel('')  # 无y轴标签
            
            # 四周用axis围上，但上和左右的axis不显示值
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)
            
            # 设置刻度 - 只显示底部刻度值，并加粗
            ax.tick_params(axis='x', which='major', labelsize=10, top=False, bottom=True, labeltop=False, labelbottom=True)
            ax.tick_params(axis='y', which='major', labelsize=10, left=False, right=False, labelleft=False, labelright=False)
            ax.set_yticks([])  # 无y轴刻度
            
            # 设置x轴刻度标签字体加粗
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            
            # 设置x轴刻度格式，小数保留两位
            from matplotlib.ticker import FuncFormatter
            def format_x_tick(x, pos):
                if isinstance(x, (int, float)):
                    if x == int(x):  # 如果是整数
                        return f'{int(x)}'
                    else:  # 如果是小数
                        return f'{x:.2f}'
                return str(x)
            
            ax.xaxis.set_major_formatter(FuncFormatter(format_x_tick))
            
            # 设置x轴范围
            ax.set_xlim(x_min, x_max)
            
            # 只在第一个子图显示图例 - 放在左上角
            if idx == 5:
                legend = ax.legend(fontsize=9.5, loc='upper right')
                legend.get_frame().set_facecolor('none')  # 无背景
                legend.get_frame().set_edgecolor('none')  # 无边框
                # 设置图例文字加粗
                for text in legend.get_texts():
                    text.set_fontweight('bold')
            
            ax.grid(True, alpha=0.3)
            
            # 标题在下方 - 匹配gaussian_plot.py
            ax.set_title(display_name, fontsize=9.3, fontweight='bold', pad=-20)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.2, wspace=0.05)  # 去掉左右留白，强制子图大小
        
        # 保存图片
        if save_plot:
            filename = 'kde_selected_features_combined.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"📊 保存组合KDE图: {filepath}")
        
        return plt

    def plot_single_feature(self, feature_name, display_name=None, save_plot=True):
        """
        绘制单个特征的详细KDE图
        
        Args:
            feature_name: 特征名称
            display_name: 显示名称，如果为None则使用feature_name
            save_plot: 是否保存图片
        """
        if display_name is None:
            display_name = feature_name
            
        feature_data = self.df[self.df['feature_name'] == feature_name]
        if feature_data.empty:
            print(f"❌ 未找到特征: {feature_name}")
            return None
        
        # 创建图形 - 匹配gaussian_plot.py的单个图形大小
        plt.figure(figsize=(1.8, 2.0))
        
        # 颜色设置 - 完全匹配gaussian_plot.py
        colors = {'burned': 'red', 'unburned': 'blue'}
        labels = {'burned': 'Burned', 'unburned': 'Not Burned'}
        
        # 按像元类型分组数据
        burned_data = feature_data[feature_data['pixel_type'] == 'burned']
        unburned_data = feature_data[feature_data['pixel_type'] == 'unburned']
        
        # 获取数据范围用于x轴
        all_values = np.concatenate([burned_data['pixel_value'].values, 
                                   unburned_data['pixel_value'].values])
        x_min, x_max = np.min(all_values), np.max(all_values)
        x_range = np.linspace(x_min, x_max, 200)
        
        for pixel_type, type_data in [('burned', burned_data), ('unburned', unburned_data)]:
            if not type_data.empty:
                # 使用pixel_value重新计算KDE，获得平滑曲线
                try:
                    pixel_values = type_data['pixel_value'].values
                    kde = gaussian_kde(pixel_values)
                    kde_values = kde(x_range)
                    
                    # 绘制KDE曲线 - 匹配gaussian_plot.py样式
                    plt.plot(x_range, kde_values, 
                            color=colors[pixel_type],
                            label=labels[pixel_type],
                            linewidth=1.5)
                    
                    # 填充区域 - 匹配gaussian_plot.py样式
                    plt.fill_between(x_range, kde_values, alpha=0.3, 
                                   color=colors[pixel_type])
                except Exception as e:
                    print(f"警告: {pixel_type} 数据KDE计算失败: {e}")
                    # 备用方案：绘制直方图
                    plt.hist(type_data['pixel_value'], bins=50, alpha=0.3, 
                           density=True, color=colors[pixel_type], 
                           label=labels[pixel_type])
        
        # 设置图形属性 - 完全匹配gaussian_plot.py
        plt.xlabel('')  # 无x轴标签
        plt.ylabel('')  # 无y轴标签
        plt.gca().set_yticks([])  # 无y轴刻度
        
        # 图例设置 - 匹配gaussian_plot.py
        legend = plt.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.035, 1))
        legend.get_frame().set_facecolor('none')  # 无背景
        legend.get_frame().set_edgecolor('none')  # 无边框
        # 设置图例文字加粗
        for text in legend.get_texts():
            text.set_fontweight('bold')
        
        plt.grid(True, alpha=0.3)
        plt.gca().tick_params(axis='both', which='major', labelsize=10)  # 刻度字体大小7
        
        # 设置x轴刻度标签字体加粗
        for label in plt.gca().get_xticklabels():
            label.set_fontweight('bold')
        
        # 设置x轴刻度格式，小数保留两位
        from matplotlib.ticker import FuncFormatter
        def format_x_tick_single(x, pos):
            if isinstance(x, (int, float)):
                if x == int(x):  # 如果是整数
                    return f'{int(x)}'
                else:  # 如果是小数
                    return f'{x:.2f}'
            return str(x)
        
        plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_tick_single))
        
        # 标题在下方 - 匹配gaussian_plot.py
        plt.title(display_name, fontsize=9, fontweight='bold', pad=-20)
        
        # 调整布局 - 匹配gaussian_plot.py
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # 保存图片
        if save_plot:
            # 清理文件名
            safe_name = display_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            filename = f'kde_{safe_name}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"📊 保存KDE图: {filepath}")
        
        return plt
    
    def get_available_features(self):
        """获取可用的特征列表"""
        return sorted(self.df['feature_name'].unique())
    
    def print_summary(self):
        """打印数据摘要"""
        print("\n" + "="*60)
        print("📊 KDE数据摘要")
        print("="*60)
        print(f"总数据点: {len(self.df)}")
        print(f"特征数量: {self.df['feature_name'].nunique()}")
        print(f"像元类型: {list(self.df['pixel_type'].unique())}")
        
        print("\n📈 各特征数据点统计:")
        feature_stats = self.df.groupby(['feature_name', 'pixel_type']).size().unstack(fill_value=0)
        for feature in sorted(self.df['feature_name'].unique()):
            burned = feature_stats.loc[feature, 'burned'] if 'burned' in feature_stats.columns else 0
            unburned = feature_stats.loc[feature, 'unburned'] if 'unburned' in feature_stats.columns else 0
            total = burned + unburned
            print(f"  {feature}: {total} (燃烧{burned}, 未燃烧{unburned})")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='基于KDE CSV数据绘制核密度估计图')
    parser.add_argument('--kde_csv', 
                       default='./canada_wildfire/pixel_driver_temporal_samples_fixed/kde_density_10pct.csv',
                       help='KDE CSV文件路径')
    parser.add_argument('--output_dir', 
                       help='输出目录（默认与CSV文件同目录）')
    parser.add_argument('--top_n', 
                       type=int, 
                       default=6,
                       help='绘制前N个数据最丰富的特征（默认6）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.kde_csv):
        print(f"❌ KDE CSV文件不存在: {args.kde_csv}")
        print("请先运行 python test_temporal_sampling_v2.py 或 python run_temporal_sampling_v2.py")
        return
    
    print("🎨 KDE核密度估计图绘制器")
    print("="*60)
    print(f"📊 KDE数据文件: {args.kde_csv}")
    print(f"📂 输出目录: {args.output_dir or os.path.dirname(args.kde_csv)}")
    print("="*60)
    
    try:
        # 创建绘图器
        plotter = KDEPlotter(args.kde_csv, args.output_dir)
        
        # 打印数据摘要
        plotter.print_summary()
        
        # 绘制指定的6个特征
        print(f"\n🎨 绘制指定的6个特征...")
        plt1 = plotter.plot_top_features(top_n=6, save_plot=True)
        plt.show()
        
        # 获取特征列表
        features = plotter.get_available_features()
        
        print(f"\n📋 可用特征列表 ({len(features)} 个):")
        for i, feature in enumerate(features, 1):
            print(f"  {i:2d}. {feature}")
        
        print("\n💡 使用说明:")
        print("  • 红色曲线: 燃烧像元")
        print("  • 青色曲线: 未燃烧像元")
        print("  • X轴: 像元数值")
        print("  • Y轴: 概率密度")
        print("  • 曲线下面积表示概率分布")
        
        print(f"\n✅ KDE图已保存到: {plotter.output_dir}")
        
    except Exception as e:
        print(f"❌ 绘图过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 