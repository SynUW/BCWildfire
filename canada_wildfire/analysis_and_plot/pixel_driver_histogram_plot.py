import pandas as pd
import numpy as np
import matplotlib
import os
import warnings

# Set matplotlib backend to avoid Qt-related errors
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings('ignore')

# Set font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


class PixelDriverHistogramPlotter:
    def __init__(self, csv_file_path, output_dir='histogram_plots'):
        """
        初始化直方图绘制器
        
        Args:
            csv_file_path: 像元驱动因素原始数据CSV文件路径
            output_dir: 输出目录
        """
        self.csv_file_path = csv_file_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_selected_driver_histograms(self, selected_drivers=None, output_filename='selected_driver_histograms.png'):
        """
        绘制选定驱动因素的分布直方图
        
        Args:
            selected_drivers: 要绘制的驱动因素列表，如果为None则自动选择重要驱动因素
            output_filename: 输出文件名
        """
        try:
            # 读取CSV文件
            print(f"正在读取数据文件: {self.csv_file_path}")
            df = pd.read_csv(self.csv_file_path)
            
            print(f"数据加载完成，共 {len(df)} 条记录")
            print(f"找到驱动因素: {df['driver_name'].unique()}")
            
            # 如果没有指定驱动因素，自动选择重要的几个
            if selected_drivers is None:
                selected_drivers = self._select_important_drivers(df)
            
            print(f"将绘制以下驱动因素: {selected_drivers}")
            
            # 准备数据
            plot_data = self._prepare_plot_data(df, selected_drivers)
            
            if not plot_data:
                print("没有找到匹配的驱动因素数据")
                return
            
            # 创建图表
            self._create_histogram_plot(plot_data, output_filename)
            
        except Exception as e:
            print(f"绘制过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _select_important_drivers(self, df):
        """自动选择重要的驱动因素进行绘制"""
        # 优先选择的驱动因素（基于重要性）
        priority_drivers = [
            'MODIS_Terra_Aqua_B20_21_merged_resampled',  # LST Day/Night
            'ERA5_multi_bands',                          # 气象数据
            'LAI_BCBoundingbox_resampled',              # 植被指数
            'NDVI_EVI',                                 # 植被指数
            'MOD21A1DN_multibands_filtered_resampled',  # 地表温度
            'Reflection_500_merge_TerraAquaWGS84_clip', # 反射率
        ]
        
        available_drivers = df['driver_name'].unique()
        selected = []
        
        # 选择存在的优先驱动因素
        for driver in priority_drivers:
            if driver in available_drivers:
                selected.append(driver)
                if len(selected) >= 6:  # 限制为6个以保持图表清晰
                    break
        
        # 如果还不够6个，添加其他可用的驱动因素
        for driver in available_drivers:
            if driver not in selected and len(selected) < 6:
                selected.append(driver)
        
        return selected[:6]  # 最多6个
    
    def _prepare_plot_data(self, df, selected_drivers):
        """准备绘图数据"""
        plot_data = {}
        
        for driver_name in selected_drivers:
            driver_data = df[df['driver_name'] == driver_name]
            
            if len(driver_data) == 0:
                continue
            
            # 按波段分组
            bands = driver_data['band_idx'].unique()
            
            for band_idx in sorted(bands):
                band_data = driver_data[driver_data['band_idx'] == band_idx]
                
                burned_values = band_data[band_data['pixel_type'] == 'burned']['value'].values
                unburned_values = band_data[band_data['pixel_type'] == 'unburned']['value'].values
                
                if len(burned_values) > 0 and len(unburned_values) > 0:
                    # 创建特征名称
                    if len(bands) == 1:
                        feature_name = driver_name.replace('_resampled', '').replace('_', ' ')
                    else:
                        feature_name = f"{driver_name.replace('_resampled', '').replace('_', ' ')} Band {band_idx}"
                    
                    plot_data[feature_name] = {
                        'burned': burned_values,
                        'unburned': unburned_values,
                        'driver_name': driver_name,
                        'band_idx': band_idx
                    }
        
        return plot_data
    
    def _create_histogram_plot(self, plot_data, output_filename):
        """创建直方图绘制"""
        n_features = len(plot_data)
        
        if n_features == 0:
            print("没有数据可以绘制")
            return
        
        print(f"准备绘制 {n_features} 个特征的直方图")
        
        # 创建子图布局 - 使用2行或3行布局
        if n_features <= 3:
            nrows, ncols = 1, n_features
            figsize = (n_features * 2.5, 3.5)
        elif n_features <= 6:
            nrows, ncols = 2, 3
            figsize = (7.5, 7)
        else:
            nrows, ncols = 3, 3
            figsize = (7.5, 10)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # 确保axes是数组格式
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # 为每个特征绘制直方图
        for i, (feature_name, data) in enumerate(plot_data.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            burned_values = data['burned']
            unburned_values = data['unburned']
            
            # 计算合适的bins
            all_values = np.concatenate([burned_values, unburned_values])
            n_bins = min(50, max(10, int(np.sqrt(len(all_values)))))
            
            # 绘制直方图
            ax.hist(unburned_values, bins=n_bins, alpha=0.6, color='red', 
                   label=f'Unburned (n={len(unburned_values)})', density=True)
            ax.hist(burned_values, bins=n_bins, alpha=0.6, color='blue', 
                   label=f'Burned (n={len(burned_values)})', density=True)
            
            # 设置标题和标签
            ax.set_title(feature_name, fontsize=10, fontweight='bold', pad=10)
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            
            # 添加网格
            ax.grid(True, alpha=0.3)
            
            # 设置刻度标签大小
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # 只在第一个子图添加图例
            if i == 0:
                legend = ax.legend(fontsize=8, loc='upper right')
                legend.get_frame().set_facecolor('none')
                legend.get_frame().set_edgecolor('none')
            
            # 添加统计信息文本
            burned_mean = np.mean(burned_values)
            unburned_mean = np.mean(unburned_values)
            ax.text(0.02, 0.98, f'Burned μ={burned_mean:.3f}\nUnburned μ={unburned_mean:.3f}', 
                   transform=ax.transAxes, fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(len(plot_data), len(axes)):
            axes[i].set_visible(False)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"直方图已保存到: {output_path}")
        
        # 创建数据统计汇总
        self._create_statistics_summary(plot_data)
    
    def _create_statistics_summary(self, plot_data):
        """创建统计汇总报告"""
        summary_file = os.path.join(self.output_dir, 'histogram_statistics_summary.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("像元驱动因素分布直方图统计汇总\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"分析特征数量: {len(plot_data)}\n\n")
            
            for feature_name, data in plot_data.items():
                burned_values = data['burned']
                unburned_values = data['unburned']
                
                f.write(f"特征: {feature_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"燃烧像元统计:\n")
                f.write(f"  样本数: {len(burned_values)}\n")
                f.write(f"  均值: {np.mean(burned_values):.6f}\n")
                f.write(f"  标准差: {np.std(burned_values):.6f}\n")
                f.write(f"  最小值: {np.min(burned_values):.6f}\n")
                f.write(f"  最大值: {np.max(burned_values):.6f}\n")
                f.write(f"  中位数: {np.median(burned_values):.6f}\n\n")
                
                f.write(f"未燃烧像元统计:\n")
                f.write(f"  样本数: {len(unburned_values)}\n")
                f.write(f"  均值: {np.mean(unburned_values):.6f}\n")
                f.write(f"  标准差: {np.std(unburned_values):.6f}\n")
                f.write(f"  最小值: {np.min(unburned_values):.6f}\n")
                f.write(f"  最大值: {np.max(unburned_values):.6f}\n")
                f.write(f"  中位数: {np.median(unburned_values):.6f}\n\n")
                
                mean_diff = np.mean(burned_values) - np.mean(unburned_values)
                f.write(f"均值差异: {mean_diff:.6f}\n")
                f.write(f"标准化差异: {mean_diff / np.std(unburned_values):.3f}\n")
                f.write("=" * 40 + "\n\n")
        
        print(f"统计汇总已保存到: {summary_file}")
    
    def plot_individual_driver_histograms(self, output_subdir='individual_histograms'):
        """为每个驱动因素创建单独的详细直方图"""
        try:
            df = pd.read_csv(self.csv_file_path)
            output_path = os.path.join(self.output_dir, output_subdir)
            os.makedirs(output_path, exist_ok=True)
            
            for driver_name in df['driver_name'].unique():
                print(f"正在绘制 {driver_name} 的详细直方图...")
                self._plot_single_driver(df, driver_name, output_path)
                
        except Exception as e:
            print(f"绘制单独直方图时出现错误: {str(e)}")
    
    def _plot_single_driver(self, df, driver_name, output_path):
        """为单个驱动因素绘制详细直方图"""
        driver_data = df[df['driver_name'] == driver_name]
        bands = sorted(driver_data['band_idx'].unique())
        
        n_bands = len(bands)
        
        if n_bands == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        else:
            ncols = min(3, n_bands)
            nrows = (n_bands + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
            if nrows == 1:
                axes = axes if n_bands > 1 else [axes]
            else:
                axes = axes.flatten()
        
        for i, band_idx in enumerate(bands):
            ax = axes[i]
            band_data = driver_data[driver_data['band_idx'] == band_idx]
            
            burned_values = band_data[band_data['pixel_type'] == 'burned']['value'].values
            unburned_values = band_data[band_data['pixel_type'] == 'unburned']['value'].values
            
            if len(burned_values) > 0 and len(unburned_values) > 0:
                # 计算bins
                all_values = np.concatenate([burned_values, unburned_values])
                n_bins = min(50, max(15, int(np.sqrt(len(all_values)))))
                
                # 绘制直方图
                ax.hist(unburned_values, bins=n_bins, alpha=0.6, color='red', 
                       label=f'Unburned (n={len(unburned_values)})', density=True)
                ax.hist(burned_values, bins=n_bins, alpha=0.6, color='blue', 
                       label=f'Burned (n={len(burned_values)})', density=True)
                
                # 设置标题和标签
                title = f"Band {band_idx}" if n_bands > 1 else driver_name
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Value', fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
                
                # 添加统计信息
                burned_mean = np.mean(burned_values)
                unburned_mean = np.mean(unburned_values)
                ax.text(0.02, 0.98, f'Burned μ={burned_mean:.4f}\nUnburned μ={unburned_mean:.4f}', 
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 隐藏多余的子图
        if n_bands < len(axes):
            for i in range(n_bands, len(axes)):
                axes[i].set_visible(False)
        
        # 设置总标题
        fig.suptitle(f"{driver_name} - Distribution Histograms", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 保存
        safe_name = driver_name.replace('/', '_').replace('\\', '_')
        filename = f"{safe_name}_histograms.png"
        plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
        plt.close()


def plot_pixel_driver_histograms(csv_file_path, output_dir='histogram_plots', selected_drivers=None):
    """
    便捷函数：绘制像元驱动因素分布直方图
    
    Args:
        csv_file_path: CSV文件路径
        output_dir: 输出目录
        selected_drivers: 选定的驱动因素列表
    """
    plotter = PixelDriverHistogramPlotter(csv_file_path, output_dir)
    
    # 绘制选定驱动因素的组合图
    plotter.plot_selected_driver_histograms(
        selected_drivers=selected_drivers,
        output_filename='selected_driver_histograms.png'
    )
    
    # 绘制每个驱动因素的详细图
    plotter.plot_individual_driver_histograms()
    
    print("所有直方图绘制完成!")


# Example usage
if __name__ == "__main__":
    # 示例用法
    csv_file = "canada_wildfire/pixel_driver_raw_samples/raw_driver_samples_10pct.csv"
    
    if os.path.exists(csv_file):
        print("开始绘制像元驱动因素分布直方图...")
        
        # 创建绘图器
        plotter = PixelDriverHistogramPlotter(csv_file, 'histogram_plots')
        
        # 绘制选定驱动因素的组合直方图
        plotter.plot_selected_driver_histograms(
            output_filename='pixel_driver_histograms.png'
        )
        
        # 绘制每个驱动因素的详细直方图
        plotter.plot_individual_driver_histograms()
        
        print("直方图绘制完成!")
    else:
        print(f"CSV文件不存在: {csv_file}")
        print("请先运行像元驱动因素原始数据抽样器生成CSV文件") 