"""
This script is used to plot the daily trends of all the features.add()
For selected features, please refer to plot_selected_trends.py.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_daily_trends(csv_path, output_dir):
    """
    根据透视表绘制每日驱动因素趋势图，分别生成燃烧和未燃烧像元的图。

    Args:
        csv_path (str): 包含透视表数据的CSV文件路径。
        output_dir (str): 保存图片的目录。
    """
    try:
        sns.set_theme(style="whitegrid")
    except ImportError:
        logging.error("Seaborn未安装，无法设置主题。请运行 'pip install seaborn'。")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(csv_path)
    logging.info(f"开始根据 {base_filename} 绘制趋势图...")
    
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        logging.error(f"文件未找到: {csv_path}")
        return

    if df.empty:
        logging.warning("透视表为空，无法绘制趋势图。")
        return
        
    # 提取唯一的特征基础名称（分别处理burned和control）
    burned_features = set()
    control_features = set()
    
    for index in df.index:
        # 匹配 'feature_name_burned_mean' 格式
        if '_burned_mean' in index:
            feature_name = index.replace('_burned_mean', '')
            burned_features.add(feature_name)
        # 匹配 'feature_name_control_mean' 格式
        elif '_control_mean' in index:
            feature_name = index.replace('_control_mean', '')
            control_features.add(feature_name)

    burned_features = sorted(list(burned_features))
    control_features = sorted(list(control_features))
    
    logging.info(f"找到 {len(burned_features)} 个燃烧像元特征")
    logging.info(f"找到 {len(control_features)} 个未燃烧像元特征")
    
    # 绘制燃烧像元图
    if burned_features:
        plot_feature_group(df, burned_features, 'burned', output_dir, base_filename)
    
    # 绘制未燃烧像元图
    if control_features:
        plot_feature_group(df, control_features, 'control', output_dir, base_filename)

def plot_feature_group(df, features, pixel_type, output_dir, base_filename):
    """
    绘制特定类型像元的特征趋势图
    
    Args:
        df: 数据框
        features: 特征名列表
        pixel_type: 像元类型 ('burned' 或 'control')
        output_dir: 输出目录
        base_filename: 基础文件名
    """
    # 设置子图网格
    cols = 4
    rows = (len(features) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        
        try:
            # 提取绘图所需数据
            mean_series = df.loc[f'{feature}_{pixel_type}_mean']
            q25_series = df.loc[f'{feature}_{pixel_type}_q25']
            q75_series = df.loc[f'{feature}_{pixel_type}_q75']
            
            # X轴是燃烧前的天数
            days = mean_series.index.astype(int)
            
            # 绘制均值实线
            ax.plot(days, mean_series.values, color='tab:blue', label='Mean')
            
            # 填充四分位间距作为阴影
            ax.fill_between(days, q25_series.values, q75_series.values, color='tab:blue', alpha=0.2, label='Interquartile Range')
            
            # 使用特征名作为子图标题
            ax.set_title(feature, fontsize=10, wrap=True)
            ax.set_xlabel("Days before fire ignition")
            if i % cols == 0:
                ax.set_ylabel("Driver Value")
            
            # 反转X轴，从10到1
            ax.invert_xaxis()
            
        except KeyError as e:
            logging.warning(f"特征 '{feature}' 缺少绘图所需数据 ({e})，跳过该子图。")
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature, fontsize=10, wrap=True)
        except Exception as e:
            logging.error(f"为特征 '{feature}' 绘图时出错: {e}")
            ax.text(0.5, 0.5, 'Error during plotting', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature, fontsize=10, wrap=True)

    # 隐藏多余的子图
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    # 根据像元类型设置标题
    title_map = {
        'burned': 'Daily Driver Trends Before Fire Ignition (Burned Pixels)',
        'control': 'Daily Driver Trends Before Fire Ignition (control Pixels)'
    }
    
    fig.suptitle(title_map[pixel_type], fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 从输入文件名生成输出文件名
    output_filename = os.path.splitext(base_filename)[0] + f'_trends_{pixel_type}.png'
    plot_file = os.path.join(output_dir, output_filename)
    
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"{pixel_type}像元逐日驱动因素趋势图已保存到: {plot_file}")

def main():
    
    csv_file='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/daily_driver_analysis_results/daily_driver_pivot_10days.csv'

    output_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/daily_driver_analysis_results'

        
    plot_daily_trends(csv_file, output_dir)

if __name__ == "__main__":
    main() 