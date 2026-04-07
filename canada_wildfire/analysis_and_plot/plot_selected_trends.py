import os
import pandas as pd
import matplotlib

# Set matplotlib backend to avoid Qt-related errors
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


def plot_selected_daily_trends(csv_path, output_dir):
    """
    根据透视表绘制选定的6个驱动因素趋势图，每种像元类型一行。
    只保留：band 20 d, band 20 n, lai, total precipitation, Volumetric Soil Water L1, Volumetric Soil Water L4

    Args:
        csv_path (str): 包含透视表数据的CSV文件路径。
        output_dir (str): 保存图片的目录。
    """
    try:
        # sns.set_theme(style="whitegrid")
        sns.set_theme(style="whitegrid", font="Arial")
    except ImportError:
        logging.error("Seaborn未安装，无法设置主题。请运行 'pip install seaborn'。")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.basename(csv_path)
    logging.info(f"开始根据 {base_filename} 绘制选定驱动因素趋势图...")

    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        logging.error(f"文件未找到: {csv_path}")
        return

    if df.empty:
        logging.warning("透视表为空，无法绘制趋势图。")
        return

    # 定义需要绘制的驱动因素（按指定顺序）
    selected_features = [
        'Temperature 2m',
        'Band 21 N',
        'V Wind 10m',
        'Total Percipitation',  # 注意CSV中有拼写错误
        'Volumetric Soil Water L4',
        'LAI',
    ]

    # 检查数据中可用的特征
    burned_available = []
    control_available = []

    for feature in selected_features:
        # 检查burned数据
        if f'{feature}_burned_mean' in df.index:
            burned_available.append(feature)

        # 检查control数据
        if f'{feature}_control_mean' in df.index:
            control_available.append(feature)

    logging.info(f"燃烧像元可用特征: {len(burned_available)} 个")
    logging.info(f"未燃烧像元可用特征: {len(control_available)} 个")

    # 创建组合图：两行，每行显示一种像元类型的6个特征
    if burned_available or control_available:
        plot_combined_trends(df, burned_available, control_available, output_dir, base_filename)


def plot_combined_trends(df, burned_features, control_features, output_dir, base_filename):
    """
    绘制组合趋势图：两行，每行6个子图

    Args:
        df: 数据框
        burned_features: 可用的燃烧像元特征列表
        control_features: 可用的未燃烧像元特征列表
        output_dir: 输出目录
        base_filename: 基础文件名
    """

    # 确定要绘制的行数
    n_cols = 6  # 固定6列
    rows_to_plot = []

    if burned_features:
        rows_to_plot.append(('burned', burned_features))
    if control_features:
        rows_to_plot.append(('control', control_features))

    if not rows_to_plot:
        logging.warning("没有可用的特征数据")
        return

    n_rows = len(rows_to_plot)

    # 创建图形 - 确保子图宽度与KDE图完全一致
    # KDE图使用figsize=(10.8, 2.0)，即每个子图实际宽度1.8英寸
    fig_width = 10.8  # 与KDE图保持完全相同的总宽度
    fig_height = n_rows * 2.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # 如果只有一行，确保axes是二维数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # 为每行绘制对应的像元类型数据
    for row_idx, (pixel_type, features) in enumerate(rows_to_plot):

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            if col_idx < len(features):
                feature = features[col_idx]

                try:
                    # 提取绘图所需数据
                    mean_series = df.loc[f'{feature}_{pixel_type}_mean']
                    q25_series = df.loc[f'{feature}_{pixel_type}_q25']
                    q75_series = df.loc[f'{feature}_{pixel_type}_q75']

                    # X轴是燃烧前的天数
                    days = mean_series.index.astype(int)

                    # 设置颜色 - 与高斯分布图保持一致
                    color = 'red' if pixel_type == 'burned' else 'blue'

                    # 绘制均值实线 - 与高斯分布图相同的线宽
                    ax.plot(days, mean_series.values, color=color, linewidth=1.5)

                    # 填充四分位间距作为阴影 - 与高斯分布图相同的透明度
                    ax.fill_between(days, q25_series.values, q75_series.values,
                                    color=color, alpha=0.3)

                    # 设置X轴范围，左右留出空白
                    ax.set_xlim(11, 0)  # x轴范围从9.8到0.8，左右留出空白

                    # 移除坐标轴标签 - 与高斯分布图保持一致
                    ax.set_xlabel('')
                    ax.set_ylabel('')

                    # 扩展Y轴范围1.5倍以提供更好的视觉效果
                    y_min, y_max = ax.get_ylim()
                    y_range = y_max - y_min
                    y_center = (y_max + y_min) / 2
                    new_range = y_range * 1.5
                    new_y_min = y_center - new_range / 2
                    new_y_max = y_center + new_range / 2
                    ax.set_ylim(new_y_min, new_y_max)
                    
                    # 在扩展后的范围内，只在2%和98%位置显示y轴刻度
                    expanded_range = new_y_max - new_y_min
                    y_02 = new_y_min + expanded_range * 0.1
                    y_98 = new_y_min + expanded_range * 0.9
                    y_ticks = [y_02, y_98]
                    ax.set_yticks(y_ticks)
                    
                    # 隐藏默认的y轴刻度标签，我们将手动放置
                    ax.set_yticklabels([])
                    
                    # 格式化y轴刻度标签，限制小数位数
                    def format_tick_label(x, pos):
                        if abs(x) >= 1:
                            return f'{x:.1f}'
                        elif abs(x) >= 0.01:
                            return f'{x:.3f}'
                        elif abs(x) >= 0.001:
                            return f'{x:.4f}'
                        else:
                            return f'{x:.1e}'
                    
                    from matplotlib.ticker import FuncFormatter
                    
                    # 手动放置y轴刻度标签，控制最左侧到坐标轴的距离
                    left_margin_points = 110  # 文字最左侧到坐标轴的距离（点数）
                    for y_pos in y_ticks:
                        formatted_label = format_tick_label(y_pos, None)
                        # 使用数据坐标系放置文本，x位置用负的像素偏移
                        ax.annotate(formatted_label, 
                                  xy=(0, y_pos), xycoords='data',
                                  xytext=(-left_margin_points, 0), textcoords='offset points',
                                  ha='left', va='center', 
                                  fontsize=11, fontweight='bold', fontfamily='Arial')

                    # 设置y轴刻度线朝内
                    ax.tick_params(axis='y', direction='in', length=4, width=1)

                    # 设置轴线 - 完全匹配KDE图的黑色边框样式
                    ax.spines['top'].set_visible(True)
                    ax.spines['bottom'].set_visible(True)
                    ax.spines['left'].set_visible(True)
                    ax.spines['right'].set_visible(True)
                    ax.spines['top'].set_color('black')
                    ax.spines['bottom'].set_color('black')
                    ax.spines['left'].set_color('black')
                    ax.spines['right'].set_color('black')

                    # 设置网格 - 与高斯分布图相同的透明度
                    ax.grid(True, alpha=0.3)

                    # 设置X轴刻度 - 确保从10到1的正确顺序
                    ax.set_xticks([10, 5, 1])
                    if row_idx == n_rows - 1:  # 最后一行显示标签
                        ax.set_xticklabels(['10', '5', '1'])  # 修正标签顺序：从左到右是10,5,1
                        ax.tick_params(axis='x', which='major', labelsize=10, top=False, bottom=True, labeltop=False, labelbottom=True)
                    else:  # 其他行不显示标签但保留刻度线
                        ax.set_xticklabels([])
                        ax.tick_params(axis='x', which='major', length=4, width=1, top=False, bottom=True)

                    # 确保字体family与高斯分布图一致，并加粗
                    for label in ax.get_xticklabels():
                        label.set_fontfamily('Arial')
                        label.set_fontweight('bold')

                    # 添加子图标题在图下方 - 与高斯分布图相同的格式
                    # ax.set_title(feature, fontsize=9, fontweight='bold', pad=-20)

                except KeyError as e:
                    logging.warning(f"特征 '{feature}' ({pixel_type}) 缺少绘图所需数据 ({e})")
                    ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
                    # ax.set_title(feature, fontsize=9, fontweight='bold', pad=-20)
                except Exception as e:
                    logging.error(f"为特征 '{feature}' ({pixel_type}) 绘图时出错: {e}")
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
                    # ax.set_title(feature, fontsize=9, fontweight='bold', pad=-20)
            else:
                # 隐藏多余的子图
                ax.set_visible(False)

    # 在最后一行最后一个子图添加图例
    last_row = n_rows - 1
    last_col = min(len(rows_to_plot[last_row][1]) - 1, n_cols - 1)
    # if last_col >= 0:
    #     # 创建虚拟线条用于图例
    #     legend_ax = axes[last_row, last_col]
    #     burned_line = plt.Line2D([0], [0], color='blue', linewidth=1.5, label='Burned')
    #     not_burned_line = plt.Line2D([0], [0], color='red', linewidth=1.5, label='Not Burned')
    #
    #     legend = legend_ax.legend(handles=[burned_line, not_burned_line],
    #                               fontsize=7, loc='upper right',
    #                               bbox_to_anchor=(0.98, 0.98))
    #     legend.get_frame().set_facecolor('none')  # 移除背景
    #     legend.get_frame().set_edgecolor('none')  # 移除边框

    # 调整布局 - 确保子图尺寸和间距与KDE图完全一致
    plt.tight_layout()
    if n_rows == 1:
        # 单行时与KDE图完全相同
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.2, wspace=0.05)
    else:
        # 多行时保持左右间距一致，添加适当的行间距
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.1, wspace=0.05, hspace=0.3)

    fig.text(0.5, -0.02, 'Days Before Ignition', ha='center', va='bottom',
             fontsize=10, fontweight='bold', fontfamily='Arial')

    # 生成输出文件名
    output_filename = os.path.splitext(base_filename)[0] + '_selected_trends_combined.png'
    plot_file = os.path.join(output_dir, output_filename)

    # 保存图片 - 与高斯分布图相同的设置
    plt.savefig(plot_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    logging.info(f"选定驱动因素组合趋势图已保存到: {plot_file}")

    # 打印数据摘要
    print(f"\n选定特征数据摘要:")
    print("-" * 60)
    for pixel_type, features in rows_to_plot:
        print(f"{pixel_type.upper()} 像元:")
        for feature in features:
            try:
                mean_series = df.loc[f'{feature}_{pixel_type}_mean']
                print(f"  {feature}: 数据点数 = {len(mean_series)}")
            except KeyError:
                print(f"  {feature}: 数据缺失")
        print()


def main():
    """主函数"""
    csv_file = r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/daily_driver_analysis_results/daily_driver_pivot_10days.csv'
    output_dir = r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/daily_driver_analysis_results/daily_driver_analysis_results'

    plot_selected_daily_trends(csv_file, output_dir)


if __name__ == "__main__":
    main()