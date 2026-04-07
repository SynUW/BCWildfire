import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime, timedelta

def get_date_from_day_of_year(year, day_of_year):
    """根据年份和一年中的第几天计算日期"""
    # 假设每年都是365天
    if day_of_year > 365:
        day_of_year = 365
    
    # 计算月份和日期
    month = 1
    day = day_of_year
    
    # 每个月的天数（非闰年）
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # 计算月份和日期
    for m, days in enumerate(days_in_month, 1):
        if day <= days:
            month = m
            break
        day -= days
    
    return datetime(year, month, day)

def plot_samples(h5_path, title, output_dir, num_samples=10):
    """从h5文件中随机抽取样本并绘制"""
    # 从文件名中提取年份
    year = int(h5_path.split('samples_')[1].split('_')[0])
    start_year = 2000
    end_year = year
    
    with h5py.File(h5_path, 'r') as f:
        # 获取所有数据集名称
        dataset_names = list(f.keys())
        
        # 随机选择样本
        selected_names = random.sample(dataset_names, num_samples)
        
        # 创建图形
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
        fig.suptitle(title, fontsize=16)
        
        for idx, name in enumerate(selected_names):
            # 获取数据
            data = f[name][:]
            time_series = data[:-1]  # 除最后一个值外都是时间序列
            label = data[-1]  # 最后一个值是标签
            
            # 绘制时间序列数据
            ax = axes[idx]
            ax.plot(time_series)
            ax.set_title(f'Sample {idx+1}')
            
            # 设置x轴标签
            # 按365天划分年份
            years = end_year - start_year + 1
            points_per_year = 365  # 固定为365天
            total_points = len(time_series)
            
            # 计算每年的位置
            x_ticks = []
            x_labels = []
            current_point = 0
            
            for y in range(start_year, end_year + 1):
                if current_point < total_points:
                    x_ticks.append(current_point)
                    x_labels.append(str(y))
                    current_point += points_per_year
            
            # 如果还有剩余的点，都算到最后一年
            if current_point < total_points:
                x_ticks.append(total_points - 1)
                x_labels.append(str(end_year))
            
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45)
            
            # 在数据值为1的位置添加日期标签
            for i, value in enumerate(time_series):
                if value == 1:
                    # 计算当前点的日期
                    days_from_start = i
                    current_year = start_year + (days_from_start // 365)
                    day_of_year = (days_from_start % 365) + 1
                    current_date = get_date_from_day_of_year(current_year, day_of_year)
                    date_str = current_date.strftime('%m-%d')
                    ax.text(i, value, date_str, 
                           bbox=dict(facecolor='white', alpha=0.8),
                           verticalalignment='bottom')
            
            ax.set_ylabel('Value')
            
            # 在右上角显示标签值
            ax.text(0.95, 0.95, f'Label: {label}', 
                   transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        # 保存到指定文件夹
        output_path = os.path.join(output_dir, f'{title.lower().replace(" ", "_")}_distribution.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

def analyze_data_distribution(h5_path):
    """分析h5文件中数据的分布情况"""
    print(f"\n分析文件: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # 获取所有数据集名称
        dataset_names = list(f.keys())
        total_samples = len(dataset_names)
        
        # 初始化计数器
        never_burned_before_burned_in_y = 0  # 数据和为0，标签不为0
        burned_before_never_burned_in_y = 0  # 数据和不为0，标签为0
        never_burned_before_never_burned_in_y = 0  # 数据和为0，标签为0
        burned_before_burned_in_y = 0  # 数据和不为0，标签不为0
        
        # 遍历所有样本
        for name in dataset_names:
            data = f[name][:]
            time_series = data[:-1]  # 除最后一个值外都是时间序列
            label = data[-1]  # 最后一个值是标签
            
            # 计算时间序列的和
            series_sum = np.sum(time_series)
            
            # 根据和与标签分类计数
            if series_sum == 0 and label != 0:
                never_burned_before_burned_in_y += 1
            elif series_sum != 0 and label == 0:
                burned_before_never_burned_in_y += 1
            elif series_sum == 0 and label == 0:
                never_burned_before_never_burned_in_y += 1
            elif series_sum != 0 and label != 0:
                burned_before_burned_in_y += 1
        
        # 打印统计结果
        print(f"总样本数: {total_samples}")
        print(f"从未燃烧过，但在y年燃烧: {never_burned_before_burned_in_y} ({never_burned_before_burned_in_y/total_samples*100:.2f}%)")
        print(f"之前燃烧过，但在y年未燃烧: {burned_before_never_burned_in_y} ({burned_before_never_burned_in_y/total_samples*100:.2f}%)")
        print(f"从未燃烧过，在y年也未燃烧: {never_burned_before_never_burned_in_y} ({never_burned_before_never_burned_in_y/total_samples*100:.2f}%)")
        print(f"之前燃烧过，在y年也燃烧: {burned_before_burned_in_y} ({burned_before_burned_in_y/total_samples*100:.2f}%)")

def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 创建输出文件夹
    output_dir = 'h5_distribution_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查训练集
    print("检查训练集...")
    train_path = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/samples_2020_sampled/train.h5'
    plot_samples(train_path, 'Training Set Distribution', output_dir)
    analyze_data_distribution(train_path)
    
    # 检查验证集
    print("\n检查验证集...")
    val_path = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/samples_2020_sampled/val.h5'
    plot_samples(val_path, 'Validation Set Distribution', output_dir)
    analyze_data_distribution(val_path)
    
    # 检查测试集
    print("\n检查测试集...")
    test_path = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/samples_2020_sampled/test.h5'
    plot_samples(test_path, 'Test Set Distribution', output_dir)
    analyze_data_distribution(test_path)
    
    print(f"\n完成！图像已保存到文件夹: {output_dir}")

if __name__ == "__main__":
    main() 