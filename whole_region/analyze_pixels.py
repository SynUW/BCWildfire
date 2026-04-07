import os
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


def analyze_pixel_distribution(data_dir):
    """
    分析所有npy文件的像元值分布
    
    参数:
        data_dir: 数据目录路径
    """
    start_time = time.time()
    
    # 获取所有npy文件
    pixel_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    print(f"找到 {len(pixel_files)} 个npy文件")
    
    # 存储每个文件的统计信息
    file_stats = []
    
    # 分析每个文件
    print("\n正在分析文件...")
    for pixel_file in tqdm(pixel_files):
        # 加载数据
        pixel_data = np.load(os.path.join(data_dir, pixel_file))
        
        # 计算统计信息
        total_sum = np.sum(pixel_data)
        mean_value = np.mean(pixel_data)
        max_value = np.max(pixel_data)
        non_zero_count = np.count_nonzero(pixel_data)
        
        file_stats.append({
            'file': pixel_file,
            'total_sum': total_sum,
            'mean': mean_value,
            'max': max_value,
            'non_zero': non_zero_count
        })
    
    # 按总和排序
    file_stats.sort(key=lambda x: x['total_sum'], reverse=True)
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"总文件数: {len(file_stats)}")
    
    # 计算总体统计信息
    total_sums = [stat['total_sum'] for stat in file_stats]
    print(f"\n总和统计:")
    print(f"最小值: {min(total_sums):.2f}")
    print(f"最大值: {max(total_sums):.2f}")
    print(f"平均值: {np.mean(total_sums):.2f}")
    print(f"中位数: {np.median(total_sums):.2f}")
    print(f"标准差: {np.std(total_sums):.2f}")
    
    # 打印前10个总和最大的文件
    print("\n前10个总和最大的文件:")
    for i, stat in enumerate(file_stats[:10], 1):
        print(f"{i}. {stat['file']}:")
        print(f"   总和: {stat['total_sum']:.2f}")
        print(f"   平均值: {stat['mean']:.2f}")
        print(f"   最大值: {stat['max']:.2f}")
        print(f"   非零值数量: {stat['non_zero']}")
    
    # 绘制分布图
    plt.figure(figsize=(12, 6))
    
    # 总和分布直方图
    plt.subplot(1, 2, 1)
    plt.hist(total_sums, bins=50)
    plt.title('像元值总和分布')
    plt.xlabel('总和')
    plt.ylabel('频数')
    
    # 总和分布箱线图
    plt.subplot(1, 2, 2)
    plt.boxplot(total_sums)
    plt.title('像元值总和箱线图')
    plt.ylabel('总和')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('pixel_distribution.png')
    print("\n分布图已保存为 'pixel_distribution.png'")
    
    # 保存详细统计信息到文件
    with open('pixel_statistics.txt', 'w') as f:
        f.write("=== 详细统计信息 ===\n\n")
        f.write(f"总文件数: {len(file_stats)}\n\n")
        
        f.write("总和统计:\n")
        f.write(f"最小值: {min(total_sums):.2f}\n")
        f.write(f"最大值: {max(total_sums):.2f}\n")
        f.write(f"平均值: {np.mean(total_sums):.2f}\n")
        f.write(f"中位数: {np.median(total_sums):.2f}\n")
        f.write(f"标准差: {np.std(total_sums):.2f}\n\n")
        
        f.write("所有文件的详细统计:\n")
        for stat in file_stats:
            f.write(f"\n文件: {stat['file']}\n")
            f.write(f"总和: {stat['total_sum']:.2f}\n")
            f.write(f"平均值: {stat['mean']:.2f}\n")
            f.write(f"最大值: {stat['max']:.2f}\n")
            f.write(f"非零值数量: {stat['non_zero']}\n")
    
    print("\n详细统计信息已保存到 'pixel_statistics.txt'")
    print(f"\n总耗时: {time.time() - start_time:.2f} 秒")


def main():
    # 数据目录
    data_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data"
    
    # 分析像元分布
    analyze_pixel_distribution(data_dir)


if __name__ == "__main__":
    main() 