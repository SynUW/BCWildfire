import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from osgeo import gdal
import pandas as pd
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def create_spatial_distribution_map(tif_path, tif_data, h5_data, output_path):
    """创建空间分布图，展示四种类别的空间分布"""
    # 创建一个与tif数据相同大小的数组，用于存储类别信息
    # 1: 从未燃烧过，但在y年燃烧
    # 2: 之前燃烧过，但在y年未燃烧
    # 3: 从未燃烧过，在y年也未燃烧
    # 4: 之前燃烧过，在y年也燃烧
    # 255: 背景值
    spatial_map = np.full_like(tif_data, 255, dtype=np.uint8)
    
    # 获取有效像素位置（非背景值255）
    valid_pixels = np.where(tif_data != 255)
    
    # 遍历每个有效像素
    for i in range(len(valid_pixels[0])):
        row, col = valid_pixels[0][i], valid_pixels[1][i]
        
        # 获取该位置的历史燃烧情况
        historical_data = h5_data[:, row, col]
        historical_sum = np.sum(historical_data)
        
        # 获取2020年的燃烧情况
        burned_in_2020 = tif_data[row, col] > 0
        
        # 根据条件分配类别
        if historical_sum == 0 and burned_in_2020:
            spatial_map[row, col] = 1
        elif historical_sum > 0 and not burned_in_2020:
            spatial_map[row, col] = 2
        elif historical_sum == 0 and not burned_in_2020:
            spatial_map[row, col] = 3
        elif historical_sum > 0 and burned_in_2020:
            spatial_map[row, col] = 4
    
    # 创建可视化
    plt.figure(figsize=(12, 8))
    
    # 定义颜色映射
    colors = ['red', 'blue', 'green', 'yellow']
    labels = ['从未燃烧过，但在y年燃烧 (1)', 
             '之前燃烧过，但在y年未燃烧 (2)',
             '从未燃烧过，在y年也未燃烧 (3)',
             '之前燃烧过，在y年也燃烧 (4)']
    
    # 创建自定义颜色映射
    cmap = mpl.colors.ListedColormap(colors)
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    # 绘制空间分布图
    plt.imshow(spatial_map, cmap=cmap, norm=norm)
    
    # 添加颜色条
    cbar = plt.colorbar(ticks=[1, 2, 3, 4])
    cbar.set_ticklabels(labels)
    
    # 设置标题
    plt.title('2020年像素燃烧情况空间分布')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存为GeoTIFF，保持原始tif的空间参考信息
    driver = gdal.GetDriverByName('GTiff')
    src_ds = gdal.Open(tif_path)
    
    # 创建输出数据集
    out_ds = driver.Create(output_path.replace('.png', '.tif'), 
                          spatial_map.shape[1], 
                          spatial_map.shape[0], 
                          1, 
                          gdal.GDT_Byte)
    
    # 复制空间参考信息
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    out_ds.SetProjection(src_ds.GetProjection())
    
    # 写入数据
    out_ds.GetRasterBand(1).WriteArray(spatial_map)
    
    # 设置NoData值
    out_ds.GetRasterBand(1).SetNoDataValue(255)
    
    # 关闭数据集
    src_ds = None
    out_ds = None

def analyze_pixel_distribution(tif_path, h5_path, output_dir):
    """分析tif文件中每个像素对应的h5文件中的历史燃烧情况"""
    print(f"\n分析文件: {tif_path}")
    
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用GDAL读取tif文件
    ds = gdal.Open(tif_path)
    if ds is None:
        print(f"无法打开tif文件: {tif_path}")
        return
    
    # 读取tif数据
    tif_data = ds.ReadAsArray()
    ds = None  # 关闭文件
    
    # 获取tif文件中的有效像素位置（非背景值255）
    valid_pixels = np.where(tif_data != 255)
    num_pixels = len(valid_pixels[0])
    print(f"tif文件中有效像素数量: {num_pixels}")
    
    # 初始化统计结果
    results = {
        'never_burned_before_burned_in_y': 0,  # 从未燃烧过，但在y年燃烧
        'burned_before_never_burned_in_y': 0,  # 之前燃烧过，但在y年未燃烧
        'never_burned_before_never_burned_in_y': 0,  # 从未燃烧过，在y年也未燃烧
        'burned_before_burned_in_y': 0,  # 之前燃烧过，在y年也燃烧
    }
    
    # 读取h5文件
    with h5py.File(h5_path, 'r') as f:
        # 获取数据
        data = f['data'][:]  # 形状: (时间步数, 高度, 宽度)
        
        # 遍历每个有效像素
        for i in range(num_pixels):
            row, col = valid_pixels[0][i], valid_pixels[1][i]
            
            # 获取该位置的历史燃烧情况（从h5文件）
            historical_data = data[:, row, col]  # 形状: (时间步数,)
            historical_sum = np.sum(historical_data)  # 历史燃烧总和
            
            # 获取2020年的燃烧情况（从tif文件）
            burned_in_2020 = tif_data[row, col] > 0  # 非0且非255的值表示燃烧
            
            # 更新统计结果
            if historical_sum == 0 and burned_in_2020:
                results['never_burned_before_burned_in_y'] += 1
            elif historical_sum > 0 and not burned_in_2020:
                results['burned_before_never_burned_in_y'] += 1
            elif historical_sum == 0 and not burned_in_2020:
                results['never_burned_before_never_burned_in_y'] += 1
            elif historical_sum > 0 and burned_in_2020:
                results['burned_before_burned_in_y'] += 1
    
    # 打印统计结果
    print("\n统计结果:")
    for key, value in results.items():
        percentage = (value / num_pixels) * 100
        print(f"{key}: {value} ({percentage:.2f}%)")
    
    # 创建可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = list(results.keys())
    values = list(results.values())
    
    # 创建柱状图
    bars = ax.bar(categories, values)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/num_pixels*100:.1f}%)',
                ha='center', va='bottom')
    
    # 设置图表属性
    ax.set_title('2020年像素燃烧情况分布')
    ax.set_ylabel('像素数量')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'pixel_distribution_2020.png'))
    plt.close()
    
    # 保存统计结果到CSV
    df = pd.DataFrame({
        'Category': categories,
        'Count': values,
        'Percentage': [v/num_pixels*100 for v in values]
    })
    df.to_csv(os.path.join(output_dir, 'pixel_distribution_2020.csv'), index=False)
    
    # 创建空间分布图
    create_spatial_distribution_map(tif_path, tif_data, data, 
                                  os.path.join(output_dir, 'spatial_distribution_2020.png'))

def main():
    # 设置文件路径
    tif_path = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/FIRMS_2020_sum.tif'
    h5_path = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/FIRMS_2000_2019.h5'
    output_dir = 'pixel_distribution_analysis'
    
    # 分析分布
    analyze_pixel_distribution(tif_path, h5_path, output_dir)
    print(f"\n分析完成！结果已保存到文件夹: {output_dir}")

if __name__ == "__main__":
    main() 