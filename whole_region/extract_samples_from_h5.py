import os
import numpy as np
import h5py
from tqdm import tqdm
import argparse
from osgeo import gdal
import pandas as pd

def read_sample_coords(csv_file):
    """读取采样点坐标"""
    df = pd.read_csv(csv_file)
    return list(zip(df['row'], df['col'])), df['label'].values

def extract_data_from_h5(data, dates, coords, labels, output_h5):
    """
    从已加载的数据中提取指定坐标的数据
    
    参数:
        data: 已加载的数据数组
        dates: 日期数组
        coords: 坐标列表 [(row1, col1), (row2, col2), ...]
        labels: 标签列表
        output_h5: 输出h5文件路径
    """
    try:
        # 创建输出文件
        with h5py.File(output_h5, 'w') as f_out:
            # 用于跟踪已使用的数据集名称
            used_names = set()
            
            # 提取数据
            for i, ((row, col), label) in enumerate(tqdm(zip(coords, labels), desc="提取数据")):
                try:
                    # 构建数据集名称，添加索引以确保唯一性
                    base_name = f"{row}_{col}_{int(label)}"
                    dataset_name = base_name
                    counter = 1
                    while dataset_name in used_names:
                        dataset_name = f"{base_name}_{counter}"
                        counter += 1
                    used_names.add(dataset_name)
                    
                    # 提取该位置的时间序列数据
                    time_series = data[:, row, col]
                    
                    # 将标签添加到数据末尾，如果标签是99则替换为0
                    label_value = 0 if label == 99 else label
                    combined_data = np.append(time_series, label_value)
                    
                    # 创建数据集
                    f_out.create_dataset(dataset_name, data=combined_data)
                except IndexError:
                    print(f"警告：坐标 ({row}, {col}) 超出数据范围")
                except Exception as e:
                    print(f"警告：处理坐标 ({row}, {col}) 时出错: {e}")
            
            # 添加属性
            f_out.attrs['description'] = 'Extracted samples from original h5 file'
            f_out.attrs['num_samples'] = len(coords)
            f_out.attrs['data_shape'] = data.shape
            f_out.attrs['dates'] = dates
    
    except Exception as e:
        print(f"处理文件时出错: {e}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='从h5文件中提取训练、验证和测试集')
    parser.add_argument('--input_h5', type=str, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/Visualization/FIRMS_2000_2021.h5",
                        help='输入h5文件路径')
    parser.add_argument('--samples_dir', type=str, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/samples_2021",
                        help='包含采样点CSV文件的目录')
    parser.add_argument('--output_dir', type=str, default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/Visualization/samples_2021_sampled",
                        help='输出h5文件保存目录')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取采样点
    print("读取采样点...")
    train_coords, train_labels = read_sample_coords(os.path.join(args.samples_dir, "train_samples.csv"))
    val_coords, val_labels = read_sample_coords(os.path.join(args.samples_dir, "val_samples.csv"))
    test_coords, test_labels = read_sample_coords(os.path.join(args.samples_dir, "test_samples.csv"))
    
    print(f"训练集样本数: {len(train_coords)}")
    print(f"验证集样本数: {len(val_coords)}")
    print(f"测试集样本数: {len(test_coords)}")
    
    # 只加载一次h5文件数据
    print("\n加载h5文件数据...")
    with h5py.File(args.input_h5, 'r') as f_in:
        data = f_in['data'][:]
        dates = f_in['dates'][:]
    
    # 提取训练集
    print("\n提取训练集...")
    extract_data_from_h5(
        data,
        dates,
        train_coords,
        train_labels,
        os.path.join(args.output_dir, "train.h5")
    )
    
    # 提取验证集
    print("\n提取验证集...")
    extract_data_from_h5(
        data,
        dates,
        val_coords,
        val_labels,
        os.path.join(args.output_dir, "val.h5")
    )
    
    # 提取测试集
    print("\n提取测试集...")
    extract_data_from_h5(
        data,
        dates,
        test_coords,
        test_labels,
        os.path.join(args.output_dir, "test.h5")
    )
    
    print("\n完成! 所有数据集已保存到:", args.output_dir)

if __name__ == "__main__":
    main()