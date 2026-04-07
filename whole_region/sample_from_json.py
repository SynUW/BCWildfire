import os
import json
import random
import argparse
from tqdm import tqdm
import numpy as np
import h5py
import re

def save_batch(memory_cache, output_dir, is_test=False):
    """
    将内存中的一批数据保存到硬盘
    """
    save_dir = os.path.join(output_dir, 'test' if is_test else 'train')
    os.makedirs(save_dir, exist_ok=True)
    
    for key, data in memory_cache.items():
        save_path = os.path.join(save_dir, key)
        np.save(save_path, data)

def process_datasets(datasets, hdf5_dir, output_dir, batch_size=50):
    """
    处理数据集并保存
    """
    memory_cache = {}
    current_file = None
    current_h5 = None
    
    print("开始处理数据...")
    for dataset in tqdm(datasets):
        # 如果切换到新的HDF5文件，关闭旧文件并打开新文件
        if current_file != dataset['file']:
            if current_h5 is not None:
                current_h5.close()
            current_file = dataset['file']
            current_h5 = h5py.File(os.path.join(hdf5_dir, current_file), 'r')
        
        try:
            # 读取数据
            data = current_h5[dataset['key']][:]
            memory_cache[dataset['key']] = data
            
            # 当缓存达到指定大小时保存
            if len(memory_cache) >= batch_size:
                save_batch(memory_cache, output_dir, is_test=(dataset['year'] == 2024))
                memory_cache.clear()
        
        except Exception as e:
            print(f"处理数据集 {dataset['key']} 时出错: {e}")
            continue
    
    # 保存剩余的数据
    if memory_cache:
        save_batch(memory_cache, output_dir, is_test=(dataset['year'] == 2024))
    
    if current_h5 is not None:
        current_h5.close()

def is_valid_negative_sample(pos_x, pos_y):
    """判断负样本的位置是否有效（x或y大于2700）"""
    return pos_x > 2700 or pos_y > 2700

def extract_info_from_key(key):
    """从文件名中提取位置、年份和y值"""
    # 匹配形如 1366_5208_year_2000_y_0.npy 的文件名
    match = re.search(r'(\d+)_(\d+)_year_(\d+)_y_([\d.]+)\.npy', key)
    if match:
        pos_x = int(match.group(1))  # 第一个位置数字
        pos_y = int(match.group(2))  # 第二个位置数字
        year = int(match.group(3))   # 年份
        y_value = float(match.group(4))  # y值
        return pos_x, pos_y, year, y_value
    return None, None, None, None

def main():
    parser = argparse.ArgumentParser(description='从JSON文件中采样数据')
    parser.add_argument('--hdf5_dir', type=str, 
                        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/merged_hdf5",
                        help='HDF5文件所在的目录')
    parser.add_argument('--output_dir', type=str,
                        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_npy_single",
                        help='NPY文件的输出目录')
    parser.add_argument('--list_dir', type=str,
                        default="dataset_lists",
                        help='数据集列表所在的目录')
    parser.add_argument('--sample_ratio', type=float,
                        default=1.0,
                        help='采样比例（0-1之间）')
    parser.add_argument('--seed', type=int,
                        default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 加载数据集列表
    json_path = os.path.join(args.list_dir, 'all_datasets.json')
    if not os.path.exists(json_path):
        print(f"错误：找不到数据集列表文件: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        all_datasets = json.load(f)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)
    
    # 准备数据集
    train_positive = all_datasets['train']['positive']
    train_negative = all_datasets['train']['negative']
    test_datasets = all_datasets['test']
    
    # 过滤负样本，确保位置坐标满足要求
    filtered_negative = []
    for sample in train_negative:
        pos_x, pos_y, year, y_value = extract_info_from_key(sample['key'])
        if pos_x is not None and is_valid_negative_sample(pos_x, pos_y):
            filtered_negative.append(sample)
    
    train_negative = filtered_negative
    
    # 打印原始统计信息
    print("\n原始数据集统计信息：")
    print(f"训练集正样本数量: {len(train_positive)}")
    print(f"训练集负样本数量: {len(train_negative)}")
    print(f"测试集样本数量: {len(test_datasets)}")
    
    # 对训练集进行均衡采样
    n_samples = min(len(train_positive), len(train_negative))
    if len(train_positive) > n_samples:
        train_positive = random.sample(train_positive, n_samples)
    if len(train_negative) > n_samples:
        train_negative = random.sample(train_negative, n_samples)
    
    # 如果指定了采样比例，进一步采样
    if args.sample_ratio < 1.0:
        n_train_samples = int(n_samples * args.sample_ratio)
        train_positive = random.sample(train_positive, n_train_samples)
        train_negative = random.sample(train_negative, n_train_samples)
        
        n_test_samples = int(len(test_datasets) * args.sample_ratio)
        test_datasets = random.sample(test_datasets, n_test_samples)
    
    # 合并训练集
    train_datasets = train_positive + train_negative
    
    # 打印采样后的统计信息
    print("\n采样后数据集统计信息：")
    print(f"训练集正样本数量: {len(train_positive)}")
    print(f"训练集负样本数量: {len(train_negative)}")
    print(f"测试集样本数量: {len(test_datasets)}")
    
    # 处理训练集
    print("\n处理训练集...")
    process_datasets(train_datasets, args.hdf5_dir, args.output_dir)
    
    # 处理测试集
    print("\n处理测试集...")
    process_datasets(test_datasets, args.hdf5_dir, args.output_dir)
    
    print(f"\n处理完成！")
    print(f"训练集保存在: {os.path.join(args.output_dir, 'train')}")
    print(f"测试集保存在: {os.path.join(args.output_dir, 'test')}")

if __name__ == "__main__":
    main() 