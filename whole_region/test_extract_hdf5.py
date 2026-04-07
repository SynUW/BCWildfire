import os
import numpy as np
import h5py
from tqdm import tqdm
import argparse
import multiprocessing as mp
from multiprocessing import Pool, Manager
from collections import defaultdict
import time
import random
import re

def generate_test_hdf5(output_path, num_samples=100):
    """
    生成测试用的HDF5文件
    """
    with h5py.File(output_path, 'w') as f:
        # 生成一些测试数据
        for i in range(num_samples):
            # 随机生成位置和年份
            pos_x = random.randint(1000, 3000)
            pos_y = random.randint(1000, 3000)
            year = random.choice([2022, 2023, 2024])
            y_value = random.choice([0, 1, 2])  # 随机标签
            
            # 创建随机数据
            data = np.random.rand(10, 10)  # 10x10的随机数据
            
            # 创建键名
            key = f"{pos_x}_{pos_y}_year_{year}_y_{y_value}.npy"
            
            # 保存数据
            f.create_dataset(key, data=data)
    
    print(f"已生成测试HDF5文件: {output_path}")

def save_batch(memory_cache, output_dir, is_test=False):
    """
    将内存中的一批数据保存到硬盘
    """
    save_dir = os.path.join(output_dir, 'test' if is_test else 'train')
    os.makedirs(save_dir, exist_ok=True)
    
    for key, data in memory_cache.items():
        save_path = os.path.join(save_dir, key)
        np.save(save_path, data)

def extract_info_from_key(key):
    """从文件名中提取位置、年份和y值"""
    match = re.search(r'(\d+)_(\d+)_year_(\d+)_y_([\d.]+)\.npy', key)
    if match:
        pos_x = int(match.group(1))
        pos_y = int(match.group(2))
        year = int(match.group(3))
        y_value = float(match.group(4))
        return pos_x, pos_y, year, y_value
    return None, None, None, None

def is_valid_negative_sample(pos_x, pos_y):
    """判断负样本的位置是否有效（x或y大于2700）"""
    return pos_x > 2700 or pos_y > 2700

def process_single_hdf5(args):
    """
    处理单个HDF5文件
    """
    hdf5_path, output_dir, process_id, progress_dict = args
    
    sample_count = 0
    year_data = defaultdict(lambda: {'positive': [], 'negative': []})
    test_data = {}
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            for key in f.keys():
                pos_x, pos_y, year, y_value = extract_info_from_key(key)
                if year is None:
                    print(f"警告：无法解析文件名格式: {key}")
                    continue
                
                data = f[key][:]
                
                if year == 2024:
                    test_data[key] = data
                else:
                    if y_value >= 1:
                        year_data[year]['positive'].append((key, data))
                    else:
                        if is_valid_negative_sample(pos_x, pos_y):
                            year_data[year]['negative'].append((key, data))
                
                sample_count += 1
                
                if sample_count % 10 == 0:  # 更频繁地更新进度
                    progress_dict['completed'] = progress_dict.get('completed', 0) + 1
            
            # 处理测试数据
            if test_data:
                save_batch(test_data, output_dir, is_test=True)
                test_data.clear()
            
            # 处理训练数据
            for year in year_data:
                pos_samples = year_data[year]['positive']
                neg_samples = year_data[year]['negative']
                
                n_samples = min(len(pos_samples), len(neg_samples))
                
                if n_samples == 0:
                    continue
                
                if len(pos_samples) > n_samples:
                    pos_samples = random.sample(pos_samples, n_samples)
                if len(neg_samples) > n_samples:
                    neg_samples = random.sample(neg_samples, n_samples)
                
                balanced_samples = dict(pos_samples + neg_samples)
                save_batch(balanced_samples, output_dir, is_test=False)
                balanced_samples.clear()
            
            year_data.clear()
        
        progress_dict['completed'] = progress_dict.get('completed', 0) + 1
        return sample_count
    except Exception as e:
        print(f"处理文件 {hdf5_path} 时出错: {e}")
        progress_dict['completed'] = progress_dict.get('completed', 0) + 1
        return 0

def main():
    # 设置测试参数
    test_hdf5_path = "test_data.h5"
    output_dir = "test_output"
    
    # 生成测试数据
    print("生成测试数据...")
    generate_test_hdf5(test_hdf5_path, num_samples=100)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # 设置并行处理
    num_processes = 2  # 测试时使用较少的进程
    print(f"使用 {num_processes} 个进程进行并行处理")
    
    # 使用Manager创建共享对象
    manager = Manager()
    progress_dict = manager.dict()
    progress_dict['completed'] = 0
    
    # 准备进程参数
    process_args = [(test_hdf5_path, output_dir, 0, progress_dict)]
    
    # 创建进度条
    total_updates = 100  # 简化进度条更新
    pbar = tqdm(total=total_updates, desc="处理进度")
    last_update = 0
    
    # 启动进程池
    with Pool(num_processes) as pool:
        results = pool.map_async(process_single_hdf5, process_args)
        
        while not results.ready():
            current = progress_dict['completed']
            if current > last_update:
                pbar.update(current - last_update)
                last_update = current
            time.sleep(0.1)
        
        final_progress = progress_dict['completed']
        if final_progress > last_update:
            pbar.update(final_progress - last_update)
        
        file_samples = results.get()
        total_samples = sum(file_samples)
    
    pbar.close()
    
    print(f"\n处理完成！总共处理了 {total_samples} 个样本")
    print(f"训练集保存在: {os.path.join(output_dir, 'train')}")
    print(f"测试集保存在: {os.path.join(output_dir, 'test')}")

if __name__ == "__main__":
    main()