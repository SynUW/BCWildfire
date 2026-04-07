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
import json

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

def is_valid_negative_sample(pos_x, pos_y):
    """判断负样本的位置是否有效（x或y大于2700）"""
    return pos_x > 2700 or pos_y > 2700

def process_hdf5_file(args):
    """
    处理单个HDF5文件，提取数据集信息
    """
    hdf5_file, hdf5_dir = args
    hdf5_path = os.path.join(hdf5_dir, hdf5_file)
    file_datasets = []
    year_stats = defaultdict(lambda: {'positive': 0, 'negative': 0, 'test': 0})
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 预先获取所有keys，避免重复遍历
            keys = list(f.keys())
            for key in keys:
                pos_x, pos_y, year, y_value = extract_info_from_key(key)
                if year is None:
                    continue
                
                dataset_info = {
                    'file': hdf5_file,
                    'key': key,
                    'pos_x': pos_x,
                    'pos_y': pos_y,
                    'year': year,
                    'y_value': y_value
                }
                
                file_datasets.append(dataset_info)
                
                if year == 2024:
                    year_stats[year]['test'] += 1
                else:
                    if y_value >= 1:
                        year_stats[year]['positive'] += 1
                    elif is_valid_negative_sample(pos_x, pos_y):
                        year_stats[year]['negative'] += 1
        
        return {
            'file': hdf5_file,
            'datasets': file_datasets,
            'stats': dict(year_stats)
        }
    
    except Exception as e:
        print(f"处理文件 {hdf5_file} 时出错: {e}")
        return None

def merge_stats(stats_list):
    """
    合并多个进程的统计信息
    """
    merged_stats = defaultdict(lambda: {'positive': 0, 'negative': 0, 'test': 0})
    for stats in stats_list:
        for year, year_data in stats.items():
            for key in ['positive', 'negative', 'test']:
                merged_stats[year][key] += year_data[key]
    return dict(merged_stats)

def backup_existing_files(list_dir):
    """
    备份已存在的json文件
    """
    backup_dir = os.path.join(list_dir, 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    files_to_backup = ['all_datasets.json', 'file_datasets.json', 'year_stats.json']
    
    for file_name in files_to_backup:
        file_path = os.path.join(list_dir, file_name)
        if os.path.exists(file_path):
            backup_path = os.path.join(backup_dir, f"{file_name}.{timestamp}")
            try:
                import shutil
                shutil.copy2(file_path, backup_path)
                print(f"已备份 {file_name} 到 {backup_path}")
            except Exception as e:
                print(f"备份 {file_name} 时出错: {e}")

def generate_file_list(hdf5_dir, list_dir):
    """
    生成HDF5文件中的数据集列表
    """
    # 获取所有HDF5文件
    hdf5_files = [f for f in os.listdir(hdf5_dir) if f.endswith('.h5')]
    
    if not hdf5_files:
        print(f"警告：在 {hdf5_dir} 中未找到HDF5文件")
        return None, None
    
    # 检查是否存在json文件
    json_files = ['all_datasets.json', 'file_datasets.json', 'year_stats.json']
    existing_files = [f for f in json_files if os.path.exists(os.path.join(list_dir, f))]
    
    if existing_files:
        print("\n发现已存在的json文件：")
        for file in existing_files:
            print(f"- {file}")
        
        response = input("\n是否要覆盖这些文件？(y/n): ").lower()
        if response != 'y':
            print("操作已取消")
            return None, None
        
        # 备份现有文件
        backup_existing_files(list_dir)
    
    # 设置并行处理
    num_processes = int(mp.cpu_count() * 0.85)
    print(f"使用 {num_processes} 个进程进行并行处理")
    
    # 准备进程参数
    process_args = [(f, hdf5_dir) for f in hdf5_files]
    
    # 用于存储所有数据集信息
    all_datasets = {
        'train': {
            'positive': [],
            'negative': []
        },
        'test': []
    }
    
    # 用于存储每个HDF5文件中的数据集
    file_datasets = {}
    
    # 用于统计每年的样本数量
    year_stats = defaultdict(lambda: {'positive': 0, 'negative': 0, 'test': 0})
    
    print("开始扫描HDF5文件...")
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_hdf5_file, process_args),
            total=len(process_args),
            desc="处理HDF5文件"
        ))
    
    # 处理结果
    for result in results:
        if result is None:
            continue
            
        file_datasets[result['file']] = result['datasets']
        
        # 更新统计信息
        for year, stats in result['stats'].items():
            for key in ['positive', 'negative', 'test']:
                year_stats[year][key] += stats[key]
        
        # 更新数据集信息
        for dataset in result['datasets']:
            if dataset['year'] == 2024:
                all_datasets['test'].append(dataset)
                    else:
                if dataset['y_value'] >= 1:
                    all_datasets['train']['positive'].append(dataset)
                elif is_valid_negative_sample(dataset['pos_x'], dataset['pos_y']):
                    all_datasets['train']['negative'].append(dataset)
    
    # 保存数据集信息
    os.makedirs(list_dir, exist_ok=True)
    print("\n保存数据集信息...")
    
    # 使用更高效的JSON序列化
    with open(os.path.join(list_dir, 'all_datasets.json'), 'w') as f:
        json.dump(all_datasets, f, indent=2)
    
    with open(os.path.join(list_dir, 'file_datasets.json'), 'w') as f:
        json.dump(file_datasets, f, indent=2)
    
    with open(os.path.join(list_dir, 'year_stats.json'), 'w') as f:
        json.dump(dict(year_stats), f, indent=2)
    
    # 打印统计信息
    print("\n数据集统计信息：")
    print(f"训练集正样本数量: {len(all_datasets['train']['positive'])}")
    print(f"训练集负样本数量: {len(all_datasets['train']['negative'])}")
    print(f"测试集样本数量: {len(all_datasets['test'])}")
    
    print("\n年度统计信息：")
    for year in sorted(year_stats.keys()):
        stats = year_stats[year]
        print(f"年份 {year}:")
        print(f"  正样本: {stats['positive']}")
        print(f"  负样本: {stats['negative']}")
        print(f"  测试样本: {stats['test']}")
    
    return all_datasets, file_datasets

def save_batch(memory_cache, output_dir, is_test=False):
    """
    将内存中的一批数据保存到硬盘
    """
    # 根据是否为测试集选择保存目录
    save_dir = os.path.join(output_dir, 'test' if is_test else 'train')
    os.makedirs(save_dir, exist_ok=True)
    
    for key, data in memory_cache.items():
        save_path = os.path.join(save_dir, key)
        np.save(save_path, data)

def process_datasets(args):
    """
    处理一批数据集
    """
    datasets, hdf5_dir, output_dir, process_id, progress_dict = args
    
    memory_cache = {}
    sample_count = 0
    
    try:
        # 按HDF5文件分组处理
        current_file = None
        current_h5 = None
        
        for dataset in datasets:
            if current_file != dataset['file']:
                if current_h5 is not None:
                    current_h5.close()
                current_file = dataset['file']
                current_h5 = h5py.File(os.path.join(hdf5_dir, current_file), 'r')
            
            try:
                data = current_h5[dataset['key']][:]
                memory_cache[dataset['key']] = data
                sample_count += 1
                
                # 每积累50个样本就保存一次
                if len(memory_cache) >= 50:
                    save_batch(memory_cache, output_dir, is_test=(dataset['year'] == 2024))
                    memory_cache.clear()
                
                if sample_count % 10 == 0:
                    progress_dict['completed'] = progress_dict.get('completed', 0) + 1
            
            except Exception as e:
                print(f"处理数据集 {dataset['key']} 时出错: {e}")
                    continue
                
        # 保存剩余的数据
        if memory_cache:
            save_batch(memory_cache, output_dir, is_test=(dataset['year'] == 2024))
        
        if current_h5 is not None:
            current_h5.close()
        
        return sample_count
    
    except Exception as e:
        print(f"处理进程 {process_id} 时出错: {e}")
        if current_h5 is not None:
            current_h5.close()
        return 0

def process_from_list(hdf5_dir, output_dir, list_dir):
    """
    从数据集列表处理数据
    """
    # 加载数据集列表
    with open(os.path.join(list_dir, 'all_datasets.json'), 'r') as f:
        all_datasets = json.load(f)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # 设置并行处理
    num_processes = min(int(mp.cpu_count() * 0.5), 16)
    print(f"使用 {num_processes} 个进程进行并行处理")
    
    # 准备数据集
    train_positive = all_datasets['train']['positive']
    train_negative = all_datasets['train']['negative']
    test_datasets = all_datasets['test']
    
    # 对训练集进行均衡采样
    n_samples = min(len(train_positive), len(train_negative))
    if len(train_positive) > n_samples:
        train_positive = random.sample(train_positive, n_samples)
    if len(train_negative) > n_samples:
        train_negative = random.sample(train_negative, n_samples)
    
    # 合并训练集
    train_datasets = train_positive + train_negative
    
    # 将数据集分成多个批次
    def chunk_list(lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]
    
    batch_size = 1000  # 每个批次处理1000个样本
    train_batches = chunk_list(train_datasets, batch_size)
    test_batches = chunk_list(test_datasets, batch_size)
    
    # 使用Manager创建共享对象
    manager = Manager()
    progress_dict = manager.dict()
    progress_dict['completed'] = 0
    
    # 准备进程参数
    process_args = []
    for i, batch in enumerate(train_batches + test_batches):
        process_args.append((batch, hdf5_dir, output_dir, i, progress_dict))
    
    # 创建进度条
    total_updates = len(process_args) * 10  # 每个批次预计更新10次
    pbar = tqdm(total=total_updates, desc="处理进度")
    last_update = 0
    
    # 启动进程池
    total_samples = 0
    with Pool(num_processes) as pool:
        results = pool.map_async(process_datasets, process_args)
        
        while not results.ready():
            current = progress_dict['completed']
            if current > last_update:
                pbar.update(current - last_update)
                last_update = current
        
        final_progress = progress_dict['completed']
        if final_progress > last_update:
            pbar.update(final_progress - last_update)
        
        file_samples = results.get()
        total_samples = sum(file_samples)
    
    pbar.close()
    
    return total_samples

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='从HDF5文件中提取数据到NPY文件')
    parser.add_argument('--hdf5_dir', type=str, 
                        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/merged_hdf5",
                        help='HDF5文件所在的目录')
    parser.add_argument('--output_dir', type=str,
                        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_npy_single",
                        help='NPY文件的输出目录')
    parser.add_argument('--list_dir', type=str,
                        default="dataset_lists",
                        help='数据集列表所在的目录')
    parser.add_argument('--skip_list_generation', action='store_true',
                        help='跳过文件列表生成步骤，直接使用现有的列表文件')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_dir):
        print(f"错误：输入目录不存在: {args.hdf5_dir}")
        return
    
    # 生成或加载文件列表
    if not args.skip_list_generation:
        print("生成文件列表...")
        all_datasets, file_datasets = generate_file_list(args.hdf5_dir, args.list_dir)
        if all_datasets is None:
            return
    else:
        print("使用现有文件列表...")
    
    # 处理数据
    print("\n开始处理数据...")
    total_samples = process_from_list(args.hdf5_dir, args.output_dir, args.list_dir)
    
    print(f"\n处理完成！总共处理了 {total_samples} 个样本")
    print(f"训练集保存在: {os.path.join(args.output_dir, 'train')}")
    print(f"测试集保存在: {os.path.join(args.output_dir, 'test')}")

if __name__ == "__main__":
    main() 