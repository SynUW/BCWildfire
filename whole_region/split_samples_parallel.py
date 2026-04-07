import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import multiprocessing as mp
from multiprocessing import Pool, Manager
from datetime import datetime
import shutil

# ========== 配置 ========== #
INPUT_DIR = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data"
FIREDETECTION_DIR = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Firms_Detection"
OUTPUT_DIR = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_split"
FILE_LIST_PATH = os.path.join(OUTPUT_DIR, "file_list.txt")

BATCH_SIZE = 50  # 每个进程一次读取的文件数
SAVE_INTERVAL = 100  # 每处理多少个文件保存一次

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_file_list():
    """获取或生成文件列表"""
    if os.path.exists(FILE_LIST_PATH):
        print("从文件加载文件名列表...")
        with open(FILE_LIST_PATH, 'r') as f:
            return [line.strip() for line in f.readlines()]
    else:
        print("生成文件名列表...")
        npy_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.npy')])
        print(f"找到 {len(npy_files)} 个npy文件")
        with open(FILE_LIST_PATH, 'w') as f:
            f.write('\n'.join(npy_files))
        return npy_files

# ========== 日期补齐相关 ========== #
print("正在加载日期信息...")
# 生成完整的日期范围
full_dates = pd.date_range('2000-01-01', '2024-12-31', freq='D')
full_dates_str = [d.strftime('%Y_%m_%d') for d in full_dates]

# 获取FIREDETECTION_DIR中的有效日期
print("获取有效日期...")
valid_dates = []
for f in os.listdir(FIREDETECTION_DIR):
    if f.endswith('.tif'):
        date = f[6:-4]  # 去掉"Firms_"前缀和".tif"后缀
        valid_dates.append(date)
valid_dates = sorted(valid_dates)
print(f"找到 {len(valid_dates)} 个有效日期")

# 创建日期到索引的映射
date2idx = {date: idx for idx, date in enumerate(valid_dates)}

def pad_npy_to_full_25years(npy_data):
    """将数据补齐到完整的25年"""
    padded = np.zeros(len(full_dates_str), dtype=npy_data.dtype)
    
    # 遍历完整日期范围
    for i, date in enumerate(full_dates_str):
        if date in date2idx:
            # 使用date2idx[date]作为npy_data的索引
            padded[i] = npy_data[date2idx[date]]
    
    return padded

def process_batch(file_batch, process_id, progress_dict):
    """处理一批文件"""
    try:
        all_samples = []
        all_data = []
        total_processed = 0
        
        # 读取所有文件到内存
        for file_name in file_batch:
            file_path = os.path.join(INPUT_DIR, file_name)
            data = np.load(file_path)
            
            # 检查数据长度是否与有效日期数量匹配
            if len(data) != len(valid_dates):
                print(f"警告：文件 {file_name} 的数据长度 ({len(data)}) 与有效日期数量 ({len(valid_dates)}) 不匹配")
                continue
            
            padded_data = pad_npy_to_full_25years(data)
            
            # 分割数据
            window_size = 365 * 20
            
            # 遍历所有可能的起始位置
            for i in range(0, len(padded_data) - window_size - 365, 365):
                x = padded_data[i:i+window_size].copy()
                y = padded_data[i+window_size:i+window_size+365].sum()
                start_year = 2000 + (i // 365)
                
                base_name = os.path.splitext(file_name)[0]
                new_file_name = f"{base_name}_year_{start_year}_y_{y}.npy"
                
                # 将y值拼接到x的末尾
                combined_data = np.append(x, y)
                
                all_samples.append(new_file_name)
                all_data.append(combined_data)
            
            total_processed += 1
            
            # 每处理SAVE_INTERVAL个文件保存一次
            if total_processed % SAVE_INTERVAL == 0:
                # 保存当前批次的数据
                for sample_name, sample_data in zip(all_samples, all_data):
                    save_path = os.path.join(OUTPUT_DIR, sample_name)
                    np.save(save_path, sample_data)
                
                # 清空列表
                all_samples = []
                all_data = []
                
                # 更新进度
                progress_dict[process_id] = total_processed
        
        # 保存剩余的数据
        if len(all_samples) > 0:
            for sample_name, sample_data in zip(all_samples, all_data):
                save_path = os.path.join(OUTPUT_DIR, sample_name)
                np.save(save_path, sample_data)
            progress_dict[process_id] = total_processed
        
        return total_processed
    
    except Exception as e:
        print(f"进程 {process_id} 处理批次时出错: {e}")
        return 0

def process_files_parallel(file_list):
    """并行处理所有文件"""
    num_processes = int(mp.cpu_count() * 0.8)
    print(f"使用 {num_processes} 个进程进行处理")
    
    # 将文件列表平均分配给每个进程
    files_per_process = len(file_list) // num_processes
    process_file_lists = [file_list[i:i + files_per_process] for i in range(0, len(file_list), files_per_process)]
    
    # 创建共享字典用于存储进度
    manager = Manager()
    progress_dict = manager.dict()
    
    # 创建进程池
    with Pool(num_processes) as pool:
        # 为每个进程分配一个ID
        process_args = [(file_list, i, progress_dict) for i, file_list in enumerate(process_file_lists)]
        
        # 启动进度显示
        pbar = tqdm(total=len(file_list), desc="总体进度")
        
        # 启动处理
        results = pool.starmap_async(process_batch, process_args)
        
        # 更新进度条
        while not results.ready():
            # 计算总体进度
            total_progress = sum(progress_dict.values())
            pbar.n = total_progress
            pbar.refresh()
            
            # 等待一小段时间
            import time
            time.sleep(0.1)
        
        # 获取结果
        results = results.get()
    
    # 关闭进度条
    pbar.close()
    
    return sum(results)  # 返回处理的总文件数

def main():
    print("开始处理数据...")
    
    # 获取所有npy文件
    npy_files = get_file_list()
    
    # 并行处理文件
    total_processed = process_files_parallel(npy_files)
    
    print(f"\n处理完成！")
    print(f"总共处理了 {total_processed} 个文件")
    print(f"结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
