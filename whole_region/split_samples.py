import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

# ========== 配置 ========== #
INPUT_DIR = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data"
FIREDETECTION_DIR = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Firms_Detection"
OUTPUT_DIR = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_split"
FILE_LIST_PATH = os.path.join(OUTPUT_DIR, "file_list.txt")
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
        # 保存文件列表
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

def process_one_file(file_name):
    """处理单个文件"""
    try:
        # 读取文件
        file_path = os.path.join(INPUT_DIR, file_name)
        data = np.load(file_path)
        
        # 检查数据长度是否与有效日期数量匹配
        if len(data) != len(valid_dates):
            print(f"警告：文件 {file_name} 的数据长度 ({len(data)}) 与有效日期数量 ({len(valid_dates)}) 不匹配")
            return []
        
        # 补齐日期
        padded_data = pad_npy_to_full_25years(data)

        print(padded_data.shape)
        
        # 分割数据
        window_size = 365 * 20
        samples = []
        
        # 遍历所有可能的起始位置
        for i in range(0, len(padded_data) - window_size - 365, 365):
            # 获取20年数据
            x = padded_data[i:i+window_size].copy()
            # 计算下一年的总和
            y = padded_data[i+window_size:i+window_size+365].sum()
            # 计算起始年份
            start_year = 2000 + (i // 365)
            
            # 构建新的文件名
            base_name = os.path.splitext(file_name)[0]
            new_file_name = f"{base_name}_year_{start_year}_y_{y}.npy"
            save_path = os.path.join(OUTPUT_DIR, new_file_name)
            
            # 保存数据
            np.save(save_path, np.concatenate([x, [y]]))
            samples.append((new_file_name, y))
        
        del data, padded_data
        gc.collect()
        return samples
    
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")
        return []

def main():
    print("开始处理数据...")
    
    # 获取所有npy文件
    npy_files = get_file_list()
    
    # 处理所有文件
    total_samples = 0
    for file_name in tqdm(npy_files, desc="处理文件"):
        samples = process_one_file(file_name)
        total_samples += len(samples)
    
    print(f"\n处理完成！")
    print(f"总共生成了 {total_samples} 个样本")
    print(f"样本保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 