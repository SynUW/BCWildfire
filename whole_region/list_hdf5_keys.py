import os
import h5py
from tqdm import tqdm
import argparse
import json
from collections import defaultdict
import time
import re
import random

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

def check_hdf5_file(hdf5_path):
    """
    检查HDF5文件中的y值
    
    参数:
    hdf5_path: HDF5文件路径
    
    返回:
    tuple: (文件中的键数量, 是否有y值>=1的样本, 最大y值)
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 获取所有键
            keys = list(f.keys())
            max_y = 0
            has_positive = False
            
            # 检查每个键
            for key in keys:
                pos_x, pos_y, year, y_value = extract_info_from_key(key)
                if y_value is not None:
                    max_y = max(max_y, y_value)
                    if y_value >= 1:
                        has_positive = True
                        print(f"发现y值>=1的样本: {key}, y值: {y_value}")
            
            return len(keys), has_positive, max_y
    
    except Exception as e:
        print(f"处理文件 {hdf5_path} 时出错: {e}")
        return 0, False, 0

def main():
    parser = argparse.ArgumentParser(description='随机检查HDF5文件中的y值')
    parser.add_argument('--hdf5_dir', type=str, 
                        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/merged_hdf5",
                        help='HDF5文件所在的目录')
    parser.add_argument('--num_files', type=int,
                        default=100,
                        help='要检查的文件数量')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_dir):
        print(f"错误：输入目录不存在: {args.hdf5_dir}")
        return
    
    # 获取所有HDF5文件
    hdf5_files = [f for f in os.listdir(args.hdf5_dir) if f.endswith('.h5')]
    print(f"找到 {len(hdf5_files)} 个HDF5文件")
    
    # 随机选择指定数量的文件
    selected_files = random.sample(hdf5_files, min(args.num_files, len(hdf5_files)))
    print(f"\n随机选择的 {len(selected_files)} 个文件:")
    for i, f in enumerate(selected_files, 1):
        print(f"{i}. {f}")
    
    # 检查每个文件
    print("\n开始检查文件...")
    total_keys = 0
    files_with_positive = 0
    
    for hdf5_file in selected_files:
        hdf5_path = os.path.join(args.hdf5_dir, hdf5_file)
        print(f"\n检查文件: {hdf5_file}")
        
        n_keys, has_positive, max_y = check_hdf5_file(hdf5_path)
        total_keys += n_keys
        
        if has_positive:
            files_with_positive += 1
            print(f"该文件包含y值>=1的样本，最大y值: {max_y}")
        else:
            print(f"该文件不包含y值>=1的样本，最大y值: {max_y}")
    
    print(f"\n检查完成！")
    print(f"总共检查了 {len(selected_files)} 个文件")
    print(f"总共包含 {total_keys} 个样本")
    print(f"其中 {files_with_positive} 个文件包含y值>=1的样本")

if __name__ == "__main__":
    main() 