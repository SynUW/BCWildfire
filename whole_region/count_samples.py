import os
import h5py
from tqdm import tqdm
import re
import argparse
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np

def process_h5_file(args):
    """处理单个H5文件并返回统计结果"""
    h5_path, h5_file = args
    total_samples = 0
    positive_samples = 0
    
    try:
        # 打开H5文件
        with h5py.File(h5_path, 'r') as f:
            # 获取所有键
            keys = list(f.keys())
            total_samples = len(keys)
            
            # 统计正样本数量
            for key in keys:
                data = f[key][:]
                y_value = data[-1]  # 最后一个值是标签
                if y_value >= 2:
                    positive_samples += 1
        
        return {
            'file': h5_file,
            'total_samples': total_samples,
            'positive_samples': positive_samples,
            'positive_ratio': positive_samples/total_samples*100 if total_samples > 0 else 0
        }
    
    except Exception as e:
        print(f"处理文件 {h5_file} 时出错: {e}")
        return None

def count_samples_by_year(h5_dir):
    """统计每个H5文件中y值大于等于2的样本数量"""
    # 获取所有H5文件
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    h5_files = [f for f in h5_files if 'year_2003' in f]
    
    # 按年份排序
    h5_files.sort()
    
    # 设置并行处理
    num_processes = int(mp.cpu_count() * 0.85)
    print(f"使用 {num_processes} 个进程进行并行处理")
    
    # 准备处理参数
    process_args = [(os.path.join(h5_dir, h5_file), h5_file) for h5_file in h5_files]
    
    # 并行处理所有文件
    print("\n开始统计样本数量...")
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_h5_file, process_args),
            total=len(process_args),
            desc="处理文件"
        ))
    
    # 打印统计结果
    print("\n统计结果:")
    for result in results:
        if result is None:
            continue
        print(f"\n文件: {result['file']}")
        print(f"总样本数: {result['total_samples']}")
        print(f"正样本数 (y >= 2): {result['positive_samples']}")
        print(f"正样本比例: {result['positive_ratio']:.2f}%")
    
    # 计算总体统计
    total_samples_all = sum(r['total_samples'] for r in results if r is not None)
    total_positive_all = sum(r['positive_samples'] for r in results if r is not None)
    
    print("\n总体统计:")
    print(f"所有文件总样本数: {total_samples_all}")
    print(f"所有文件正样本数: {total_positive_all}")
    print(f"所有文件正样本比例: {total_positive_all/total_samples_all*100:.2f}%")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='统计H5文件中的样本数量')
    parser.add_argument('--h5_dir', type=str, 
                        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/resampled_by_year",
                        help='H5文件所在的目录')
    parser.add_argument('--num_processes', type=int, default=None,
                        help='并行处理的进程数，默认为CPU核心数的85%')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.h5_dir):
        print(f"错误：目录不存在: {args.h5_dir}")
        return
    
    # 统计样本数量
    count_samples_by_year(args.h5_dir)
    
    print("\n统计完成！")

if __name__ == "__main__":
    main() 