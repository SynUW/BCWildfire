import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import random

def analyze_file(path):
    arr = np.load(path, mmap_mode='r')
    return arr.sum(), arr.mean(), arr.max(), np.count_nonzero(arr)

def analyze_npy_dir(data_dir, num_workers=8):
    npy_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

    # 随机抽样
    sample_ratio = 1  # 抽样比例,可以根据需要修改
    sample_size = int(len(npy_files) * sample_ratio)
    npy_files = np.random.choice(npy_files, size=sample_size, replace=False)
    print(f"随机抽取了 {sample_size} 个文件进行分析 (抽样比例: {sample_ratio*100}%)")
    print(f"共找到 {len(npy_files)} 个npy文件。")

    all_sums, all_means, all_maxs, all_nonzeros = [], [], [], []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(analyze_file, f): f for f in npy_files}
        for future in tqdm(as_completed(futures), total=len(npy_files), desc="统计中"):
            s, m, mx, nz = future.result()
            all_sums.append(s)
            all_means.append(m)
            all_maxs.append(mx)
            all_nonzeros.append(nz)

    print("\n==== 统计结果 ====")
    print(f"所有像素文件总和的最小值: {np.min(all_sums)}")
    print(f"所有像素文件总和的最大值: {np.max(all_sums)}")
    print(f"所有像素文件总和的均值: {np.mean(all_sums):.2f}")
    print(f"所有像素文件总和的中位数: {np.median(all_sums):.2f}")
    print(f"所有像素文件总和的标准差: {np.std(all_sums):.2f}")
    print(f"所有像素文件均值的均值: {np.mean(all_means):.4f}")
    print(f"所有像素文件最大值的均值: {np.mean(all_maxs):.2f}")
    print(f"所有像素文件非零值数量的均值: {np.mean(all_nonzeros):.2f}")

    # 可选：输出前10个总和最大的文件
    top_idx = np.argsort(all_sums)[-10:][::-1]
    print("\n总和最大的前10个文件：")
    for idx in top_idx:
        print(f"{os.path.basename(npy_files[idx])}: {all_sums[idx]}")

def copy_with_sum(src_path, dst_dir, value_sum):
    base = os.path.basename(src_path)
    name, ext = os.path.splitext(base)
    new_name = f"{name}_{int(value_sum)}{ext}"
    dst_path = os.path.join(dst_dir, new_name)
    shutil.copy(src_path, dst_path)

def main(data_dir, out_dir_pos, out_dir_neg):
    os.makedirs(out_dir_pos, exist_ok=True)
    os.makedirs(out_dir_neg, exist_ok=True)

    npy_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    positive_files = []
    negative_files = []

    print("筛选正样本（总和>0）和负样本（总和=0）...")
    for f in tqdm(npy_files):
        arr = np.load(f, mmap_mode='r')
        s = arr.sum()
        if s > 0:
            positive_files.append((f, s))
        else:
            negative_files.append((f, s))

    print(f"正样本数量: {len(positive_files)}")
    print(f"负样本数量: {len(negative_files)}")

    # 复制正样本并重命名
    for f, s in tqdm(positive_files, desc="复制正样本"):
        copy_with_sum(f, out_dir_pos, s)

    # 随机抽取与正样本数量相同的负样本
    sampled_neg = random.sample(negative_files, min(len(positive_files), len(negative_files)))
    for f, s in tqdm(sampled_neg, desc="复制负样本"):
        copy_with_sum(f, out_dir_neg, s)

    print("复制并重命名完成！")

if __name__ == "__main__":
    data_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data"  # 原始npy目录
    out_dir_pos = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_selection"
    out_dir_neg = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_selection"
    main(data_dir, out_dir_pos, out_dir_neg)