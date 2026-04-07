import os
import h5py
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import concurrent.futures
import shutil

def normalize_data(data, norm_params):
    """对数据进行归一化处理：
    1. 第一个波段不归一化
    2. 其他波段根据参数进行归一化
    3. 对于有ignore value的波段，先clip再归一化，最后将ignore value设为0
    4. 对于没有ignore value的波段，直接归一化
    """
    # data: t, c, h, w
    normalized = np.zeros_like(data, dtype=np.float32)
    t, c, h, w = data.shape
    for band_idx in range(c):
        params = norm_params[band_idx]
        band_data = data[:, band_idx, :, :]
        ignore_val = params.get('ignore', None)
        # mask: 有效值（不是ignore且是有限数）
        if ignore_val is not None:
            mask = (band_data != ignore_val) & np.isfinite(band_data)
        else:
            mask = np.isfinite(band_data)
        if np.any(mask):
            valid_data = np.clip(band_data[mask], params['min'], params['max'])
            normalized[:, band_idx, :, :][mask] = (valid_data - params['min']) / (params['max'] - params['min'])
        # 其余位置自动为0
    return normalized

def process_h5_file(h5_path, norm_params, output_dir):
    """处理单个h5文件，对所有数据集进行归一化并保存到新文件夹"""
    try:
        # 创建输出文件路径，保持train/val/test的目录结构
        rel_path = os.path.relpath(h5_path, os.path.dirname(os.path.dirname(h5_path)))
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with h5py.File(h5_path, 'r') as f_in:
            with h5py.File(output_path, 'w') as f_out:
                # 遍历所有数据集
                for key in f_in.keys():
                    data = f_in[key][:]  # 读取数据，形状为 t*39*h*w
                    normalized = normalize_data(data, norm_params)
                    # 创建新的归一化数据集
                    f_out.create_dataset(key, data=normalized)
        return True
    except Exception as e:
        print(f"处理文件 {h5_path} 时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='使用ignore.xlsx中的参数对h5文件进行归一化')
    parser.add_argument('--h5_dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/samples_merged_right', type=str, required=False, help='h5文件所在目录')
    parser.add_argument('--ignore_xlsx', default='ignore.xlsx', type=str, help='归一化参数文件路径')
    parser.add_argument('--output_dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/samples_merged_right_normalized', type=str, required=False, help='归一化后的h5文件保存目录')
    args = parser.parse_args()

    # 读取归一化参数
    try:
        df = pd.read_excel(args.ignore_xlsx)
        norm_params = []
        for _, row in df.iterrows():
            params = {
                'min': row['min_value'],
                'max': row['max_value']
            }
            # 如果存在ignore_value列且值不为空，则添加到参数中
            if 'ignore_value' in row and pd.notna(row['ignore']):
                params['ignore'] = row['ignore']
            norm_params.append(params)
    except Exception as e:
        print(f"读取归一化参数文件出错: {e}")
        return

    # 获取所有h5文件，保持train/val/test的目录结构
    h5_files = []
    for dataset in ['train', 'val', 'test']:
        dataset_dir = os.path.join(args.h5_dir, dataset)
        if os.path.exists(dataset_dir):
            for root, _, files in os.walk(dataset_dir):
                for file in files:
                    if file.endswith('.h5'):
                        h5_files.append(os.path.join(root, file))
    
    # 并行处理h5文件
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_h5_file, h5_path, norm_params, args.output_dir) for h5_path in h5_files]
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="归一化处理h5文件"):
            results.append(future.result())
        success_count = sum(results)

    # 统计处理结果
    print(f"\n处理完成:")
    print(f"成功: {success_count}")
    print(f"失败: {len(results) - success_count}")

if __name__ == "__main__":
    main()
