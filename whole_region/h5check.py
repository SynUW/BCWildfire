import os
import h5py
import numpy as np
from tqdm import tqdm

def get_h5_info(file_path):
    """获取HDF5文件的信息"""
    with h5py.File(file_path, 'r') as f:
        # 获取数据集信息
        X = f['X']
        y = f['y']
        
        # 获取形状和数据类型
        X_shape = X.shape
        y_shape = y.shape
        X_dtype = X.dtype
        y_dtype = y.dtype
        
        # 计算文件大小（MB）
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        
        return {
            'file_size_mb': file_size,
            'X_shape': X_shape,
            'y_shape': y_shape,
            'X_dtype': X_dtype,
            'y_dtype': y_dtype
        }

def main():
    # 设置HDF5文件目录
    h5_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_split_hdf5"
    
    # 获取所有h5文件
    h5_files = sorted([f for f in os.listdir(h5_dir) if f.endswith('.hdf5')])
    print(f"找到 {len(h5_files)} 个HDF5文件")
    
    # 统计信息
    total_size = 0
    total_samples = 0
    
    # 检查每个文件
    for file_name in tqdm(h5_files, desc="检查文件"):
        file_path = os.path.join(h5_dir, file_name)
        info = get_h5_info(file_path)
        
        total_size += info['file_size_mb']
        total_samples += info['X_shape'][0]
        
        print(f"\n文件: {file_name}")
        print(f"文件大小: {info['file_size_mb']:.2f} MB")
        print(f"样本数量: {info['X_shape'][0]}")
        print(f"X形状: {info['X_shape']}")
        print(f"y形状: {info['y_shape']}")
        print(f"X数据类型: {info['X_dtype']}")
        print(f"y数据类型: {info['y_dtype']}")
        break
    
    print("\n总体统计:")
    print(f"总文件数: {len(h5_files)}")
    print(f"总大小: {total_size:.2f} MB")
    print(f"总样本数: {total_samples}")
    print(f"平均每个文件大小: {total_size/len(h5_files):.2f} MB")
    print(f"平均每个样本大小: {total_size*1024*1024/total_samples:.2f} bytes")

if __name__ == "__main__":
    main()
