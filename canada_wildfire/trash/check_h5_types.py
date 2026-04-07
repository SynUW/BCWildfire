#!/usr/bin/env python3
import h5py
import sys

def check_h5_types(file_path):
    """检查HDF5文件中的对象类型"""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"检查文件: {file_path}")
            print(f"总对象数: {len(f.keys())}")
            
            # 检查前20个对象
            keys = list(f.keys())[:20]
            for key in keys:
                obj = f[key]
                print(f"  {key}: {type(obj)}")
                
            # 检查是否有group类型
            groups = []
            datasets = []
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Group):
                    groups.append(key)
                elif isinstance(obj, h5py.Dataset):
                    datasets.append(key)
                    
            print(f"\n统计:")
            print(f"  Group数量: {len(groups)}")
            print(f"  Dataset数量: {len(datasets)}")
            
            if groups:
                print(f"  Group列表: {groups[:10]}")  # 只显示前10个
                
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    file_path = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples/2024_0713_0719_ERA5_multi_bands_full.h5"
    check_h5_types(file_path) 