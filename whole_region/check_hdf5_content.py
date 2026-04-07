import os
import h5py
import random
import argparse

def check_hdf5_content(hdf5_dir):
    """
    随机选择一个HDF5文件并显示其内容
    """
    # 获取所有HDF5文件
    hdf5_files = [f for f in os.listdir(hdf5_dir) if f.endswith('.h5')]
    
    if not hdf5_files:
        print(f"警告：在 {hdf5_dir} 中未找到HDF5文件")
        return
    
    # 随机选择一个文件
    selected_file = random.choice(hdf5_files)
    print(f"\n选中的文件: {selected_file}")
    
    # 读取文件内容
    file_path = os.path.join(hdf5_dir, selected_file)
    try:
        with h5py.File(file_path, 'r') as f:
            # 获取所有键
            keys = list(f.keys())
            total_keys = len(keys)
            print(f"\n文件中共有 {total_keys} 个数据集")
            
            # 显示前10个键
            print("\n前10个数据集的名称:")
            for i, key in enumerate(keys[:10]):
                print(f"{i+1}. {key}")
            
            # 如果数据集超过10个，显示最后一个键
            if total_keys > 10:
                print(f"\n... 省略 {total_keys-11} 个数据集 ...")
                print(f"最后一个数据集: {keys[-1]}")
            
            # 随机选择一个数据集显示其形状
            random_key = random.choice(keys)
            data = f[random_key][:]
            print(f"\n随机选择的数据集 '{random_key}' 的形状: {data.shape}")
            
    except Exception as e:
        print(f"读取文件时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='查看HDF5文件内容')
    parser.add_argument('--hdf5_dir', type=str, 
                       default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/merged_hdf5",
                       help='HDF5文件所在的目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_dir):
        print(f"错误：目录不存在: {args.hdf5_dir}")
        return
    
    check_hdf5_content(args.hdf5_dir)

if __name__ == "__main__":
    main() 