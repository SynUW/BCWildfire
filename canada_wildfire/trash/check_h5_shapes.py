import os
import h5py
import argparse
import shutil
from typing import Dict, List, Tuple


def copy_attributes(source, target):
    """复制属性"""
    for key, value in source.attrs.items():
        target.attrs[key] = value


def get_dataset_shapes(file_path: str) -> Dict[str, Tuple]:
    """
    递归获取 h5 文件内所有数据集的 shape，并创建新文件只保留符合要求的数据集。
    :param file_path: h5 文件路径
    :return: 字典，键为数据集路径，值为 shape 元组
    """
    shapes = {}
    temp_file = file_path + '.temp'
    
    with h5py.File(file_path, 'r') as source:
        with h5py.File(temp_file, 'w') as target:
            # 复制文件级别的属性
            copy_attributes(source, target)
            
            def visit(name, obj):
                if isinstance(obj, h5py.Group):
                    # 创建组并复制其属性
                    if name not in target:
                        grp = target.create_group(name)
                        copy_attributes(obj, grp)
                elif isinstance(obj, h5py.Dataset):
                    shape = obj.shape
                    if shape in [(7, 39, 10, 10), (10, 39, 10, 10)]:
                        # 复制符合要求的数据集
                        source.copy(name, target)
                        shapes[name] = shape
                    else:
                        print(f"  跳过不符合要求的数据集: {name}, shape: {shape}")
            
            source.visititems(visit)
    
    os.remove(file_path)
    os.rename(temp_file, file_path)
    return shapes


def check_shapes_consistency(shapes: Dict[str, Tuple]) -> bool:
    """
    检查所有数据集的 shape 是否一致。
    :param shapes: 数据集 shape 字典
    :return: 是否一致
    """
    if not shapes:
        return True
    first_shape = next(iter(shapes.values()))
    return all(shape == first_shape for shape in shapes.values())


def main():
    parser = argparse.ArgumentParser(description='检查 h5 文件内数据集 shape 是否一致')
    parser.add_argument('--dir', type=str, default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/samples_merged_normalized/test',
                        help='目标文件夹路径')
    args = parser.parse_args()

    target_dir = args.dir
    if not os.path.isdir(target_dir):
        print(f"错误：{target_dir} 不是有效目录")
        return

    h5_files = [f for f in os.listdir(target_dir) if f.endswith(('.h5', '.hdf5'))]
    if not h5_files:
        print(f"在 {target_dir} 中未找到 h5 文件")
        return

    for h5_file in h5_files:
        file_path = os.path.join(target_dir, h5_file)
        print(f"\n检查文件: {h5_file}")
        try:
            shapes = get_dataset_shapes(file_path)
            if not shapes:
                print("  文件内没有数据集")
                continue
            if check_shapes_consistency(shapes):
                print("  所有数据集 shape 一致:", next(iter(shapes.values())))
            else:
                print("  数据集 shape 不一致:")
                for name, shape in shapes.items():
                    if shape not in [(7, 39, 10, 10), (10, 39, 10, 10)]:
                        print(f"    {name}: {shape}")
        except Exception as e:
            print(f"  检查文件时出错: {e}")


if __name__ == '__main__':
    main() 