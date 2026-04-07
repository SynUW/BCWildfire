import os
import h5py
import numpy as np
import logging
from tqdm import tqdm


def check_normalized_range(h5_root_dir):
    """递归检查所有h5文件中所有数据集的值是否都在0-1之间，只打印异常"""
    for root, _, files in os.walk(h5_root_dir):
        for file in files:
            if file.endswith('.h5'):
                h5_path = os.path.join(root, file)
                with h5py.File(h5_path, 'r') as f:
                    for dset_name in f.keys():
                        data = f[dset_name][:]
                        if np.any(data < 0) or np.any(data > 1):
                            print(f"异常: {h5_path} 数据集 {dset_name} 存在超出[0,1]范围的值，min={data.min()}, max={data.max()}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # 新增归一化范围检查
    norm_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/samples_merged_normalized'  # 可根据实际情况修改
    logger.info(f"开始检查归一化后数据是否在[0,1]范围: {norm_dir}")
    check_normalized_range(norm_dir)
    logger.info("归一化范围检查完成")

if __name__ == "__main__":
    main()
