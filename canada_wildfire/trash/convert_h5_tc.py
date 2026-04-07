import os
import h5py
import numpy as np
from tqdm import tqdm

def convert_h5_tc(input_h5_path, output_h5_path):
    """
    将 h5 文件从 t, c, h, w 转换为 t, c。
    对于每个 t, c，取空间维度非0的中值；对于第一个 c（c=0），所有 t 使用该 t 的最大值。
    保留所有 key，数据为 float。
    """
    with h5py.File(input_h5_path, 'r') as f_in:
        with h5py.File(output_h5_path, 'w') as f_out:
            for key in tqdm(f_in.keys(), desc="处理数据集"):
                data = f_in[key][:]
                if len(data.shape) != 4:
                    print(f"警告：数据集 {key} 的 shape 不是 t, c, h, w，跳过处理。")
                    f_out.create_dataset(key, data=data)
                    continue
                t, c, h, w = data.shape
                new_data = np.zeros((t, c), dtype=np.float32)
                for i in range(t):
                    for j in range(c):
                        if j == 0:
                            # 对于第一个 c，取该 t 的最大值
                            new_data[i, j] = np.max(data[i, j])
                        else:
                            # 对于其他 c，取非0的中值
                            non_zero = data[i, j][data[i, j] != 0]
                            if non_zero.size > 0:
                                new_data[i, j] = np.median(non_zero)
                            else:
                                new_data[i, j] = 0
                f_out.create_dataset(key, data=new_data)
    print(f"转换完成，输出文件：{output_h5_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="将 h5 文件从 t, c, h, w 转换为 t, c")
    parser.add_argument("--input_h5", dest="input_h5", default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/samples_merged_normalized/train_2/wildfire_samples_2024_merged.h5", help="输入 h5 文件路径")
    parser.add_argument("--output_h5", dest="output_h5", default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/samples_merged_normalized/train_2_series", help="输出 h5 文件路径")
    args = parser.parse_args()
    convert_h5_tc(args.input_h5, args.output_h5)