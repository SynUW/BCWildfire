import h5py
import numpy as np
from collections import Counter

def check_first_element(h5_file):
    """检查h5文件中时间序列数据之和和标签值的分布"""
    try:
        with h5py.File(h5_file, 'r') as f:
            # 获取所有键
            keys = list(f.keys())
            print(f"文件中的样本总数: {len(keys)}")
            
            # 用于存储统计信息
            time_series_sums = []
            labels = []
            
            # 用于存储特定和的样本信息
            target_sum = 1862775.00
            target_samples = []
            
            # 遍历所有元素
            for key in keys:
                # 获取数据
                data = f[key][:]
                
                # 计算前n-1个数值的和
                time_series_sum = np.sum(data[:-1])
                last_value = data[-1]
                
                time_series_sums.append(time_series_sum)
                labels.append(last_value)
                
                # 检查是否匹配目标和
                if abs(time_series_sum - target_sum) < 0.01:  # 使用小的误差范围来处理浮点数比较
                    target_samples.append((key, last_value))
            
            # 计算时间序列数据之和的统计信息
            print("\n时间序列数据之和的统计信息:")
            print(f"最小值: {np.min(time_series_sums):.2f}")
            print(f"最大值: {np.max(time_series_sums):.2f}")
            print(f"平均值: {np.mean(time_series_sums):.2f}")
            print(f"中位数: {np.median(time_series_sums):.2f}")
            print(f"标准差: {np.std(time_series_sums):.2f}")
            
            # 计算标签值的分布
            label_counts = Counter(labels)
            print("\n标签值的分布:")
            for label, count in sorted(label_counts.items()):
                percentage = (count / len(labels)) * 100
                print(f"标签 {int(label)}: {count} 个样本 ({percentage:.2f}%)")
            
            # 显示特定和的样本信息
            print(f"\n时间序列数据之和为 {target_sum} 的样本:")
            if target_samples:
                print(f"找到 {len(target_samples)} 个匹配的样本")
                for key, label in target_samples:
                    print(f"样本 {key}: 标签值 = {int(label)}")
            else:
                print("未找到匹配的样本")
            
    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == "__main__":
    h5_file = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/samples_2024_sampled/train.h5"
    check_first_element(h5_file)
    h5_file = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/samples_2024_sampled/val.h5"
    check_first_element(h5_file)
    h5_file = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/samples_2024_sampled/test.h5"
    check_first_element(h5_file)  