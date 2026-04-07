import os
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path
import random
import pandas as pd

def clean_and_balance_data(input_dir, output_dir, test_dir, seed=42):
    """
    清洗和平衡数据集，使用21年的滑动窗口，每年移动一次
    参数:
        input_dir: 输入数据目录
        output_dir: 训练和验证数据目录
        test_dir: 测试数据目录
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取所有npy文件
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    print(f"找到 {len(npy_files)} 个npy文件")
    
    # 用于存储不同类别的样本
    train_positive_samples = []  # y >= 2 (非2024年)
    train_negative_samples = []  # y = 0 (非2024年)
    test_samples = []  # 2024年的数据
    
    # 计算2000年需要补充的天数（从1月1日到2月23日）
    days_to_pad = 31 + 23  # 1月31天 + 2月23天
    
    # 生成完整日期序列
    full_dates = pd.date_range('2000-01-01', '2024-12-31', freq='D')
    full_dates_str = [d.strftime('%Y_%m_%d') for d in full_dates]
    # 获取firms_detection文件夹下的实际日期序列
    firms_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Firms_Detection'
    actual_files = sorted([f for f in os.listdir(firms_dir) if f.startswith('Firms_') and f.endswith('.tif')])
    actual_dates = [f[6:-4] for f in actual_files]
    # 构建实际日期到npy索引的映射
    date2idx = {date: idx for idx, date in enumerate(actual_dates)}
    def pad_npy_to_full_25years(npy_data):
        padded = np.zeros(len(full_dates_str), dtype=npy_data.dtype)
        for i, date in enumerate(full_dates_str):
            if date in date2idx:
                padded[i] = npy_data[date2idx[date]]
        return padded
    
    # 遍历所有文件
    print("正在读取和分类数据...")
    for file_name in tqdm(npy_files):
        file_path = os.path.join(input_dir, file_name)
        data = np.load(file_path)
        # 补齐为完整25年序列
        data = pad_npy_to_full_25years(data)
        # 在数据前面补充缺失的天数（此步可省略，因已补齐）
        padded_data = data
        
        # 计算每个时间窗口的y值，每年移动一次
        window_size = 365 * 20  # 20年的窗口
        for i in range(0, len(padded_data) - window_size - 365, 365):  # 步长为365天
            x = padded_data[i:i+window_size]  # 前20年
            y = padded_data[i+window_size:i+window_size+365].sum()  # 第21年
            
            # 判断是否是2024年的数据
            # 计算当前窗口的起始年份
            start_year = 2000 + (i // 365)
            end_year = start_year + 20  # 窗口跨度20年
            
            if end_year == 2024:  # 如果窗口结束于2024年，则y是2024年的数据
                test_samples.append((file_name, i, x, y))
            else:
                # 非2024年的数据用于训练和验证
                if y >= 1:
                    train_positive_samples.append((file_name, i, x, y))
                elif y == 0:
                    train_negative_samples.append((file_name, i, x, y))
    
    print(f"\n数据统计:")
    print(f"训练集正样本数量 (y >= 2): {len(train_positive_samples)}")
    print(f"训练集负样本数量 (y = 0): {len(train_negative_samples)}")
    print(f"测试集样本数量: {len(test_samples)}")
    
    # 随机选择与正样本相同数量的负样本
    selected_negative = random.sample(train_negative_samples, len(train_positive_samples))
    print(f"选择的训练集负样本数量: {len(selected_negative)}")
    
    # 合并所有选中的训练样本
    train_samples = train_positive_samples + selected_negative
    random.shuffle(train_samples)  # 随机打乱顺序
    
    # 保存训练和验证数据
    print("\n正在保存训练和验证数据...")
    for idx, (file_name, pos, x, y) in enumerate(tqdm(train_samples)):
        base_name = os.path.splitext(file_name)[0]
        new_file_name = f"{base_name}_{pos}_{idx}.npy"
        save_path = os.path.join(output_dir, new_file_name)
        np.save(save_path, np.concatenate([x, [y]]))

    # ========== 测试集保存 ========== #
    # 1. 完整测试集
    test_dir_full = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_selection_clean_gt1_test_full"
    os.makedirs(test_dir_full, exist_ok=True)
    print("\n正在保存完整测试数据...")
    for idx, (file_name, pos, x, y) in enumerate(tqdm(test_samples)):
        base_name = os.path.splitext(file_name)[0]
        new_file_name = f"{base_name}_{pos}_{idx}.npy"
        save_path = os.path.join(test_dir_full, new_file_name)
        np.save(save_path, np.concatenate([x, [y]]))

    # 2. 均衡采样测试集
    test_dir_balanced = test_dir
    os.makedirs(test_dir_balanced, exist_ok=True)
    test_positive = [s for s in test_samples if s[3] >= 2]
    test_negative = [s for s in test_samples if s[3] == 0]
    num_pos = len(test_positive)
    num_neg = min(len(test_negative), num_pos)
    test_negative_sampled = random.sample(test_negative, num_neg) if num_neg > 0 else []
    test_balanced = test_positive + test_negative_sampled
    random.shuffle(test_balanced)
    print("\n正在保存均衡采样测试数据...")
    for idx, (file_name, pos, x, y) in enumerate(tqdm(test_balanced)):
        base_name = os.path.splitext(file_name)[0]
        new_file_name = f"{base_name}_{pos}_{idx}.npy"
        save_path = os.path.join(test_dir_balanced, new_file_name)
        np.save(save_path, np.concatenate([x, [y]]))
    # ========== END ========== #

    print(f"\n处理完成！")
    print(f"训练集总样本数: {len(train_samples)}")
    print(f"训练集正样本数: {len(train_positive_samples)}")
    print(f"训练集负样本数: {len(selected_negative)}")
    print(f"测试集样本数: {len(test_samples)}")
    print(f"训练和验证数据已保存到: {output_dir}")
    print(f"测试数据已保存到: {test_dir}")

if __name__ == "__main__":
    # 设置输入输出路径
    input_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_selection"
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_selection_gt1_balanced_without_test"
    test_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_selection_gt1_balanced_test"
    
    # 执行数据清洗和平衡
    clean_and_balance_data(input_dir, output_dir, test_dir) 