"""
用于生成验证和测试数据，数据转为20年数据，并进行均衡采样
要求负样本全部来自图像右半部分
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import shutil
import gc
import multiprocessing
import psutil

# ========== 配置 ========== #
INPUT_DIR = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data"
OUTPUT_DIR_GT1 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_gt1_balanced_without_test"
OUTPUT_DIR_GT2 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_gt2_balanced_without_test"
TEST_DIR_GT1 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_gt1_balanced_test"
TEST_DIR_GT2 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_gt2_balanced_test"
TEST_DIR_FULL = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_test_full"
FIRMS_DIR = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Firms_Detection"
SEED = 42
MAX_WORKERS = 48  # 留出部分CPU核心给系统
BATCH_SIZE = 10000  # 每批处理的文件数
CHUNK_SIZE = 2000  # 分块大小

# 图像尺寸信息
IMAGE_WIDTH = 5565  # 图像总宽度
RIGHT_HALF_START = IMAGE_WIDTH // 2  # 右半部分起始x坐标

random.seed(SEED)
np.random.seed(SEED)

def print_memory_usage():
    """打印当前内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"当前进程内存使用: {memory_info.rss / 1024 / 1024 / 1024:.2f} GB")

def print_system_memory():
    """打印系统内存使用情况"""
    memory = psutil.virtual_memory()
    print(f"系统内存使用: {memory.used / 1024 / 1024 / 1024:.2f} GB / {memory.total / 1024 / 1024 / 1024:.2f} GB ({memory.percent}%)")

# 创建输出目录
for dir_path in [OUTPUT_DIR_GT1, OUTPUT_DIR_GT2, TEST_DIR_GT1, TEST_DIR_GT2, TEST_DIR_FULL]:
    os.makedirs(dir_path, exist_ok=True)

# ========== 日期补齐相关 ========== #
print("正在加载日期信息...")
full_dates = pd.date_range('2000-01-01', '2024-12-31', freq='D')
full_dates_str = [d.strftime('%Y_%m_%d') for d in full_dates]
actual_files = sorted([f for f in os.listdir(FIRMS_DIR) if f.startswith('Firms_') and f.endswith('.tif')])
actual_dates = [f[6:-4] for f in actual_files]
date2idx = {date: idx for idx, date in enumerate(actual_dates)}

def pad_npy_to_full_25years(npy_data):
    padded = np.zeros(len(full_dates_str), dtype=npy_data.dtype)
    for i, date in enumerate(full_dates_str):
        if date in date2idx:
            padded[i] = npy_data[date2idx[date]]
    return padded

# ========== 检查是否在右半部分 ========== #
def is_in_right_half(file_name):
    """检查文件对应的坐标是否在图像右半部分"""
    try:
        base_name = os.path.splitext(file_name)[0]
        parts = base_name.split('_')
        if len(parts) >= 2:
            x = int(parts[0])  # x坐标对应图像高度方向
            y = int(parts[1])  # y坐标对应图像宽度方向
            return y >= RIGHT_HALF_START
    except:
        pass
    return False

# ========== 从文件名获取y值 ========== #
def get_y_from_filename(file_name):
    """从文件名中获取y值"""
    try:
        base_name = os.path.splitext(file_name)[0]
        parts = base_name.split('_')
        if len(parts) >= 2:
            return int(parts[1])  # 返回宽度值
    except:
        pass
    return 0

# ========== 并行处理单个文件 ========== #
def process_one_file(file_name):
    file_path = os.path.join(INPUT_DIR, file_name)
    try:
        # 分块读取
        data = np.load(file_path, mmap_mode='r')  # 使用内存映射
        data = pad_npy_to_full_25years(data)
        # 滑窗处理
        window_size = 365 * 20
        samples = []
        for i in range(0, len(data) - window_size - 365, 365):
            x = data[i:i+window_size].copy()  # 复制数据避免内存映射问题
            y = data[i+window_size:i+window_size+365].sum()
            start_year = 2000 + (i // 365)
            end_year = start_year + 20
            if end_year == 2024:
                # 测试集样本
                samples.append(('test', file_name, i, x, y, start_year))
            else:
                # 训练/验证样本
                if y >= 1:  # 只使用阈值1
                    samples.append(('pos', file_name, i, x, y, start_year))
                elif y == 0:
                    samples.append(('neg', file_name, i, x, y, start_year))
        del data  # 释放内存
        print(f"文件 {file_name} 处理完成，得到 {len(samples)} 个样本")
        return samples
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")
        return []

# ========== 批量保存数据 ========== #
def batch_save_samples(samples, output_dir, desc="保存数据"):
    """批量保存样本数据"""
    if len(samples) == 0:
        print(f"警告: {desc} 没有样本需要保存")
        return
        
    print(f"\n开始{desc}，总样本数: {len(samples)}")
    # 按批次保存
    for i in range(0, len(samples), CHUNK_SIZE):
        batch = samples[i:i+CHUNK_SIZE]
        print(f"\n处理批次 {i//CHUNK_SIZE + 1}，样本数: {len(batch)}")
        print_memory_usage()  # 打印批次开始时的内存使用
        print_system_memory()  # 打印系统内存使用
        
        saved_count = 0
        for idx, (_, file_name, pos, x, y, year) in enumerate(tqdm(batch, desc=f"{desc} (批次 {i//CHUNK_SIZE + 1})")):
            try:
                base_name = os.path.splitext(file_name)[0]
                parts = base_name.split('_')
                height = parts[0] if len(parts) > 0 else "unknown"
                width = parts[1] if len(parts) > 1 else "unknown"
                new_file_name = f"{base_name}_year_{year}_h_{height}_w_{width}.npy"
                save_path = os.path.join(output_dir, new_file_name)
                np.save(save_path, np.concatenate([x, [y]]))
                saved_count += 1
                del x  # 释放内存
            except Exception as e:
                print(f"保存样本时出错: {e}")
        
        print(f"批次 {i//CHUNK_SIZE + 1} 成功保存 {saved_count} 个样本")
        print_memory_usage()  # 打印批次结束时的内存使用
        print_system_memory()  # 打印系统内存使用
        gc.collect()  # 每批次后清理内存

# ========== 主流程 ========== #
def main():
    print("开始处理数据...")
    print_memory_usage()  # 打印初始内存使用
    print_system_memory()  # 打印系统内存使用
    
    npy_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.npy')]
    print(f"找到 {len(npy_files)} 个npy文件")
    
    # 分批处理文件
    all_pos, all_neg, all_test = [], [], []
    for i in range(0, len(npy_files), BATCH_SIZE):
        batch_files = npy_files[i:i+BATCH_SIZE]
        print(f"\n处理第 {i//BATCH_SIZE + 1} 批文件 ({len(batch_files)} 个文件)...")
        print_memory_usage()  # 打印每批开始时的内存使用
        print_system_memory()  # 打印系统内存使用
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_one_file, file_name) for file_name in batch_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
                samples = future.result()
                for s in samples:
                    if s[0] == 'pos':
                        all_pos.append(s)
                    elif s[0] == 'neg':
                        all_neg.append(s)
                    elif s[0] == 'test':
                        all_test.append(s)
        
        print(f"当前批次处理完成，累计正样本: {len(all_pos)}，负样本: {len(all_neg)}，测试集: {len(all_test)}")
        print_memory_usage()  # 打印每批结束时的内存使用
        print_system_memory()  # 打印系统内存使用
        gc.collect()  # 清理内存
    
    print(f"\n所有文件处理完成！")
    print(f"正样本总数: {len(all_pos)}")
    print(f"负样本总数: {len(all_neg)}")
    print(f"测试集总数: {len(all_test)}")

    # 处理阈值1的样本
    print("\n正在处理阈值1的样本...")
    right_half_neg = [s for s in all_neg if is_in_right_half(s[1])]
    print(f"右半部分负样本数: {len(right_half_neg)}")
    num_pos_gt1 = len(all_pos)
    num_neg_gt1 = min(len(right_half_neg), num_pos_gt1)
    print(f"阈值1正样本数: {num_pos_gt1}，负样本数: {num_neg_gt1}")
    selected_neg_gt1 = random.sample(right_half_neg, num_neg_gt1) if num_neg_gt1 > 0 else []
    train_samples_gt1 = all_pos + selected_neg_gt1
    random.shuffle(train_samples_gt1)
    print(f"阈值1训练集总样本数: {len(train_samples_gt1)}")

    # 保存阈值1的训练/验证数据
    batch_save_samples(train_samples_gt1, OUTPUT_DIR_GT1, "保存阈值1训练数据")
    del train_samples_gt1, selected_neg_gt1, right_half_neg  # 释放内存

    # 从阈值1的目录中筛选阈值2的样本
    print("\n正在处理阈值2的样本...")
    gt1_files = os.listdir(OUTPUT_DIR_GT1)
    print(f"阈值1目录中的文件数: {len(gt1_files)}")
    pos_gt2_files = []
    for file_name in gt1_files:
        y_value = get_y_from_filename(file_name)
        if y_value >= 2:
            pos_gt2_files.append(file_name)
    print(f"阈值2正样本数: {len(pos_gt2_files)}")
    
    # 从右半部分选择负样本
    neg_gt2_files = []
    for file_name in gt1_files:
        y_value = get_y_from_filename(file_name)
        if y_value == 0 and is_in_right_half(file_name):
            neg_gt2_files.append(file_name)
    print(f"阈值2右半部分负样本数: {len(neg_gt2_files)}")
    
    # 随机选择负样本
    num_pos_gt2 = len(pos_gt2_files)
    num_neg_gt2 = min(len(neg_gt2_files), num_pos_gt2)
    selected_neg_gt2_files = random.sample(neg_gt2_files, num_neg_gt2) if num_neg_gt2 > 0 else []
    print(f"阈值2最终负样本数: {len(selected_neg_gt2_files)}")
    
    # 复制文件到阈值2目录
    print("\n正在复制阈值2的训练和验证数据...")
    for file_name in tqdm(pos_gt2_files + selected_neg_gt2_files, desc="复制阈值2数据"):
        src_path = os.path.join(OUTPUT_DIR_GT1, file_name)
        dst_path = os.path.join(OUTPUT_DIR_GT2, file_name)
        shutil.copy2(src_path, dst_path)

    # 保存完整测试集
    print("\n正在保存完整测试数据...")
    batch_save_samples(all_test, TEST_DIR_FULL, "保存完整测试数据")

    # 均衡采样测试集 - 分别处理阈值1和2
    print("\n正在均衡采样测试集...")
    # 阈值1的测试集
    test_pos_gt1 = [s for s in all_test if s[4] >= 1]
    test_neg_gt1 = [s for s in all_test if s[4] == 0 and is_in_right_half(s[1])]
    num_test_pos_gt1 = len(test_pos_gt1)
    num_test_neg_gt1 = min(len(test_neg_gt1), num_test_pos_gt1)
    print(f"阈值1测试集正样本数: {num_test_pos_gt1}，负样本数: {num_test_neg_gt1}")
    test_neg_sampled_gt1 = random.sample(test_neg_gt1, num_test_neg_gt1) if num_test_neg_gt1 > 0 else []
    test_balanced_gt1 = test_pos_gt1 + test_neg_sampled_gt1
    random.shuffle(test_balanced_gt1)
    print(f"阈值1测试集总样本数: {len(test_balanced_gt1)}")

    # 阈值2的测试集
    test_pos_gt2 = [s for s in all_test if s[4] >= 2]
    test_neg_gt2 = [s for s in all_test if s[4] == 0 and is_in_right_half(s[1])]
    num_test_pos_gt2 = len(test_pos_gt2)
    num_test_neg_gt2 = min(len(test_neg_gt2), num_test_pos_gt2)
    print(f"阈值2测试集正样本数: {num_test_pos_gt2}，负样本数: {num_test_neg_gt2}")
    test_neg_sampled_gt2 = random.sample(test_neg_gt2, num_test_neg_gt2) if num_test_neg_gt2 > 0 else []
    test_balanced_gt2 = test_pos_gt2 + test_neg_sampled_gt2
    random.shuffle(test_balanced_gt2)
    print(f"阈值2测试集总样本数: {len(test_balanced_gt2)}")
    
    # 保存均衡测试数据
    batch_save_samples(test_balanced_gt1, TEST_DIR_GT1, "保存阈值1均衡测试数据")
    batch_save_samples(test_balanced_gt2, TEST_DIR_GT2, "保存阈值2均衡测试数据")

    print("\n处理完成！")
    print(f"阈值1训练集正样本数: {len(all_pos)}")
    print(f"阈值1训练集负样本数: {len(selected_neg_gt1)}")
    print(f"阈值2训练集正样本数: {len(pos_gt2_files)}")
    print(f"阈值2训练集负样本数: {len(selected_neg_gt2_files)}")
    print(f"测试集样本数: {len(all_test)}")
    print(f"阈值1测试集均衡采样数: {len(test_balanced_gt1)}")
    print(f"阈值2测试集均衡采样数: {len(test_balanced_gt2)}")
    print(f"阈值1训练和验证数据已保存到: {OUTPUT_DIR_GT1}")
    print(f"阈值2训练和验证数据已保存到: {OUTPUT_DIR_GT2}")
    print(f"阈值1均衡采样测试数据已保存到: {TEST_DIR_GT1}")
    print(f"阈值2均衡采样测试数据已保存到: {TEST_DIR_GT2}")
    print(f"完整测试数据已保存到: {TEST_DIR_FULL}")

if __name__ == "__main__":
    main()