import os
import json
import re
from collections import defaultdict

def extract_info_from_key(key):
    """从文件名中提取位置、年份和y值"""
    # 匹配形如 1366_5208_year_2000_y_0.npy 的文件名
    match = re.search(r'(\d+)_(\d+)_year_(\d+)_y_([\d.]+)\.npy', key)
    if match:
        pos_x = int(match.group(1))  # 第一个位置数字
        pos_y = int(match.group(2))  # 第二个位置数字
        year = int(match.group(3))   # 年份
        y_value = float(match.group(4))  # y值
        return pos_x, pos_y, year, y_value
    return None, None, None, None

def is_valid_negative_sample(pos_x, pos_y):
    """判断负样本的位置是否有效（x或y大于2700）"""
    return pos_x > 2700 or pos_y > 2700

def main():
    # 设置JSON文件路径
    json_path = os.path.join("dataset_lists", "all_datasets.json")
    
    if not os.path.exists(json_path):
        print(f"错误：找不到数据集列表文件: {json_path}")
        return
    
    # 加载数据集
    with open(json_path, 'r') as f:
        all_datasets = json.load(f)
    
    # 打印JSON文件的基本结构
    print("\n=== JSON文件结构 ===")
    print(f"JSON文件中的键: {list(all_datasets.keys())}")
    if 'train' in all_datasets:
        print(f"训练集中的键: {list(all_datasets['train'].keys())}")
        print(f"训练集正样本数量: {len(all_datasets['train']['positive'])}")
        print(f"训练集负样本数量: {len(all_datasets['train']['negative'])}")
        print(f"测试集样本数量: {len(all_datasets['test'])}")
    
    # 获取原始数据
    train_positive = all_datasets['train']['positive']
    train_negative = all_datasets['train']['negative']
    test_datasets = all_datasets['test']
    
    # 限制样本数量
    max_samples = 100000  # 最多读取100万个样本
    print(f"\n限制每个类别最多读取 {max_samples} 个样本")
    
    train_positive = train_positive[:max_samples]
    train_negative = train_negative[:max_samples]
    test_datasets = test_datasets[:max_samples]
    
    # 打印原始数据的样本信息
    print("\n=== 原始数据样本信息 ===")
    if len(train_positive) > 0:
        print(f"正样本示例: {train_positive[0]}")
        print(f"正样本类型: {type(train_positive[0])}")
    else:
        print("警告：没有找到正样本")
    
    if len(train_negative) > 0:
        print(f"负样本示例: {train_negative[0]}")
        print(f"负样本类型: {type(train_negative[0])}")
    else:
        print("警告：没有找到负样本")
    
    if len(test_datasets) > 0:
        print(f"测试集示例: {test_datasets[0]}")
        print(f"测试集类型: {type(test_datasets[0])}")
    else:
        print("警告：没有找到测试集样本")
    
    # 按年份统计正样本
    positive_by_year = defaultdict(int)
    for sample in train_positive:
        key = sample['key'] if isinstance(sample, dict) else sample
        _, _, year, _ = extract_info_from_key(key)
        if year is not None:
            positive_by_year[year] += 1
    
    # 按年份统计负样本（包括位置验证）
    negative_by_year = defaultdict(int)
    valid_negative = []
    for sample in train_negative:
        key = sample['key'] if isinstance(sample, dict) else sample
        pos_x, pos_y, year, _ = extract_info_from_key(key)
        if year is not None and is_valid_negative_sample(pos_x, pos_y):
            negative_by_year[year] += 1
            valid_negative.append(sample)
    
    # 按年份统计测试集
    test_by_year = defaultdict(int)
    for sample in test_datasets:
        key = sample['key'] if isinstance(sample, dict) else sample
        _, _, year, _ = extract_info_from_key(key)
        if year is not None:
            test_by_year[year] += 1
    
    # 打印统计信息
    print("\n=== 数据集统计信息 ===")
    print(f"\n原始训练集正样本数量: {len(train_positive)}")
    print(f"原始训练集负样本数量: {len(train_negative)}")
    print(f"测试集样本数量: {len(test_datasets)}")
    
    print("\n=== 按年份统计训练集正样本 ===")
    for year in sorted(positive_by_year.keys()):
        print(f"{year}年: {positive_by_year[year]}个样本")
    
    print("\n=== 按年份统计训练集负样本（经过位置验证）===")
    for year in sorted(negative_by_year.keys()):
        print(f"{year}年: {negative_by_year[year]}个样本")
    
    print("\n=== 按年份统计测试集 ===")
    for year in sorted(test_by_year.keys()):
        print(f"{year}年: {test_by_year[year]}个样本")
    
    # 计算最小样本数（用于均衡采样）
    min_samples = min(len(train_positive), len(valid_negative))
    print(f"\n均衡采样后的样本数: {min_samples}（正样本） + {min_samples}（负样本） = {min_samples * 2}")

if __name__ == "__main__":
    main()