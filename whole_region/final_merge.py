import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from tqdm import tqdm
from collections import defaultdict

"""
https://chat.deepseek.com/a/chat/s/822910d5-318c-4b79-b86e-1aba49956799
搜索：
根据新需求，我将修改代码以实现按日期合并不同特征（存储在根目录的一级子文件夹中）到多波段TIFF文件中，每个日期对应一个输出文件，包含当日的所有特征数据。
"""

def organize_files_by_date(root_dir):
    """
    组织文件结构为:
    {
        "yyyy_mm_dd": {
            "ndvi": "path/to/yyyy_mm_dd_ndvi.tif",
            "evi": "path/to/yyyy_mm_dd_evi.tif",
            ...
        },
        ...
    }
    """
    date_feature_dict = defaultdict(dict)

    # 只遍历一级子文件夹
    for feature_name in os.listdir(root_dir):
        feature_dir = os.path.join(root_dir, feature_name)
        if not os.path.isdir(feature_dir):
            continue

        # 遍历特征文件夹中的TIFF文件
        for fname in os.listdir(feature_dir):
            if not fname.lower().endswith(('.tif', '.tiff')):
                continue

            # 解析日期和特征名
            try:
                date_part, actual_feature = os.path.splitext(fname)[0].rsplit('_', 1)
                if date_part.count('_') != 2:  # 确保是yyyy_mm_dd格式
                    continue

                # 添加到字典
                full_path = os.path.join(feature_dir, fname)
                date_feature_dict[date_part][actual_feature] = full_path
            except ValueError:
                continue

    return date_feature_dict


def find_common_extent(file_paths):
    """找出所有文件的最小共同范围"""
    min_bounds = None
    min_area = float('inf')

    for file in file_paths:
        with rasterio.open(file) as src:
            bounds = src.bounds
            area = (bounds.right - bounds.left) * (bounds.top - bounds.bottom)
            if area < min_area:
                min_area = area
                min_bounds = bounds

    return min_bounds


def process_date(date_str, feature_dict, output_dir):
    """处理单个日期的所有特征"""
    file_paths = list(feature_dict.values())
    if not file_paths:
        return

    # 1. 确定共同范围
    target_bounds = find_common_extent(file_paths)

    # 2. 获取参考元数据（使用第一个文件）
    with rasterio.open(file_paths[0]) as ref:
        ref_meta = ref.meta.copy()
        ref_meta.update(count=len(file_paths))  # 设置波段数为特征数

    # 3. 初始化输出数组
    out_data = np.zeros((len(file_paths), ref.height, ref.width), dtype=ref.dtypes[0])

    # 4. 处理每个特征
    for i, (feature, path) in enumerate(feature_dict.items()):
        with rasterio.open(path) as src:
            window = from_bounds(*target_bounds, transform=src.transform)
            out_data[i] = src.read(
                1,
                window=window,
                out_shape=(ref_meta['height'], ref_meta['width']),
                resampling=Resampling.bilinear
            )

    # 5. 写入输出文件
    output_path = os.path.join(output_dir, f"{date_str}.tif")
    with rasterio.open(output_path, 'w', **ref_meta) as dst:
        dst.write(out_data)
        # 设置波段名称
        for band_idx, feature_name in enumerate(feature_dict.keys(), 1):
            dst.set_band_description(band_idx, feature_name)

    return output_path


def merge_features_by_date(root_dir, output_dir):
    """主处理函数"""
    # 1. 组织文件结构
    date_feature_dict = organize_files_by_date(root_dir)
    if not date_feature_dict:
        print("未找到符合命名规范的文件!")
        return

    print(f"找到 {len(date_feature_dict)} 个有效日期")

    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 3. 处理每个日期
    success_dates = []
    for date_str, feature_dict in tqdm(date_feature_dict.items(), desc="处理日期"):
        try:
            out_path = process_date(date_str, feature_dict, output_dir)
            success_dates.append((date_str, out_path))
        except Exception as e:
            print(f"处理 {date_str} 失败: {str(e)}")

    # 4. 打印报告
    print("\n处理完成!")
    print(f"成功处理 {len(success_dates)} 个日期:")
    for date, path in success_dates[:5]:  # 只显示前5个示例
        features = list(date_feature_dict[date].keys())
        print(f"- {date}.tif: {len(features)}个特征 ({', '.join(features[:3])}...)")

    if len(success_dates) > 5:
        print(f"...等 {len(success_dates) - 5} 个其他日期")


if __name__ == "__main__":
    # 配置参数
    input_root = "./wildfire_features"  # 包含ndvi/, evi/等子文件夹的根目录
    output_dir = "./output/daily_merged"

    # 执行处理
    merge_features_by_date(input_root, output_dir)