import os
import re
from tqdm import tqdm


def rename_era5_files(folder_path):
    """
    将文件夹中的yyyy_mm_dd.tif文件重命名为ERA5_yyyy_mm_dd.tif格式

    参数:
        folder_path (str): 目标文件夹路径
    """
    # 编译正则表达式匹配日期格式
    date_pattern = re.compile(r'^(\d{4})_(\d{2})_(\d{2})\.tif$')

    # 获取文件夹中所有文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

    # 用于统计重命名数量
    renamed_count = 0

    for filename in tqdm(files, desc="重命名文件中"):
        # 检查是否已经是ERA5_开头
        if filename.startswith('ERA5_'):
            continue

        # 尝试匹配日期格式
        match = date_pattern.match(filename)
        if match:
            # 构造新文件名
            new_name = f"ERA5_{filename}"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            # 执行重命名
            try:
                os.rename(old_path, new_path)
                renamed_count += 1
                print(f"已重命名: {filename} -> {new_name}")
            except Exception as e:
                print(f"重命名 {filename} 失败: {e}")

    print(f"\n完成! 共重命名了 {renamed_count} 个文件")


if __name__ == "__main__":
    # 设置你的文件夹路径
    target_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/DAILY_500_clip"  # 替换为你的实际文件夹路径

    # 执行重命名操作
    rename_era5_files(target_folder)