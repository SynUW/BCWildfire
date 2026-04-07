import os
import shutil
from datetime import datetime, timedelta
import re


def generate_daily_data(input_dir, output_dir):
    """
    将每年的单景MODIS LUIC数据复制为前一年全年每日数据

    参数:
        input_dir (str): 输入目录路径，包含原始年度数据文件
        output_dir (str): 输出目录路径，将保存完整的每日时间序列
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 正则表达式匹配文件名中的日期
    pattern = re.compile(r'clipped_MODIS_LULC_(\d{4})-01-01\.tif$')

    # 获取所有年度数据文件
    annual_files = []
    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            year = int(match.group(1))
            annual_files.append((year, filename))

    # 按年份排序
    annual_files.sort()

    if not annual_files:
        print("错误：输入目录中没有找到符合格式的MODIS LUIC文件")
        return

    # 处理每个年度文件
    for i, (year, filename) in enumerate(annual_files):
        input_path = os.path.join(input_dir, filename)

        # 当前文件对应的前一年
        prev_year = year - 1

        # 为该年度生成前一年的每日数据
        current_date = datetime(prev_year, 1, 1)
        end_date = datetime(prev_year, 12, 31) + timedelta(days=1)  # 包含12月31日

        while current_date < end_date:
            # 生成输出文件名
            output_filename = f"MODIS_LULC_{current_date.strftime('%Y-%m-%d')}.tif"
            output_path = os.path.join(output_dir, output_filename)

            # 检查文件是否已存在（避免重复生成）
            if not os.path.exists(output_path):
                # 复制文件（所有日期都使用当前年份1月1日数据）
                shutil.copy2(input_path, output_path)
                print(f"已生成: {output_filename} (源数据: {filename})")
            else:
                print(f"已跳过: {output_filename} (文件已存在)")

            current_date += timedelta(days=1)


if __name__ == "__main__":
    # 设置路径
    input_directory = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LULC_BCBoundingbox_resampled"
    output_directory = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data/Topo_Distance_WGS84_resize_resampled"

    # 执行处理
    generate_daily_data(input_directory, output_directory)
    print("每日数据生成完成")