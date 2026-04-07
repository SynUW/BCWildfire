import os
import shutil
from datetime import datetime, timedelta
import re


def generate_full_year_data(input_dir, output_dir):
    """
    将每年的单景MODIS LUIC数据复制为全年每月数据

    参数:
        input_dir (str): 输入目录路径，包含原始年度数据文件
        output_dir (str): 输出目录路径，将保存完整的月度时间序列
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 正则表达式匹配文件名中的日期
    pattern = re.compile(r'MODIS_LUIC_(\d{4})-01-01\.tif$')

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

        # 确定下一年份用于12月数据的填补
        next_year = annual_files[i + 1][0] if i + 1 < len(annual_files) else year

        # 为该年度生成每月数据
        for month in range(1, 13):
            # 生成目标日期
            if month < 12:
                target_date = datetime(year, month, 1)
            else:
                # 12月使用下一年度的1月数据（如果存在）
                target_date = datetime(next_year, 1, 1)

            # 生成输出文件名
            output_filename = f"MODIS_LUIC_{target_date.strftime('%Y-%m-%d')}.tif"
            output_path = os.path.join(output_dir, output_filename)

            # 复制文件
            shutil.copy2(input_path, output_path)
            print(f"已生成: {output_filename} (源数据: {filename})")


if __name__ == "__main__":
    # 设置路径
    input_directory = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LULC_500"  # 替换为你的输入目录
    output_directory = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LULC_500_interpolated"  # 替换为输出目录

    # 执行处理
    generate_full_year_data(input_directory, output_directory)
    print("全年数据生成完成")