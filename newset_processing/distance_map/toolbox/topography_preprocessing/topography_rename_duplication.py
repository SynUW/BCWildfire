"""
将裁剪后的topograhpy文件复制并重命名为每一天的文件
"""
import os
import shutil
from datetime import datetime, timedelta
import glob
import re


def parse_filename(filename):
    """解析文件名中的日期信息"""
    # 移除扩展名
    basename = os.path.splitext(filename)[0]

    # 使用正则表达式匹配文件名中的日期部分
    match = re.match(r'(\d{4})_(\d+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})', basename)
    if not match:
        raise ValueError(f"文件名格式不正确: {filename}")

    year = match.group(1)
    fire_code = match.group(2)
    start_date_str = match.group(3)
    end_date_str = match.group(4)

    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"日期解析错误: {str(e)}")

    return f"{year}_{fire_code}", start_date, end_date


def generate_date_range(start_date, end_date, buffer_days=6):
    """生成日期范围（前后各加buffer_days天）"""
    extended_start = start_date - timedelta(days=buffer_days)
    extended_end = end_date + timedelta(days=buffer_days)

    # 生成日期范围内的每一天
    current_date = extended_start
    while current_date <= extended_end:
        yield current_date
        current_date += timedelta(days=1)


def copy_and_rename_tif_files(input_dir, output_dir):
    """复制并重命名TIFF文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有TIFF文件
    tif_files = glob.glob(os.path.join(input_dir, '*.tif'))

    total_files = len(tif_files)
    processed_files = 0

    for tif_file in tif_files:
        try:
            filename = os.path.basename(tif_file)
            fire_id, start_date, end_date = parse_filename(filename)

            # 生成日期范围内的每一天
            date_range = list(generate_date_range(start_date, end_date))

            print(f"处理文件: {filename} (共 {len(date_range)} 天)")

            for current_date in date_range:
                # 创建新文件名
                date_str = current_date.strftime("%Y-%m-%d")
                new_filename = f"{fire_id}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_{date_str}_Topography.tif"
                output_path = os.path.join(output_dir, new_filename)

                # 复制文件
                shutil.copy2(tif_file, output_path)
                print(f"  创建副本: {new_filename}")

            processed_files += 1
            print(f"完成处理: {filename} ({processed_files}/{total_files})\n")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            continue

    print(f"\n处理完成! 共处理 {processed_files}/{total_files} 个文件")

if __name__ == '__main__':
    # 配置路径 (根据实际情况修改)
    input_directory = r"D:\wildfire_dataset\self_built\drivers\Topography_clip"  # 输入的TIFF文件目录
    output_directory = r"D:\wildfire_dataset\self_built\drivers\Topography_clip_everyday"  # 输出目录

    # 执行复制和重命名
    copy_and_rename_tif_files(input_directory, output_directory)