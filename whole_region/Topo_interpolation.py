import os
import shutil
from datetime import datetime, timedelta


def generate_all_dates(start_year=2000, end_year=2024):
    """生成从2000-01-01到2024-12-31的所有日期"""
    dates = []
    current_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    return dates


def copy_tif_with_dates(input_tif, output_dir, dates):
    """
    复制TIFF文件并按照日期重命名
    :param input_tif: 输入的TIFF文件路径
    :param output_dir: 输出目录
    :param dates: 日期列表(datetime对象)
    """
    if not os.path.exists(input_tif):
        print(f"错误: 输入文件不存在 {input_tif}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = "TopoDis"
    total = len(dates)
    success = 0

    for i, date in enumerate(dates, 1):
        # 格式化日期为yyyy_mm_dd
        date_str = date.strftime("%Y_%m_%d")
        output_name = f"{base_name}_{date_str}.tif"
        output_path = os.path.join(output_dir, output_name)

        # 如果文件已存在则跳过
        if os.path.exists(output_path):
            print(f"文件已存在，跳过: {output_path}")
            continue

        try:
            shutil.copy2(input_tif, output_path)
            success += 1
            if i % 100 == 0 or i == total:  # 每100个文件或最后打印进度
                print(f"进度: {i}/{total} ({i / total:.1%}) - 最新创建: {output_name}")
        except Exception as e:
            print(f"复制 {output_name} 失败: {str(e)}")

    print(f"\n操作完成! 成功创建 {success} 个文件，跳过 {total - success} 个已存在文件")


if __name__ == "__main__":
    # 配置参数
    input_directory = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x/Topo_Distance_WGS84_resize_resampled_original/TopoDis_2000_09_27.tif"
    output_directory = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x/Topo_Distance_WGS84_resize_resampled"

    # 生成所有日期
    print("正在生成日期列表...")
    all_dates = generate_all_dates()
    print(f"共生成 {len(all_dates)} 个日期")

    # 执行复制
    copy_tif_with_dates(input_directory, output_directory, all_dates)