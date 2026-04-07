from osgeo import gdal
import os
from datetime import datetime, timedelta
import numpy as np

def split_multi_band_tif_gdal(input_tif, output_dir):
    """
    使用 GDAL 将多波段 TIFF 文件的每个波段保存为一个单独的 TIFF 文件，
    文件名根据原始文件名中的日期信息生成，并保存到指定的输出文件夹。

    Args:
        input_tif (str): 输入的多波段 TIFF 文件路径。
        output_dir (str): 输出文件夹路径。
    """
    try:
        src_ds = gdal.Open(input_tif)
        if src_ds is None:
            print(f"错误：无法打开文件 '{input_tif}'。请检查文件路径是否正确。")
            return

        num_bands = src_ds.RasterCount

        # 从文件名中提取日期信息
        filename_base = os.path.splitext(os.path.basename(input_tif))[0]
        parts = filename_base.split('_')

        if len(parts) < 3:
            print(f"警告：文件名 '{filename_base}' 无法解析出日期信息，将使用波段号命名。")
            use_band_number = True
        else:
            prefix = parts[0]
            sensor = parts[1]
            date_part1 = parts[2]
            date_part2 = parts[3] if len(parts) > 3 else None
            use_band_number = False

        driver = gdal.GetDriverByName("GTiff")

        for i in range(1, num_bands + 1):
            band = src_ds.GetRasterBand(i)
            if band is None:
                print(f"警告：无法读取文件 '{input_tif}' 的第 {i} 个波段。")
                continue

            gt = src_ds.GetGeoTransform()
            proj = src_ds.GetProjection()
            cols = src_ds.RasterXSize
            rows = src_ds.RasterYSize

            if use_band_number:
                output_filename = f"{filename_base}_band_{i}.tif"
            else:
                try:
                    start_date = datetime.strptime(date_part1, '%Y%m%d')
                    if date_part2:
                        end_date = datetime.strptime(date_part2, '%Y%m%d')
                        delta = timedelta(days=1)
                        current_date = start_date
                        date_list = []
                        while current_date <= end_date:
                            date_list.append(current_date.strftime('%Y%m%d'))
                            current_date += delta

                        if i - 1 < len(date_list):
                            output_filename = f"{prefix}_{sensor}_{date_list[i-1]}.tif"
                        else:
                            print(f"警告：波段数超过文件名中日期范围，将使用波段号命名波段 {i}。")
                            output_filename = f"{filename_base}_band_{i}.tif"
                    else:
                        output_filename = f"{prefix}_{sensor}_{start_date.strftime('%Y%m%d')}.tif"
                except ValueError:
                    print(f"警告：文件名 '{filename_base}' 中的日期格式不正确，将使用波段号命名波段 {i}。")
                    output_filename = f"{filename_base}_band_{i}.tif"

            output_path = os.path.join(output_dir, output_filename)
            os.makedirs(output_dir, exist_ok=True)  # 如果输出文件夹不存在则创建

            # 创建输出文件，使用GDT_Byte (uint8)类型
            dst_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte)
            if dst_ds is None:
                print(f"错误：无法创建输出文件 '{output_path}'。")
                continue

            dst_ds.SetGeoTransform(gt)
            dst_ds.SetProjection(proj)
            dst_band = dst_ds.GetRasterBand(1)
            
            # 读取数据并确保是uint8类型
            data = band.ReadAsArray()
            data = data.astype(np.uint8)
            dst_band.WriteArray(data)

            dst_band.FlushCache()
            dst_band = None
            dst_ds = None

        src_ds = None
        print(f"成功将 '{input_tif}' 分割为 {num_bands} 个单波段文件 (使用 GDAL)。")

    except Exception as e:
        print(f"发生未知错误 (GDAL)：{e}")

if __name__ == "__main__":
    input_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/Firms_Terra"  # 输入文件夹路径
    output_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/Firms_Terra_Daily"  # 输出文件夹路径

    if not os.path.isdir(input_folder):
        print(f"错误：输入文件夹 '{input_folder}' 不存在。")
    elif not os.path.isdir(output_folder):
        print(f"警告：输出文件夹 '{output_folder}' 不存在，将尝试创建。")

    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".TIF"):
            input_tif_path = os.path.join(input_folder, filename)
            print(f"正在处理文件: {input_tif_path}")
            split_multi_band_tif_gdal(input_tif_path, output_folder)

    print("所有 TIFF 文件处理完成。")