import os
import re
import numpy as np
from osgeo import gdal
import collections

# 设置工作目录和输出目录
work_dir = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500"
output_dir = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500_merge"

# 创建输出目录（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建输出目录: {output_dir}")

# 正则表达式匹配文件名模式
pattern = r"MODIS_(Terra|Aqua)_sur_refl_b(0[1237])_(\d{4})_(\d{2})_(\d{2})\.tif"

# 用于存储文件的字典，按传感器和日期分组
file_groups = collections.defaultdict(dict)

# 遍历目录中的所有文件
files = os.listdir(work_dir)
files = files[len(files)//2:]

for filename in files:
    try:
        match = re.match(pattern, filename)
        if match:
            sensor = match.group(1)  # Terra 或 Aqua
            band = match.group(2)  # 01, 02, 03 或 07
            year = match.group(3)
            month = match.group(4)
            day = match.group(5)

            date_key = f"{year}_{month}_{day}"
            sensor_date_key = f"{sensor}_{date_key}"

            # 将文件按波段存储到对应的传感器和日期组中
            if sensor_date_key not in file_groups:
                file_groups[sensor_date_key] = {}

            # 存储完整路径
            file_groups[sensor_date_key][band] = os.path.join(work_dir, filename)
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {str(e)}")
        continue

# 处理每个传感器和日期组
for sensor_date_key, band_files in file_groups.items():
    try:
        # 检查是否包含所有需要的波段
        required_bands = ["01", "02", "03", "07"]
        if all(band in band_files for band in required_bands):
            sensor, date = sensor_date_key.split("_", 1)
            output_filename = f"{sensor}_{date}.tif"
            output_path = os.path.join(output_dir, output_filename)

            # 检查输出文件是否已存在
            if os.path.exists(output_path):
                print(f"输出文件已存在，跳过处理组: {sensor_date_key}")
                continue

            print(f"处理组: {sensor_date_key}")
            print(f"输出文件: {output_path}")

            # 读取第一个波段文件以获取尺寸信息
            try:
                first_ds = gdal.Open(band_files["01"])
                if not first_ds:
                    print(f"无法打开文件: {band_files['01']}")
                    continue

                width = first_ds.RasterXSize
                height = first_ds.RasterYSize
                geo_transform = first_ds.GetGeoTransform()
                projection = first_ds.GetProjection()
            except Exception as e:
                print(f"读取基础信息时出错: {str(e)}")
                continue

            # 创建输出文件
            try:
                driver = gdal.GetDriverByName("GTiff")
                out_ds = driver.Create(output_path, width, height, 4, gdal.GDT_Float32)

                if not out_ds:
                    print(f"无法创建输出文件: {output_path}")
                    continue

                out_ds.SetGeoTransform(geo_transform)
                out_ds.SetProjection(projection)
            except Exception as e:
                print(f"创建输出文件时出错: {str(e)}")
                continue

            # 按顺序读取波段并写入输出文件
            success = True
            for i, band in enumerate(required_bands, start=1):
                try:
                    if band in band_files:
                        print(f"  添加波段 {band} 作为输出文件的第 {i} 通道")
                        ds = gdal.Open(band_files[band])
                        if ds:
                            data = ds.GetRasterBand(1).ReadAsArray()
                            out_ds.GetRasterBand(i).WriteArray(data)
                            out_ds.GetRasterBand(i).SetDescription(f"Band {band}")
                            ds = None  # 关闭数据集
                        else:
                            print(f"  无法打开文件: {band_files[band]}")
                            success = False
                    else:
                        print(f"  缺少波段 {band}")
                        success = False
                except Exception as e:
                    print(f"  处理波段 {band} 时出错: {str(e)}")
                    success = False
                    continue

            # 关闭输出数据集，将其写入磁盘
            try:
                out_ds = None
                if success:
                    print(f"已完成 {output_filename} 的创建")
                else:
                    print(f"部分完成 {output_filename}，存在错误")
                    # 删除不完整的输出文件
                    if os.path.exists(output_path):
                        os.remove(output_path)
            except Exception as e:
                print(f"关闭输出文件时出错: {str(e)}")
                if os.path.exists(output_path):
                    os.remove(output_path)
        else:
            missing_bands = [band for band in required_bands if band not in band_files]
            print(f"跳过组 {sensor_date_key}，缺少波段: {', '.join(missing_bands)}")
    except Exception as e:
        print(f"处理组 {sensor_date_key} 时发生意外错误: {str(e)}")
        continue

print("处理完成!")