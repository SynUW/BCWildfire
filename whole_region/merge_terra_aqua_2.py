import os
import numpy as np
from osgeo import gdal, gdalconst
from datetime import datetime
from tqdm import tqdm


def process_tif_files_with_gdal(input_folder, output_folder):
    """
    使用GDAL处理同一日期不同传感器的TIFF文件，合并波段并保存结果

    Args:
        input_folder (str): 输入文件夹路径
        output_folder (str): 输出文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有文件并按日期分组
    file_dict = {}
    files = os.listdir(input_folder)
    files = sorted(files)
    files.reverse()

    for filename in files:
        if filename.endswith('.tif'):
            try:
                # 解析文件名
                parts = filename.split('_')
                sensor = parts[0]
                date_str = '_'.join(parts[1:4]).replace('.tif', '')
                date_obj = datetime.strptime(date_str, '%Y_%m_%d')

                # 按日期分组
                if date_str not in file_dict:
                    file_dict[date_str] = {'Terra': None, 'Aqua': None, 'date_obj': date_obj}

                file_dict[date_str][sensor] = os.path.join(input_folder, filename)
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
                continue

    # 处理每个日期的文件
    for date_str, files in tqdm(file_dict.items(), desc="Processing dates"):
        output_path = os.path.join(output_folder, f"Reflect_{date_str}.tif")

        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            print(f"Skipping {date_str} - output file already exists")
            continue

        terra_path = files['Terra']
        aqua_path = files['Aqua']

        # 如果只有一个传感器有数据
        if terra_path is None or aqua_path is None:
            src_path = terra_path if terra_path is not None else aqua_path
            if src_path is None:
                continue

            # 直接复制文件
            src_ds = gdal.Open(src_path, gdalconst.GA_ReadOnly)
            if src_ds is None:
                print(f"Failed to open {src_path}")
                continue

            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.CreateCopy(output_path, src_ds, 0)

            # 释放资源
            dst_ds = None
            src_ds = None
            continue

        # 两个传感器都有数据，需要合并
        terra_ds = gdal.Open(terra_path, gdalconst.GA_ReadOnly)
        aqua_ds = gdal.Open(aqua_path, gdalconst.GA_ReadOnly)

        if terra_ds is None or aqua_ds is None:
            print(f"Failed to open one of the input files for date {date_str}")
            if terra_ds: terra_ds = None
            if aqua_ds: aqua_ds = None
            continue

        # 检查两个文件是否具有相同的形状和投影
        if (terra_ds.RasterXSize != aqua_ds.RasterXSize or
                terra_ds.RasterYSize != aqua_ds.RasterYSize):
            print(f"Size mismatch for date {date_str}, skipping...")
            terra_ds = None
            aqua_ds = None
            continue

        # 获取nodata值
        terra_nodata = terra_ds.GetRasterBand(1).GetNoDataValue()
        aqua_nodata = aqua_ds.GetRasterBand(1).GetNoDataValue()

        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            output_path,
            terra_ds.RasterXSize,
            terra_ds.RasterYSize,
            terra_ds.RasterCount,
            gdal.GDT_Float32
        )

        if dst_ds is None:
            print(f"Failed to create output file for date {date_str}")
            terra_ds = None
            aqua_ds = None
            continue

        # 设置地理参考和投影
        dst_ds.SetGeoTransform(terra_ds.GetGeoTransform())
        dst_ds.SetProjection(terra_ds.GetProjection())

        # 处理每个波段
        for band in range(1, terra_ds.RasterCount + 1):
            # 读取波段数据
            terra_band = terra_ds.GetRasterBand(band)
            aqua_band = aqua_ds.GetRasterBand(band)

            terra_data = terra_band.ReadAsArray()
            aqua_data = aqua_band.ReadAsArray()

            # 创建掩膜
            terra_mask = terra_data != terra_nodata if terra_nodata is not None else np.ones_like(terra_data,
                                                                                                  dtype=bool)
            aqua_mask = aqua_data != aqua_nodata if aqua_nodata is not None else np.ones_like(aqua_data, dtype=bool)

            # 计算合并后的数据
            combined_data = np.zeros_like(terra_data, dtype=np.float32)

            # 两个传感器都有数据的位置
            both_valid = terra_mask & aqua_mask
            combined_data[both_valid] = (terra_data[both_valid] + aqua_data[both_valid]) / 2

            # 只有Terra有数据的位置
            terra_only = terra_mask & ~aqua_mask
            combined_data[terra_only] = terra_data[terra_only]

            # 只有Aqua有数据的位置
            aqua_only = ~terra_mask & aqua_mask
            combined_data[aqua_only] = aqua_data[aqua_only]

            # 两个都没有数据的位置
            neither = ~terra_mask & ~aqua_mask
            if terra_nodata is not None:
                combined_data[neither] = terra_nodata
            elif aqua_nodata is not None:
                combined_data[neither] = aqua_nodata

            # 写入输出波段
            dst_band = dst_ds.GetRasterBand(band)
            dst_band.WriteArray(combined_data)

            # 设置nodata值
            if terra_nodata is not None:
                dst_band.SetNoDataValue(terra_nodata)
            elif aqua_nodata is not None:
                dst_band.SetNoDataValue(aqua_nodata)

            # 计算统计信息
            dst_band.FlushCache()
            dst_band.ComputeStatistics(False)

        # 释放资源
        dst_ds = None
        terra_ds = None
        aqua_ds = None


if __name__ == '__main__':
    # 注册所有GDAL驱动
    gdal.AllRegister()

    input_folder = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500_merge'  # 替换为你的输入文件夹路径
    output_folder = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500_merge_mergeTerraAqua'  # 替换为你的输出文件夹路径

    # 验证输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        exit(1)

    process_tif_files_with_gdal(input_folder, output_folder)
    print("Processing completed!")