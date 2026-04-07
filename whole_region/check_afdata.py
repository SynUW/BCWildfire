import os
from osgeo import gdal
import numpy as np


def check_non_zero_geotiff(folder_path):
    """
    检查文件夹内的GeoTIFF文件是否不全为0
    :param folder_path: 包含GeoTIFF文件的文件夹路径
    """
    # 获取文件夹内所有tif文件
    tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    tif_files.sort()
    tif_files.reverse()

    for tif_file in tif_files:
        file_path = os.path.join(folder_path, tif_file)

        # 打开文件
        dataset = gdal.Open(file_path)
        if dataset is None:
            print(f"无法打开文件: {tif_file}")
            continue

        # 读取第一个波段（可根据需要修改为多波段处理）
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()

        # 检查是否全为0
        if not np.all(data == 0):
            print(f"非全零文件: {tif_file}")

        # 关闭数据集
        dataset = None


if __name__ == "__main__":
    import argparse

    file_path = r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Firms_Aqua_500_Daily_WGS84_Binary'

    check_non_zero_geotiff(file_path)