import os
import numpy as np
import geopandas as gpd
from pathlib import Path
import time
from tqdm import tqdm
from osgeo import gdal, ogr
from multiprocessing import Pool, cpu_count


def process_block(args):
    """
    处理一个数据块
    
    参数:
        args: (block_data, start_row, start_col, transform, output_dir)
    返回:
        保存的像素数量
    """
    block_data, start_row, start_col, transform, output_dir = args
    saved_count = 0
    
    # 获取数据块大小
    bands, height, width = block_data.shape
    
    # 处理每个像素
    for i in range(height):
        for j in range(width):
            pixel_data = block_data[:, i, j]
            
            # 如果像素值全为0，跳过
            # if np.all(pixel_data == 0):
            #     continue
            
            # 计算全局坐标
            global_row = i + start_row + int(transform[5])
            global_col = j + start_col + int(transform[2])
            
            # 保存像素数据
            output_file = os.path.join(output_dir, f"{global_row}_{global_col}.npy")
            np.save(output_file, pixel_data)
            saved_count += 1
    
    return saved_count


def extract_pixels_from_shapefile(shapefile_path, tif_file, output_dir):
    """
    使用shapefile对TIF图像进行裁切，然后保存每个像素的数据
    
    参数:
        shapefile_path: shapefile文件路径
        tif_file: TIF文件路径
        output_dir: 输出目录
    """
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== 步骤1: 读取数据 ===")
    print("正在读取shapefile...")
    shapefile_gdf = gpd.read_file(shapefile_path)
    print(f"shapefile包含 {len(shapefile_gdf)} 个多边形")
    
    print("\n正在读取TIF文件...")
    # 使用GDAL打开TIF文件
    dataset = gdal.Open(tif_file, gdal.GA_ReadOnly)
    if dataset is None:
        raise Exception("无法打开TIF文件")
    
    # 获取图像信息
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    total_bands = dataset.RasterCount
    print(f"图像大小: {width}x{height}")
    print(f"总波段数: {total_bands}")
    
    # 获取地理变换参数
    transform = dataset.GetGeoTransform()
    
    # 计算分块大小
    block_size = 512  # 由于使用uint8，可以增加块大小
    n_blocks_h = (height + block_size - 1) // block_size
    n_blocks_w = (width + block_size - 1) // block_size
    total_blocks = n_blocks_h * n_blocks_w
    
    print(f"\n=== 步骤2: 分块处理数据 ===")
    print(f"将数据分为 {total_blocks} 个块进行处理")
    
    # 准备并行处理
    n_processes = min(cpu_count(), 16)  # 使用更多进程
    print(f"使用 {n_processes} 个进程并行处理")
    
    # 使用进程池并行处理
    total_pixels = 0
    with Pool(n_processes) as pool:
        with tqdm(total=total_blocks, desc="处理进度") as pbar:
            for i in range(n_blocks_h):
                for j in range(n_blocks_w):
                    # 计算当前块的范围
                    start_row = i * block_size
                    start_col = j * block_size
                    end_row = min(start_row + block_size, height)
                    end_col = min(start_col + block_size, width)
                    
                    # 读取当前块的数据
                    block_data = np.zeros((total_bands, end_row - start_row, end_col - start_col), dtype=np.uint8)
                    for band_idx in range(total_bands):
                        band = dataset.GetRasterBand(band_idx + 1)
                        block_data[band_idx] = band.ReadAsArray(start_col, start_row, end_col - start_col, end_row - start_row)
                    
                    # 处理当前块
                    saved_count = process_block((block_data, start_row, start_col, transform, output_dir))
                    total_pixels += saved_count
                    pbar.update(1)
                    pbar.set_postfix({'已保存像素': total_pixels})
    
    # 清理GDAL资源
    dataset = None
    
    print(f"\n=== 处理完成 ===")
    print(f"共保存了 {total_pixels} 个像素点的数据")
    
    print(f"\n总耗时: {time.time() - start_time:.2f} 秒")


def main():
    # 注册所有GDAL驱动
    gdal.AllRegister()
    
    # 输入文件路径
    shapefile_path = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_boundary/bc_boundary_without_sea.shp"
    tif_file = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/FIRMS_all_band.tif"
    
    # 创建输出目录
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_gt1"
    
    # 提取像素数据
    extract_pixels_from_shapefile(shapefile_path, tif_file, output_dir)


if __name__ == "__main__":
    main()