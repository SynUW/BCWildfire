import os
import numpy as np
import geopandas as gpd
from osgeo import gdal
from tqdm import tqdm
import time
import shutil
from multiprocessing import Pool, cpu_count
import functools

def process_chunk(chunk_data, output_dir):
    """处理一个数据块"""
    i_start, i_end, all_bands = chunk_data
    count = 0
    # 计算每个像元的和
    sums = np.sum(all_bands, axis=0)
    # 找出和大于等于1的位置
    valid_mask = sums >= 1
    
    # 批量保存数据
    for i in range(i_start, i_end):
        for j in range(all_bands.shape[2]):
            if valid_mask[i-i_start, j]:
                pixel_data = all_bands[:, i-i_start, j]
                output_file = os.path.join(output_dir, f"{i}_{j}.npy")
                np.save(output_file, pixel_data)
                count += 1
    return count

def extract_pixels_from_shapefile(shapefile_path, tif_file, output_dir):
    """
    使用shapefile对TIF图像进行裁切，然后保存每个像素的数据（不分块，直接全图遍历）
    """
    start_time = time.time()
    # 清空保存目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print("\n=== 步骤1: 读取数据 ===")
    print("正在读取shapefile...")
    shapefile_gdf = gpd.read_file(shapefile_path)
    print(f"shapefile包含 {len(shapefile_gdf)} 个多边形")
    
    print("\n正在读取TIF文件...")
    # 设置GDAL缓存
    gdal.SetCacheMax(1024 * 1024 * 1024)  # 设置1GB缓存
    
    # 打开数据集并设置读取选项
    dataset = gdal.Open(tif_file, gdal.GA_ReadOnly)
    if dataset is None:
        raise Exception("无法打开TIF文件")
    
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    total_bands = dataset.RasterCount
    print(f"图像大小: {width}x{height}")
    print(f"总波段数: {total_bands}")
    
    # 一次性读取所有波段数据到内存
    print("正在加载所有波段数据到内存...")
    all_bands = np.zeros((total_bands, height, width), dtype=np.uint8)
    for band_idx in range(total_bands):
        band = dataset.GetRasterBand(band_idx + 1)
        # 一次性读取整个波段
        all_bands[band_idx] = band.ReadAsArray(0, 0, width, height)
    
    print("\n正在保存像元数据...")
    
    # 将数据分成多个块进行并行处理
    num_processes = cpu_count()
    chunk_size = height // num_processes
    chunks = []
    
    for i in range(0, height, chunk_size):
        i_end = min(i + chunk_size, height)
        chunk_data = (i, i_end, all_bands[:, i:i_end, :])
        chunks.append(chunk_data)
    
    # 使用进程池并行处理
    with Pool(num_processes) as pool:
        process_func = functools.partial(process_chunk, output_dir=output_dir)
        results = list(tqdm(pool.imap(process_func, chunks), total=len(chunks)))
    
    total_count = sum(results)
    print(f"\n处理完成！共保存了 {total_count} 个像元数据")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    shapefile_path = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_boundary/bc_boundary_without_sea.shp"
    tif_file = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/FIRMS_all_band.tif"
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_gt1"
    extract_pixels_from_shapefile(shapefile_path, tif_file, output_dir) 