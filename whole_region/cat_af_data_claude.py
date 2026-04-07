import os
import re
import rasterio
import numpy as np
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import psutil
import multiprocessing
from rasterio.windows import Window
import time


def natural_sort_key(s):
    """提供自然排序的键函数，确保temp_batch_0.tif排在temp_batch_1.tif前面"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def read_tif_window(tif_file, window):
    """读取单个GeoTIFF文件的指定窗口区域"""
    with rasterio.open(tif_file) as src:
        return src.read(window=window), src.descriptions


def get_optimal_thread_count():
    """获取最优线程数，考虑CPU核心数和可用内存"""
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    # 根据可用内存和CPU核心数确定线程数
    memory_based_threads = int(memory_gb / 2)
    return min(cpu_count * 2, memory_based_threads)


def stack_geotiffs(input_folder, output_file):
    """
    在波段维度上合并多个GeoTIFF文件，按文件名顺序保留所有波段
    使用分块处理方式避免内存溢出
    """
    start_time = time.time()
    
    # 获取所有tif文件并自然排序
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
    tif_files.sort(key=natural_sort_key)

    if not tif_files:
        print(f"在{input_folder}中未找到GeoTIFF文件")
        return

    print(f"找到{len(tif_files)}个GeoTIFF文件，按以下顺序合并:")
    for i, file in enumerate(tif_files):
        print(f"{i + 1}. {os.path.basename(file)}")

    # 打开第一个文件获取元数据模板
    with rasterio.open(tif_files[0]) as first_src:
        meta = first_src.meta.copy()
        height = first_src.height
        width = first_src.width
        first_transform = first_src.transform
        first_crs = first_src.crs

    # 获取最优线程数
    optimal_threads = get_optimal_thread_count()
    print(f"\n系统信息:")
    print(f"CPU核心数: {multiprocessing.cpu_count()}")
    print(f"可用内存: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.2f} GB")
    print(f"使用线程数: {optimal_threads}")

    # 计算总波段数
    total_bands = 0
    band_descriptions = []
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            total_bands += src.count
            band_descriptions.extend(src.descriptions)

    # 设置分块大小
    block_size = 1024  # 可以根据实际情况调整
    blocks = [(i, j) for i in range(0, height, block_size) 
             for j in range(0, width, block_size)]

    # 更新元数据
    meta.update({
        'count': total_bands,
        'driver': 'GTiff',
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': block_size,
        'blockysize': block_size,
        'bigtiff': 'YES',
        'interleave': 'band',
    })

    print(f"\n正在创建输出文件: {output_file} (总波段数: {total_bands})")
    print(f"分块大小: {block_size}x{block_size}")
    print(f"总块数: {len(blocks)}")

    # 创建输出文件
    with rasterio.open(output_file, 'w', **meta) as dst:
        # 设置所有波段的描述
        for i, desc in enumerate(band_descriptions, 1):
            dst.set_band_description(i, desc)

        # 分块处理
        for block_idx, (i, j) in enumerate(blocks, 1):
            print(f"\n处理块 {block_idx}/{len(blocks)}")
            block_start = time.time()
            
            # 计算当前块的大小
            block_height = min(block_size, height - i)
            block_width = min(block_size, width - j)
            window = Window(j, i, block_width, block_height)
            
            # 读取所有文件在当前窗口的数据
            block_data = []
            with ThreadPoolExecutor(max_workers=optimal_threads) as executor:
                futures = [executor.submit(read_tif_window, tif_file, window) 
                          for tif_file in tif_files]
                for future in tqdm(futures, total=len(tif_files), 
                                 desc=f"读取块 {block_idx}"):
                    data, _ = future.result()
                    block_data.append(data)
            
            # 合并当前块的数据
            stacked_block = np.concatenate(block_data, axis=0)
            
            # 写入当前块
            dst.write(stacked_block, window=window)
            
            print(f"块 {block_idx} 处理完成，耗时: {time.time() - block_start:.2f} 秒")
            print(f"当前内存使用: {psutil.Process().memory_info().rss / (1024 * 1024 * 1024):.2f} GB")

    total_time = time.time() - start_time
    print(f"\n处理完成!")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"最终内存使用: {psutil.Process().memory_info().rss / (1024 * 1024 * 1024):.2f} GB")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    # 用户输入路径
    input_folder = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/temp_batches"
    output_file ="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/FIRMS_all_band.tif"

    # 处理相对路径和绝对路径
    input_folder = os.path.abspath(input_folder)
    output_file = os.path.abspath(output_file)

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stack_geotiffs(input_folder, output_file)