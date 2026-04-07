import os
import rasterio
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


def process_block(args):
    """处理单个数据块"""
    block_data, block_idx, total_blocks = args
    # 计算所有波段的总和
    sum_all = np.sum(block_data, axis=0)
    
    # 计算前N-365和最后365个波段的总和
    if block_data.shape[0] > 365:
        sum_first = np.sum(block_data[:-365], axis=0)
        sum_last = np.sum(block_data[-365:], axis=0)
        return block_idx, sum_all, sum_first, sum_last
    else:
        return block_idx, sum_all, None, None


def sum_bands(tif_file, output_dir):
    """
    计算波段值的总和并生成新的单波段图像，使用分块处理方式
    
    参数:
        tif_file: 输入的TIF文件路径
        output_dir: 输出目录
    """
    start_time = time.time()
    
    print("正在读取TIF文件...")
    with rasterio.open(tif_file) as src:
        # 获取文件信息
        total_bands = src.count
        height = src.height
        width = src.width
        print(f"总波段数: {total_bands}")
        print(f"图像大小: {width}x{height}")
        
        # 获取元数据
        meta = src.meta.copy()
        meta.update({
            'count': 1,  # 单波段
            'dtype': 'uint32',  # 使用uint32以处理较大的和值
            'compress': 'lzw'
        })
        
        # 设置分块大小
        block_size = 1024  # 可以根据实际情况调整
        blocks = [(i, j) for i in range(0, height, block_size) 
                 for j in range(0, width, block_size)]
        
        print(f"分块大小: {block_size}x{block_size}")
        print(f"总块数: {len(blocks)}")
        
        # 创建输出文件
        output_file1 = os.path.join(output_dir, "sum_all_bands.tif")
        output_file2 = os.path.join(output_dir, "sum_first_bands.tif")
        output_file3 = os.path.join(output_dir, "sum_last_365_bands.tif")
        
        with rasterio.open(output_file1, 'w', **meta) as dst1, \
             rasterio.open(output_file2, 'w', **meta) as dst2, \
             rasterio.open(output_file3, 'w', **meta) as dst3:
            
            # 使用线程池并行处理数据块
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                # 准备任务
                tasks = []
                for block_idx, (i, j) in enumerate(blocks):
                    # 计算当前块的大小
                    block_height = min(block_size, height - i)
                    block_width = min(block_size, width - j)
                    # 读取数据块
                    block_data = src.read(window=((i, i + block_height), (j, j + block_width)))
                    tasks.append((block_data, block_idx, len(blocks)))
                
                # 提交任务并获取结果
                results = []
                for result in executor.map(process_block, tasks):
                    results.append(result)
            
            # 按块索引排序结果
            results.sort(key=lambda x: x[0])
            
            # 将结果写入输出文件
            for block_idx, (i, j) in enumerate(blocks):
                block_height = min(block_size, height - i)
                block_width = min(block_size, width - j)
                window = ((i, i + block_height), (j, j + block_width))
                
                # 获取当前块的结果
                _, sum_all, sum_first, sum_last = results[block_idx]
                
                # 写入所有波段总和的图像
                dst1.write(sum_all.astype('uint32'), 1, window=window)
                
                # 如果有365波段分割的结果，也写入相应文件
                if sum_first is not None and sum_last is not None:
                    dst2.write(sum_first.astype('uint32'), 1, window=window)
                    dst3.write(sum_last.astype('uint32'), 1, window=window)
                
                if block_idx % 10 == 0:  # 每处理10个块显示一次进度
                    print(f"已处理 {block_idx}/{len(blocks)} 个数据块")
    
    print(f"\n处理完成! 总耗时: {time.time() - start_time:.2f} 秒")


def main():
    # 输入文件路径
    tif_file = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/FIRMS_all_band.tif"
    
    # 创建输出目录
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/sum_bands"
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算波段总和
    sum_bands(tif_file, output_dir)


if __name__ == "__main__":
    main()