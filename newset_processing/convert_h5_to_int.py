#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5文件浮点型转整型转换器 - 多进程版本
每个进程处理一个文件，避免文件锁定冲突
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import re
import time
import sys
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import argparse

def setup_logging(process_id=None):
    """设置日志"""
    if process_id:
        format_str = f'%(asctime)s - 进程{process_id} - %(levelname)s - %(message)s'
    else:
        format_str = '%(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=logging.INFO,
        format=format_str,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def parse_channel_mapping_from_h5(h5_file):
    """从H5文件中解析通道映射信息"""
    channel_mapping = {}
    driver_order = []
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # 获取驱动因素顺序
            if 'driver_names' in f.attrs:
                driver_order = f.attrs['driver_names'].tolist() if hasattr(f.attrs['driver_names'], 'tolist') else f.attrs['driver_names']
            else:
                # 从channel_mapping属性中提取
                for key in f.attrs.keys():
                    if key.startswith('channel_mapping_'):
                        driver_name = key.replace('channel_mapping_', '')
                        driver_order.append(driver_name)
            
            # 解析每个驱动的通道范围
            current_channel = 0
            for driver_name in driver_order:
                mapping_key = f'channel_mapping_{driver_name}'
                if mapping_key in f.attrs:
                    mapping_str = f.attrs[mapping_key]
                    if isinstance(mapping_str, bytes):
                        mapping_str = mapping_str.decode('utf-8')
                    
                    # 解析格式 "start-end"
                    match = re.match(r'(\d+)-(\d+)', mapping_str)
                    if match:
                        start_idx = int(match.group(1))
                        end_idx = int(match.group(2)) + 1  # 转换为exclusive end
                        channels = end_idx - start_idx
                    else:
                        # 如果解析失败，使用默认值
                        channels = 1
                        start_idx = current_channel
                        end_idx = current_channel + 1
                else:
                    # 如果没有映射信息，使用默认值
                    channels = 1
                    start_idx = current_channel
                    end_idx = current_channel + 1
                
                channel_mapping[driver_name] = {
                    'start': start_idx,
                    'end': end_idx,
                    'channels': channels,
                    'needs_scaling': driver_name not in [driver_order[0], driver_order[-1]]  # 第0个和最后一个不需要缩放
                }
                current_channel = end_idx
            
    except Exception as e:
        print(f"解析通道映射失败: {e}")
        return {}, []
    
    return channel_mapping, driver_order

def convert_data_vectorized_batch(data_batch, channel_mapping):
    """批量向量化转换数据"""
    if data_batch is None or data_batch.size == 0:
        return data_batch
    
    # data_batch shape: (batch_size, channels, time_steps)
    batch_size, channels, time_steps = data_batch.shape
    converted_data = np.zeros((batch_size, channels, time_steps), dtype=np.uint8)
    
    # 按驱动因素分组处理
    for driver_name, info in channel_mapping.items():
        start_ch = info['start']
        end_ch = info['end']
        needs_scaling = info['needs_scaling']
        
        if start_ch >= channels:
            continue
            
        # 确保不超出数据范围
        actual_end = min(end_ch, channels)
        driver_data = data_batch[:, start_ch:actual_end, :]  # (batch_size, driver_channels, time_steps)
        
        if needs_scaling:
            # 0-1范围的数据乘以255取整（向量化操作）
            data_clipped = np.clip(driver_data, 0.0, 1.0)
            data_scaled = data_clipped * 255.0
            converted_data[:, start_ch:actual_end, :] = np.round(data_scaled).astype(np.uint8)
        else:
            # 直接转为整型（向量化操作）
            converted_data[:, start_ch:actual_end, :] = np.round(driver_data).astype(np.uint8)
    
    return converted_data

def process_single_file(input_file, output_file, batch_size=1000):
    """处理单个文件"""
    process_id = os.getpid()
    setup_logging(process_id)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"开始处理 {os.path.basename(input_file)}")
        start_time = time.time()
        
        # 解析通道映射
        channel_mapping, driver_order = parse_channel_mapping_from_h5(input_file)
        if not channel_mapping:
            logger.error(f"无法解析通道映射")
            return False
        
        logger.info(f"解析到 {len(driver_order)} 个驱动因素")
        
        # 转换文件
        with h5py.File(input_file, 'r') as src:
            # 获取数据集信息
            dataset_keys = [key for key in src.keys() if not key.startswith('channel_')]
            total_pixels = len(dataset_keys)
            
            logger.info(f"开始转换 {total_pixels} 个像素数据集")
            
            with h5py.File(output_file, 'w') as dst:
                # 复制所有属性
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                
                # 添加转换信息
                dst.attrs['converted_to_int'] = True
                dst.attrs['conversion_date'] = datetime.now().isoformat()
                dst.attrs['original_file'] = os.path.basename(input_file)
                dst.attrs['conversion_method'] = 'multiprocess_float_to_int'
                
                # 批量处理像素数据集
                processed_count = 0
                batch_data = []
                batch_keys = []
                
                for pixel_key in tqdm(dataset_keys, desc=f"转换 {os.path.basename(input_file)}", position=process_id % 16):
                    src_dataset = src[pixel_key]
                    shape = src_dataset.shape
                    
                    if len(shape) == 2:  # [channels, time]
                        # 读取数据
                        data = src_dataset[:, :]  # (channels, time)
                        batch_data.append(data)
                        batch_keys.append(pixel_key)
                        
                        # 当批次满了或者到达最后一个像素时，处理批次
                        if len(batch_data) >= batch_size or pixel_key == dataset_keys[-1]:
                            # 将批次数据堆叠成 (batch_size, channels, time_steps)
                            batch_array = np.stack(batch_data, axis=0)
                            
                            # 批量向量化转换
                            converted_batch = convert_data_vectorized_batch(batch_array, channel_mapping)
                            
                            # 写入转换后的数据
                            for i, (pixel_key_batch, converted_data) in enumerate(zip(batch_keys, converted_batch)):
                                # 创建目标数据集（整型）
                                dst_dataset = dst.create_dataset(
                                    pixel_key_batch,
                                    shape=shape,
                                    dtype=np.uint8,
                                    chunks=src_dataset.chunks,
                                    compression=src_dataset.compression,
                                    compression_opts=src_dataset.compression_opts,
                                    shuffle=src_dataset.shuffle
                                )
                                
                                # 复制属性
                                for attr_key, attr_value in src_dataset.attrs.items():
                                    dst_dataset.attrs[attr_key] = attr_value
                                
                                # 写入转换后的数据
                                dst_dataset[:, :] = converted_data
                                processed_count += 1
                            
                            # 清理批次数据
                            batch_data.clear()
                            batch_keys.clear()
                            
                            # 强制垃圾回收
                            gc.collect()
                            
                            # 每处理10000个像素输出一次进度
                            if processed_count % 10000 == 0:
                                logger.info(f"已处理 {processed_count}/{total_pixels} 个像素数据集")
                        else:
                            continue
                    else:
                        logger.warning(f"跳过非标准形状的数据集: {pixel_key} - {shape}")
                        continue
                
                # 如果有channel_mapping，也复制过来
                if 'channel_mapping' in src:
                    src.copy('channel_mapping', dst)
                
                logger.info(f"转换完成 {processed_count} 个像素数据集")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"成功处理 {os.path.basename(input_file)} (耗时: {processing_time:.2f}秒)")
        return True
        
    except Exception as e:
        logger.error(f"处理异常 {os.path.basename(input_file)} - {e}")
        return False

def process_file_wrapper(args):
    """多进程包装函数"""
    input_file, output_file, batch_size = args
    return process_single_file(input_file, output_file, batch_size)

def main():
    parser = argparse.ArgumentParser(description='H5文件浮点型转整型转换器 - 多进程版本')
    parser.add_argument('--input_dir', 
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/newset_dataset_v5',
                       help='输入H5文件目录')
    parser.add_argument('--output_dir',
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/newset_dataset_v5_int',
                       help='输出H5文件目录')
    parser.add_argument('--test_file', type=str, default=None,
                       help='测试单个文件转换（可选）')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='批处理大小（默认: 1000）')
    parser.add_argument('--max_processes', type=int, default=16,
                       help='最大并行进程数（默认: 16）')
    
    args = parser.parse_args()
    
    # 设置主进程日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.test_file:
        # 测试单个文件
        if not os.path.exists(args.test_file):
            logger.error(f"测试文件不存在: {args.test_file}")
            return
        
        output_file = os.path.join(args.output_dir, os.path.basename(args.test_file))
        success = process_single_file(args.test_file, output_file, args.batch_size)
        
        if success:
            print(f"✅ 测试成功: {os.path.basename(args.test_file)}")
        else:
            print(f"❌ 测试失败: {os.path.basename(args.test_file)}")
    else:
        # 查找所有H5文件
        h5_files = []
        for filename in os.listdir(args.input_dir):
            if filename.endswith('_year_dataset.h5'):
                h5_files.append(filename)
        
        if not h5_files:
            logger.warning(f"在 {args.input_dir} 中未找到H5文件")
            return
        
        logger.info(f"找到 {len(h5_files)} 个H5文件")
        
        # 准备参数
        file_args = []
        for filename in sorted(h5_files):
            input_file = os.path.join(args.input_dir, filename)
            output_file = os.path.join(args.output_dir, filename)
            file_args.append((input_file, output_file, args.batch_size))
        
        logger.info(f"使用多进程并行处理，最大进程数: {args.max_processes}")
        
        # 使用ProcessPoolExecutor进行并行处理
        success_count = 0
        failed_files = []
        total_processing_time = 0
        
        with ProcessPoolExecutor(max_workers=args.max_processes) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(process_file_wrapper, args): args[0] 
                for args in file_args
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(file_args), desc="并行转换H5文件") as pbar:
                for future in as_completed(future_to_file):
                    success = future.result()
                    input_file = future_to_file[future]
                    filename = os.path.basename(input_file)
                    
                    if success:
                        success_count += 1
                        logger.info(f"✅ 成功转换: {filename}")
                    else:
                        failed_files.append(filename)
                        logger.error(f"❌ 转换失败: {filename}")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        '成功': success_count,
                        '失败': len(failed_files)
                    })
        
        # 输出转换结果统计
        logger.info(f"🎉 转换完成: {success_count}/{len(h5_files)} 个文件成功")
        if failed_files:
            logger.warning(f"⚠️  失败文件: {len(failed_files)} 个")
            for filename in failed_files:
                logger.warning(f"  - {filename}")

if __name__ == "__main__":
    main()
