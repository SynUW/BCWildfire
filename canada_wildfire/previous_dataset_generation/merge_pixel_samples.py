"""
合并像素样本数据，将分散的驱动因素H5文件合并为统一格式
应该是需要和pixel_sampling.py一起使用
"""

import h5py
import numpy as np
import os
import glob
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
import gc

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PixelSamplesMerger:
    def __init__(self, pixel_samples_dir, output_dir, driver_order=None, batch_size=100):
        """
        Args:
            pixel_samples_dir: 包含分散H5文件的目录
            output_dir: 合并后文件的输出目录
            driver_order: 驱动因素顺序列表，如果为None则使用默认顺序
            batch_size: 批处理大小，优化内存使用
        """
        self.pixel_samples_dir = pixel_samples_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置驱动因素顺序
        if driver_order is None:
            # 默认驱动因素顺序，确保与数据处理流程一致
            self.driver_order = [
                'Firms_Detection_resampled',
                'ERA5_multi_bands',
                'LULC_BCBoundingbox_resampled',
                'Topo_Distance_WGS84_resize_resampled',
                'NDVI_EVI',
                'Reflection_500_merge_TerraAquaWGS84_clip',
                'MODIS_Terra_Aqua_B20_21_merged_resampled',
                'MOD21A1DN_multibands_filtered_resampled',
                'LAI_BCBoundingbox_resampled'
            ]
        else:
            self.driver_order = driver_order
        
        # 获取实际可用的驱动因素名称
        available_drivers = self._get_driver_names()
        
        # 过滤出实际存在的驱动因素，保持顺序
        self.driver_names = [driver for driver in self.driver_order if driver in available_drivers]
        
        logger.info(f"指定的驱动因素顺序: {self.driver_order}")
        logger.info(f"实际可用的驱动因素: {self.driver_names}")
        
        if len(self.driver_names) == 0:
            logger.warning("没有找到任何可用的驱动因素数据文件，可能目录为空或文件格式不匹配")
            # 不抛出异常，允许在空目录中测试
    
    def _get_driver_names(self):
        """从文件名中提取驱动因素名称"""
        driver_names = set()
        
        # 扫描所有H5文件
        h5_files = glob.glob(os.path.join(self.pixel_samples_dir, '*.h5'))
        
        for file_path in h5_files:
            filename = os.path.basename(file_path)
            
            if filename.endswith('_full.h5'):
                # 完整数据文件格式可能有两种：
                # 1. 年份模式: YYYY_DriverName_full.h5
                # 2. 时间窗口模式: YYYY_MMDD_MMDD_DriverName_full.h5
                parts = filename[:-8].split('_')  # 移除_full.h5
                
                if len(parts) >= 2:
                    # 检查是否是时间窗口格式 (YYYY_MMDD_MMDD_DriverName)
                    if len(parts) >= 4 and parts[1].isdigit() and len(parts[1]) == 4 and parts[2].isdigit() and len(parts[2]) == 4:
                        # 时间窗口格式：YYYY_MMDD_MMDD_DriverName_full.h5
                        driver_name = '_'.join(parts[3:])  # 驱动因素名称可能包含下划线
                        driver_names.add(driver_name)
                    else:
                        # 年份格式：YYYY_DriverName_full.h5
                        driver_name = '_'.join(parts[1:])  # 驱动因素名称可能包含下划线
                        driver_names.add(driver_name)
            else:
                # 样本数据文件格式可能有两种：
                # 1. 年份模式: YYYY_DriverName.h5
                # 2. 时间窗口模式: YYYY_MMDD_MMDD_DriverName.h5
                parts = filename[:-3].split('_')  # 移除.h5
                
                if len(parts) >= 2:
                    # 检查是否是时间窗口格式 (YYYY_MMDD_MMDD_DriverName)
                    if len(parts) >= 4 and parts[1].isdigit() and len(parts[1]) == 4 and parts[2].isdigit() and len(parts[2]) == 4:
                        # 时间窗口格式：YYYY_MMDD_MMDD_DriverName.h5
                        driver_name = '_'.join(parts[3:])  # 驱动因素名称可能包含下划线
                        driver_names.add(driver_name)
                    else:
                        # 年份格式：YYYY_DriverName.h5
                        driver_name = '_'.join(parts[1:])  # 驱动因素名称可能包含下划线
                        driver_names.add(driver_name)
        
        return driver_names
    
    def _get_available_years(self):
        """获取可用的年份"""
        years = set()
        
        # 扫描所有H5文件
        h5_files = glob.glob(os.path.join(self.pixel_samples_dir, '*.h5'))
        
        for file_path in h5_files:
            filename = os.path.basename(file_path)
            
            # 提取年份
            if filename.endswith('_full.h5'):
                parts = filename[:-8].split('_')
            else:
                parts = filename[:-3].split('_')
            
            if len(parts) >= 1:
                try:
                    year = int(parts[0])
                    years.add(year)
                except ValueError:
                    continue
        
        return sorted(list(years))

    def _get_available_time_windows(self):
        """获取可用的时间窗口"""
        time_windows = set()
        
        # 扫描所有H5文件
        h5_files = glob.glob(os.path.join(self.pixel_samples_dir, '*.h5'))
        
        for file_path in h5_files:
            filename = os.path.basename(file_path)
            
            # 检查是否是时间窗口格式
            if filename.endswith('_full.h5'):
                parts = filename[:-8].split('_')
            else:
                parts = filename[:-3].split('_')
            
            # 时间窗口格式: YYYY_MMDD_MMDD_DriverName
            if len(parts) >= 4 and parts[1].isdigit() and len(parts[1]) == 4 and parts[2].isdigit() and len(parts[2]) == 4:
                window_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
                time_windows.add(window_id)
        
        return sorted(list(time_windows))
    
    def merge_year_data(self, year, include_full_data=True):
        """
        合并指定年份的数据
        
        Args:
            year: 年份
            include_full_data: 是否包含完整数据处理（默认True）
        
        输出文件: 
        - {year}_samples.h5: 合并后的样本数据（抽样的正样本和n倍负样本）
        - {year}_full.h5: 合并后的完整数据（仅2020-2024年，且include_full_data=True时）
        """
        logger.info(f"开始合并年份 {year} 的数据...")
        
        # 验证原始数据的时间一致性 (已禁用以加快处理速度)
        # self._validate_time_consistency(year)
        
        # 优先合并样本数据（抽样数据）
        logger.info(f"处理年份 {year} 的抽样数据...")
        self._merge_sample_data(year)
        
        # 根据参数决定是否处理完整数据
        if include_full_data:
            logger.info(f"处理年份 {year} 的完整数据...")
            self._merge_full_data_optimized(year)
        else:
            logger.info(f"跳过年份 {year} 的完整数据处理")
    
    def _validate_time_consistency(self, year):
        """验证原始数据的时间一致性"""
        logger.info(f"验证年份 {year} 的数据时间一致性...")
        
        # 获取该年份的所有样本文件
        driver_files = {}
        for driver_name in self.driver_names:
            sample_file = os.path.join(self.pixel_samples_dir, f'{year}_{driver_name}.h5')
            if os.path.exists(sample_file):
                driver_files[driver_name] = sample_file
        
        if not driver_files:
            logger.warning(f"年份 {year} 没有找到样本文件")
            return
        
        # 检查每个驱动因素的时间长度
        time_length_stats = {}
        
        for driver_name, file_path in driver_files.items():
            with h5py.File(file_path, 'r') as f:
                past_lengths = []
                future_lengths = []
                
                # 检查所有数据集的时间长度
                for dataset_key in f.keys():
                    if '_past_' in dataset_key:
                        data = f[dataset_key][:]
                        if data.ndim == 1:
                            past_lengths.append(len(data))
                        else:
                            past_lengths.append(data.shape[-1])  # 最后一个维度是时间
                    elif '_future_' in dataset_key:
                        data = f[dataset_key][:]
                        if data.ndim == 1:
                            future_lengths.append(len(data))
                        else:
                            future_lengths.append(data.shape[-1])  # 最后一个维度是时间
                
                time_length_stats[driver_name] = {
                    'past_lengths': set(past_lengths),
                    'future_lengths': set(future_lengths),
                    'past_days': f.attrs.get('past_days', 'unknown'),
                    'future_days': f.attrs.get('future_days', 'unknown')
                }
        
        # 报告时间长度统计
        logger.info("时间长度统计:")
        for driver_name, stats in time_length_stats.items():
            logger.info(f"  {driver_name}:")
            logger.info(f"    配置: past_days={stats['past_days']}, future_days={stats['future_days']}")
            logger.info(f"    实际past长度: {stats['past_lengths']}")
            logger.info(f"    实际future长度: {stats['future_lengths']}")
            
            # 检查是否有不一致的长度
            if len(stats['past_lengths']) > 1:
                logger.warning(f"    {driver_name} 的past数据长度不一致!")
            if len(stats['future_lengths']) > 1:
                logger.warning(f"    {driver_name} 的future数据长度不一致!")
        
        # 检查不同驱动因素之间的一致性
        all_past_lengths = set()
        all_future_lengths = set()
        for stats in time_length_stats.values():
            all_past_lengths.update(stats['past_lengths'])
            all_future_lengths.update(stats['future_lengths'])
        
        if len(all_past_lengths) > 1:
            logger.warning(f"不同驱动因素的past数据长度不一致: {all_past_lengths}")
        if len(all_future_lengths) > 1:
            logger.warning(f"不同驱动因素的future数据长度不一致: {all_future_lengths}")
        
        if len(all_past_lengths) == 1 and len(all_future_lengths) == 1:
            logger.info(f"✓ 时间长度一致: past={list(all_past_lengths)[0]}, future={list(all_future_lengths)[0]}")
        
        return time_length_stats
    
    def _merge_sample_data(self, year):
        """合并样本数据"""
        logger.info(f"合并年份 {year} 的样本数据...")
        
        # 获取该年份的所有样本数据文件
        driver_files = {}
        for driver_name in self.driver_names:
            sample_file = os.path.join(self.pixel_samples_dir, f'{year}_{driver_name}.h5')
            if os.path.exists(sample_file):
                driver_files[driver_name] = sample_file
        
        if not driver_files:
            logger.warning(f"年份 {year} 没有找到样本数据文件")
            return
        
        logger.info(f"找到 {len(driver_files)} 个样本数据文件: {list(driver_files.keys())}")
        
        # 读取所有数据集信息
        all_dataset_keys = set()
        geo_reference = None
        
        # 从第一个文件获取所有数据集的键和地理参考信息
        first_driver = list(driver_files.keys())[0]
        with h5py.File(driver_files[first_driver], 'r') as f:
            all_dataset_keys = set(f.keys())
            
            # 获取地理参考信息
            if 'geotransform' in f.attrs:
                geo_reference = {
                    'geotransform': f.attrs['geotransform'],
                    'projection': f.attrs['projection'],
                    'raster_size': f.attrs['raster_size'],
                    'raster_count': f.attrs['raster_count']
                }
        
        logger.info(f"找到 {len(all_dataset_keys)} 个样本数据集")
        
        # 创建合并文件 - 样本数据
        output_path = os.path.join(self.output_dir, f'{year}_samples.h5')
        
        # 检查文件是否已存在
        if os.path.exists(output_path):
            logger.info(f"样本数据合并文件已存在，跳过: {output_path}")
            return
        
        with h5py.File(output_path, 'w') as output_f:
            # 写入全局属性
            output_f.attrs['year'] = str(year)
            output_f.attrs['num_drivers'] = len(driver_files)
            output_f.attrs['driver_names'] = list(driver_files.keys())
            output_f.attrs['data_format'] = 'bands_x_time'
            output_f.attrs['data_type'] = 'merged_samples'
            output_f.attrs['dataset_naming'] = 'YYYYMMDD_{past/future}_{firms_value}_row_col'
            
            # 保存地理参考信息
            if geo_reference:
                output_f.attrs['geotransform'] = geo_reference['geotransform']
                output_f.attrs['projection'] = geo_reference['projection']
                output_f.attrs['raster_size'] = geo_reference['raster_size']
                output_f.attrs['raster_count'] = geo_reference['raster_count']
            
            # 从第一个驱动因素文件复制其他属性
            with h5py.File(driver_files[first_driver], 'r') as first_f:
                for attr_name in ['past_days', 'future_days', 'negative_ratio', 'sampling_type']:
                    if attr_name in first_f.attrs:
                        output_f.attrs[attr_name] = first_f.attrs[attr_name]
            
            merged_count = 0
            total_bands = 0
            
            # 处理每个数据集
            for dataset_key in tqdm(all_dataset_keys, desc="合并样本数据"):
                # 收集所有驱动因素的数据
                driver_data_list = []
                current_bands = 0
                time_steps_list = []  # 记录每个数据的时间步数
                
                for driver_name in self.driver_names:
                    if driver_name not in driver_files:
                        continue
                        
                    with h5py.File(driver_files[driver_name], 'r') as driver_f:
                        if dataset_key not in driver_f:
                            continue
                            
                        dataset = driver_f[dataset_key]
                        data = dataset[:]  # shape: (time_steps,) 或 (bands, time_steps)
                        
                        # 处理多波段数据
                        if len(data.shape) == 2:  # 多波段数据: (bands, time_steps)
                            # 将每个波段作为单独的通道
                            for band_idx in range(data.shape[0]):
                                band_data = data[band_idx]
                                driver_data_list.append(band_data)
                                time_steps_list.append(len(band_data))
                                current_bands += 1
                        else:  # 单波段数据: (time_steps,)
                            driver_data_list.append(data)
                            time_steps_list.append(len(data))
                            current_bands += 1
                
                # 检查是否所有驱动因素都有数据
                if len(driver_data_list) != current_bands:
                    continue
                
                # 检查时间步数是否一致
                if len(set(time_steps_list)) > 1:
                    logger.warning(f"数据集 {dataset_key} 的时间步数不一致: {set(time_steps_list)}")
                    # 找到最小的时间步数，截断到统一长度
                    min_time_steps = min(time_steps_list)
                    driver_data_list = [data[:min_time_steps] for data in driver_data_list]
                    logger.info(f"将所有数据截断到 {min_time_steps} 个时间步")
                
                # 合并数据: shape = (total_bands, time_steps)
                merged_data = np.stack(driver_data_list, axis=0)
                
                # 保持原始数据集名称格式
                dataset = output_f.create_dataset(
                    dataset_key,  # 保持原始名称: YYYYMMDD_{past/future}_{firms_value}_row_col
                    data=merged_data,
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=6,
                    shuffle=True
                )
                
                # 添加新的属性（不保存冗余信息）
                dataset.attrs['data_shape'] = f"{merged_data.shape[0]}x{merged_data.shape[1]}"
                dataset.attrs['total_bands'] = current_bands
                
                merged_count += 1
                total_bands = current_bands  # 所有数据集的波段数应该相同
            
            output_f.attrs['total_datasets'] = merged_count
            output_f.attrs['total_bands'] = total_bands
            
        logger.info(f"样本数据合并完成: {output_path}, 共 {merged_count} 个数据集")
    
    def _merge_full_data_optimized(self, year):
        """优化的完整数据合并方法 - 适应新的按日期文件结构"""
        logger.info(f"检查年份 {year} 的完整数据...")
        
        # 检查新的按日期文件结构
        year_dir = os.path.join(self.pixel_samples_dir, str(year))
        if not os.path.exists(year_dir):
            logger.info(f"年份 {year} 目录不存在，跳过完整数据合并")
            return
        
        # 获取该年份目录下的所有完整数据文件
        full_files = glob.glob(os.path.join(year_dir, '*_full.h5'))
        
        if not full_files:
            logger.info(f"年份 {year} 没有完整数据文件，跳过完整数据合并")
            return
        
        logger.info(f"找到 {len(full_files)} 个完整数据文件")
        
        # 创建合并文件 - 完整数据
        output_path = os.path.join(self.output_dir, f'{year}_full.h5')
        
        # 检查文件是否已存在
        if os.path.exists(output_path):
            logger.info(f"完整数据合并文件已存在，跳过: {output_path}")
            return
        
        # 预读取所有数据集键和元数据
        all_dataset_keys = []
        geo_reference = None
        file_handles = {}
        
        # 保持文件句柄打开，避免重复打开关闭
        for file_path in full_files:
            filename = os.path.basename(file_path)
            file_handles[filename] = h5py.File(file_path, 'r')
        
        try:
            # 从第一个文件获取地理参考信息和样本数据集结构
            first_filename = list(file_handles.keys())[0]
            first_file = file_handles[first_filename]
            
            # 获取地理参考信息
            if 'geotransform' in first_file.attrs:
                geo_reference = {
                    'geotransform': first_file.attrs['geotransform'],
                    'projection': first_file.attrs['projection'],
                    'raster_size': first_file.attrs['raster_size'],
                    'raster_count': first_file.attrs['raster_count']
                }
            
            # 收集所有文件中的数据集键
            for filename, file_handle in file_handles.items():
                file_dataset_keys = list(file_handle.keys())
                all_dataset_keys.extend(file_dataset_keys)
            
            logger.info(f"找到 {len(all_dataset_keys)} 个数据集")
            
            # 计算总波段数（从第一个数据集获取）
            if all_dataset_keys:
                sample_dataset = first_file[all_dataset_keys[0]]
                if len(sample_dataset.shape) == 2:  # (bands, time_steps)
                    total_bands = sample_dataset.shape[0]
                else:  # (time_steps,) - 单波段
                    total_bands = 1
            else:
                total_bands = 0
            
            logger.info(f"预计总波段数: {total_bands}")
            
            with h5py.File(output_path, 'w') as output_f:
                # 写入全局属性
                output_f.attrs['year'] = str(year)
                output_f.attrs['num_files'] = len(full_files)
                output_f.attrs['data_format'] = 'bands_x_time'
                output_f.attrs['data_type'] = 'merged_full_data_by_date'
                output_f.attrs['dataset_naming'] = 'YYYY_MM_DD_{past/future}_{firms_value}_row_col'
                output_f.attrs['source_structure'] = 'daily_files'
                
                # 保存地理参考信息
                if geo_reference:
                    output_f.attrs['geotransform'] = geo_reference['geotransform']
                    output_f.attrs['projection'] = geo_reference['projection']
                    output_f.attrs['raster_size'] = geo_reference['raster_size']
                    output_f.attrs['raster_count'] = geo_reference['raster_count']
                
                # 从第一个文件复制其他属性
                for attr_name in ['past_days', 'future_days']:
                    if attr_name in first_file.attrs:
                        output_f.attrs[attr_name] = first_file.attrs[attr_name]
                
                merged_count = 0
                
                # 批量处理数据集以优化内存使用
                dataset_batches = [all_dataset_keys[i:i+self.batch_size] 
                                 for i in range(0, len(all_dataset_keys), self.batch_size)]
                
                for batch_idx, batch_keys in enumerate(dataset_batches):
                    logger.info(f"处理批次 {batch_idx+1}/{len(dataset_batches)}, "
                              f"包含 {len(batch_keys)} 个数据集")
                    
                    # 批量处理数据集
                    for dataset_key in tqdm(batch_keys, desc=f"批次 {batch_idx+1}"):
                        try:
                            # 查找包含此数据集的文件
                            source_data = None
                            for filename, file_handle in file_handles.items():
                                if dataset_key in file_handle:
                                    source_data = file_handle[dataset_key][:]
                                    break
                            
                            if source_data is not None:
                                # 直接复制数据（因为每个日期文件已经包含了所有驱动因素的合并数据）
                                dataset = output_f.create_dataset(
                                    dataset_key,
                                    data=source_data,
                                    dtype=np.float32,
                                    compression='lzf',  # 更快的压缩算法
                                    shuffle=False,      # 关闭shuffle以提高速度
                                    chunks=True,        # 启用分块以优化I/O
                                    fletcher32=False    # 关闭校验和以提高速度
                                )
                                
                                dataset.attrs['data_shape'] = f"{source_data.shape[0]}x{source_data.shape[1]}"
                                dataset.attrs['total_bands'] = total_bands
                                
                                merged_count += 1
                                
                                # 立即释放内存
                                del source_data
                                
                        except Exception as e:
                            logger.error(f"处理数据集 {dataset_key} 时出错: {e}")
                            continue
                    
                    # 每个批次后强制垃圾回收
                    gc.collect()
                
                output_f.attrs['total_datasets'] = merged_count
                output_f.attrs['total_bands'] = total_bands
                
        finally:
            # 关闭所有文件句柄
            for file_handle in file_handles.values():
                file_handle.close()
        
        logger.info(f"完整数据合并完成: {output_path}, 共 {merged_count} 个数据集")
    
    def _calculate_total_bands(self, file_handles, sample_dataset_key):
        """计算总波段数"""
        total_bands = 0
        
        for driver_name in self.driver_names:
            if driver_name not in file_handles:
                continue
                
            file_handle = file_handles[driver_name]
            if sample_dataset_key not in file_handle:
                continue
                
            dataset = file_handle[sample_dataset_key]
            data_shape = dataset.shape
            
            if len(data_shape) == 2:  # 多波段数据: (bands, time_steps)
                total_bands += data_shape[0]
            else:  # 单波段数据: (time_steps,)
                total_bands += 1
        
        return total_bands
    
    def _merge_single_dataset_optimized(self, file_handles, dataset_key, expected_bands):
        """优化的单个数据集合并方法"""
        driver_data_list = []
        current_bands = 0
        time_steps_list = []
        
        for driver_name in self.driver_names:
            if driver_name not in file_handles:
                continue
                
            file_handle = file_handles[driver_name]
            if dataset_key not in file_handle:
                continue
                
            dataset = file_handle[dataset_key]
            data = dataset[:]  # shape: (time_steps,) 或 (bands, time_steps)
            
            # 处理多波段数据
            if len(data.shape) == 2:  # 多波段数据: (bands, time_steps)
                for band_idx in range(data.shape[0]):
                    band_data = data[band_idx]  # shape: (time_steps,)
                    driver_data_list.append(band_data)
                    time_steps_list.append(len(band_data))
                    current_bands += 1
            else:  # 单波段数据: (time_steps,)
                driver_data_list.append(data)
                time_steps_list.append(len(data))
                current_bands += 1
        
        # 检查是否有足够的数据
        if current_bands == 0 or len(driver_data_list) != current_bands:
            return None
        
        # 检查时间步数是否一致
        if len(set(time_steps_list)) > 1:
            # 找到最小的时间步数，截断到统一长度
            min_time_steps = min(time_steps_list)
            driver_data_list = [data[:min_time_steps] for data in driver_data_list]
        
        # 合并数据: shape = (total_bands, time_steps)
        try:
            merged_data = np.stack(driver_data_list, axis=0)
            return merged_data
        except Exception as e:
            logger.error(f"合并数据集 {dataset_key} 时出错: {e}")
            return None
    
    def _merge_full_data(self, year):
        """保持原有的完整数据合并方法作为备用"""
        logger.info(f"使用传统方法合并年份 {year} 的完整数据...")
        self._merge_full_data_optimized(year)

    def merge_all_years(self, samples_only=False):
        """
        合并所有年份的数据
        
        Args:
            samples_only: 是否只处理抽样数据（默认False，处理所有数据）
        """
        years = self._get_available_years()
        logger.info(f"发现年份: {years}")
        
        if samples_only:
            logger.info("=" * 60)
            logger.info("仅处理抽样数据模式")
            logger.info("=" * 60)
            
            # 只处理抽样数据，跳过完整数据
            for year in years:
                logger.info(f"处理年份 {year} 的抽样数据...")
                self.merge_year_data(year, include_full_data=False)
            
            logger.info("=" * 60)
            logger.info("抽样数据处理完成！")
            logger.info("如需处理完整数据，请运行: merger.merge_full_data_only()")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("分阶段处理模式：先处理抽样数据，再处理完整数据")
            logger.info("=" * 60)
            
            # 第一阶段：处理所有年份的抽样数据
            logger.info("第一阶段：处理所有年份的抽样数据")
            for year in years:
                logger.info(f"处理年份 {year} 的抽样数据...")
                self.merge_year_data(year, include_full_data=False)
            
            logger.info("抽样数据处理完成！")
            
            # 第二阶段：处理2020-2024年的完整数据
            full_data_years = [year for year in years if 2020 <= year <= 2024]
            if full_data_years:
                logger.info(f"第二阶段：处理 {full_data_years} 年份的完整数据")
                for year in full_data_years:
                    logger.info(f"处理年份 {year} 的完整数据...")
                    self._merge_full_data(year)
                logger.info("完整数据处理完成！")
            else:
                logger.info("没有找到2020-2024年的数据，跳过完整数据处理")
    
    def merge_samples_only(self):
        """只合并抽样数据"""
        logger.info("开始合并抽样数据...")
        self.merge_all_years(samples_only=True)
    
    def merge_full_data_only(self):
        """只合并完整数据（2020-2024年）"""
        years = self._get_available_years()
        full_data_years = [year for year in years if 2020 <= year <= 2024]
        
        if not full_data_years:
            logger.warning("没有找到2020-2024年的数据")
            return
        
        logger.info("=" * 60)
        logger.info(f"开始处理 {full_data_years} 年份的完整数据")
        logger.info("=" * 60)
        
        for year in full_data_years:
            logger.info(f"处理年份 {year} 的完整数据...")
            self._merge_full_data(year)
        
        logger.info("完整数据处理完成！")

    def merge_time_windows_data(self, include_full_data=True):
        """合并时间窗口数据
        
        Args:
            include_full_data: 是否包含完整数据处理（默认True）
        """
        available_windows = self._get_available_time_windows()
        
        if not available_windows:
            logger.info("没有找到时间窗口格式的文件")
            return
        
        logger.info(f"找到 {len(available_windows)} 个时间窗口")
        
        for window_id in available_windows:
            logger.info(f"开始合并时间窗口 {window_id} 的数据...")
            
            # 合并这个时间窗口的数据
            if include_full_data:
                logger.info(f"处理时间窗口 {window_id} 的完整数据...")
                self._merge_time_window_full_data(window_id)
            else:
                logger.info(f"跳过时间窗口 {window_id} 的完整数据处理")

    def _merge_time_window_full_data(self, window_id):
        """合并指定时间窗口的完整数据
        
        Args:
            window_id: 时间窗口ID，格式为 YYYY_MMDD_MMDD
        """
        logger.info(f"开始合并时间窗口 {window_id} 的完整数据...")
        
        # 检查输出文件是否已存在
        output_full_file = os.path.join(self.output_dir, f'{window_id}_full.h5')
        if os.path.exists(output_full_file):
            logger.info(f"时间窗口完整数据文件已存在，跳过: {output_full_file}")
            return
        
        # 获取该时间窗口的所有驱动因素文件
        driver_files = {}
        for driver_name in self.driver_names:
            full_file = os.path.join(self.pixel_samples_dir, f'{window_id}_{driver_name}_full.h5')
            if os.path.exists(full_file):
                driver_files[driver_name] = full_file
            else:
                logger.warning(f"时间窗口 {window_id} 缺少驱动因素文件: {driver_name}")
        
        if not driver_files:
            logger.warning(f"时间窗口 {window_id} 没有找到任何驱动因素完整数据文件")
            return
        
        logger.info(f"时间窗口 {window_id} 找到 {len(driver_files)} 个驱动因素文件")
        
        # 读取所有数据集信息
        all_dataset_keys = set()
        geo_reference = None
        
        # 从第一个文件获取所有数据集的键和地理参考信息
        first_driver = list(driver_files.keys())[0]
        with h5py.File(driver_files[first_driver], 'r') as f:
            all_dataset_keys = set(f.keys())
            
            # 获取地理参考信息（检查是否为group或属性）
            if 'geo_reference' in f:
                # 如果是group，记录路径
                geo_reference = {'type': 'group', 'path': 'geo_reference'}
            elif 'geotransform' in f.attrs:
                # 如果是属性
                geo_reference = {
                    'type': 'attrs',
                    'geotransform': f.attrs['geotransform'],
                    'projection': f.attrs['projection'],
                    'raster_size': f.attrs['raster_size'],
                    'raster_count': f.attrs['raster_count']
                }
        
        logger.info(f"时间窗口 {window_id} 总共有 {len(all_dataset_keys)} 个数据集")
        
        # 创建输出文件并合并数据
        with h5py.File(output_full_file, 'w') as output_file:
            # 写入全局属性
            output_file.attrs['window_id'] = window_id
            output_file.attrs['num_drivers'] = len(driver_files)
            output_file.attrs['driver_names'] = list(driver_files.keys())
            output_file.attrs['data_format'] = 'bands_x_time'
            output_file.attrs['data_type'] = 'merged_time_window_full'
            output_file.attrs['dataset_naming'] = 'YYYYMMDD_{past/future}_{firms_value}_row_col'
            
            # 复制地理参考信息
            if geo_reference:
                if geo_reference['type'] == 'group':
                    # 如果是group，从第一个文件复制整个group
                    with h5py.File(driver_files[first_driver], 'r') as first_f:
                        if 'geo_reference' in first_f:
                            first_f.copy('geo_reference', output_file)
                            logger.info("已复制geo_reference group")
                else:
                    # 如果是属性，复制属性
                    output_file.attrs['geotransform'] = geo_reference['geotransform']
                    output_file.attrs['projection'] = geo_reference['projection']
                    output_file.attrs['raster_size'] = geo_reference['raster_size']
                    output_file.attrs['raster_count'] = geo_reference['raster_count']
                    logger.info("已复制geo_reference属性")
            
            # 从第一个驱动因素文件复制其他属性
            with h5py.File(driver_files[first_driver], 'r') as first_f:
                for attr_name in ['past_days', 'future_days', 'sampling_type']:
                    if attr_name in first_f.attrs:
                        output_file.attrs[attr_name] = first_f.attrs[attr_name]
            
            merged_count = 0
            total_bands = 0
            
            # 处理每个数据集
            for dataset_key in tqdm(all_dataset_keys, desc=f"合并时间窗口 {window_id} 数据"):
                # 收集所有驱动因素的数据
                driver_data_list = []
                current_bands = 0
                time_steps_list = []  # 记录每个数据的时间步数
                
                for driver_name in self.driver_names:
                    if driver_name not in driver_files:
                        continue
                        
                    with h5py.File(driver_files[driver_name], 'r') as driver_f:
                        if dataset_key not in driver_f:
                            continue
                            
                        obj = driver_f[dataset_key]
                        
                        # 检查对象类型，只处理dataset，跳过group
                        if not isinstance(obj, h5py.Dataset):
                            logger.debug(f"跳过非数据集对象: {dataset_key} (类型: {type(obj)})")
                            continue
                            
                        data = obj[:]  # shape: (time_steps,) 或 (bands, time_steps)
                        
                        # 处理多波段数据
                        if len(data.shape) == 2:  # 多波段数据: (bands, time_steps)
                            # 将每个波段作为单独的通道
                            for band_idx in range(data.shape[0]):
                                band_data = data[band_idx]
                                driver_data_list.append(band_data)
                                time_steps_list.append(len(band_data))
                                current_bands += 1
                        else:  # 单波段数据: (time_steps,)
                            driver_data_list.append(data)
                            time_steps_list.append(len(data))
                            current_bands += 1
                
                # 检查是否所有驱动因素都有数据
                if len(driver_data_list) != current_bands:
                    continue
                
                # 检查时间步数是否一致
                if len(set(time_steps_list)) > 1:
                    logger.warning(f"数据集 {dataset_key} 的时间步数不一致: {set(time_steps_list)}")
                    # 找到最小的时间步数，截断到统一长度
                    min_time_steps = min(time_steps_list)
                    driver_data_list = [data[:min_time_steps] for data in driver_data_list]
                    logger.info(f"将所有数据截断到 {min_time_steps} 个时间步")
                
                # 合并数据: shape = (total_bands, time_steps)
                merged_data = np.stack(driver_data_list, axis=0)
                
                # 保持原始数据集名称格式
                dataset = output_file.create_dataset(
                    dataset_key,  # 保持原始名称: YYYYMMDD_{past/future}_{firms_value}_row_col
                    data=merged_data,
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=4,
                    shuffle=True
                )
                
                # 添加新的属性
                dataset.attrs['data_shape'] = f"{merged_data.shape[0]}x{merged_data.shape[1]}"
                dataset.attrs['total_bands'] = current_bands
                
                merged_count += 1
                total_bands = current_bands  # 所有数据集的波段数应该相同
            
            output_file.attrs['total_datasets'] = merged_count
            output_file.attrs['total_bands'] = total_bands
            
            logger.info(f"时间窗口 {window_id} 完整数据合并完成: {output_full_file}, 共 {merged_count} 个数据集")

    def _merge_time_window_single_dataset(self, file_handles, dataset_key, output_file):
        """合并时间窗口的单个数据集 - 已废弃，使用新的合并逻辑"""
        # 这个方法已不再使用，保留是为了兼容性
        pass

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='合并像素样本数据')
    parser.add_argument('--mode', choices=['all', 'samples', 'full', 'windows'], default='all',
                       help='处理模式: all=分阶段处理所有数据, samples=仅抽样数据, full=仅完整数据, windows=时间窗口数据')
    parser.add_argument('--pixel_samples_dir', 
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples',
                       help='像素样本数据目录')
    parser.add_argument('--output_dir',
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples_merged',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建合并器
    merger = PixelSamplesMerger(
        pixel_samples_dir=args.pixel_samples_dir,
        output_dir=args.output_dir
    )
    
    # 根据模式执行不同的处理
    if args.mode == 'samples':
        logger.info("=" * 60)
        logger.info("模式：仅处理抽样数据")
        logger.info("=" * 60)
        merger.merge_samples_only()
    elif args.mode == 'full':
        logger.info("=" * 60)
        logger.info("模式：仅处理完整数据")
        logger.info("=" * 60)
        merger.merge_full_data_only()
    elif args.mode == 'windows':
        logger.info("=" * 60)
        logger.info("模式：处理时间窗口数据")
        logger.info("=" * 60)
        merger.merge_time_windows_data()
    elif args.mode == 'all':
        logger.info("=" * 60)
        logger.info("模式：分阶段处理所有数据")
        logger.info("=" * 60)
        
        # 检查是否有时间窗口数据
        available_windows = merger._get_available_time_windows()
        if available_windows:
            logger.info("检测到时间窗口数据，优先处理时间窗口...")
            merger.merge_time_windows_data()
        else:
            logger.info("未检测到时间窗口数据，处理年份数据...")
        merger.merge_all_years(samples_only=False)
    
    logger.info("数据合并完成！")

if __name__ == "__main__":
    main() 