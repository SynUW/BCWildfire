import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import re
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
import glob

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesPixelDataset(Dataset):
    """
    适配merge_pixel_samples.py生成的H5文件的数据加载器
    
    数据格式:
    - 文件名: {year}_samples.h5 (抽样数据) 或 {year}_full.h5 (完整数据)
    - 数据集名: YYYYMMDD_{past/future}_{firms_value}_row_col (合并后的数据)
    - 数据形状: (total_bands, total_time_steps) 其中 total_time_steps = past_days + future_days
    - 数据类型: float32 (已标准化)
    - 默认配置: past_days=365, future_days=30, total_bands=39
    
    特性: 
    - 默认分离过去和未来数据，返回 (past_data, future_data)
    - 数据已经标准化，无需额外处理
    - 支持抽样数据和完整数据的区分
    - 支持按年份进行训练/验证/测试划分
    - 要求所有数据尺寸一致，不一致则报错
    """
    
    def __init__(self, h5_dir, years=None, firms_values=None, return_metadata=False, 
                 use_full_data=False):
        """
        初始化时间序列像素数据集
        
        Args:
            h5_dir: H5文件目录
            years: 要加载的年份列表，None表示加载所有年份
            firms_values: 要加载的FIRMS值列表，None表示加载所有值
            return_metadata: 是否返回元数据（日期、坐标、FIRMS值等）
            use_full_data: 是否使用完整数据（True: {year}_full.h5, False: {year}_samples.h5）
        """
        self.h5_dir = h5_dir
        self.years = years
        self.firms_values = firms_values
        self.return_metadata = return_metadata
        self.use_full_data = use_full_data
        
        # 获取H5文件列表
        self.h5_files = self._get_h5_files()
        
        # 构建样本索引
        self.sample_index = []  # (h5_path, dataset_key, metadata)
        self.dataset_info = {}  # 存储数据集信息
        
        self._build_index()
        
        logger.info(f"数据集初始化完成，共 {len(self.sample_index)} 个样本")
    
    def custom_collate_fn(self, batch):
        """
        自定义的collate函数，要求所有数据尺寸一致，不一致则报错
        默认分离过去和未来数据
        """
        if self.return_metadata:
            past_data_list, future_data_list, metadata_list = zip(*batch)
        else:
            past_data_list, future_data_list = zip(*batch)
        
        # 检查所有tensor的形状
        past_shapes = [data.shape for data in past_data_list]
        future_shapes = [data.shape for data in future_data_list]
        
        # 检查past数据形状一致性
        if len(set(past_shapes)) > 1:
            raise ValueError(f"Past数据形状不一致: {set(past_shapes)}")
        
        # 检查future数据形状一致性
        if len(set(future_shapes)) > 1:
            raise ValueError(f"Future数据形状不一致: {set(future_shapes)}")
        
        # 直接堆叠，形状必须一致
        past_batch = torch.stack(past_data_list, dim=0)
        future_batch = torch.stack(future_data_list, dim=0)
        
        if self.return_metadata:
            return past_batch, future_batch, metadata_list
        else:
            return past_batch, future_batch
    
    def _get_h5_files(self):
        """获取符合条件的H5文件列表"""
        h5_files = []
        
        # 根据use_full_data参数选择文件类型
        file_suffix = '_full.h5' if self.use_full_data else '_samples.h5'
        pattern = r'(\d{4})_full\.h5' if self.use_full_data else r'(\d{4})_samples\.h5'
        
        for filename in os.listdir(self.h5_dir):
            if not filename.endswith(file_suffix):
                continue
                
            # 提取年份
            year_match = re.match(pattern, filename)
            if not year_match:
                continue
                
            year = int(year_match.group(1))
            
            # 检查年份过滤条件
            if self.years is not None and year not in self.years:
                continue
                
            h5_path = os.path.join(self.h5_dir, filename)
            h5_files.append((h5_path, year))
        
        data_type = "完整数据" if self.use_full_data else "抽样数据"
        logger.info(f"找到 {len(h5_files)} 个{data_type}文件")
        return h5_files
    
    def _build_index(self):
        """构建样本索引"""
        logger.info("构建样本索引...")
        
        for h5_path, year in tqdm(self.h5_files, desc="扫描H5文件"):
            try:
                with h5py.File(h5_path, 'r') as f:
                    # 获取数据集信息
                    if h5_path not in self.dataset_info:
                        self.dataset_info[h5_path] = {
                            'year': year,
                            'total_bands': f.attrs.get('total_bands', 0),
                            'driver_names': f.attrs.get('driver_names', []),
                            'past_days': f.attrs.get('past_days', 0),
                            'future_days': f.attrs.get('future_days', 0),
                            'data_format': f.attrs.get('data_format', 'unknown')
                        }
                    
                    # 扫描所有数据集
                    for dataset_key in f.keys():
                        # 解析数据集名称: YYYYMMDD_{past/future}_{firms_value}_row_col
                        metadata = self._parse_dataset_key(dataset_key)
                        if metadata is None:
                            continue
                        
                        # 检查FIRMS值过滤条件
                        if (self.firms_values is not None and 
                            metadata['firms_value'] not in self.firms_values):
                            continue
                        
                        # 添加到索引
                        self.sample_index.append((h5_path, dataset_key, metadata))
                        
            except Exception as e:
                logger.error(f"处理文件 {h5_path} 时出错: {str(e)}")
                continue
    
    def _parse_dataset_key(self, dataset_key):
        """
        解析数据集键名
        格式: YYYYMMDD_{past/future}_{firms_value}_row_col
        """
        try:
            # 使用正则表达式解析
            pattern = r'(\d{8})_(past|future)_(\d+(?:\.\d+)?)_(\d+)_(\d+)'
            match = re.match(pattern, dataset_key)
            
            if not match:
                return None
            
            date_str, time_type, firms_value_str, row_str, col_str = match.groups()
            
            return {
                'date': datetime.strptime(date_str, '%Y%m%d'),
                'time_type': time_type,
                'firms_value': float(firms_value_str),
                'row': int(row_str),
                'col': int(col_str),
                'pixel_coord': (int(row_str), int(col_str))
            }
            
        except Exception as e:
            logger.debug(f"解析数据集键名失败: {dataset_key}, 错误: {str(e)}")
            return None
    
    def _is_valid_sample(self, sample_group):
        """验证样本数据是否有效"""
        try:
            # 检查必要的属性
            if not all(attr in sample_group.attrs for attr in ['year', 'driver']):
                return False
            
            # 检查年份是否符合要求
            year = int(sample_group.attrs['year'])
            if self.years and year not in self.years:
                return False
            
            # 检查数据维度
            if 'data' not in sample_group:
                return False
                
            data = sample_group['data'][:]
            if len(data.shape) != 2:  # 应该是 (bands, time_steps)
                return False
                
            # 根据样本类型检查时间步数
            time_steps = data.shape[1]
            sample_id = sample_group.name.split('/')[-1]
            
            # 解析数据集名称格式: YYYYMMDD_{past/future}_{firms_value}_row_col
            parts = sample_id.split('_')
            if len(parts) < 4:
                return False
            
            data_type = parts[1]  # past 或 future
            
            # 检查时间步数
            if data_type == 'past' and time_steps != 365:
                return False
            elif data_type == 'future' and time_steps != 30:
                return False
            elif data_type not in ['past', 'future']:
                return False
            
            # 如果指定了FIRMS值过滤，检查FIRMS值
            if self.firms_values:
                try:
                    firms_value = int(parts[2])
                    if firms_value not in self.firms_values:
                        return False
                except (ValueError, IndexError):
                    return False
                
            return True
            
        except Exception as e:
            logger.error(f"验证样本时出错: {str(e)}")
            return False
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            - past_data: (bands, past_time_steps) 
            - future_data: (bands, future_time_steps)
            - metadata (可选): 包含日期、坐标等信息
        """
        h5_path, dataset_key, metadata = self.sample_index[idx]
        
        try:
            with h5py.File(h5_path, 'r') as f:
                data = f[dataset_key][:]  # shape: (total_bands, time_steps)
                
                # 确保数据是2D格式 (bands, time_steps)
                if data.ndim == 1:
                    data = data[np.newaxis, :]  # 添加波段维度
                elif data.ndim > 2:
                    logger.warning(f"数据维度异常: {data.shape}, 数据集: {dataset_key}")
                    data = data.reshape(data.shape[0], -1)  # 展平为2D
                
                # 转换为torch tensor
                data = torch.from_numpy(data).float()
                
                # 根据数据集类型返回对应的数据
                time_type = metadata['time_type']  # 'past' 或 'future'
                
                if time_type == 'past':
                    # 对于past数据，必须找到对应的future数据
                    future_dataset_key = dataset_key.replace('_past_', '_future_')
                    if future_dataset_key not in f:
                        raise ValueError(f"Past数据 {dataset_key} 缺少对应的Future数据 {future_dataset_key}")
                    
                    future_data = f[future_dataset_key][:]
                    if future_data.ndim == 1:
                        future_data = future_data[np.newaxis, :]
                    elif future_data.ndim > 2:
                        future_data = future_data.reshape(future_data.shape[0], -1)
                    future_data = torch.from_numpy(future_data).float()
                    
                    past_data = data
                    
                elif time_type == 'future':
                    # 对于future数据，必须找到对应的past数据
                    past_dataset_key = dataset_key.replace('_future_', '_past_')
                    if past_dataset_key not in f:
                        raise ValueError(f"Future数据 {dataset_key} 缺少对应的Past数据 {past_dataset_key}")
                    
                    past_data = f[past_dataset_key][:]
                    if past_data.ndim == 1:
                        past_data = past_data[np.newaxis, :]
                    elif past_data.ndim > 2:
                        past_data = past_data.reshape(past_data.shape[0], -1)
                    past_data = torch.from_numpy(past_data).float()
                    
                    future_data = data
                
                else:
                    raise ValueError(f"未知的时间类型: {time_type}")
                
                # 验证数据形状的合理性
                expected_past_steps = 365
                expected_future_steps = 30
                
                if past_data.shape[1] != expected_past_steps:
                    raise ValueError(f"Past数据时间步数错误: 期望{expected_past_steps}, 实际{past_data.shape[1]}, 数据集: {dataset_key}")
                
                if future_data.shape[1] != expected_future_steps:
                    raise ValueError(f"Future数据时间步数错误: 期望{expected_future_steps}, 实际{future_data.shape[1]}, 数据集: {dataset_key}")
                
                if past_data.shape[0] != future_data.shape[0]:
                    raise ValueError(f"Past和Future数据波段数不匹配: Past={past_data.shape[0]}, Future={future_data.shape[0]}, 数据集: {dataset_key}")
                
                if self.return_metadata:
                    return past_data, future_data, metadata
                else:
                    return past_data, future_data
                        
        except Exception as e:
            logger.error(f"读取样本失败: {dataset_key}, 错误: {str(e)}")
            raise e  # 重新抛出异常，不要掩盖问题
    
    def get_dataset_info(self):
        """获取数据集信息"""
        return self.dataset_info
    
    def get_sample_by_criteria(self, year=None, firms_value=None, date_range=None):
        """
        根据条件筛选样本
        
        Args:
            year: 年份
            firms_value: FIRMS值
            date_range: 日期范围 (start_date, end_date)
        
        Returns:
            符合条件的样本索引列表
        """
        matching_indices = []
        
        for idx, (h5_path, dataset_key, metadata) in enumerate(self.sample_index):
            # 检查年份
            if year is not None and self.dataset_info[h5_path]['year'] != year:
                continue
            
            # 检查FIRMS值
            if firms_value is not None and metadata['firms_value'] != firms_value:
                continue
            
            # 检查日期范围
            if date_range is not None:
                start_date, end_date = date_range
                if not (start_date <= metadata['date'] <= end_date):
                    continue
            
            matching_indices.append(idx)
        
        return matching_indices
    
    def get_statistics(self):
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.sample_index),
            'years': set(),
            'firms_values': set(),
            'time_types': set(),
            'files': len(self.h5_files)
        }
        
        for h5_path, dataset_key, metadata in self.sample_index:
            stats['years'].add(self.dataset_info[h5_path]['year'])
            stats['firms_values'].add(metadata['firms_value'])
            stats['time_types'].add(metadata['time_type'])
        
        # 转换为列表并排序
        stats['years'] = sorted(list(stats['years']))
        stats['firms_values'] = sorted(list(stats['firms_values']))
        stats['time_types'] = sorted(list(stats['time_types']))
        
        return stats


class TimeSeriesDataLoader:
    """时间序列数据加载器的便捷包装类"""
    
    def __init__(self, h5_dir, **dataset_kwargs):
        """
        初始化数据加载器
        
        Args:
            h5_dir: H5文件目录
            **dataset_kwargs: 传递给TimeSeriesPixelDataset的参数
        """
        self.dataset = TimeSeriesPixelDataset(h5_dir, **dataset_kwargs)
    
    def create_dataloader(self, batch_size=32, shuffle=True, num_workers=4, **dataloader_kwargs):
        """
        创建PyTorch DataLoader
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            **dataloader_kwargs: 传递给DataLoader的其他参数
        
        Returns:
            torch.utils.data.DataLoader
        """
        # 使用自定义的collate函数来处理尺寸不一致的问题
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.dataset.custom_collate_fn,
            **dataloader_kwargs
        )
    
    def get_year_based_split(self, train_years, val_years, test_years, test_full_years=None):
        """
        基于年份创建训练、验证和测试数据集分割
        
        Args:
            train_years: 训练年份列表（使用抽样数据）
            val_years: 验证年份列表（使用抽样数据）
            test_years: 测试年份列表（使用抽样数据）
            test_full_years: 完整数据测试年份列表（使用完整数据，可选）
        
        Returns:
            如果test_full_years为None: (train_indices, val_indices, test_indices)
            如果test_full_years不为None: (train_indices, val_indices, test_indices, test_full_indices)
        """
        train_indices = []
        val_indices = []
        test_indices = []
        
        for idx, (h5_path, dataset_key, metadata) in enumerate(self.dataset.sample_index):
            year = self.dataset.dataset_info[h5_path]['year']
            
            if year in train_years:
                train_indices.append(idx)
            elif year in val_years:
                val_indices.append(idx)
            elif year in test_years:
                test_indices.append(idx)
        
        data_type = "完整数据" if self.dataset.use_full_data else "抽样数据"
        logger.info(f"年份划分结果 ({data_type}):")
        logger.info(f"  训练集: {len(train_indices)} 样本 (年份: {train_years})")
        logger.info(f"  验证集: {len(val_indices)} 样本 (年份: {val_years})")
        logger.info(f"  测试集: {len(test_indices)} 样本 (年份: {test_years})")
        
        # 如果指定了完整数据测试年份，创建完整数据测试集
        if test_full_years is not None:
            # 创建完整数据加载器
            full_dataset = TimeSeriesPixelDataset(
                h5_dir=self.dataset.h5_dir,
                years=test_full_years,
                firms_values=self.dataset.firms_values,
                return_metadata=self.dataset.return_metadata,
                use_full_data=True
            )
            
            # 获取完整数据的所有索引
            test_full_indices = list(range(len(full_dataset)))
            
            logger.info(f"完整数据测试集: {len(test_full_indices)} 样本 (年份: {test_full_years})")
            
            return train_indices, val_indices, test_indices, test_full_indices, full_dataset
        
        return train_indices, val_indices, test_indices


if __name__ == '__main__':
    """使用示例"""
    print("=" * 60)
    print("TimeSeriesPixelDataset 使用示例")
    print("=" * 60)
    
    h5_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples_merged'
    
    # 1. 创建抽样数据加载器
    print("\n1. 创建抽样数据加载器")
    data_loader = TimeSeriesDataLoader(
        h5_dir=h5_dir,
        years=None,  # 加载所有年份
        firms_values=None,  # 加载所有FIRMS值
        return_metadata=False,  # 不返回元数据（训练时通常不需要）
        use_full_data=False  # 使用抽样数据
    )
    
    # 2. 查看数据集统计信息
    print("\n2. 数据集统计信息")
    stats = data_loader.dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 3. 基于年份划分数据集
    print("\n3. 基于年份划分数据集")
    train_years = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 
                   2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    val_years = [2019, 2020]
    test_years = [2021, 2022]
    test_full_years = [2019, 2020]  # 使用完整数据进行最终测试
    
    result = data_loader.get_year_based_split(
        train_years, val_years, test_years, test_full_years
    )
    
    if len(result) == 5:
        train_indices, val_indices, test_indices, test_full_indices, full_dataset = result
    else:
        train_indices, val_indices, test_indices = result
        test_full_indices = []
        full_dataset = None
    
    # 4. 创建PyTorch DataLoader
    print("\n4. 创建PyTorch DataLoader")
    from torch.utils.data import Subset
    
    # 训练集
    train_dataset = Subset(data_loader.dataset, train_indices)
    train_dataloader = data_loader.create_dataloader(
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # 验证集
    val_dataset = Subset(data_loader.dataset, val_indices)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=data_loader.dataset.custom_collate_fn
    )
    
    # 测试集
    test_dataset = Subset(data_loader.dataset, test_indices)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=data_loader.dataset.custom_collate_fn
    )
    
    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  验证集大小: {len(val_dataset)}")
    print(f"  测试集大小: {len(test_dataset)}")    
    # 完整数据测试集
    if full_dataset is not None:
        test_full_dataset = Subset(full_dataset, test_full_indices)
        test_full_dataloader = DataLoader(
            test_full_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=full_dataset.custom_collate_fn
        )
        print(f"  完整数据测试集大小: {len(test_full_dataset)}")
