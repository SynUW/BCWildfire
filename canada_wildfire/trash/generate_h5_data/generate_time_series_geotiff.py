#!/usr/bin/env python3
"""
时间序列GeoTIFF生成器
- 针对每个驱动因素的每个波段，生成整个时间尺度的GeoTIFF文件
- 输出格式：h*w*t的GeoTIFF文件，其中t是时间维度
- 支持多波段数据的分离处理
- 保持地理参考信息
- 每个波段以日期命名（YYYY_MM_DD格式）
"""

import os
import glob
import numpy as np
from osgeo import gdal, osr
import logging
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from collections import defaultdict
import gc
import time
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('TimeSeriesGeoTIFF')

# 设置GDAL配置
gdal.SetConfigOption('GDAL_CACHEMAX', '2048')
gdal.SetConfigOption('GDAL_NUM_THREADS', '4')
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
gdal.SetConfigOption('VSI_CACHE', 'TRUE')

class TimeSeriesGeoTIFFGenerator:
    def __init__(self, data_dir, output_dir, years=None, max_workers=8, 
                 compression='LZW', tiled=True, bigtiff=True):
        """
        时间序列GeoTIFF生成器
        
        Args:
            data_dir: 驱动因素数据根目录
            output_dir: 输出目录
            years: 要处理的年份列表，None表示处理所有年份
            max_workers: 最大线程数
            compression: 压缩方式 ('LZW', 'DEFLATE', 'PACKBITS', None)
            tiled: 是否使用分块存储
            bigtiff: 是否使用BigTIFF格式（支持>4GB文件）
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.years = years
        self.max_workers = max_workers
        self.compression = compression
        self.tiled = tiled
        self.bigtiff = bigtiff
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有驱动因素目录
        self.driver_dirs = self._get_driver_directories()
        logger.info(f"找到 {len(self.driver_dirs)} 个驱动因素目录: {list(self.driver_dirs.keys())}")
        
    def _get_driver_directories(self):
        """获取所有驱动因素目录"""
        driver_dirs = {}
        
        # 要排除的目录列表
        # Topo_Distance_WGS84_resize_resampled: 地形距离数据，通常是静态数据，不需要时间序列处理
        excluded_dirs = {'Topo_Distance_WGS84_resize_resampled'}
        
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                # 检查是否在排除列表中
                if item in excluded_dirs:
                    logger.info(f"跳过排除的驱动因素目录: {item}")
                    continue
                
                # 检查目录中是否有tif文件
                tif_files = glob.glob(os.path.join(item_path, '*.tif'))
                if tif_files:
                    driver_dirs[item] = item_path
        
        return driver_dirs
    
    def _extract_date_from_filename(self, filename):
        """从文件名提取日期"""
        try:
            # 文件名格式为 drivers_yyyy_mm_dd.tif 或 *_yyyy_mm_dd.tif
            basename = os.path.basename(filename)
            parts = basename.split('_')
            
            # 寻找日期模式 YYYY_MM_DD
            for i in range(len(parts) - 2):
                if (len(parts[i]) == 4 and parts[i].isdigit() and
                    len(parts[i+1]) == 2 and parts[i+1].isdigit() and
                    len(parts[i+2]) == 2 and parts[i+2].split('.')[0].isdigit()):
                    year = int(parts[i])
                    month = int(parts[i+1])
                    # 处理最后一部分可能包含.tif的情况
                    day_part = parts[i+2].split('.')[0]
                    day = int(day_part)
                    return datetime(year, month, day)
            
            # 如果上面的方法失败，尝试其他常见格式
            # 移除.tif扩展名
            name_without_ext = basename.replace('.tif', '').replace('.TIF', '')
            
            # 尝试匹配末尾的日期格式
            # 匹配 YYYY_MM_DD 或 YYYYMMDD 格式
            date_patterns = [
                r'(\d{4})_(\d{2})_(\d{2})$',  # YYYY_MM_DD
                r'(\d{4})(\d{2})(\d{2})$',    # YYYYMMDD
                r'.*_(\d{4})_(\d{2})_(\d{2})$',  # 任意前缀_YYYY_MM_DD
                r'.*_(\d{4})(\d{2})(\d{2})$'     # 任意前缀_YYYYMMDD
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, name_without_ext)
                if match:
                    year, month, day = map(int, match.groups())
                    return datetime(year, month, day)
            
            logger.debug(f"无法从文件名提取日期: {filename}")
            return None
            
        except Exception as e:
            logger.debug(f"从文件名提取日期失败: {filename}, 错误: {str(e)}")
            return None
    
    def _get_file_info(self, file_path):
        """获取文件的基本信息"""
        try:
            ds = gdal.Open(file_path, gdal.GA_ReadOnly)
            if ds is None:
                return None
            
            info = {
                'width': ds.RasterXSize,
                'height': ds.RasterYSize,
                'bands': ds.RasterCount,
                'geotransform': ds.GetGeoTransform(),
                'projection': ds.GetProjection(),
                'datatype': ds.GetRasterBand(1).DataType
            }
            
            ds = None
            return info
            
        except Exception as e:
            logger.error(f"获取文件信息失败: {file_path}, 错误: {str(e)}")
            return None
    
    def _load_single_file(self, file_path):
        """加载单个文件的所有波段数据"""
        try:
            ds = gdal.Open(file_path, gdal.GA_ReadOnly)
            if ds is None:
                return None
            
            # 获取文件信息
            bands = ds.RasterCount
            height = ds.RasterYSize
            width = ds.RasterXSize
            
            # 读取所有波段数据
            data = ds.ReadAsArray()
            ds = None
            
            # 确保数据是3D格式 (bands, height, width)
            if data.ndim == 2:
                # 单波段数据，添加波段维度
                data = data[np.newaxis, :, :]  # (height, width) -> (1, height, width)
            elif data.ndim == 3:
                # 多波段数据，检查维度顺序
                if data.shape == (bands, height, width):
                    # 已经是正确的格式 (bands, height, width)
                    pass
                elif data.shape == (height, width, bands):
                    # 需要转置 (height, width, bands) -> (bands, height, width)
                    data = np.transpose(data, (2, 0, 1))
                else:
                    # 其他情况，根据已知的文件信息来判断
                    logger.warning(f"文件 {file_path} 的数据形状 {data.shape} 与预期不符，"
                                 f"预期: ({bands}, {height}, {width})")
                    # 尝试重新整形
                    if data.size == bands * height * width:
                        data = data.reshape(bands, height, width)
                    else:
                        logger.error(f"无法重新整形数据: {data.shape}")
                        return None
            
            return data
            
        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {str(e)}")
            return None
    
    def _create_time_series_geotiff(self, driver_name, band_idx, time_series_data, 
                                   dates, geo_info, output_path):
        """
        创建时间序列GeoTIFF文件
        
        Args:
            driver_name: 驱动因素名称
            band_idx: 波段索引
            time_series_data: 时间序列数据 (time_steps, height, width)
            dates: 日期列表
            geo_info: 地理参考信息
            output_path: 输出文件路径
        """
        try:
            time_steps, height, width = time_series_data.shape
            
            # 创建输出驱动
            driver = gdal.GetDriverByName('GTiff')
            
            # 设置创建选项
            creation_options = []
            if self.compression:
                creation_options.append(f'COMPRESS={self.compression}')
            if self.tiled:
                creation_options.append('TILED=YES')
                creation_options.append('BLOCKXSIZE=512')
                creation_options.append('BLOCKYSIZE=512')
            if self.bigtiff:
                creation_options.append('BIGTIFF=YES')
            
            # 添加其他优化选项
            creation_options.extend([
                'PREDICTOR=2',  # 对于浮点数据使用预测器
                'NUM_THREADS=ALL_CPUS',
                'INTERLEAVE=BAND'
            ])
            
            # 创建数据集
            ds = driver.Create(
                output_path,
                width, height, time_steps,
                geo_info['datatype'],
                creation_options
            )
            
            if ds is None:
                raise Exception(f"无法创建输出文件: {output_path}")
            
            # 设置地理参考信息
            ds.SetGeoTransform(geo_info['geotransform'])
            ds.SetProjection(geo_info['projection'])
            
            # 写入每个时间步的数据
            for t in range(time_steps):
                band = ds.GetRasterBand(t + 1)
                band.WriteArray(time_series_data[t])
                
                # 设置波段名称为日期格式
                if t < len(dates):
                    date_str = dates[t].strftime('%Y_%m_%d')
                    band_name = f'{date_str}'
                else:
                    band_name = f'time_{t:04d}'
                
                band.SetDescription(band_name)
                
                # 设置波段的元数据
                band_metadata = {
                    'DATE': dates[t].strftime('%Y-%m-%d') if t < len(dates) else f'unknown',
                    'DRIVER': driver_name,
                    'BAND_INDEX': str(band_idx),
                    'TIME_STEP': str(t)
                }
                band.SetMetadata(band_metadata)
                
                # 设置NoData值（如果需要）
                # band.SetNoDataValue(-9999)
                
                band.FlushCache()
            
            # 设置元数据
            metadata = {
                'DRIVER_NAME': driver_name,
                'BAND_INDEX': str(band_idx),
                'TIME_STEPS': str(time_steps),
                'START_DATE': dates[0].strftime('%Y-%m-%d') if dates else 'unknown',
                'END_DATE': dates[-1].strftime('%Y-%m-%d') if dates else 'unknown',
                'CREATION_TIME': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            ds.SetMetadata(metadata)
            
            # 强制写入并关闭
            ds.FlushCache()
            ds = None
            
            logger.info(f"成功创建时间序列GeoTIFF: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建时间序列GeoTIFF失败: {output_path}, 错误: {str(e)}")
            return False
    
    def _process_driver_band(self, driver_name, band_idx, file_date_pairs, geo_info):
        """
        处理单个驱动因素的单个波段
        
        Args:
            driver_name: 驱动因素名称
            band_idx: 波段索引
            file_date_pairs: (文件路径, 日期) 对的列表
            geo_info: 地理参考信息
        
        Returns:
            处理结果
        """
        logger.info(f"开始处理 {driver_name} 的波段 {band_idx}")
        
        try:
            # 按日期排序
            file_date_pairs.sort(key=lambda x: x[1])
            
            # 收集时间序列数据
            time_series_data = []
            valid_dates = []
            
            for file_path, date in tqdm(file_date_pairs, 
                                      desc=f"加载 {driver_name} 波段 {band_idx}"):
                # 加载文件数据
                data = self._load_single_file(file_path)
                if data is None:
                    logger.warning(f"跳过无效文件: {file_path}")
                    continue
                
                # 检查波段索引是否有效
                if band_idx >= data.shape[0]:
                    logger.warning(f"波段索引 {band_idx} 超出范围，文件 {file_path} 只有 {data.shape[0]} 个波段")
                    continue
                
                # 提取指定波段的数据
                band_data = data[band_idx]  # shape: (height, width)
                time_series_data.append(band_data)
                valid_dates.append(date)
            
            if not time_series_data:
                logger.warning(f"{driver_name} 波段 {band_idx} 没有有效数据")
                return {'status': 'no_data', 'driver': driver_name, 'band': band_idx}
            
            # 转换为numpy数组
            time_series_array = np.stack(time_series_data, axis=0)  # shape: (time_steps, height, width)
            
            logger.info(f"{driver_name} 波段 {band_idx} 时间序列形状: {time_series_array.shape}")
            
            # 创建输出文件名
            output_filename = f"{driver_name}_band_{band_idx}_timeseries.tif"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 创建时间序列GeoTIFF
            success = self._create_time_series_geotiff(
                driver_name, band_idx, time_series_array, 
                valid_dates, geo_info, output_path
            )
            
            if success:
                return {
                    'status': 'success',
                    'driver': driver_name,
                    'band': band_idx,
                    'time_steps': len(valid_dates),
                    'output_path': output_path,
                    'file_size_mb': os.path.getsize(output_path) / (1024 * 1024)
                }
            else:
                return {
                    'status': 'error',
                    'driver': driver_name,
                    'band': band_idx,
                    'error': 'Failed to create GeoTIFF'
                }
                
        except Exception as e:
            logger.error(f"处理 {driver_name} 波段 {band_idx} 时出错: {str(e)}")
            return {
                'status': 'error',
                'driver': driver_name,
                'band': band_idx,
                'error': str(e)
            }
    
    def _process_single_driver(self, driver_name):
        """
        处理单个驱动因素的所有波段
        
        Args:
            driver_name: 驱动因素名称
        
        Returns:
            处理结果列表
        """
        logger.info(f"开始处理驱动因素: {driver_name}")
        
        driver_path = self.driver_dirs[driver_name]
        
        # 获取所有tif文件
        tif_files = glob.glob(os.path.join(driver_path, '*.tif'))
        if not tif_files:
            logger.warning(f"驱动因素 {driver_name} 没有找到tif文件")
            return []
        
        # 提取日期并过滤年份
        file_date_pairs = []
        invalid_files = 0
        filtered_files = 0
        
        for file_path in tif_files:
            date = self._extract_date_from_filename(file_path)
            if date is None:
                invalid_files += 1
                logger.debug(f"无法提取日期: {os.path.basename(file_path)}")
                continue
            
            # 年份过滤
            if self.years is not None and date.year not in self.years:
                filtered_files += 1
                continue
            
            file_date_pairs.append((file_path, date))
        
        if not file_date_pairs:
            logger.warning(f"驱动因素 {driver_name} 没有有效的日期文件")
            return []
        
        # 按日期排序并显示日期范围
        file_date_pairs.sort(key=lambda x: x[1])
        start_date = file_date_pairs[0][1]
        end_date = file_date_pairs[-1][1]
        
        logger.info(f"{driver_name}: 找到 {len(file_date_pairs)} 个有效文件")
        logger.info(f"  - 日期范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"  - 无效文件: {invalid_files} 个, 年份过滤: {filtered_files} 个")
        
        # 获取第一个文件的信息来确定波段数和地理参考
        first_file = file_date_pairs[0][0]
        geo_info = self._get_file_info(first_file)
        if geo_info is None:
            logger.error(f"无法获取 {driver_name} 的地理参考信息")
            return []
        
        logger.info(f"{driver_name}: 图像尺寸 {geo_info['width']}x{geo_info['height']}, "
                   f"波段数 {geo_info['bands']}")
        
        # 为每个波段创建时间序列
        results = []
        for band_idx in range(geo_info['bands']):
            result = self._process_driver_band(driver_name, band_idx, file_date_pairs, geo_info)
            results.append(result)
            
            # 清理内存
            gc.collect()
        
        return results
    
    def generate_all_time_series(self):
        """
        生成所有驱动因素的时间序列GeoTIFF文件
        """
        logger.info(f"开始生成时间序列GeoTIFF文件，共 {len(self.driver_dirs)} 个驱动因素")
        
        all_results = []
        total_success = 0
        total_errors = 0
        
        for i, driver_name in enumerate(self.driver_dirs.keys()):
            logger.info(f"处理进度: {i+1}/{len(self.driver_dirs)} - {driver_name}")
            
            start_time = time.time()
            results = self._process_single_driver(driver_name)
            elapsed_time = time.time() - start_time
            
            # 统计结果
            success_count = len([r for r in results if r['status'] == 'success'])
            error_count = len([r for r in results if r['status'] == 'error'])
            
            total_success += success_count
            total_errors += error_count
            all_results.extend(results)
            
            logger.info(f"完成 {driver_name}: 成功 {success_count} 个波段，"
                       f"错误 {error_count} 个波段，耗时 {elapsed_time:.1f} 秒")
        
        # 生成处理报告
        self._generate_report(all_results, total_success, total_errors)
        
        logger.info(f"时间序列GeoTIFF生成完成！"
                   f"成功 {total_success} 个，错误 {total_errors} 个")
    
    def _generate_report(self, results, total_success, total_errors):
        """
        生成处理报告
        
        Args:
            results: 所有处理结果
            total_success: 成功数量
            total_errors: 错误数量
        """
        report_path = os.path.join(self.output_dir, 'time_series_generation_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("时间序列GeoTIFF生成报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入目录: {self.data_dir}\n")
            f.write(f"输出目录: {self.output_dir}\n")
            f.write(f"处理年份: {self.years if self.years else '所有年份'}\n\n")
            
            f.write(f"总体统计:\n")
            f.write(f"- 驱动因素数量: {len(self.driver_dirs)}\n")
            f.write(f"- 成功生成文件: {total_success}\n")
            f.write(f"- 生成错误文件: {total_errors}\n")
            f.write(f"- 成功率: {total_success/(total_success+total_errors)*100:.2f}%\n\n")
            
            # 按驱动因素分组统计
            driver_stats = defaultdict(lambda: {'success': 0, 'error': 0, 'files': []})
            total_size_mb = 0
            
            for result in results:
                driver = result['driver']
                status = result['status']
                driver_stats[driver][status] += 1
                
                if status == 'success':
                    file_info = {
                        'band': result['band'],
                        'time_steps': result['time_steps'],
                        'output_path': result['output_path'],
                        'file_size_mb': result['file_size_mb']
                    }
                    driver_stats[driver]['files'].append(file_info)
                    total_size_mb += result['file_size_mb']
            
            f.write("各驱动因素处理详情:\n")
            f.write("-" * 50 + "\n")
            
            for driver, stats in driver_stats.items():
                f.write(f"\n驱动因素: {driver}\n")
                f.write(f"  成功生成: {stats['success']} 个波段\n")
                f.write(f"  生成错误: {stats['error']} 个波段\n")
                
                if stats['files']:
                    f.write(f"  生成的文件:\n")
                    for file_info in stats['files']:
                        filename = os.path.basename(file_info['output_path'])
                        f.write(f"    {filename}: "
                               f"{file_info['time_steps']} 个时间步, "
                               f"{file_info['file_size_mb']:.1f} MB\n")
            
            f.write(f"\n总文件大小: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)\n")
        
        logger.info(f"处理报告已保存到: {report_path}")



def main():
    """主函数"""
    # 配置参数
    config = {
        'data_dir': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked_normalized',
        'output_dir': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/masked_time_series_geotiff',
        'years': None,  # 指定要处理的年份，None表示处理所有年份
        'max_workers': 24,
        'compression': 'LZW',  # 压缩方式
        'tiled': True,         # 使用分块存储
        'bigtiff': True        # 使用BigTIFF格式
    }
    
    logger.info("开始生成时间序列GeoTIFF文件...")
    logger.info(f"配置参数: {config}")
    
    # 创建生成器
    generator = TimeSeriesGeoTIFFGenerator(**config)
    
    # 开始生成
    generator.generate_all_time_series()
    
    logger.info("时间序列GeoTIFF生成完成！")


if __name__ == '__main__':
    main() 