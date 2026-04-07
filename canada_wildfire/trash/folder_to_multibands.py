"""
将多个文件夹中相同日期的数据按通道拼接成一个多波段GeoTIFF文件。
如果某个文件夹中某个日期缺失数据，则用全0波段补充。
"""
import os
import glob
from osgeo import gdal
import numpy as np
from tqdm import tqdm
import logging
import re
from datetime import datetime
import multiprocessing
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('MultiBand_Merger')

# 设置GDAL缓存大小
gdal.SetCacheMax(1024 * 1024 * 1024)  # 1GB缓存

def get_date_from_filename(filename):
    """
    从文件名中提取日期
    例如：从 'driver_yyyy_mm_dd.tif' 提取 'yyyy_mm_dd'
    """
    # date_pattern = r'(\d{4}_\d{2}_\d{2})'
    date_pattern = r'(\d{8})'
    match = re.search(date_pattern, filename)
    if match:
        return match.group(1)
    return None

def create_empty_band(width, height, data_type):
    """
    创建全0波段
    
    参数:
        width (int): 图像宽度
        height (int): 图像高度
        data_type: GDAL数据类型
    
    返回:
        numpy.ndarray: 全0数组
    """
    # 将GDAL数据类型转换为numpy数据类型
    dtype_map = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
        gdal.GDT_CInt16: np.complex64,
        gdal.GDT_CInt32: np.complex64,
        gdal.GDT_CFloat32: np.complex64,
        gdal.GDT_CFloat64: np.complex128
    }
    
    numpy_dtype = dtype_map.get(data_type, np.float32)  # 默认使用float32
    return np.zeros((height, width), dtype=numpy_dtype)

def merge_bands_for_date(date, input_dirs, output_dir, output_driver_name):
    """
    合并指定日期的所有波段数据
    
    参数:
        date (str): 日期字符串 (yyyy_mm_dd)
        input_dirs (list): 输入目录列表
        output_dir (str): 输出目录
        output_driver_name (str): 输出文件的driver名称
    """
    try:
        # 检查输出文件是否已存在
        output_file = os.path.join(output_dir, f'{output_driver_name}_{date}.tif')
        if os.path.exists(output_file):
            logger.info(f"文件已存在，跳过处理: {output_file}")
            return True

        # 查找所有目录中该日期的文件
        date_files = {}
        ref_ds = None
        total_bands = 0
        
        # 首先找到参考数据集
        for input_dir in input_dirs:
            files = glob.glob(os.path.join(input_dir, f'*{date}.tif'))
            if files:
                ref_ds = gdal.Open(files[0])
                if ref_ds:
                    break
        
        if ref_ds is None:
            logger.warning(f"日期 {date} 没有找到任何数据")
            return False

        # 获取参考数据集的信息
        geo_transform = ref_ds.GetGeoTransform()
        projection = ref_ds.GetProjection()
        width = ref_ds.RasterXSize
        height = ref_ds.RasterYSize
        data_type = ref_ds.GetRasterBand(1).DataType

        # 统计总波段数
        for input_dir in input_dirs:
            files = glob.glob(os.path.join(input_dir, f'*{date}.tif'))
            if files:
                ds = gdal.Open(files[0])
                if ds:
                    total_bands += ds.RasterCount
                    date_files[input_dir] = files[0]
                ds = None

        # 创建输出文件
        output_file = os.path.join(output_dir, f'{output_driver_name}_{date}.tif')
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            output_file,
            width,
            height,
            total_bands,
            data_type,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
        )

        if dst_ds is None:
            raise Exception(f"无法创建输出文件: {output_file}")

        # 设置地理信息
        dst_ds.SetGeoTransform(geo_transform)
        dst_ds.SetProjection(projection)

        # 合并波段
        current_band = 1
        for input_dir in input_dirs:
            if input_dir in date_files:
                # 如果有数据，复制所有波段
                src_ds = gdal.Open(date_files[input_dir])
                for i in range(src_ds.RasterCount):
                    band = src_ds.GetRasterBand(i + 1)
                    data = band.ReadAsArray()
                    dst_ds.GetRasterBand(current_band).WriteArray(data)
            
                    # 复制波段描述
                    band_desc = band.GetDescription()
                    if band_desc:
                        dst_ds.GetRasterBand(current_band).SetDescription(band_desc)
            
                    current_band += 1
            src_ds = None
            else:
                # 如果没有数据，创建全0波段
                for i in range(2):  # 假设每个文件有2个波段
                    empty_band = create_empty_band(width, height, data_type)
                    dst_ds.GetRasterBand(current_band).WriteArray(empty_band)
                    dst_ds.GetRasterBand(current_band).SetDescription(f"Empty_Band_{i+1}")
                    current_band += 1

        # 清理
        dst_ds = None
        ref_ds = None

        # logger.info(f"成功合并文件: {output_file}")
        return True

    except Exception as e:
        logger.error(f"处理日期 {date} 时出错: {str(e)}")
        return False

def process_date_wrapper(args):
    """
    包装函数，用于多进程处理
    """
    date, input_dirs, output_dir, output_driver_name = args
    return merge_bands_for_date(date, input_dirs, output_dir, output_driver_name)

def main():
    # 设置输入目录列表
    input_dirs = [
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data/ERA5_resampled',
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data/ERA5_soil_water_resampled'
    ]
    
    # 设置输出目录
    output_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data/ERA5_multi_bands'
    os.makedirs(output_dir, exist_ok=True)

    # 设置输出文件的driver名称
    output_driver_name = 'ERA5'  # 统一的输出文件名前缀
    
    # 获取所有日期
    dates = set()
    for input_dir in input_dirs:
        tif_files = glob.glob(os.path.join(input_dir, '*.tif'))
        for file in tif_files:
            date = get_date_from_filename(os.path.basename(file))
            if date:
                dates.add(date)
    
    if not dates:
        logger.error("没有找到任何有效日期")
        return

    # 准备参数
    args = [(date, input_dirs, output_dir, output_driver_name) for date in sorted(dates)]
    
    # 设置进程数为CPU内核数的85%
    num_processes = max(1, int(multiprocessing.cpu_count() * 0.3))
    logger.info(f"使用 {num_processes} 个进程进行并行处理")
    
    # 使用进程池处理日期
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度
        list(tqdm(
            pool.imap(process_date_wrapper, args),
            total=len(args),
            desc="处理日期"
        ))
    
    logger.info("所有日期处理完成")

if __name__ == '__main__':
    main() 
