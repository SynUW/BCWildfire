"""
将Terra和Aqua的数据合并成一个4波段的GeoTIFF文件。
如果某个日期的数据缺失，则用全0波段补充。
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
logger = logging.getLogger('Terra_Aqua_Merger')

# 设置GDAL缓存大小
gdal.SetCacheMax(1024 * 1024 * 1024)  # 1GB缓存

def get_date_from_filename(filename):
    """
    从文件名中提取日期
    例如：从 'Terra_B20_21_20241203_20241204.tif' 提取 '20241203'
    """
    date_pattern = r'(\d{8})_\d{8}'
    match = re.search(date_pattern, filename)
    if match:
        date_str = match.group(1)
        # 将日期格式化为 yyyy_mm_dd
        return f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:]}"
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

def merge_terra_aqua(terra_file, aqua_file, output_file):
    """
    合并Terra和Aqua的数据
    
    参数:
        terra_file (str): Terra数据文件路径
        aqua_file (str): Aqua数据文件路径
        output_file (str): 输出文件路径
    """
    try:
        # 打开Terra文件
        terra_ds = gdal.Open(terra_file) if terra_file else None
        # 打开Aqua文件
        aqua_ds = gdal.Open(aqua_file) if aqua_file else None
        
        # 获取参考数据集（优先使用Terra）
        ref_ds = terra_ds if terra_ds else aqua_ds
        if ref_ds is None:
            raise Exception("没有可用的参考数据集")
            
        # 获取地理信息
        geo_transform = ref_ds.GetGeoTransform()
        projection = ref_ds.GetProjection()
        width = ref_ds.RasterXSize
        height = ref_ds.RasterYSize
        data_type = ref_ds.GetRasterBand(1).DataType
        
        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            output_file,
            width,
            height,
            4,  # 4个波段
            data_type,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
        )
        
        if dst_ds is None:
            raise Exception(f"无法创建输出文件: {output_file}")
            
        # 设置地理信息
        dst_ds.SetGeoTransform(geo_transform)
        dst_ds.SetProjection(projection)
        
        # 处理Terra数据
        if terra_ds:
            for i in range(2):  # Terra数据占前两个波段
                band = terra_ds.GetRasterBand(i + 1)
                data = band.ReadAsArray()
                dst_ds.GetRasterBand(i + 1).WriteArray(data)
                # 复制波段描述
                band_desc = band.GetDescription()
                if band_desc:
                    dst_ds.GetRasterBand(i + 1).SetDescription(band_desc)
        else:
            # 如果Terra数据缺失，用全0波段补充
            for i in range(2):
                empty_band = create_empty_band(width, height, data_type)
                dst_ds.GetRasterBand(i + 1).WriteArray(empty_band)
                dst_ds.GetRasterBand(i + 1).SetDescription(f"Terra_Band_{i+1}")
                
        # 处理Aqua数据
        if aqua_ds:
            for i in range(2):  # Aqua数据占后两个波段
                band = aqua_ds.GetRasterBand(i + 1)
                data = band.ReadAsArray()
                dst_ds.GetRasterBand(i + 3).WriteArray(data)  # 从第3个波段开始
                # 复制波段描述
                band_desc = band.GetDescription()
                if band_desc:
                    dst_ds.GetRasterBand(i + 3).SetDescription(band_desc)
        else:
            # 如果Aqua数据缺失，用全0波段补充
            for i in range(2):
                empty_band = create_empty_band(width, height, data_type)
                dst_ds.GetRasterBand(i + 3).WriteArray(empty_band)
                dst_ds.GetRasterBand(i + 3).SetDescription(f"Aqua_Band_{i+1}")
        
        # 清理
        dst_ds = None
        if terra_ds:
            terra_ds = None
        if aqua_ds:
            aqua_ds = None
            
        logger.info(f"成功合并文件: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
        return False

def process_date_wrapper(args):
    """
    包装函数，用于多进程处理
    """
    date, terra_dir, aqua_dir, output_dir = args
    try:
        # 查找对应日期的文件
        terra_files = glob.glob(os.path.join(terra_dir, f'*{date.replace("_", "")}*.tif'))
        aqua_files = glob.glob(os.path.join(aqua_dir, f'*{date.replace("_", "")}*.tif'))
        
        terra_file = terra_files[0] if terra_files else None
        aqua_file = aqua_files[0] if aqua_files else None
        
        if not terra_file and not aqua_file:
            logger.warning(f"日期 {date} 没有找到任何数据")
            return False
            
        # 创建输出文件名
        output_file = os.path.join(output_dir, f'B20B21_{date}.tif')
        
        # 合并数据
        return merge_terra_aqua(terra_file, aqua_file, output_file)
        
    except Exception as e:
        logger.error(f"处理日期 {date} 时出错: {str(e)}")
        return False

def main():
    # 设置输入输出目录
    terra_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection/MODIS_B20_21/Terra_B20_21'
    aqua_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection/MODIS_B20_21/Aqua_B20_21'
    output_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection/MODIS_Terra_Aqua_B20_21_merged'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有日期
    terra_files = glob.glob(os.path.join(terra_dir, '*.tif'))
    aqua_files = glob.glob(os.path.join(aqua_dir, '*.tif'))
    
    # 提取所有日期
    dates = set()
    for file in terra_files + aqua_files:
        date = get_date_from_filename(os.path.basename(file))
        if date:
            dates.add(date)
    
    if not dates:
        logger.error("没有找到任何有效日期")
        return
        
    # 准备参数
    args = [(date, terra_dir, aqua_dir, output_dir) for date in sorted(dates)]
    
    # 设置进程数为CPU内核数的45%
    num_processes = max(1, int(multiprocessing.cpu_count() * 0.45))
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