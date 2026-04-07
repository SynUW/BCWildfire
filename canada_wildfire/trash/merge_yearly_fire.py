import os
import glob
import numpy as np
from osgeo import gdal
import logging
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
from functools import partial

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('FireDetectionMerger')

def get_date_from_filename(filename):
    """从文件名提取日期 (格式: *_YYYY_MM_DD.tif)"""
    basename = os.path.basename(filename)
    date_part = basename.split('_')[-3:]
    try:
        return datetime.strptime('_'.join(date_part), '%Y_%m_%d.tif')
    except ValueError:
        return None

def process_single_file(file):
    """处理单个文件并返回数据"""
    try:
        ds = gdal.Open(file)
        if ds is None:
            logger.warning(f"无法打开文件: {file}")
            return None
        
        data = ds.ReadAsArray()
        if data is None:
            logger.warning(f"无法读取数据: {file}")
            return None
        
        # 如果是多波段数据，只取第一个波段
        if len(data.shape) == 3:
            data = data[0]
        
        return data
    except Exception as e:
        logger.error(f"处理文件 {file} 时出错: {str(e)}")
        return None

def process_year_data(files, height, width):
    """处理一年的数据"""
    yearly_sum = np.zeros((height, width), dtype=np.float32)
    
    # 使用线程池处理文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_single_file, file): file for file in files}
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                         total=len(files),
                         desc="处理文件"):
            file = future_to_file[future]
            try:
                data = future.result()
                if data is not None:
                    yearly_sum += data
            except Exception as e:
                logger.error(f"处理文件 {file} 时出错: {str(e)}")
    
    return yearly_sum

def merge_yearly_fire_detection(input_dir, output_file):
    """
    合并每年的火灾检测数据
    
    参数:
        input_dir: 输入目录，包含每年的火灾检测GeoTIFF文件
        output_file: 输出文件路径
    """
    logger.info(f"开始处理火灾检测数据...")
    
    # 获取所有火灾检测文件
    fire_files = glob.glob(os.path.join(input_dir, '*.tif'))
    if not fire_files:
        logger.error(f"在目录 {input_dir} 中未找到任何GeoTIFF文件")
        return
    
    # 按年份分组文件
    yearly_files = {}
    for file in fire_files:
        date = get_date_from_filename(file)
        if date:
            year = date.year
            if year not in yearly_files:
                yearly_files[year] = []
            yearly_files[year].append(file)
    
    if not yearly_files:
        logger.error("未找到任何有效的火灾检测文件")
        return
    
    # 获取第一个文件的地理信息
    first_file = fire_files[0]
    ds = gdal.Open(first_file)
    if ds is None:
        logger.error(f"无法打开文件: {first_file}")
        return
    
    # 获取地理信息
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    width = ds.RasterXSize
    height = ds.RasterYSize
    
    # 创建输出文件
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_file,
        width,
        height,
        len(yearly_files),  # 通道数等于年份数
        gdal.GDT_Float32,
        options=['COMPRESS=LZW']
    )
    
    if out_ds is None:
        logger.error("无法创建输出文件")
        return
    
    # 设置地理信息
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    # 使用线程池处理每年的数据
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as year_executor:
        # 提交所有年份的处理任务
        future_to_year = {
            year_executor.submit(process_year_data, files, height, width): year 
            for year, files in yearly_files.items()
        }
        
        # 处理每年的结果
        for future in tqdm(concurrent.futures.as_completed(future_to_year), 
                         total=len(yearly_files),
                         desc="处理年度数据"):
            year = future_to_year[future]
            try:
                yearly_sum = future.result()
                
                # 写入输出文件
                band_idx = list(yearly_files.keys()).index(year)
                out_band = out_ds.GetRasterBand(band_idx + 1)
                out_band.WriteArray(yearly_sum)
                out_band.SetDescription(f"Fire_Detection_{year}")
                
                # 计算统计信息
                out_band.ComputeStatistics(False)
                
                logger.info(f"完成 {year} 年数据处理")
            except Exception as e:
                logger.error(f"处理 {year} 年数据时出错: {str(e)}")
    
    # 清理
    out_ds = None
    logger.info(f"处理完成，输出文件: {output_file}")

def main():
    # 配置参数
    input_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data/Firms_Detection_resampled'
    output_file = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/yearly_sum/yearly_fire_detection.tif'
    
    try:
        logger.info("开始运行合并程序...")
        merge_yearly_fire_detection(input_dir, output_file)
        logger.info("程序运行完成")
    except Exception as e:
        logger.error(f"程序异常退出: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main() 