import os
from osgeo import gdal
import glob
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime, timedelta
import re
import numpy as np

def get_date_from_filename(filename):
    """从文件名中解析日期信息"""
    basename = os.path.basename(filename)
    pattern = r'ERA5_(\d{4})_(\d{2})_half(\d)_stacked\.tif'
    match = re.match(pattern, basename)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        half = int(match.group(3))
        return year, month, half
    return None

def get_days_in_month(year, month):
    """获取指定年月的天数"""
    if month == 2:
        return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    return 30 if month in [4, 6, 9, 11] else 31

def get_days_in_half_month(year, month, half):
    """获取指定年月半月的天数"""
    days_in_month = get_days_in_month(year, month)
    if half == 1:
        return min(15, days_in_month)
    else:
        return days_in_month - 15

def merge_monthly_tifs(input_dir, output_dir, year, month):
    """
    合并一个月的上下半月数据
    
    参数:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        year: 年份
        month: 月份
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建文件名模式
    half1_file = f"ERA5_{year}_{month:02d}_half1_stacked.tif"
    half2_file = f"ERA5_{year}_{month:02d}_half2_stacked.tif"
    
    # 获取文件路径
    half1_path = os.path.join(input_dir, half1_file)
    half2_path = os.path.join(input_dir, half2_file)
    
    # 检查文件是否存在
    if not os.path.exists(half1_path) or not os.path.exists(half2_path):
        print(f"警告: {year}年{month}月的数据不完整")
        return None
    
    # 打开文件
    ds1 = gdal.Open(half1_path)
    ds2 = gdal.Open(half2_path)
    
    if ds1 is None or ds2 is None:
        print(f"无法打开文件: {half1_path} 或 {half2_path}")
        return None
    
    # 获取数据
    data1 = ds1.ReadAsArray()
    data2 = ds2.ReadAsArray()
    
    # 检查数据维度
    if data1 is None or data2 is None:
        print(f"无法读取数据: {half1_path} 或 {half2_path}")
        return None
    
    # 确保数据是3D的 (bands, height, width)
    if len(data1.shape) == 2:
        data1 = np.expand_dims(data1, axis=0)
    if len(data2.shape) == 2:
        data2 = np.expand_dims(data2, axis=0)
    
    # 合并数据
    merged_data = np.concatenate([data1, data2], axis=0)
    
    # 创建输出文件
    output_file = os.path.join(output_dir, f"ERA5_{year}_{month:02d}_merged.tif")
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(
        output_file,
        ds1.RasterXSize,
        ds1.RasterYSize,
        merged_data.shape[0],
        ds1.GetRasterBand(1).DataType,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
    )
    
    # 复制地理信息
    dst_ds.SetGeoTransform(ds1.GetGeoTransform())
    dst_ds.SetProjection(ds1.GetProjection())
    
    # 写入数据
    for i in range(merged_data.shape[0]):
        dst_ds.GetRasterBand(i + 1).WriteArray(merged_data[i])
    
    # 清理
    ds1 = None
    ds2 = None
    dst_ds = None
    
    return output_file

def split_daily_tifs(input_tif, output_dir, year, month):
    """
    将合并后的月度数据分割为每日数据
    
    参数:
        input_tif: 输入tif文件路径
        output_dir: 输出文件夹路径
        year: 年份
        month: 月份
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开文件
    ds = gdal.Open(input_tif)
    if ds is None:
        print(f"无法打开文件: {input_tif}")
        return
    
    # 获取数据
    data = ds.ReadAsArray()
    
    # 检查数据维度
    if data is None:
        print(f"无法读取数据: {input_tif}")
        return
    
    # 确保数据是3D的 (bands, height, width)
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    
    # 获取每天的数据
    days_in_month = get_days_in_month(year, month)
    bands_per_day = 6
    
    # 检查数据波段数是否足够
    total_bands_needed = days_in_month * bands_per_day
    if data.shape[0] < total_bands_needed:
        print(f"警告: 数据波段数({data.shape[0]})小于所需波段数({total_bands_needed})")
        return
    
    # 创建输出文件
    driver = gdal.GetDriverByName('GTiff')
    
    for day in range(1, days_in_month + 1):
        # 计算当天的波段范围
        start_band = (day - 1) * bands_per_day
        end_band = day * bands_per_day
        
        # 获取当天的数据
        daily_data = data[start_band:end_band]
        
        # 创建输出文件
        output_file = os.path.join(output_dir, f"ERA5_{year}_{month:02d}_{day:02d}.tif")
        dst_ds = driver.Create(
            output_file,
            ds.RasterXSize,
            ds.RasterYSize,
            bands_per_day,
            ds.GetRasterBand(1).DataType,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
        )
        
        # 复制地理信息
        dst_ds.SetGeoTransform(ds.GetGeoTransform())
        dst_ds.SetProjection(ds.GetProjection())
        
        # 写入数据
        for i in range(bands_per_day):
            dst_ds.GetRasterBand(i + 1).WriteArray(daily_data[i])
        
        # 清理
        dst_ds = None
    
    # 清理
    ds = None

def process_year(input_dir, output_dir, year):
    """
    处理一年的数据
    
    参数:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        year: 年份
    """
    # 创建临时目录用于存储合并后的月度数据
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 处理每个月的数据
    for month in range(1, 13):
        print(f"\n处理 {year}年{month}月 的数据...")
        
        # 合并月度数据
        merged_file = merge_monthly_tifs(input_dir, temp_dir, year, month)
        if merged_file is None:
            continue
        
        # 分割为每日数据
        split_daily_tifs(merged_file, output_dir, year, month)
        
        # 删除临时文件
        os.remove(merged_file)
    
    # 删除临时目录
    os.rmdir(temp_dir)

def main():
    # 设置GDAL配置
    gdal.SetConfigOption('GDAL_CACHEMAX', '20000')  # 80GB内存
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 使用所有可用CPU线程
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')  # 可选：提高多线程效率
    gdal.UseExceptions()
    
    # 设置输入输出路径
    input_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/raw_data_with_issues/raw_data"
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/raw_data_with_issues/raw_data_daily"
    
    # 处理2000-2024年的数据
    for year in range(2000, 2025):
        print(f"\n处理 {year}年 的数据...")
        process_year(input_dir, output_dir, year)

if __name__ == "__main__":
    main() 