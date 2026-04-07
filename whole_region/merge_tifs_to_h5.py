import os
import glob
from osgeo import gdal
import h5py
import numpy as np
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def get_date_from_filename(filename):
    """从文件名中解析日期"""
    # 尝试两种可能的格式：FIRMS_20000220.tif 或 FIRMS_2000_02_20.tif
    basename = os.path.basename(filename)
    if '_' in basename:
        try:
            # 尝试解析带下划线的格式
            date_str = basename.split('_')[1].replace('.tif', '')
            if len(date_str) == 8:  # yyyymmdd格式
                return datetime.strptime(date_str, '%Y%m%d')
            else:  # yyyy_mm_dd格式
                return datetime.strptime(date_str, '%Y_%m_%d')
        except:
            return None
    return None


def read_tif_data(tif_file):
    """读取TIF文件数据"""
    try:
        ds = gdal.Open(tif_file)
        if ds is None:
            return None
        
        # 获取数据
        data = ds.ReadAsArray()
        ds = None
        return data
    except Exception as e:
        print(f"读取文件 {tif_file} 时发生错误: {str(e)}")
        return None


def merge_tifs_to_h5(input_dir, output_h5, start_year=2000, end_year=2019):
    """
    将指定年份范围内的TIF文件合并为一个H5文件
    
    参数:
        input_dir: 输入文件夹路径
        output_h5: 输出H5文件路径
        start_year: 开始年份
        end_year: 结束年份
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_h5)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有tif文件
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    tif_files.extend(glob.glob(os.path.join(input_dir, "*.TIF")))
    
    if not tif_files:
        print(f"在 {input_dir} 中没有找到tif文件")
        return
    
    # 过滤指定年份的文件
    filtered_files = []
    for tif_file in tif_files:
        date = get_date_from_filename(tif_file)
        if date and start_year <= date.year <= end_year:
            filtered_files.append(tif_file)
    
    if not filtered_files:
        print(f"在 {start_year}-{end_year} 年间没有找到tif文件")
        return
    
    print(f"找到 {len(filtered_files)} 个符合条件的tif文件")
    
    # 按日期排序
    filtered_files.sort(key=get_date_from_filename)
    
    # 读取第一个文件获取数据形状
    first_data = read_tif_data(filtered_files[0])
    if first_data is None:
        print("无法读取第一个文件")
        return
    
    # 创建H5文件
    with h5py.File(output_h5, 'w') as f:
        # 创建数据集
        data_shape = (len(filtered_files),) + first_data.shape
        dataset = f.create_dataset('data', shape=data_shape, dtype=first_data.dtype)
        
        # 创建日期属性
        dates = [get_date_from_filename(f).strftime('%Y%m%d') for f in filtered_files]
        f.create_dataset('dates', data=np.array(dates, dtype='S8'))
        
        # 写入数据
        print("开始写入数据...")
        for i, tif_file in enumerate(tqdm(filtered_files)):
            data = read_tif_data(tif_file)
            if data is not None:
                dataset[i] = data
        
        # 添加属性
        f.attrs['start_date'] = dates[0]
        f.attrs['end_date'] = dates[-1]
        f.attrs['description'] = f'FIRMS data from {start_year} to {end_year}'
    
    print(f"\n完成! 数据已保存到 {output_h5}")


def process_time_periods(input_dir, base_output_dir):
    """
    处理多个时间段的数据
    
    参数:
        input_dir: 输入文件夹路径
        base_output_dir: 输出文件夹的基础路径
    """
    # 确保输出目录存在
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 定义要处理的时间段
    time_periods = [
        (2000, 2021, 2021),
        # (2002, 2021, 2022),
        # (2003, 2022, 2023),
        # (2004, 2023, 2024)
    ]
    
    for start_year, end_year, next_year in time_periods:
        print(f"\n处理时间段: {start_year}-{end_year} 和 {next_year}")
        
        # 生成H5文件路径
        output_h5 = os.path.join(base_output_dir, f"FIRMS_{start_year}_{end_year}.h5")
        # 生成TIF文件路径
        output_tif = os.path.join(base_output_dir, f"FIRMS_{next_year}_sum.tif")
        
        # 处理H5文件
        print(f"正在生成H5文件: {output_h5}")
        merge_tifs_to_h5(input_dir, output_h5, start_year, end_year)
        
        # 处理TIF文件
        print(f"正在生成TIF文件: {output_tif}")
        sum_2020_data(input_dir, output_tif, next_year)


def sum_2020_data(input_dir, output_tif, target_year=2020):
    """
    将指定年份的数据按位置相加，生成一个TIF文件
    
    参数:
        input_dir: 输入文件夹路径
        output_tif: 输出TIF文件路径
        target_year: 目标年份，默认为2020
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_tif)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有tif文件
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    tif_files.extend(glob.glob(os.path.join(input_dir, "*.TIF")))
    
    if not tif_files:
        print(f"在 {input_dir} 中没有找到tif文件")
        return
    
    # 过滤指定年份的文件
    filtered_files = []
    for tif_file in tif_files:
        date = get_date_from_filename(tif_file)
        if date and date.year == target_year:
            filtered_files.append(tif_file)
    
    if not filtered_files:
        print(f"没有找到{target_year}年的tif文件")
        return
    
    print(f"找到 {len(filtered_files)} 个{target_year}年的tif文件")
    
    # 读取第一个文件获取数据形状和地理信息
    first_ds = gdal.Open(filtered_files[0])
    if first_ds is None:
        print("无法读取第一个文件")
        return
    
    # 获取地理信息
    geo_transform = first_ds.GetGeoTransform()
    projection = first_ds.GetProjection()
    rows = first_ds.RasterYSize
    cols = first_ds.RasterXSize
    data_type = first_ds.GetRasterBand(1).DataType
    first_ds = None
    
    # 创建输出文件
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_tif,
        cols,
        rows,
        1,  # 单波段
        data_type,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
    )
    
    # 设置地理信息
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    
    # 初始化结果数组
    result = np.zeros((rows, cols), dtype=np.float32)
    
    # 处理每个文件
    print(f"开始处理{target_year}年数据...")
    for tif_file in tqdm(filtered_files):
        data = read_tif_data(tif_file)
        if data is not None:
            # 创建掩膜，排除NoData值（255）
            mask = (data != 255)
            # 累加有效值
            result[mask] += data[mask]
    
    # 将背景区域设置为255
    result[result == 0] = 255
    
    # 写入结果
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(result)
    out_band.SetNoDataValue(255)
    
    # 清理
    out_ds = None
    
    print(f"\n完成! {target_year}年数据已保存到 {output_tif}")


if __name__ == "__main__":
    # 设置GDAL配置
    gdal.SetConfigOption('GDAL_CACHEMAX', '20000')  # 80GB内存
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 使用所有可用CPU线程
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')  # 可选：提高多线程效率
    gdal.UseExceptions()
    
    # 设置输入输出路径
    input_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/Firms_Detection_BC_Cropped"
    base_output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/Visualization"
    
    # 处理所有时间段
    process_time_periods(input_dir, base_output_dir) 