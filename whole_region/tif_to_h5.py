import os
from osgeo import gdal
import h5py
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm


def is_leap_year(year):
    """判断是否为闰年"""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def get_days_in_year(year):
    """获取指定年份的天数"""
    return 366 if is_leap_year(year) else 365

def tif_to_h5(input_tif, output_h5, start_year=2000, end_year=2024):
    """
    将合并后的TIF文件转换为H5格式
    
    参数:
        input_tif: 输入TIF文件路径
        output_h5: 输出H5文件路径
        start_year: 开始年份
        end_year: 结束年份
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_h5)
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开TIF文件
    ds = gdal.Open(input_tif)
    if ds is None:
        print(f"无法打开文件: {input_tif}")
        return
    
    # 获取数据
    data = ds.ReadAsArray()
    if data is None:
        print("无法读取数据")
        return
    
    # 获取波段数
    num_bands = ds.RasterCount
    print(f"数据波段数: {num_bands}")
    
    # 计算总天数
    total_days = sum(get_days_in_year(year) for year in range(start_year, end_year + 1))
    print(f"2000-2024年总天数: {total_days}")
    
    # 创建日期列表
    dates = []
    current_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    while current_date <= end_date:
        dates.append(current_date.strftime('%Y%m%d'))
        current_date += timedelta(days=1)
    
    # 确保数据长度与日期数量匹配
    if len(dates) != num_bands:
        print(f"警告：数据波段数({num_bands})与日期数量({len(dates)})不匹配")
        print(f"日期数量: {len(dates)}")
        # 截断较长的那个
        if len(dates) > num_bands:
            dates = dates[:num_bands]
            print(f"截断日期列表至 {num_bands} 个日期")
        else:
            data = data[:len(dates)]
            print(f"截断数据至 {len(dates)} 个波段")
    
    # 创建H5文件
    with h5py.File(output_h5, 'w') as f:
        # 创建数据集
        dataset = f.create_dataset('data', data=data)
        
        # 创建日期属性
        f.create_dataset('dates', data=np.array(dates, dtype='S8'))
        
        # 添加属性
        f.attrs['start_date'] = dates[0]
        f.attrs['end_date'] = dates[-1]
        f.attrs['description'] = f'ERA5 data from {start_year} to {end_year}'
        f.attrs['total_days'] = total_days
        f.attrs['leap_years'] = [year for year in range(start_year, end_year + 1) if is_leap_year(year)]
    
    # 清理
    ds = None
    
    print(f"\n完成! 数据已保存到 {output_h5}")
    print(f"开始日期: {dates[0]}")
    print(f"结束日期: {dates[-1]}")
    print(f"闰年: {f.attrs['leap_years']}")

def main():
    # 设置GDAL配置
    gdal.SetConfigOption('GDAL_CACHEMAX', '20000')  # 80GB内存
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 使用所有可用CPU线程
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')  # 可选：提高多线程效率
    gdal.UseExceptions()
    
    # 设置输入输出路径
    input_tif = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/raw_data_with_issues/raw_data_merged.tif"
    output_h5 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/raw_data_with_issues/raw_data_merged.h5"
    
    # 转换文件
    tif_to_h5(input_tif, output_h5)

if __name__ == "__main__":
    main() 