import os
from osgeo import gdal
import numpy as np
from scipy import ndimage
from tqdm import tqdm

def remove_isolated_pixels(input_tif, output_tif):
    """
    消除TIF文件中的孤立像素点，但保留值大于等于2的像素
    
    参数:
        input_tif: 输入TIF文件路径
        output_tif: 输出TIF文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_tif)
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取输入文件
    print(f"正在读取文件: {input_tif}")
    ds = gdal.Open(input_tif)
    if ds is None:
        print("无法打开输入文件")
        return
    
    # 获取地理信息
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    data_type = ds.GetRasterBand(1).DataType
    
    # 读取数据
    data = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    
    # 获取NoData值
    no_data_value = 255
    
    print("正在处理数据...")
    # 创建二值掩膜，非NoData的区域为1，NoData的区域为0
    binary_mask = (data != no_data_value).astype(np.uint8)
    
    # 使用3x3卷积核计算每个像素的邻域和
    kernel = np.ones((3,3), np.uint8)
    kernel[1,1] = 0  # 中心像素不参与计算
    neighbor_sum = ndimage.convolve(binary_mask, kernel, mode='constant', cval=0)
    
    # 创建结果数组
    result = data.copy()
    
    # 找出孤立像素（邻域和为0）且值小于2的像素
    isolated_pixels = (neighbor_sum == 0) & (binary_mask == 1) & (data < 2)
    
    # 将这些像素设置为NoData
    result[isolated_pixels] = no_data_value
    
    # 创建输出文件
    print("正在保存结果...")
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
    
    # 写入数据
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(result)
    out_band.SetNoDataValue(no_data_value)
    
    # 清理
    out_ds = None
    
    # 统计信息
    total_pixels = np.sum(binary_mask)
    removed_pixels = np.sum(isolated_pixels)
    preserved_pixels = total_pixels - removed_pixels
    
    print(f"\n完成! 结果已保存到: {output_tif}")
    print(f"总非NoData像素数: {total_pixels}")
    print(f"移除的孤立像素数: {removed_pixels}")
    print(f"保留的像素数: {preserved_pixels}")

if __name__ == "__main__":
    # 设置GDAL配置
    gdal.SetConfigOption('GDAL_CACHEMAX', '20000')  # 80GB内存
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 使用所有可用CPU线程
    gdal.UseExceptions()
    
    # 设置输入输出路径
    input_tif = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/FIRMS_2024_sum.tif"
    output_tif = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/FIRMS_2024_sum_cleaned.tif"
    
    # 执行处理
    remove_isolated_pixels(input_tif, output_tif) 