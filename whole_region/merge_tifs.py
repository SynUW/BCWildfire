import os
from osgeo import gdal
import numpy as np
from tqdm import tqdm

def merge_tifs(input_dir, output_path):
    """
    将目录中的所有tif文件按文件名顺序合并成一个多通道的tif文件
    
    Args:
        input_dir: 输入目录，包含所有tif文件
        output_path: 输出文件路径
    """
    # 获取所有tif文件并按名称排序
    tif_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])
    
    if not tif_files:
        print(f"在目录 {input_dir} 中没有找到tif文件")
        return
    
    print(f"找到 {len(tif_files)} 个tif文件")
    
    # 读取第一个文件获取元数据
    src_ds = gdal.Open(os.path.join(input_dir, tif_files[0]))
    if src_ds is None:
        print(f"无法打开文件: {tif_files[0]}")
        return
    
    # 获取元数据
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    projection = src_ds.GetProjection()
    geotransform = src_ds.GetGeoTransform()
    
    # 计算总通道数
    total_bands = 0
    for tif_file in tif_files:
        src_ds = gdal.Open(os.path.join(input_dir, tif_file))
        if src_ds is not None:
            total_bands += src_ds.RasterCount
    
    print(f"总通道数: {total_bands}")
    
    # 创建输出文件
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(
        output_path,
        width,
        height,
        total_bands,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES']
    )
    
    if dst_ds is None:
        print("无法创建输出文件")
        return
    
    # 设置地理参考信息
    dst_ds.SetProjection(projection)
    dst_ds.SetGeoTransform(geotransform)
    
    # 逐个读取并写入通道
    current_band = 1
    for tif_file in tqdm(tif_files, desc="合并通道"):
        src_ds = gdal.Open(os.path.join(input_dir, tif_file))
        if src_ds is None:
            print(f"无法打开文件: {tif_file}")
            continue
        
        # 读取所有通道
        for band in range(src_ds.RasterCount):
            data = src_ds.GetRasterBand(band + 1).ReadAsArray()
            # 写入到对应的通道
            dst_ds.GetRasterBand(current_band).WriteArray(data)
            dst_ds.GetRasterBand(current_band).SetNoDataValue(-9999)
            current_band += 1
    
    # 清理
    dst_ds = None
    src_ds = None
    
    print(f"合并完成，输出文件: {output_path}")

def main():
    # 设置输入输出路径
    input_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/raw_data_with_issues/DAILY_500_clip'
    output_path = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/raw_data_with_issues/raw_data_merged.tif'
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 合并tif文件
    merge_tifs(input_dir, output_path)

if __name__ == "__main__":
    main() 