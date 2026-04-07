import os
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.mask import mask
from pathlib import Path
import time
from shapely.geometry import Point


def extract_pixels_from_shapefile(tif_file, shp_file, output_dir):
    """
    提取位于shpfile范围内的像素点数据
    
    参数:
        tif_file: 输入的TIF文件路径
        shp_file: 输入的shp文件路径
        output_dir: 输出目录
    """
    start_time = time.time()
    
    # 读取shp文件
    print("正在读取shp文件...")
    gdf = gpd.read_file(shp_file)
    
    # 读取TIF文件
    print("正在读取TIF文件...")
    with rasterio.open(tif_file) as src:
        # 获取TIF文件的元数据
        meta = src.meta.copy()
        meta.update({
            'dtype': 'uint8',
            'count': 1,  # 每个文件只保存一个波段
            'compress': 'lzw'
        })
        
        # 对每个波段进行处理
        total_bands = src.count
        print(f"总波段数: {total_bands}")
        
        # 遍历每个波段
        for band_idx in range(1, total_bands + 1):
            band_start = time.time()
            print(f"\n处理波段 {band_idx}/{total_bands}")
            
            # 读取当前波段数据
            band_data = src.read(band_idx)
            
            # 获取非零像素的位置
            rows, cols = np.where(band_data > 0)
            
            if len(rows) == 0:
                print(f"波段 {band_idx} 没有有效像素")
                continue
                
            # 将像素坐标转换为地理坐标
            pixel_coords = []
            for row, col in zip(rows, cols):
                x, y = rasterio.transform.xy(src.transform, row, col)
                pixel_coords.append((x, y))
            
            # 创建GeoDataFrame
            pixel_gdf = gpd.GeoDataFrame(
                geometry=[Point(xy) for xy in pixel_coords],
                crs=src.crs
            )
            
            # 找出在shpfile范围内的像素
            pixels_in_shape = gpd.sjoin(pixel_gdf, gdf, how='inner', predicate='within')
            
            if len(pixels_in_shape) == 0:
                print(f"波段 {band_idx} 在shpfile范围内没有像素")
                continue
            
            # 保存每个像素点的数据
            for idx, row in pixels_in_shape.iterrows():
                # 获取像素的行列号
                pixel_row = rows[idx]
                pixel_col = cols[idx]
                
                # 创建输出文件名
                output_file = os.path.join(output_dir, f"{pixel_row}_{pixel_col}.tif")
                
                # 保存像素数据
                with rasterio.open(output_file, 'w', **meta) as dst:
                    dst.write(band_data[pixel_row:pixel_row+1, pixel_col:pixel_col+1].astype('uint8'), 1)
            
            print(f"波段 {band_idx} 处理完成，耗时: {time.time() - band_start:.2f} 秒")
            print(f"保存了 {len(pixels_in_shape)} 个像素点")
    
    print(f"\n处理完成! 总耗时: {time.time() - start_time:.2f} 秒")


def main():
    # 输入文件路径
    tif_file = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/FIRMS_all_band.tif"
    shp_file = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_boundary/bc_boundary_without_sea.shp"  # 请替换为实际的shp文件路径
    
    # 创建输出目录
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/extracted_pixels"
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取像素
    extract_pixels_from_shapefile(tif_file, shp_file, output_dir)


if __name__ == "__main__":
    main()