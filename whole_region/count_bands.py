import os
import rasterio
import glob
from pathlib import Path


def get_file_info(tif_file):
    """
    读取TIF文件并返回其波段数和像元数
    
    参数:
        tif_file: TIF文件路径
    
    返回:
        (波段数, 像元数, 宽度, 高度)
    """
    with rasterio.open(tif_file) as src:
        bands = src.count
        width = src.width
        height = src.height
        pixels = width * height
        return bands, pixels, width, height


def main():
    # 用户输入路径
    input_folder = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS"
    
    # 获取所有tif文件
    tif_files = glob.glob(os.path.join(input_folder, "*FIRMS_all_band.tif"))
    
    if not tif_files:
        print(f"在{input_folder}中未找到GeoTIFF文件")
        return
    
    print(f"找到{len(tif_files)}个GeoTIFF文件:")
    total_bands = 0
    total_pixels = 0
    
    # 遍历所有文件并统计信息
    for tif_file in tif_files:
        bands, pixels, width, height = get_file_info(tif_file)
        total_bands += bands
        total_pixels = pixels  # 所有文件应该有相同的像元数
        
        print(f"文件: {os.path.basename(tif_file)}")
        print(f"波段数: {bands}")
        print(f"像元数: {pixels:,}")  # 使用千位分隔符格式化数字
        print(f"空间分辨率: {width} x {height}")
        print("-" * 50)
    
    print(f"\n统计信息:")
    print(f"总波段数: {total_bands}")
    print(f"总像元数: {total_pixels:,}")
    print(f"总数据量: {total_bands * total_pixels:,} 个像元值")


if __name__ == "__main__":
    main()