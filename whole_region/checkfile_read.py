import os
import rasterio
import glob
from pathlib import Path


def count_bands(tif_file):
    """
    读取TIF文件并返回其波段数
    
    参数:
        tif_file: TIF文件路径
    
    返回:
        波段数
    """
    with rasterio.open(tif_file) as src:
        return src.count


def main():
    # 用户输入路径
    input_folder = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS"
    
    # 获取所有tif文件
    tif_files = glob.glob(os.path.join(input_folder, "FIRMS_all_band.tif"))
    
    if not tif_files:
        print(f"在{input_folder}中未找到GeoTIFF文件")
        return
    
    print(f"找到{len(tif_files)}个GeoTIFF文件:")
    total_bands = 0
    
    # 遍历所有文件并统计波段数
    for tif_file in tif_files:
        bands = count_bands(tif_file)
        total_bands += bands
        print(f"文件: {os.path.basename(tif_file)}")
        print(f"波段数: {bands}")
        print("-" * 50)
    
    print(f"\n总波段数: {total_bands}")


if __name__ == "__main__":
    main()