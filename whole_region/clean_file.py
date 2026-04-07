import os
import numpy as np
from osgeo import gdal
from tqdm import tqdm

def calculate_ndvi(red, nir):
    """计算NDVI (Normalized Difference Vegetation Index)"""
    ndvi = (nir - red) / (nir + red + 1e-10)  # 添加小量避免除以零
    return ndvi

def calculate_evi(blue, red, nir):
    """计算EVI (Enhanced Vegetation Index)"""
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1 + 1e-10)
    return evi

def process_modis_tif(input_path, output_dir):
    """处理MODIS TIFF文件，计算NDVI和EVI"""
    # 提取日期信息
    date_part = os.path.basename(input_path).split('_')[1:]
    date_str = '_'.join(date_part).replace('.tif', '')
    output_path = os.path.join(output_dir, f'Vidx_{date_str}.tif')

    # 检查输出文件是否已存在
    if os.path.exists(output_path):
        print(f"文件已存在，跳过: {output_path}")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 打开输入文件
    ds = gdal.Open(input_path)
    if ds is None:
        print(f"无法打开文件: {input_path}")
        return

    # 获取波段信息
    red_band = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nir_band = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
    blue_band = ds.GetRasterBand(3).ReadAsArray().astype(np.float32)

    # 计算指数
    ndvi = calculate_ndvi(red_band, nir_band)
    evi = calculate_evi(blue_band, red_band, nir_band)

    # 创建输出文件(只使用GDAL支持的GDT_Float32)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_path,
        ds.RasterXSize,
        ds.RasterYSize,
        2,  # 两个波段: NDVI和EVI
        gdal.GDT_Float32  # 仅使用GDAL支持的Float32
    )

    # 设置地理参考和投影
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    # 写入NDVI (波段1)
    out_band1 = out_ds.GetRasterBand(1)
    out_band1.WriteArray(ndvi.astype(np.float32))  # 确保使用float32
    out_band1.SetDescription('NDVI')
    out_band1.SetNoDataValue(np.nan)

    # 写入EVI (波段2)
    out_band2 = out_ds.GetRasterBand(2)
    out_band2.WriteArray(evi.astype(np.float32))  # 确保使用float32
    out_band2.SetDescription('EVI')
    out_band2.SetNoDataValue(np.nan)

    # 关闭数据集
    out_ds = None
    ds = None
    print(f"处理完成: {output_path}")

if __name__ == "__main__":
    input_root = r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500_merge_mergeTerraAqua_clip'
    output_root = r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/NDVI_EVI'

    # 获取并排序所有tif文件
    files = [f for f in os.listdir(input_root) if f.endswith('.tif')]
    files.sort()

    # 使用tqdm显示进度条
    for file in tqdm(files, desc="Processing MODIS files"):
        process_modis_tif(os.path.join(input_root, file), output_root)