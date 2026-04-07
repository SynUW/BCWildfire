import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RioResampling
import glob
from tqdm import tqdm

# 配置参数
input_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LAI_LULC/MODIS_LULC"  # 输入文件夹路径
output_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LULC_500"  # 输出文件夹路径
target_crs = "EPSG:32611"  # 目标坐标系
target_resolution = 500  # 目标分辨率（米）

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹内所有栅格文件（支持.tif/.img等）
raster_files = glob.glob(os.path.join(input_folder, "*.tif")) + \
               glob.glob(os.path.join(input_folder, "*.img"))

def reproject_and_resample(input_path, output_path):
    """投影转换与重采样函数"""
    with rasterio.open(input_path) as src:
        # 计算目标变换参数
        transform, width, height = calculate_default_transform(
            src.crs, target_crs,
            src.width, src.height,
            *src.bounds,
            resolution=target_resolution
        )

        # 设置输出元数据
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'driver': 'GTiff'  # 强制输出为GeoTIFF
        })

        # 执行重投影和重采样
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest  # 双线性插值
                )


# 批量处理所有文件
for file in tqdm(raster_files):
    output_path = os.path.join(output_folder, os.path.basename(file))

    # 检查输出文件是否已存在
    if not os.path.exists(output_path):
        reproject_and_resample(file, output_path)
        print(f"处理完成: {os.path.basename(file)} -> EPSG:32611 500m")
    else:
        print(f"跳过处理: {os.path.basename(file)}，因目标文件已存在")

print("全部文件处理完毕！")