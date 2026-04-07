import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob

# 配置参数
input_root = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection"  # 输入文件夹路径
output_root = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500"  # 输出文件夹路径
target_crs = "EPSG:32611"  # 目标坐标系
target_resolution = 500  # 目标分辨率（米）
output_format = "GTiff"  # 输出格式（GeoTIFF）

# 创建输出文件夹（不清空已存在内容）
os.makedirs(output_root, exist_ok=True)


def find_deepest_tiffs(root_dir):
    """递归查找最深层的TIFF文件"""
    tiff_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.tif', '.tiff')):
                tiff_files.append(os.path.join(dirpath, f))
    return tiff_files


def reproject_resample(input_path, output_path):
    """执行投影转换和重采样"""
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
            'driver': output_format
        })

        # 执行操作
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )


# 查找所有最深层的TIFF文件
all_tiffs = find_deepest_tiffs(input_root)
all_tiffs.reverse()
print(f"找到 {len(all_tiffs)} 个TIFF文件待处理...")


# 批量处理
processed_count = 0
skipped_count = 0
for tif_path in all_tiffs:
    # 获取输出文件夹中已存在的文件列表（用于跳过处理）
    existing_files = set(os.listdir(output_root))

    base_name = os.path.basename(tif_path)
    output_path = os.path.join(output_root, base_name)

    # 检查是否已存在同名文件
    if base_name in existing_files:
        print(f"跳过已存在文件: {base_name}")
        skipped_count += 1
        continue

    try:
        reproject_resample(tif_path, output_path)
        print(f"处理成功: {base_name}")
        processed_count += 1
    except Exception as e:
        print(f"处理失败 {base_name}: {str(e)}")

print(f"\n处理完成！\n已处理: {processed_count} 个文件\n跳过: {skipped_count} 个已存在文件")
print(f"输出目录: {output_root} (未删除任何已有文件)")