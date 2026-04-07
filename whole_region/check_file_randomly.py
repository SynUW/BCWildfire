import os
import random
from osgeo import gdal, osr
import numpy as np


def get_raster_info(filepath):
    """获取栅格文件的基本信息"""
    ds = gdal.Open(filepath)
    if ds is None:
        raise ValueError(f"无法打开文件: {filepath}")

    # 获取行列数
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    # 获取地理变换和投影
    transform = ds.GetGeoTransform()
    projection = ds.GetProjection()

    # 计算边界坐标
    x_min = transform[0]
    y_max = transform[3]
    x_max = x_min + cols * transform[1]
    y_min = y_max + rows * transform[5]

    return {
        'rows': rows,
        'cols': cols,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'projection': projection,
        'transform': transform,
        'dataset': ds  # 保持数据集打开状态
    }


def process_folders(folder_paths):
    """处理多个文件夹，从每个文件夹随机抽取一个样本"""
    samples = []

    for folder in folder_paths:
        # 获取文件夹中所有栅格文件
        raster_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(('.tif', '.tiff', '.img', '.hdf', '.h5')):
                    raster_files.append(os.path.join(root, file))

        if not raster_files:
            print(f"警告: 文件夹 {folder} 中没有找到栅格文件")
            continue

        # 随机选择一个文件
        selected_file = random.choice(raster_files)
        print(f"从文件夹 {folder} 中选择了文件: {selected_file}")

        try:
            # 获取文件信息
            info = get_raster_info(selected_file)
            samples.append(info)

            print(f"文件信息: 行数={info['rows']}, 列数={info['cols']}")
            print(f"空间范围: x({info['x_min']}, {info['x_max']}), y({info['y_min']}, {info['y_max']})")
            print("-" * 50)
        except Exception as e:
            print(f"处理文件 {selected_file} 时出错: {str(e)}")

    return samples


def calculate_common_extent(samples):
    """计算所有样本的共同空间范围"""
    if not samples:
        return None

    # 初始化最大最小范围
    common = {
        'x_min': max(s['x_min'] for s in samples),
        'x_max': min(s['x_max'] for s in samples),
        'y_min': max(s['y_min'] for s in samples),
        'y_max': min(s['y_max'] for s in samples),
        'projection': samples[0]['projection']  # 假设所有投影相同
    }

    # 检查是否有有效交集
    if common['x_min'] >= common['x_max'] or common['y_min'] >= common['y_max']:
        return None

    return common


def clip_raster_to_extent(input_path, output_path, extent, resolution=None):
    """
    裁剪栅格到指定范围
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :param extent: 包含x_min, x_max, y_min, y_max, projection的字典
    :param resolution: 可选，输出分辨率(x_res, y_res)
    """
    # 检查输出文件是否已存在
    if os.path.exists(output_path):
        print(f"文件已存在，跳过: {output_path}")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 使用gdal.Warp进行裁剪
    warp_options = gdal.WarpOptions(
        outputBounds=(extent['x_min'], extent['y_min'], extent['x_max'], extent['y_max']),
        dstSRS=extent['projection'],
        resampleAlg=gdal.GRA_NearestNeighbour,
        format='GTiff',
        multithread=True
    )

    if resolution:
        warp_options = gdal.WarpOptions(
            outputBounds=(extent['x_min'], extent['y_min'], extent['x_max'], extent['y_max']),
            dstSRS=extent['projection'],
            xRes=resolution[0],
            yRes=resolution[1],
            resampleAlg=gdal.GRA_NearestNeighbour,
            format='GTiff',
            multithread=True
        )

    try:
        gdal.Warp(output_path, input_path, options=warp_options)
        print(f"已裁剪并保存: {output_path}")
    except Exception as e:
        print(f"裁剪文件 {input_path} 时出错: {str(e)}")
        # 如果出错，删除可能创建的不完整文件
        if os.path.exists(output_path):
            os.remove(output_path)


def process_all_files(folder_paths, common_extent, resolution=None):
    """处理所有文件夹中的所有文件，裁剪到共同范围"""
    for folder in folder_paths:
        # 创建输出文件夹
        output_folder = folder + '_clip'

        # 获取文件夹中所有栅格文件
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(('.tif', '.tiff', '.img', '.hdf', '.h5')):
                    input_path = os.path.join(root, file)

                    # 构建输出路径，保持原始目录结构
                    relative_path = os.path.relpath(root, folder)
                    output_root = os.path.join(output_folder, relative_path)
                    output_path = os.path.join(output_root, file)

                    # 检查文件扩展名，确保输出为.tif格式
                    if not output_path.lower().endswith('.tif'):
                        output_path = os.path.splitext(output_path)[0] + '.tif'

                    clip_raster_to_extent(input_path, output_path, common_extent, resolution)


def main():
    # 输入文件夹列表
    folder_paths = [
        r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500_merge_mergeTerraAqua',

        # r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Terra_Aqua_Daily',
        # r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Topo_Distance_500',
        #
        # r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LULC_500_interpolated',
        #
        # r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LAI_500_interpolated',
        #
        # r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/DAILY_500',


        # 添加更多文件夹...
    ]

    # 处理文件夹获取样本
    samples = process_folders(folder_paths)

    if not samples:
        print("没有找到有效的栅格样本")
        return

    # 计算共同范围
    common_extent = calculate_common_extent(samples)

    if not common_extent:
        print("\n警告: 样本之间没有共同的空间范围交集")
        return

    print("\n所有样本的共同空间范围:")
    print(f"x范围: {common_extent['x_min']} - {common_extent['x_max']}")
    print(f"y范围: {common_extent['y_min']} - {common_extent['y_max']}")
    print(f"投影: {common_extent['projection']}")

    # 可选: 设置输出分辨率 (None表示保持原始分辨率)
    output_resolution = None  # 例如 (500, 500) 表示500m分辨率

    # 处理所有文件
    print("\n开始裁剪所有文件...")
    process_all_files(folder_paths, common_extent, output_resolution)

    # 关闭所有打开的数据集
    for sample in samples:
        if 'dataset' in sample:
            sample['dataset'] = None

    print("\n处理完成!")


if __name__ == "__main__":
    main()