import os
import numpy as np
from osgeo import gdal, osr
import glob

# 配置GDAL使用大内存和多线程
gdal.SetConfigOption('GDAL_CACHEMAX', '80000')  # 80GB内存
gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 使用所有CPU线程
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')  # 提高多线程效率
gdal.UseExceptions()


def process_sensor_data(input_folder, output_folder):
    """处理传感器数据，生成合并结果并转换为WGS84"""
    # 获取所有Aqua和Terra文件
    aqua_files = glob.glob(os.path.join(input_folder, 'Aqua_*.tif'))
    terra_files = glob.glob(os.path.join(input_folder, 'Terra_*.tif'))

    # 创建日期到文件路径的映射
    date_to_files = {}

    # 处理Aqua文件
    for file in aqua_files:
        date_str = os.path.basename(file).split('_')[1:4]
        date_key = '_'.join(date_str)
        date_to_files.setdefault(date_key, {})['Aqua'] = file

    # 处理Terra文件
    for file in terra_files:
        date_str = os.path.basename(file).split('_')[1:4]
        date_key = '_'.join(date_str)
        date_to_files.setdefault(date_key, {})['Terra'] = file

    # 处理每个日期的数据
    for date_key, files in date_to_files.items():
        output_path = os.path.join(output_folder, f'Reflectance_{date_key}.tif')

        # 检查输出文件是否已存在且有效
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            print(f'Skipping {date_key}: Output file already exists')
            continue

        if 'Aqua' in files and 'Terra' in files:
            # 两个传感器都有数据
            print(f'Processing {date_key}: Both sensors available')
            process_dual_sensor(files['Aqua'], files['Terra'], output_path)
        elif 'Aqua' in files:
            # 只有Aqua有数据
            print(f'Processing {date_key}: Only Aqua available')
            process_single_sensor(files['Aqua'], output_path)
        elif 'Terra' in files:
            # 只有Terra有数据
            print(f'Processing {date_key}: Only Terra available')
            process_single_sensor(files['Terra'], output_path)


def process_single_sensor(input_path, output_path):
    """处理单个传感器数据，转换为WGS84"""
    # 检查输出文件是否已存在
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        print(f'Skipping single-sensor processing: {os.path.basename(output_path)} already exists')
        return

    # 创建临时文件进行投影转换
    temp_path = output_path + '.temp.tif'

    # 定义目标坐标系(WGS84)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)

    # 设置投影转换选项
    warp_options = gdal.WarpOptions(
        dstSRS=target_srs,
        resampleAlg=gdal.GRA_Bilinear,
        multithread=True,  # 启用多线程
        dstNodata=None
    )

    # 执行投影转换
    gdal.Warp(
        destNameOrDestDS=temp_path,
        srcDSOrSrcDSTab=input_path,
        options=warp_options
    )

    # 重命名临时文件
    os.rename(temp_path, output_path)


def process_dual_sensor(aqua_path, terra_path, output_path):
    """处理两个传感器数据，使用最小值合成并转换为WGS84"""
    # 检查输出文件是否已存在
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        print(f'Skipping dual-sensor processing: {os.path.basename(output_path)} already exists')
        return

    # 创建临时合并文件
    temp_merge_path = output_path + '.merge.tif'

    # 打开两个数据集
    aqua_ds = gdal.Open(aqua_path)
    terra_ds = gdal.Open(terra_path)

    if aqua_ds is None or terra_ds is None:
        raise ValueError("无法打开一个或两个输入文件")

    # 验证数据集一致性
    if (aqua_ds.RasterXSize != terra_ds.RasterXSize or
            aqua_ds.RasterYSize != terra_ds.RasterYSize or
            aqua_ds.RasterCount != terra_ds.RasterCount):
        raise ValueError("输入数据集具有不同的尺寸或波段数")

    # 获取第一个波段信息
    band = aqua_ds.GetRasterBand(1)
    dtype = band.DataType
    nodata = band.GetNoDataValue()

    # 创建临时合并数据集
    driver = gdal.GetDriverByName('GTiff')
    merge_ds = driver.Create(
        temp_merge_path,
        aqua_ds.RasterXSize,
        aqua_ds.RasterYSize,
        aqua_ds.RasterCount,
        dtype,
        options=['BIGTIFF=YES', 'COMPRESS=LZW', 'TILED=YES']
    )

    # 设置地理参考和投影
    merge_ds.SetGeoTransform(aqua_ds.GetGeoTransform())
    merge_ds.SetProjection(aqua_ds.GetProjection())

    # 处理每个波段
    for b in range(1, aqua_ds.RasterCount + 1):
        # 读取波段数据
        aqua_band = aqua_ds.GetRasterBand(b)
        aqua_data = aqua_band.ReadAsArray()
        aqua_nodata = aqua_band.GetNoDataValue()

        terra_band = terra_ds.GetRasterBand(b)
        terra_data = terra_band.ReadAsArray()
        terra_nodata = terra_band.GetNoDataValue()

        # 创建有效值掩膜
        aqua_valid = (aqua_data != aqua_nodata) if aqua_nodata is not None else np.ones_like(aqua_data, dtype=bool)
        terra_valid = (terra_data != terra_nodata) if terra_nodata is not None else np.ones_like(terra_data, dtype=bool)

        # 初始化输出数组
        output_data = np.full_like(aqua_data, fill_value=nodata if nodata is not None else -9999)

        # 创建组合掩膜
        both_valid = aqua_valid & terra_valid
        only_aqua = aqua_valid & ~terra_valid
        only_terra = ~aqua_valid & terra_valid

        # 处理不同情况
        if np.any(both_valid):
            aqua_positive = (aqua_data > 0) & both_valid
            terra_positive = (terra_data > 0) & both_valid

            # 两个都是正值，取最小值
            both_positive = aqua_positive & terra_positive
            output_data[both_positive] = np.minimum(aqua_data[both_positive], terra_data[both_positive])

            # 只有一个是正值，取正值
            a_pos_only = aqua_positive & ~terra_positive
            output_data[a_pos_only] = aqua_data[a_pos_only]

            t_pos_only = ~aqua_positive & terra_positive
            output_data[t_pos_only] = terra_data[t_pos_only]

            # 两个都是负值，设为nodata
            both_negative = (~aqua_positive & ~terra_positive) & both_valid
            output_data[both_negative] = nodata if nodata is not None else -9999

        if np.any(only_aqua):
            output_data[only_aqua] = aqua_data[only_aqua]

        if np.any(only_terra):
            output_data[only_terra] = terra_data[only_terra]

        # 写入输出波段
        out_band = merge_ds.GetRasterBand(b)
        out_band.WriteArray(output_data)

        if nodata is not None:
            out_band.SetNoDataValue(nodata)

        out_band.FlushCache()
        out_band.ComputeStatistics(False)

    # 清理数据集
    merge_ds = None
    aqua_ds = None
    terra_ds = None

    # 将合并结果转换为WGS84
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)

    warp_options = gdal.WarpOptions(
        dstSRS=target_srs,
        resampleAlg=gdal.GRA_Bilinear,
        multithread=True,  # 启用多线程
        dstNodata=nodata
    )

    gdal.Warp(
        destNameOrDestDS=output_path,
        srcDSOrSrcDSTab=temp_merge_path,
        options=warp_options
    )

    # 删除临时文件
    if os.path.exists(temp_merge_path):
        os.remove(temp_merge_path)


if __name__ == '__main__':
    input_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500_merge"
    output_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500_merge_TerraAquaWGS84"

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    process_sensor_data(input_folder, output_folder)
    print("处理完成！")