import os
import numpy as np
from osgeo import gdal
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import functools


def binarize_geotiff(input_path, output_path):
    """二值化单个GeoTIFF文件"""
    try:
        src_ds = gdal.Open(input_path)
        if src_ds is None:
            print(f"无法打开文件: {input_path}")
            return False

        # 读取数据和元数据
        band = src_ds.GetRasterBand(1)
        data = band.ReadAsArray()
        geo_transform = src_ds.GetGeoTransform()
        projection = src_ds.GetProjection()

        # 二值化处理 (8或9→1，其他→0)
        binary_data = np.where((data == 8) | (data == 9), 1, 0)

        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path,
                            src_ds.RasterXSize,
                            src_ds.RasterYSize,
                            1,
                            gdal.GDT_Byte)

        out_ds.SetGeoTransform(geo_transform)
        out_ds.SetProjection(projection)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(binary_data)

        # 关闭数据集
        out_ds = None
        src_ds = None
        return True
    except Exception as e:
        print(f"处理文件 {input_path} 时发生错误: {str(e)}")
        return False


def process_single_file(args):
    """处理单个文件的包装函数，用于并行处理"""
    input_path, output_path = args
    return binarize_geotiff(input_path, output_path)


def process_folder(input_folder, output_folder):
    """使用并行处理处理文件夹中的所有GeoTIFF文件"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有需要处理的文件
    files_to_process = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.tif', '.tiff', '.geotiff')):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"binary_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            files_to_process.append((input_path, output_path))

    # 使用进程池进行并行处理
    num_processes = int(cpu_count() *0.85)  # 保留一个CPU核心给系统
    print(f"使用 {num_processes} 个进程进行并行处理")

    with Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度条
        results = list(tqdm(
            pool.imap(process_single_file, files_to_process),
            total=len(files_to_process),
            desc="处理文件进度"
        ))

    # 统计处理结果
    processed = sum(1 for r in results if r)
    failed = len(results) - processed

    print(f"\n处理完成: 成功 {processed} 个, 失败 {failed} 个")


# 使用示例
if __name__ == '__main__':
    gdal.SetConfigOption('GDAL_CACHEMAX', '20000')  # 80GB内存
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 使用所有可用CPU线程
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')  # 可选：提高多线程效率
    gdal.UseExceptions()

    input_folder = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/Firms_Terra_Daily"
    output_folder = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/Firms_Terra_Daily_Binary"

    process_folder(input_folder, output_folder)