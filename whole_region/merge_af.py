import os
import glob
import shutil
from osgeo import gdal, gdal_array
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import functools


def process_single_date(args):
    """
    处理单个日期的数据
    参数:
        args: 包含 (date, aqua_file, terra_file, output_dir) 的元组
    """
    date, aqua_file, terra_file, output_dir = args
    output_filename = f"FIRMS_{date}.tif"
    output_path = os.path.join(output_dir, output_filename)

    try:
        if aqua_file and terra_file:  # 双传感器数据
            # 读取Aqua数据
            aqua_ds = gdal.Open(aqua_file)
            aqua_band = aqua_ds.GetRasterBand(1)
            aqua_array = aqua_band.ReadAsArray()

            # 读取Terra数据
            terra_ds = gdal.Open(terra_file)
            terra_band = terra_ds.GetRasterBand(1)
            terra_array = terra_band.ReadAsArray()

            # 取并集 (逻辑或运算)并转换为uint8
            union_array = np.logical_or(aqua_array, terra_array).astype(np.uint8)

            # 获取原始文件的投影和地理变换信息
            geo_transform = aqua_ds.GetGeoTransform()
            projection = aqua_ds.GetProjection()

            # 创建输出文件
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(
                output_path,
                aqua_ds.RasterXSize,
                aqua_ds.RasterYSize,
                1,
                gdal.GDT_Byte  # 使用GDT_Byte (uint8)
            )

            # 设置地理信息
            out_ds.SetGeoTransform(geo_transform)
            out_ds.SetProjection(projection)

            # 写入数据
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(union_array)

            # 关闭数据集
            out_ds = None
            aqua_ds = None
            terra_ds = None

        elif aqua_file:  # 只有Aqua数据
            # 读取并确保数据类型为uint8
            aqua_ds = gdal.Open(aqua_file)
            aqua_band = aqua_ds.GetRasterBand(1)
            aqua_array = aqua_band.ReadAsArray().astype(np.uint8)
            
            # 创建输出文件
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(
                output_path,
                aqua_ds.RasterXSize,
                aqua_ds.RasterYSize,
                1,
                gdal.GDT_Byte
            )
            
            # 设置地理信息
            out_ds.SetGeoTransform(aqua_ds.GetGeoTransform())
            out_ds.SetProjection(aqua_ds.GetProjection())
            
            # 写入数据
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(aqua_array)
            
            # 关闭数据集
            out_ds = None
            aqua_ds = None
            
        elif terra_file:  # 只有Terra数据
            # 读取并确保数据类型为uint8
            terra_ds = gdal.Open(terra_file)
            terra_band = terra_ds.GetRasterBand(1)
            terra_array = terra_band.ReadAsArray().astype(np.uint8)
            
            # 创建输出文件
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(
                output_path,
                terra_ds.RasterXSize,
                terra_ds.RasterYSize,
                1,
                gdal.GDT_Byte
            )
            
            # 设置地理信息
            out_ds.SetGeoTransform(terra_ds.GetGeoTransform())
            out_ds.SetProjection(terra_ds.GetProjection())
            
            # 写入数据
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(terra_array)
            
            # 关闭数据集
            out_ds = None
            terra_ds = None

        return True, date
    except Exception as e:
        print(f"处理日期 {date} 时发生错误: {str(e)}")
        return False, date


def process_terra_aqua_union(input_terra, input_aqua, output_dir):
    """
    处理Terra和Aqua的野火检测结果:
    - 同一天有两者数据则取并集
    - 只有单一传感器数据则直接复制

    参数:
        input_terra: Terra数据输入目录
        input_aqua: Aqua数据输入目录
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有Aqua和Terra文件
    aqua_files = glob.glob(os.path.join(input_aqua, 'binary_Firms_Aqua_*.tif'))
    terra_files = glob.glob(os.path.join(input_terra, 'binary_Firms_Terra_*.tif'))

    # 创建日期到文件的映射
    aqua_dict = {}
    for f in aqua_files:
        date = f.split('_')[-1].split('.')[0]  # 提取yyyymmdd
        aqua_dict[date] = f

    terra_dict = {}
    for f in terra_files:
        date = f.split('_')[-1].split('.')[0]  # 提取yyyymmdd
        terra_dict[date] = f

    # 找出所有日期
    all_dates = set(aqua_dict.keys()).union(set(terra_dict.keys()))
    common_dates = set(aqua_dict.keys()) & set(terra_dict.keys())
    aqua_only_dates = set(aqua_dict.keys()) - common_dates
    terra_only_dates = set(terra_dict.keys()) - common_dates

    print(f"找到 {len(all_dates)} 个有效日期")
    print(f"其中 {len(common_dates)} 天有双传感器数据")
    print(f"{len(aqua_only_dates)} 天只有Aqua数据")
    print(f"{len(terra_only_dates)} 天只有Terra数据")

    # 准备并行处理的参数
    process_args = []
    for date in all_dates:
        aqua_file = aqua_dict.get(date)
        terra_file = terra_dict.get(date)
        process_args.append((date, aqua_file, terra_file, output_dir))

    # 设置进程数（使用85%的CPU核心）
    num_processes = max(1, int(cpu_count() * 0.85))
    print(f"使用 {num_processes} 个进程进行并行处理")

    # 使用进程池进行并行处理
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_date, process_args),
            total=len(process_args),
            desc="处理文件进度"
        ))

    # 统计处理结果
    success_count = sum(1 for success, _ in results if success)
    failed_count = len(results) - success_count

    print(f"\n处理完成: 成功 {success_count} 个, 失败 {failed_count} 个")
    if failed_count > 0:
        print("\n失败的日期:")
        for success, date in results:
            if not success:
                print(f"- {date}")


if __name__ == "__main__":
    gdal.SetConfigOption('GDAL_CACHEMAX', '20000')  # 80GB内存
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 使用所有可用CPU线程
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')  # 可选：提高多线程效率
    gdal.UseExceptions()

    # 设置输入和输出目录
    input_terra = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/Firms_Terra_Daily_Binary"
    input_aqua = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/Firms_Aqua_Daily_Binary"
    output_directory = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/Firms_Detection"

    process_terra_aqua_union(input_terra, input_aqua, output_directory)