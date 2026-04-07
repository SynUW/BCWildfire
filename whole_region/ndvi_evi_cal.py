import os
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import multiprocessing
from functools import partial
import warnings

# 配置GDAL内存使用
gdal.SetCacheMax(2 ** 30*10)  # 设置GDAL缓存为10GB
warnings.filterwarnings('ignore', category=RuntimeWarning)  # 忽略除以零警告


def calculate_ndvi(red, nir):
    """计算NDVI (Normalized Difference Vegetation Index)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = np.clip(ndvi, -1, 1)
    return ndvi


def calculate_evi(blue, red, nir):
    """计算EVI (Enhanced Vegetation Index)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1 + 1e-10)
    evi = np.clip(evi, 0, 10)
    return evi


def process_single_file(input_path, output_dir):
    """处理单个MODIS TIFF文件，计算NDVI和EVI"""
    try:
        # 提取日期信息
        date_part = os.path.basename(input_path).split('_')[1:]
        date_str = '_'.join(date_part).replace('.tif', '')
        output_path = os.path.join(output_dir, f'Vidx_{date_str}.tif')

        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            return (input_path, "skipped", None)

        # 打开输入文件
        ds = gdal.Open(input_path)
        if ds is None:
            return (input_path, "failed", "无法打开文件")

        # 获取波段信息 - 使用更高效的读取方式
        red_band = ds.GetRasterBand(1).ReadAsArray(buf_obj=np.empty((ds.RasterYSize, ds.RasterXSize), dtype=np.float32))
        nir_band = ds.GetRasterBand(2).ReadAsArray(buf_obj=np.empty((ds.RasterYSize, ds.RasterXSize), dtype=np.float32))
        blue_band = ds.GetRasterBand(3).ReadAsArray(
            buf_obj=np.empty((ds.RasterYSize, ds.RasterXSize), dtype=np.float32))

        # 计算指数
        ndvi = calculate_ndvi(red_band, nir_band)
        evi = calculate_evi(blue_band, red_band, nir_band)

        # 创建输出文件 - 优化写入性能
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_path,
            ds.RasterXSize,
            ds.RasterYSize,
            2,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'BIGTIFF=IF_SAFER', 'TILED=YES', 'NUM_THREADS=ALL_CPUS']
        )

        # 设置地理参考和投影
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())

        # 写入NDVI (波段1)
        out_band1 = out_ds.GetRasterBand(1)
        out_band1.WriteArray(ndvi)
        out_band1.SetDescription('NDVI')
        out_band1.SetNoDataValue(np.nan)
        out_band1.FlushCache()

        # 写入EVI (波段2)
        out_band2 = out_ds.GetRasterBand(2)
        out_band2.WriteArray(evi)
        out_band2.SetDescription('EVI')
        out_band2.SetNoDataValue(np.nan)
        out_band2.FlushCache()

        # 关闭数据集
        out_ds = None
        ds = None

        return (input_path, "success", None)

    except Exception as e:
        return (input_path, "failed", str(e))


def process_modis_files(input_root, output_root, num_processes=None):
    """并行处理MODIS TIFF文件"""
    # 确保输出目录存在
    os.makedirs(output_root, exist_ok=True)

    # 获取并排序所有tif文件
    files = [os.path.join(input_root, f) for f in os.listdir(input_root) if f.endswith('.tif')]
    files.sort()

    # 设置进程数
    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(), 8)  # 默认最多8个进程

    print(f"开始处理 {len(files)} 个文件，使用 {num_processes} 个并行进程...")

    # 创建进程池
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用partial固定输出目录参数
        worker_func = partial(process_single_file, output_dir=output_root)

        # 使用tqdm显示进度条
        results = []
        for result in tqdm(pool.imap_unordered(worker_func, files),
                           total=len(files),
                           desc="处理进度"):
            results.append(result)

    # 统计结果
    success_count = sum(1 for r in results if r[1] == "success")
    skipped_count = sum(1 for r in results if r[1] == "skipped")
    failed_count = sum(1 for r in results if r[1] == "failed")

    print(f"\n处理完成: 成功 {success_count}, 跳过 {skipped_count}, 失败 {failed_count}")

    # 打印失败文件
    if failed_count > 0:
        print("\n失败文件列表:")
        for file, status, error in results:
            if status == "failed":
                print(f"{os.path.basename(file)}: {error}")


if __name__ == "__main__":
    # 设置环境变量优化GDAL性能
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'TRUE'
    os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tif'
    os.environ['VSI_CACHE'] = 'TRUE'
    os.environ['VSI_CACHE_SIZE'] = '80000'  # 80GB缓存

    # 输入输出路径
    input_root = r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500_merge_TerraAqua'
    output_root = r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/NDVI_EVI'

    # 启动处理
    process_modis_files(input_root, output_root, num_processes=32)  # 可以调整进程数