import os
import re
from osgeo import gdal, osr
from collections import defaultdict
from datetime import datetime
import numpy as np
from multiprocessing import Pool, cpu_count
import tempfile

# 配置参数
GDAL_CACHE_SIZE = 2 ** 30*150  # 1GB缓存
COMPRESS = 'LZW'  # 压缩方式
PREDICTOR = 2  # 预测器（浮点数据用2）
NUM_THREADS = max(1, cpu_count() - 2)  # 使用大部分CPU核心
RESAMPLING_METHOD = gdal.GRA_NearestNeighbour  # 重采样方法


def parse_date_from_filename(filename):
    """优化版日期解析"""
    match = re.search(r'_(\d{4})[-_](\d{2})[-_](\d{2})\.tif$', filename.lower())
    return datetime.strptime(f"{match.group(1)}-{match.group(2)}-{match.group(3)}",
                             "%Y-%m-%d").date() if match else None


def check_spatial_consistency(datasets):
    """检查空间参考一致性"""
    if not datasets:
        return False, None

    # 以最后一个数据集为基准
    ref_ds = datasets[-1]
    ref_proj = ref_ds.GetProjection()
    ref_gt = ref_ds.GetGeoTransform()
    ref_xsize = ref_ds.RasterXSize
    ref_ysize = ref_ds.RasterYSize

    # 检查参数
    for i, ds in enumerate(datasets[:-1]):
        if (ds.RasterXSize != ref_xsize or
                ds.RasterYSize != ref_ysize or
                ds.GetProjection() != ref_proj or
                not np.allclose(ds.GetGeoTransform(), ref_gt, atol=1e-6)):
            return False, ref_ds

    return True, None


def reproject_to_match(src_path, ref_ds, temp_dir):
    """将源数据重投影/重采样到参考数据集"""
    temp_path = os.path.join(temp_dir, os.path.basename(src_path))

    # 使用内存优化参数
    gdal.Warp(temp_path, src_path,
              format='GTiff',
              outputBounds=[
                  ref_ds.GetGeoTransform()[0],  # minX
                  ref_ds.GetGeoTransform()[3] + ref_ds.GetGeoTransform()[5] * ref_ds.RasterYSize,  # minY (注意Y方向)
                  ref_ds.GetGeoTransform()[0] + ref_ds.GetGeoTransform()[1] * ref_ds.RasterXSize,  # maxX
                  ref_ds.GetGeoTransform()[3]  # maxY
              ],
              xRes=abs(ref_ds.GetGeoTransform()[1]),
              yRes=abs(ref_ds.GetGeoTransform()[5]),
              dstSRS=ref_ds.GetProjection(),
              resampleAlg=RESAMPLING_METHOD,
              outputType=gdal.GDT_Float32,
              creationOptions=[
                  f'COMPRESS={COMPRESS}',
                  f'PREDICTOR={PREDICTOR}',
                  'BIGTIFF=IF_SAFER',
                  'TILED=YES'
              ],
              multithread=True)


    return temp_path


def process_date(args):
    """增强版日期处理函数"""
    date, file_infos, output_dir, temp_dir = args
    output_path = os.path.join(output_dir, f'stacked_{date}.tif')

    if os.path.exists(output_path):
        return f"跳过 {date}: 文件已存在"

    try:
        # 1. 打开所有数据集
        datasets = []
        temp_files = []
        for _, filepath, _ in file_infos:
            ds = gdal.Open(filepath, gdal.GA_ReadOnly)
            if not ds:
                raise ValueError(f"无法打开文件: {filepath}")
            datasets.append(ds)

        # 2. 检查空间一致性
        is_consistent, ref_ds = check_spatial_consistency(datasets)

        # 3. 处理不一致的情况
        if not is_consistent:
            if not ref_ds:
                raise ValueError("无法确定参考空间基准")

            # 重新处理所有非基准数据集
            for i in range(len(datasets) - 1):
                if datasets[i].GetProjection() != ref_ds.GetProjection() or \
                        not np.allclose(datasets[i].GetGeoTransform(), ref_ds.GetGeoTransform(), atol=1e-6):
                    reproj_path = reproject_to_match(file_infos[i][1], ref_ds, temp_dir)
                    temp_files.append(reproj_path)
                    datasets[i] = gdal.Open(reproj_path, gdal.GA_ReadOnly)

        # 4. 计算总波段数
        total_bands = sum(ds.RasterCount for ds in datasets)

        # 5. 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_path,
            ref_ds.RasterXSize,
            ref_ds.RasterYSize,
            total_bands,
            gdal.GDT_Float32,  # 统一输出为float32
            options=[
                f'COMPRESS={COMPRESS}',
                f'PREDICTOR={PREDICTOR}',
                'BIGTIFF=IF_SAFER',
                'TILED=YES',
                f'NUM_THREADS={NUM_THREADS}'
            ]
        )
        out_ds.SetProjection(ref_ds.GetProjection())
        out_ds.SetGeoTransform(ref_ds.GetGeoTransform())

        # 6. 写入数据
        current_band = 1
        for ds in datasets:
            for band_num in range(1, ds.RasterCount + 1):
                src_band = ds.GetRasterBand(band_num)
                dst_band = out_ds.GetRasterBand(current_band)

                # 使用分块读取提高大文件处理效率
                for y in range(0, ds.RasterYSize, 512):  # 512行为一个块
                    height = min(512, ds.RasterYSize - y)
                    arr = src_band.ReadAsArray(0, y, ds.RasterXSize, height)
                    dst_band.WriteArray(arr, 0, y)

                # 设置波段元数据
                desc = src_band.GetDescription()
                dst_band.SetDescription(desc if desc else
                                        f"{os.path.splitext(os.path.basename(ds.GetDescription()))[0]}_b{band_num}")

                current_band += 1

        return f"完成 {date}: 共 {total_bands} 波段"

    except Exception as e:
        # 出错时清理
        if 'out_ds' in locals():
            out_ds = None
            if os.path.exists(output_path):
                os.remove(output_path)
        return f"错误 {date}: {str(e)}"
    finally:
        # 确保关闭和清理
        for ds in datasets:
            ds = None
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass


def stack_bands_optimized(folders, output_dir, start_date, end_date):
    """最终优化版主函数"""
    os.makedirs(output_dir, exist_ok=True)
    gdal.SetCacheMax(GDAL_CACHE_SIZE)

    # 使用临时目录处理重投影文件
    with tempfile.TemporaryDirectory() as temp_dir:
        print("正在扫描和验证文件...")

        # 第一阶段：快速扫描和验证
        date_files = defaultdict(list)
        valid_dates = []

        for folder_idx, folder in enumerate(folders):
            if not os.path.isdir(folder):
                print(f"警告：跳过不存在文件夹 {folder}")
                continue

            for filename in os.listdir(folder):
                if not filename.lower().endswith('.tif'):
                    continue

                date = parse_date_from_filename(filename)
                if date and start_date <= date <= end_date:
                    filepath = os.path.join(folder, filename)
                    date_files[date].append((folder_idx, filepath, None))

        # 第二阶段：准备任务
        tasks = []
        for date, files_info in date_files.items():
            if len(files_info) == len(folders):
                files_info.sort(key=lambda x: x[0])  # 按文件夹顺序排序
                tasks.append((date, files_info, output_dir, temp_dir))

        print(f"找到 {len(tasks)} 个有效日期")

        # 第三阶段：并行处理
        print(f"开始并行处理（使用 {NUM_THREADS} 线程）...")
        with Pool(processes=NUM_THREADS) as pool:
            results = list(pool.imap(process_date, tasks))

        # 打印结果摘要
        success = sum(1 for r in results if r.startswith("完成"))
        skipped = sum(1 for r in results if r.startswith("跳过"))
        errors = sum(1 for r in results if r.startswith("错误"))

        print(f"\n处理完成: 成功 {success}, 跳过 {skipped}, 错误 {errors}")


if __name__ == "__main__":
    # 配置
    input_folders = [
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500_merge_mergeTerraAqua_clip',
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/NDVI_EVI',
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LAI_500_interpolated',
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/DAILY_500_clip',
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LULC_500_interpolated',
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Topo_Distance_500_clip_daily',
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Terra_Aqua_Daily',
    ]
    output_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/final_result'

    # 日期范围
    start_date = datetime.strptime("2000-02-24", "%Y-%m-%d").date()
    end_date = datetime.strptime("2024-12-31", "%Y-%m-%d").date()

    # 运行优化版本
    stack_bands_optimized(input_folders, output_dir, start_date, end_date)