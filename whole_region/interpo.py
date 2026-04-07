import os
import numpy as np
from osgeo import gdal, gdalconst
from datetime import datetime, timedelta
from tqdm import tqdm


def generate_empty_tif(template_path, output_path, date_str):
    """根据模板生成一个全为nodata的TIFF文件"""
    try:
        template_ds = gdal.Open(template_path, gdalconst.GA_ReadOnly)
        if template_ds is None:
            print(f"Failed to open template file: {template_path}")
            return False

        cols = template_ds.RasterXSize
        rows = template_ds.RasterYSize
        bands = template_ds.RasterCount
        dtype = template_ds.GetRasterBand(1).DataType
        nodata = template_ds.GetRasterBand(1).GetNoDataValue()
        projection = template_ds.GetProjection()
        geotransform = template_ds.GetGeoTransform()

        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_path,
            cols,
            rows,
            bands,
            dtype,
            options=['COMPRESS=LZW']
        )

        if out_ds is None:
            print(f"Failed to create output file: {output_path}")
            template_ds = None
            return False

        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)

        empty_data = np.zeros((rows, cols), dtype=np.float32)
        if nodata is not None:
            empty_data[:] = nodata

        for band in range(1, bands + 1):
            out_band = out_ds.GetRasterBand(band)
            out_band.WriteArray(empty_data)
            if nodata is not None:
                out_band.SetNoDataValue(nodata)
            out_band.FlushCache()

        out_ds.SetDescription(f"Empty LA data for {date_str}")
        out_ds = None
        template_ds = None
        return True

    except Exception as e:
        print(f"Error generating empty TIFF: {e}")
        if 'out_ds' in locals() and out_ds: out_ds = None
        if 'template_ds' in locals() and template_ds: template_ds = None
        return False


def process_continuous_firms_data(input_folder, output_folder):
    """
    处理连续日期的FIRMS数据，填充缺失日期

    Args:
        input_folder (str): 输入文件夹路径（包含原始FIRMS文件）
        output_folder (str): 输出文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有有效日期和模板路径
    file_dates = []
    template_path = None

    for filename in os.listdir(input_folder):
        if filename.startswith('LAI_') and filename.endswith('.tif'):
            try:
                date_str = filename[6:-4]
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                file_dates.append(date_obj)

                if template_path is None:
                    template_path = os.path.join(input_folder, filename)
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
                continue

    if not file_dates:
        print("No valid LA files found in input folder")
        return

    min_date = min(file_dates)
    max_date = max(file_dates)
    print(f"Processing date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    if template_path is None:
        print("No template file found")
        return

    # 预处理：收集所有需要跳过的日期（原始文件已存在）
    existing_dates = set()
    for filename in os.listdir(input_folder):
        if filename.startswith('LAI_') and filename.endswith('.tif'):
            try:
                date_str = filename[6:-4]
                existing_dates.add(date_str)
            except:
                continue

    current_date = min_date
    total_days = (max_date - min_date).days + 1

    with tqdm(total=total_days, desc="Processing days") as pbar:
        while current_date <= max_date:
            date_str = current_date.strftime('%Y-%m-%d')
            output_path = os.path.join(output_folder, f"LAI_{date_str}.tif")

            # 跳过条件1：输出文件已存在
            if os.path.exists(output_path):
                pbar.update(1)
                current_date += timedelta(days=1)
                continue

            # 跳过条件2：原始文件已存在（新增）
            input_path = os.path.join(input_folder, f"LAI_{date_str}.tif")
            if os.path.exists(input_path):
                pbar.update(1)
                current_date += timedelta(days=1)
                continue

            # 生成空数据文件
            if not generate_empty_tif(template_path, output_path, date_str):
                print(f"Failed to generate empty file for {date_str}")

            pbar.update(1)
            current_date += timedelta(days=1)


if __name__ == '__main__':
    gdal.AllRegister()

    input_folder = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LAI_500_interpolated_clip'  # 替换为你的输入文件夹路径
    output_folder = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LAI_500_interpolated_clip_missingDays'  # 替换为你的输出文件夹路径

    if not os.path.exists(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        exit(1)

    process_continuous_firms_data(input_folder, output_folder)
    print("Processing completed!")