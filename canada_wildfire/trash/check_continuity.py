import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
from osgeo import gdal
import numpy as np
from tqdm import tqdm


def parse_date_from_filename(filename):
    """从文件名中解析日期，支持两种格式：
    1. FIRMS_20000220.tif
    2. FIRMS_2000_02_20.tif
    """
    # 尝试第一种格式：FIRMS_20000220.tif
    pattern1 = re.compile(r'^(.+)_(\d{8})\.tif$')
    match1 = pattern1.match(filename)
    if match1:
        factor = match1.group(1)
        date_str = match1.group(2)
        try:
            date = datetime.strptime(date_str, "%Y%m%d")
            return factor, date
        except ValueError:
            pass

    # 尝试第二种格式：FIRMS_2000_02_20.tif
    pattern2 = re.compile(r'^(.+)_(\d{4})_(\d{2})_(\d{2})\.tif$')
    match2 = pattern2.match(filename)
    if match2:
        factor = match2.group(1)
        date_str = f"{match2.group(2)}-{match2.group(3)}-{match2.group(4)}"
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            return factor, date
        except ValueError:
            pass

    return None, None


def find_missing_dates(folder_path, start_date_str, end_date_str):
    """识别文件夹中的缺失日期，基于指定的日期范围"""
    factor_dates = defaultdict(list)

    # 解析输入日期范围
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    # 收集所有文件的日期
    for filename in os.listdir(folder_path):
        factor, date = parse_date_from_filename(filename)
        if factor and date:
            # 只收集在指定日期范围内的日期
            if start_date <= date <= end_date:
                factor_dates[factor].append(date)

    # 找出每个因素的缺失日期
    missing_info = {}
    for factor, dates in factor_dates.items():
        if not dates:
            # 如果没有任何文件，则整个日期范围都缺失
            missing_dates = [start_date + timedelta(days=x)
                             for x in range((end_date - start_date).days + 1)]
            missing_info[factor] = {
                'template': None,  # 没有模板文件
                'missing_dates': missing_dates,
                'geoinfo': None
            }
            continue

        dates.sort()
        # 生成完整日期序列（从输入的开始日期到结束日期）
        all_dates = {start_date + timedelta(days=x)
                     for x in range((end_date - start_date).days + 1)}
        existing_dates = set(dates)
        missing_dates = sorted(all_dates - existing_dates)

        if missing_dates:
            # 使用该因素最早的文件作为模板
            template_date = min(d for d in dates if d >= start_date)
            missing_info[factor] = {
                'template': template_date,
                'missing_dates': missing_dates,
                'geoinfo': None  # 稍后填充
            }

    return missing_info


def fill_missing_with_zeros(folder_path, start_date_str, end_date_str):
    """主函数：填充缺失日期为全零文件"""
    # 1. 识别缺失日期
    print("正在分析缺失日期...")
    missing_info = find_missing_dates(folder_path, start_date_str, end_date_str)

    if not missing_info:
        print("没有发现缺失日期")
        return

    # 2. 加载模板的地理信息
    print("加载模板地理信息...")
    for factor, info in missing_info.items():
        if info['template'] is None:
            print(f"警告: 因素 {factor} 在指定日期范围内没有任何文件，无法创建模板")
            continue

        # 尝试两种可能的文件名格式
        template_file1 = f"{factor}_{info['template'].strftime('%Y%m%d')}.tif"
        template_file2 = f"{factor}_{info['template'].strftime('%Y_%m_%d')}.tif"
        template_path1 = os.path.join(folder_path, template_file1)
        template_path2 = os.path.join(folder_path, template_file2)

        # 尝试打开模板文件
        ds = None
        if os.path.exists(template_path1):
            ds = gdal.Open(template_path1)
        elif os.path.exists(template_path2):
            ds = gdal.Open(template_path2)

        if ds:
            info['geoinfo'] = {
                'cols': ds.RasterXSize,
                'rows': ds.RasterYSize,
                'bands': ds.RasterCount,
                'projection': ds.GetProjection(),
                'geotransform': ds.GetGeoTransform(),
                'data_type': ds.GetRasterBand(1).DataType,
                'descriptions': [ds.GetRasterBand(b + 1).GetDescription() for b in range(ds.RasterCount)]
            }
            ds = None
        else:
            print(f"无法打开模板文件 {template_path1} 或 {template_path2}")

    # 3. 为每个缺失日期创建全零文件
    print("开始创建缺失文件...")
    total_created = 0

    for factor, info in missing_info.items():
        if not info['geoinfo']:
            print(f"无法获取 {factor} 的模板信息，跳过")
            continue

        # 判断原始文件日期格式
        date_fmt = None
        for filename in os.listdir(folder_path):
            if filename.startswith(factor + "_"):
                if re.match(rf"{factor}_\d{{8}}\.tif", filename):
                    date_fmt = "%Y%m%d"
                    break
                elif re.match(rf"{factor}_\d{{4}}_\d{{2}}_\d{{2}}\.tif", filename):
                    date_fmt = "%Y_%m_%d"
                    break
        if date_fmt is None:
            date_fmt = "%Y%m%d"  # 默认

        for missing_date in tqdm(info['missing_dates'], desc=f"处理 {factor}"):
            output_file = f"{factor}_{missing_date.strftime(date_fmt)}.tif"
            output_path = os.path.join(folder_path, output_file)

            if os.path.exists(output_path):
                continue

            # 使用内存创建避免磁盘I/O
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(
                output_path,
                info['geoinfo']['cols'],
                info['geoinfo']['rows'],
                info['geoinfo']['bands'],
                info['geoinfo']['data_type']
            )
            out_ds.SetGeoTransform(info['geoinfo']['geotransform'])
            out_ds.SetProjection(info['geoinfo']['projection'])

            zero_array = np.zeros((info['geoinfo']['rows'], info['geoinfo']['cols']), dtype=np.float32)
            for band in range(info['geoinfo']['bands']):
                out_band = out_ds.GetRasterBand(band + 1)
                out_band.WriteArray(zero_array)
                if band < len(info['geoinfo']['descriptions']):
                    out_band.SetDescription(info['geoinfo']['descriptions'][band])

            out_ds = None
            total_created += 1

    print(f"\n完成! 共创建了 {total_created} 个缺失日期的全零文件")


if __name__ == "__main__":
    input_folder = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LAI_BCBoundingbox"
    start_date = "2000-01-01"  # 指定开始日期
    end_date = "2024-12-31"  # 指定结束日期

    fill_missing_with_zeros(input_folder, start_date, end_date)