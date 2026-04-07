import os
from osgeo import gdal
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

def count_valid_pixels(tif_file):
    """
    统计tif文件中的有效值像素个数
    
    参数:
        tif_file: tif文件路径
    返回:
        valid_count: 有效值像素个数
        nodata: NoData值
    """
    ds = gdal.Open(tif_file)
    if ds is None:
        raise Exception(f"无法打开文件: {tif_file}")
    
    # 获取第一个波段的NoData值
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    
    # 读取数据
    data = band.ReadAsArray()
    
    # 统计有效值像素个数
    if nodata is not None:
        valid_count = np.sum(data != nodata)
    else:
        valid_count = np.sum(~np.isnan(data))
    
    ds = None
    return valid_count, nodata

def analyze_folders(input_dir1, input_dir2):
    """
    分析两个文件夹中每个图像的有效值像素个数
    
    参数:
        input_dir1: 第一个输入文件夹路径
        input_dir2: 第二个输入文件夹路径
    """
    # 获取所有tif文件
    tif_files1 = glob.glob(os.path.join(input_dir1, "*.tif"))
    tif_files1.extend(glob.glob(os.path.join(input_dir1, "*.TIF")))
    tif_files2 = glob.glob(os.path.join(input_dir2, "*.tif"))
    tif_files2.extend(glob.glob(os.path.join(input_dir2, "*.TIF")))
    
    # 确保文件名匹配
    tif_files1.sort()
    tif_files2.sort()
    
    if len(tif_files1) != len(tif_files2):
        print(f"警告：两个文件夹中的文件数量不匹配: {len(tif_files1)} vs {len(tif_files2)}")
    
    # 创建结果列表
    results = []
    
    # 处理第一个文件夹
    print(f"\n处理文件夹1: {input_dir1}")
    for f in tqdm(tif_files1, desc="处理文件夹1"):
        try:
            valid_count, nodata = count_valid_pixels(f)
            results.append({
                'folder': 'folder1',
                'file': os.path.basename(f),
                'valid_pixels': valid_count,
                'nodata_value': nodata
            })
        except Exception as e:
            print(f"\n处理文件时发生错误:")
            print(f"文件: {f}")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print("-" * 50)
    
    # 处理第二个文件夹
    print(f"\n处理文件夹2: {input_dir2}")
    for f in tqdm(tif_files2, desc="处理文件夹2"):
        try:
            valid_count, nodata = count_valid_pixels(f)
            results.append({
                'folder': 'folder2',
                'file': os.path.basename(f),
                'valid_pixels': valid_count,
                'nodata_value': nodata
            })
        except Exception as e:
            print(f"\n处理文件时发生错误:")
            print(f"文件: {f}")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print("-" * 50)
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 获取每个文件夹的有效像素数唯一值
    folder1_unique = sorted(df[df['folder'] == 'folder1']['valid_pixels'].unique())
    folder2_unique = sorted(df[df['folder'] == 'folder2']['valid_pixels'].unique())
    
    # 打印结果
    print("\n文件夹1的有效像素数唯一值:")
    print(folder1_unique)
    print(f"数量: {len(folder1_unique)}")
    
    print("\n文件夹2的有效像素数唯一值:")
    print(folder2_unique)
    print(f"数量: {len(folder2_unique)}")
    
    # 检查两个文件夹的唯一值是否一致
    if set(folder1_unique) == set(folder2_unique):
        print("\n两个文件夹的有效像素数唯一值完全一致")
    else:
        print("\n两个文件夹的有效像素数唯一值不一致:")
        print("仅在文件夹1中存在的值:", set(folder1_unique) - set(folder2_unique))
        print("仅在文件夹2中存在的值:", set(folder2_unique) - set(folder1_unique))
    
    # 保存详细结果
    output_file = 'valid_pixels_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    # 设置GDAL配置
    gdal.SetConfigOption('GDAL_CACHEMAX', '20000')  # 80GB内存
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 使用所有可用CPU线程
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')  # 可选：提高多线程效率
    gdal.UseExceptions()
    
    # 设置输入路径
    output_dir1 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/ERA5_with_moisture_BC_Cropped_FIRMS10KM_aligned"
    output_dir2 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/Firms_Detection_BC_Cropped_10times_downsampled_FIRMS10KM_aligned"
    
    # 执行分析
    analyze_folders(output_dir1, output_dir2) 