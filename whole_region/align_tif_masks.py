import os
from osgeo import gdal
import glob
from tqdm import tqdm
import multiprocessing as mp
import numpy as np

def get_valid_mask(tif_file):
    """
    获取tif文件的有效值掩膜（非NoData值的区域）
    
    参数:
        tif_file: tif文件路径
    返回:
        valid_mask: 有效值掩膜（布尔数组）
    """
    ds = gdal.Open(tif_file)
    if ds is None:
        raise Exception(f"无法打开文件: {tif_file}")
    
    # 获取第一个波段的NoData值
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    
    # 读取数据
    data = band.ReadAsArray()
    
    # 创建有效值掩膜
    if nodata is not None:
        valid_mask = (data != nodata)
    else:
        valid_mask = ~np.isnan(data)
    
    ds = None
    return valid_mask

def apply_mask_to_tif(input_tif, output_tif, mask):
    """
    将掩膜应用到tif文件
    
    参数:
        input_tif: 输入tif文件路径
        output_tif: 输出tif文件路径
        mask: 掩膜（布尔数组）
    """
    try:
        # 打开输入文件
        ds = gdal.Open(input_tif)
        if ds is None:
            raise Exception(f"无法打开文件: {input_tif}")
        
        # 获取文件信息
        geo_transform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        data_type = ds.GetRasterBand(1).DataType
        band_count = ds.RasterCount
        
        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_tif,
            ds.RasterXSize,
            ds.RasterYSize,
            band_count,
            data_type,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
        )
        
        # 设置地理信息
        out_ds.SetGeoTransform(geo_transform)
        out_ds.SetProjection(projection)
        
        # 处理每个波段
        for band_idx in range(1, band_count + 1):
            # 读取数据
            data = ds.GetRasterBand(band_idx).ReadAsArray()
            
            # 获取NoData值
            nodata = ds.GetRasterBand(band_idx).GetNoDataValue()
            
            # 应用掩膜
            data[~mask] = nodata
            
            # 写入数据
            out_ds.GetRasterBand(band_idx).WriteArray(data)
            
            # 设置NoData值
            if nodata is not None:
                out_ds.GetRasterBand(band_idx).SetNoDataValue(nodata)
        
        # 清理
        ds = None
        out_ds = None
        return True
        
    except Exception as e:
        print(f"\n处理文件时发生错误:")
        print(f"文件: {input_tif}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("-" * 50)
        return False

def process_single_file(args):
    """处理单个文件的包装函数，用于并行处理"""
    input_tif, output_dir, mask = args
    output_filename = os.path.basename(input_tif)
    output_path = os.path.join(output_dir, output_filename)
    return apply_mask_to_tif(input_tif, output_path, mask)

def verify_folder_masks_consistency(input_dir, num_samples=5):
    """
    验证文件夹内所有文件的有效值区域是否一致
    
    参数:
        input_dir: 输入文件夹路径
        num_samples: 要检查的样本数量
    返回:
        bool: 是否一致
        mask: 如果一致，返回掩膜；否则返回None
    """
    # 获取所有tif文件
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    tif_files.extend(glob.glob(os.path.join(input_dir, "*.TIF")))
    tif_files.sort()
    
    if len(tif_files) == 0:
        print(f"错误：文件夹 {input_dir} 中没有找到tif文件")
        return False, None
    
    # 随机选择样本
    if len(tif_files) > num_samples:
        indices = np.random.choice(len(tif_files), num_samples, replace=False)
        sample_files = [tif_files[i] for i in indices]
    else:
        sample_files = tif_files
    
    print(f"正在验证文件夹 {input_dir} 中 {len(sample_files)} 个文件的掩膜一致性...")
    
    # 获取第一个文件的掩膜作为参考
    ref_mask = get_valid_mask(sample_files[0])
    
    # 检查其他样本
    for i, f in enumerate(sample_files[1:], 1):
        print(f"检查样本 {i+1}/{len(sample_files)}...")
        
        # 获取当前样本的掩膜
        mask = get_valid_mask(f)
        
        # 比较掩膜
        if not np.array_equal(ref_mask, mask):
            print(f"发现不一致:")
            print(f"参考文件: {sample_files[0]}")
            print(f"当前文件: {f}")
            print(f"掩膜差异像素数: {np.sum(ref_mask != mask)}")
            return False, None
    
    print(f"验证完成：文件夹 {input_dir} 中所有检查的样本掩膜完全一致")
    return True, ref_mask

def batch_align_masks(input_dir1, input_dir2, output_dir1, output_dir2):
    """
    批量处理两个文件夹中的tif文件
    
    参数:
        input_dir1: 第一个输入文件夹路径
        input_dir2: 第二个输入文件夹路径
        output_dir1: 第一个输出文件夹路径
        output_dir2: 第二个输出文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    
    # 首先验证两个文件夹内部的掩膜一致性
    print("开始验证文件夹1的掩膜一致性...")
    is_consistent1, mask1 = verify_folder_masks_consistency(input_dir1)
    if not is_consistent1:
        print("错误：文件夹1的掩膜不一致，终止处理")
        return
    
    print("\n开始验证文件夹2的掩膜一致性...")
    is_consistent2, mask2 = verify_folder_masks_consistency(input_dir2)
    if not is_consistent2:
        print("错误：文件夹2的掩膜不一致，终止处理")
        return
    
    # 计算两个文件夹掩膜的交集
    print("\n计算两个文件夹掩膜的交集...")
    intersection_mask = mask1 & mask2
    print("掩膜交集计算完成")
    
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
    
    # 准备并行处理的参数
    process_args1 = [(f, output_dir1, intersection_mask) for f in tif_files1]
    process_args2 = [(f, output_dir2, intersection_mask) for f in tif_files2]
    
    # 设置进程数（使用85%的CPU核心）
    num_processes = max(1, int(mp.cpu_count() * 0.85))
    print(f"使用 {num_processes} 个进程进行并行处理")
    
    # 使用进程池进行并行处理
    with mp.Pool(processes=num_processes) as pool:
        # 处理第一个文件夹
        results1 = list(tqdm(
            pool.imap(process_single_file, process_args1),
            total=len(process_args1),
            desc="处理文件夹1进度"
        ))
        
        # 处理第二个文件夹
        results2 = list(tqdm(
            pool.imap(process_single_file, process_args2),
            total=len(process_args2),
            desc="处理文件夹2进度"
        ))
    
    # 统计处理结果
    success_count1 = sum(1 for r in results1 if r)
    failed_count1 = len(results1) - success_count1
    success_count2 = sum(1 for r in results2 if r)
    failed_count2 = len(results2) - success_count2
    
    print(f"\n处理完成:")
    print(f"文件夹1: 成功 {success_count1} 个, 失败 {failed_count1} 个")
    print(f"文件夹2: 成功 {success_count2} 个, 失败 {failed_count2} 个")

if __name__ == "__main__":
    # 设置GDAL配置
    gdal.SetConfigOption('GDAL_CACHEMAX', '20000')  # 80GB内存
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 使用所有可用CPU线程
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')  # 可选：提高多线程效率
    gdal.UseExceptions()
    
    # 设置输入输出路径
    input_dir1 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/ERA5_with_moisture_BC_Cropped_ResizeFIRMS10KM"
    input_dir2 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/Firms_Detection_BC_Cropped_10times_downsampled"
    output_dir1 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/ERA5_with_moisture_BC_Cropped_FIRMS10KM_aligned"
    output_dir2 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/Firms_Detection_BC_Cropped_10times_downsampled_FIRMS10KM_aligned"
    
    # 执行批量处理
    batch_align_masks(input_dir1, input_dir2, output_dir1, output_dir2) 