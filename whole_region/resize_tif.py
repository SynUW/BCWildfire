import os
from osgeo import gdal
import glob
from tqdm import tqdm
import multiprocessing as mp
import traceback

def resize_tif(input_tif, output_tif, target_width, target_height):
    """
    调整tif图像大小并更新地理信息
    
    参数:
        input_tif: 输入的tif文件路径
        output_tif: 输出的tif文件路径
        target_width: 目标宽度（像素）
        target_height: 目标高度（像素）
    """
    try:
        # 打开输入文件
        src_ds = gdal.Open(input_tif)
        if src_ds is None:
            raise Exception(f"无法打开输入文件: {input_tif}")
        
        # 获取输入文件的信息
        src_geo_transform = src_ds.GetGeoTransform()
        src_projection = src_ds.GetProjection()
        src_band = src_ds.GetRasterBand(1)
        src_data_type = src_band.DataType
        src_nodata = src_band.GetNoDataValue()
        src_band_count = src_ds.RasterCount  # 获取波段数
        
        # 打印输入文件信息
        # print(f"\n处理文件: {input_tif}")
        # print(f"原始尺寸: {src_ds.RasterXSize}x{src_ds.RasterYSize}")
        # print(f"原始分辨率: x={src_geo_transform[1]:.6f}度, y={abs(src_geo_transform[5]):.6f}度")
        
        # 计算新的地理变换参数
        # 保持左上角坐标不变，调整分辨率
        new_geo_transform = list(src_geo_transform)
        new_geo_transform[1] = (src_geo_transform[1] * src_ds.RasterXSize) / target_width  # 新的x方向分辨率
        new_geo_transform[5] = (src_geo_transform[5] * src_ds.RasterYSize) / target_height  # 新的y方向分辨率
        
        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            output_tif,
            target_width,
            target_height,
            src_band_count,  # 使用原始文件的波段数
            src_data_type,
            options=[
                'COMPRESS=LZW',
                'TILED=YES',
                'BIGTIFF=YES'
            ]
        )
        
        if dst_ds is None:
            raise Exception(f"无法创建输出文件: {output_tif}")
        
        # 设置地理信息
        dst_ds.SetGeoTransform(new_geo_transform)
        dst_ds.SetProjection(src_projection)
        
        # 设置每个波段的NoData值
        for band_idx in range(1, src_band_count + 1):
            src_band = src_ds.GetRasterBand(band_idx)
            dst_band = dst_ds.GetRasterBand(band_idx)
            src_nodata = src_band.GetNoDataValue()
            if src_nodata is not None:
                dst_band.SetNoDataValue(src_nodata)
        
        # 执行重采样
        result = gdal.ReprojectImage(
            src_ds,
            dst_ds,
            src_projection,
            src_projection,
            gdal.GRA_Bilinear  # 使用双线性插值
        )
        
        if result != 0:
            raise Exception(f"重采样失败，错误代码: {result}")
        
        # 检查输出文件尺寸
        dst_ds = gdal.Open(output_tif)
        if dst_ds is None:
            raise Exception(f"无法打开输出文件进行验证: {output_tif}")
        
        if dst_ds.RasterXSize != target_width or dst_ds.RasterYSize != target_height:
            raise Exception(f"输出文件尺寸不符合要求: 期望 {target_width}x{target_height}, 实际 {dst_ds.RasterXSize}x{dst_ds.RasterYSize}")
        
        if dst_ds.RasterCount != src_band_count:
            raise Exception(f"输出文件波段数不符合要求: 期望 {src_band_count}, 实际 {dst_ds.RasterCount}")
        
        # 清理
        src_ds = None
        dst_ds = None
        
        # print(f"调整后尺寸: {target_width}x{target_height}")
        # print(f"新的分辨率: x={new_geo_transform[1]:.6f}度, y={abs(new_geo_transform[5]):.6f}度")
        # print("处理成功！")
        # print("-" * 50)
        return True
        
    except Exception as e:
        print(f"\n处理文件 {input_tif} 时发生错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("详细错误信息:")
        print(traceback.format_exc())
        print("-" * 50)
        return False

def process_single_file(args):
    """处理单个文件的包装函数，用于并行处理"""
    input_tif, output_dir, target_width, target_height = args
    output_filename = os.path.basename(input_tif)
    output_path = os.path.join(output_dir, output_filename)
    return resize_tif(input_tif, output_path, target_width, target_height)

def batch_resize_tifs(input_dir, output_dir, target_width, target_height):
    """
    批量处理文件夹中的所有tif文件
    
    参数:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        target_width: 目标宽度（像素）
        target_height: 目标高度（像素）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有tif文件
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    tif_files.extend(glob.glob(os.path.join(input_dir, "*.TIF")))
    
    if not tif_files:
        print(f"在 {input_dir} 中没有找到tif文件")
        return
    
    print(f"找到 {len(tif_files)} 个tif文件")
    
    # 准备并行处理的参数
    process_args = [(f, output_dir, target_width, target_height) for f in tif_files]
    
    # 设置进程数（使用85%的CPU核心）
    num_processes = max(1, int(mp.cpu_count() * 0.85))
    print(f"使用 {num_processes} 个进程进行并行处理")
    
    # 使用进程池进行并行处理
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, process_args),
            total=len(process_args),
            desc="处理文件进度"
        ))
    
    # 统计处理结果
    success_count = sum(1 for r in results if r)
    failed_count = len(results) - success_count
    
    print(f"\n处理完成: 成功 {success_count} 个, 失败 {failed_count} 个")
    
    # 如果有失败的文件，列出它们
    if failed_count > 0:
        print("\n失败的文件列表:")
        for i, (result, file) in enumerate(zip(results, tif_files)):
            if not result:
                print(f"{i+1}. {file}")

if __name__ == "__main__":
    # 设置GDAL配置
    gdal.SetConfigOption('GDAL_CACHEMAX', '20000')  # 80GB内存
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 使用所有可用CPU线程
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')  # 可选：提高多线程效率
    gdal.UseExceptions()
    
    # 设置输入输出路径
    input_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/ERA5_with_moisture_BC_Cropped"
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/ERA5_with_moisture_BC_Cropped_ResizeFIRMS10KM"
    target_width = 250  # 目标宽度（像素）
    target_height = 116  # 目标高度（像素）
    
    # 执行批量调整大小
    batch_resize_tifs(input_dir, output_dir, target_width, target_height) 