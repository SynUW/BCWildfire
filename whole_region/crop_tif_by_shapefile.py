import os
from osgeo import gdal
import glob
from tqdm import tqdm
import multiprocessing as mp


def crop_tif_by_shapefile(input_tif, shapefile, output_tif):
    """
    使用shapefile裁剪tif图像，边界外的区域设置为255
    
    参数:
        input_tif: 输入的tif文件路径
        shapefile: shapefile文件路径
        output_tif: 输出的tif文件路径
    """
    try:
        # 打开输入文件以获取数据类型和分辨率
        src_ds = gdal.Open(input_tif)
        if src_ds is None:
            raise Exception(f"无法打开输入文件: {input_tif}")
        
        # 获取输入文件的数据类型和分辨率
        data_type = src_ds.GetRasterBand(1).DataType
        geo_transform = src_ds.GetGeoTransform()
        x_res = geo_transform[1]  # 获取x方向分辨率
        y_res = abs(geo_transform[5])  # 获取y方向分辨率（取绝对值）
        src_ds = None

        # 设置裁剪选项
        warp_options = gdal.WarpOptions(
            format='GTiff',
            cutlineDSName=shapefile,  # 裁剪边界文件
            cropToCutline=True,       # 裁剪到边界
            dstNodata=255,            # 设置边界外的值为255
            xRes=x_res,              # 使用原始x方向分辨率
            yRes=y_res,              # 使用原始y方向分辨率
            outputType=data_type,     # 保持原始数据类型
            creationOptions=[
                'COMPRESS=LZW',        # 使用LZW压缩
                'TILED=YES',          # 使用分块存储
                'BIGTIFF=YES'         # 支持大文件
            ]
        )
        
        # 执行裁剪
        gdal.Warp(output_tif, input_tif, options=warp_options)
        return True
    except Exception as e:
        print(f"处理文件 {input_tif} 时发生错误: {str(e)}")
        return False


def process_single_file(args):
    """处理单个文件的包装函数，用于并行处理"""
    input_tif, shapefile, output_dir = args
    output_filename = os.path.basename(input_tif)
    output_path = os.path.join(output_dir, output_filename)
    return crop_tif_by_shapefile(input_tif, shapefile, output_path)


def batch_crop_tifs(input_dir, shapefile, output_dir):
    """
    批量处理文件夹中的所有tif文件
    
    参数:
        input_dir: 输入文件夹路径
        shapefile: shapefile文件路径
        output_dir: 输出文件夹路径
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
    process_args = [(f, shapefile, output_dir) for f in tif_files]
    
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


if __name__ == "__main__":
    # 设置GDAL配置
    gdal.SetConfigOption('GDAL_CACHEMAX', '20000')  # 80GB内存
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 使用所有可用CPU线程
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES')  # 可选：提高多线程效率
    gdal.UseExceptions()
    
    # 设置输入输出路径
    input_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/ERA5_with_moisture/ERA5"
    shapefile = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_boundary/british_columnbia_no_crs.shp"
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/ERA5_with_moisture_BC_Cropped"
    
    # 执行批量裁剪
    batch_crop_tifs(input_dir, shapefile, output_dir)
