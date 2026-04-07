import os
import glob
from osgeo import gdal
import numpy as np
from tqdm import tqdm
import logging
import multiprocessing
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('GeoTIFF_Resampler')

# 设置GDAL缓存大小
gdal.SetCacheMax(1024 * 1024 * 1024)  # 1GB缓存

def get_resample_method(input_dir):
    """
    根据输入目录确定重采样方法
    
    参数:
        input_dir (str): 输入目录路径
    
    返回:
        int: GDAL重采样方法
    """
    # 使用最近邻重采样的目录列表
    nearest_dirs = [
        'Firms_Detection_aligned',
        'LULC_BCBoundingbox_interpolated',
        'Topo_Distance_WGS84_resize_daily'
    ]
    
    # 检查输入目录是否包含需要最近邻重采样的关键词
    for dir_name in nearest_dirs:
        if dir_name in input_dir:
            return gdal.GRA_NearestNeighbour
    
    # 默认使用双线性插值
    return gdal.GRA_Bilinear

def resample_geotiff(input_file, output_file, target_width, target_height, resample_method):
    """
    重采样GeoTIFF文件
    
    参数:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        target_width (int): 目标宽度
        target_height (int): 目标高度
        resample_method (int): GDAL重采样方法
    """
    try:
        # 打开输入文件
        src_ds = gdal.Open(input_file)
        if src_ds is None:
            raise Exception(f"无法打开输入文件: {input_file}")
            
        # 获取输入文件的地理信息
        src_geo_transform = src_ds.GetGeoTransform()
        projection = src_ds.GetProjection()
        
        # 计算新的地理变换参数
        # 保持左上角坐标不变，调整分辨率以适应新的尺寸
        new_geo_transform = list(src_geo_transform)
        new_geo_transform[1] = (src_geo_transform[1] * src_ds.RasterXSize) / target_width  # 新的x分辨率
        new_geo_transform[5] = (src_geo_transform[5] * src_ds.RasterYSize) / target_height  # 新的y分辨率
        
        # 读取源 NoData，若未设置则使用 -9999（该数据集约定）
        src_band_1 = src_ds.GetRasterBand(1)
        src_nodata = src_band_1.GetNoDataValue()
        if src_nodata is None:
            src_nodata = -9999.0

        # 使用 gdal.Warp 完成重采样与写出，并在过程中屏蔽无效值
        warp_result = gdal.Warp(
            destNameOrDestDS=output_file,
            srcDSOrSrcDSTab=src_ds,
            format='GTiff',
            width=target_width,
            height=target_height,
            resampleAlg=resample_method,
            srcNodata=src_nodata,
            dstNodata=src_nodata,
            multithread=True,
            creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
            warpOptions=['UNIFIED_SRC_NODATA=YES', 'INIT_DEST=NO_DATA']
        )
        if warp_result is None:
            raise Exception(f"gdal.Warp 失败: {output_file}")
        warp_result = None
        
        # 清理
        src_ds = None
        
        return True
        
    except Exception as e:
        logger.error(f"处理文件 {input_file} 时出错: {str(e)}")
        return False

def get_output_dir(input_dir, base_output_dir):
    """
    根据输入目录生成输出目录路径
    
    参数:
        input_dir (str): 输入目录路径
        base_output_dir (str): 基础输出目录
    
    返回:
        str: 输出目录路径
    """
    # 获取输入目录的最后一层
    dir_name = os.path.basename(input_dir)
    # 替换后缀
    output_name = dir_name.replace('_aligned', '_resampled').replace('_interpolated', '_resampled').replace('_resize_daily', '_resampled')
    # 构建输出路径
    return os.path.join(base_output_dir, output_name)

def process_file(args):
    """
    处理单个文件的函数
    """
    input_file, output_file, target_width, target_height, resample_method = args
    # 新增：跳过已存在的文件
    if os.path.exists(output_file):
        logger.info(f"已存在，跳过: {output_file}")
        return True
    try:
        return resample_geotiff(input_file, output_file, target_width, target_height, resample_method)
    except Exception as e:
        logger.error(f"处理文件 {input_file} 时出错: {str(e)}")
        return False

def process_directory(input_dir, base_output_dir, target_width, target_height, pool):
    """
    处理单个目录的函数
    """
    try:
        # 获取所有tif文件
        tif_files = glob.glob(os.path.join(input_dir, '*.tif'))
        
        if not tif_files:
            logger.warning(f"在目录 {input_dir} 中没有找到tif文件")
            return False
            
        # 确定重采样方法
        resample_method = get_resample_method(input_dir)
        method_name = "最近邻" if resample_method == gdal.GRA_NearestNeighbour else "双线性插值"
        logger.info(f"目录 {input_dir} 使用{method_name}重采样方法")
        
        # 创建输出目录
        output_dir = get_output_dir(input_dir, base_output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"创建输出目录: {output_dir}")
        
        # 准备参数
        args = []
        for tif_file in tif_files:
            output_file = os.path.join(output_dir, os.path.basename(tif_file))
            args.append((tif_file, output_file, target_width, target_height, resample_method))
        
        # 使用进程池并行处理所有文件
        list(tqdm(
            pool.imap(process_file, args),
            total=len(args),
            desc=f"处理目录 {os.path.basename(input_dir)}"
        ))
        
        return True
        
    except Exception as e:
        logger.error(f"处理目录 {input_dir} 时出错: {str(e)}")
        return False

def main():
    # 设置输入目录列表
    input_dirs = [
        # '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/Firms_Detection_aligned',
        # '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/ERA5_with_moisture/ERA5_multi_bands',
        # '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LAI_BCBoundingbox_interpolated',
        # '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/MODIS_LST_29_31_32_QC/MOD21A1D_filtered_aligned',
        # '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/MODIS_LST_29_31_32_QC/MOD21A1N_filtered_aligned',
        # '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/NDVI_EVI',
        # '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection/MODIS_Terra_Aqua_B20_21_merged_aligned',
        # '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LULC_BCBoundingbox_interpolated',
        # '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Topo_Distance_WGS84_resize'
        # '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500_merge_TerraAquaWGS84_clip'
        
        
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/MODIS_LST_29_31_32_QC/MOD21A1DN_multibands_withoutFiltering_merged'
    ]
    
    # 设置基础输出目录
    base_output_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 设置目标尺寸
    target_width = 2783
    target_height = 1301
    
    # 设置进程数为CPU内核数的85%
    num_processes = max(1, int(multiprocessing.cpu_count() * 0.3))
    logger.info(f"使用 {num_processes} 个进程进行并行处理")
    
    # 创建进程池
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 顺序处理每个目录，但目录内的文件并行处理
        for input_dir in tqdm(input_dirs, desc="处理文件夹"):
            process_directory(input_dir, base_output_dir, target_width, target_height, pool)
    
    logger.info("所有目录处理完成")

if __name__ == '__main__':
    main() 