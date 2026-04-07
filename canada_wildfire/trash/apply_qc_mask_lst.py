"""
使用QC波段对MODIS LST数据进行掩膜过滤
QC波段是一个16位的整数值，每个位或位组合代表不同的质量信息：
- Bits 0-1: 强制QA标志
  * 0: 像素已生成，质量好
  * 1: 像素已生成，质量一般
  * 2: 由于云层未生成像素
  * 3: 由于其他原因未生成像素
- Bits 4-5: 云标志
  * 0: 无云
  * 1: 薄卷云
  * 2: 像素在最近云层2像素范围内
  * 3: 有云像素
- Bits 12-13: 发射率精度
  * 0: >0.02 (性能差)
  * 1: 0.015-0.02 (性能一般)
  * 2: 0.01-0.015 (性能好)
  * 3: <0.01 (性能优秀)
- Bits 14-15: LST精度
  * 0: >2K (性能差)
  * 1: 1.5-2K (性能一般)
  * 2: 1-1.5K (性能好)
  * 3: <1K (性能优秀)

过滤条件：
1. Bits 0-1 <= 1: 保留已生成的像素（质量好或一般）
2. Bits 4-5 <= 1: 允许无云或薄卷云（避免完全遮云）
3. Bits 12-13 >= 1: 允许发射率精度一般及以上（marginal、good、excellent）
4. Bits 14-15 >= 1: 允许LST精度一般及以上（marginal、good、excellent）

不符合以上条件的像素将被设置为-9999（无效值）
"""
import os
import glob
from osgeo import gdal
import numpy as np
from tqdm import tqdm
import logging
import re
from datetime import datetime
import multiprocessing
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('QC_Mask_Applier')

# 设置GDAL缓存大小
gdal.SetCacheMax(1024 * 1024 * 1024)  # 1GB缓存

def extract_bits(qc_value, start_bit, end_bit):
    """
    从QC值中提取指定位范围的值
    
    参数:
        qc_value (int): QC值
        start_bit (int): 起始位
        end_bit (int): 结束位
    
    返回:
        int: 提取的位值
    """
    mask = (1 << (end_bit - start_bit + 1)) - 1
    return (qc_value >> start_bit) & mask

def create_mask(qc_array):
    """
    根据QC值创建掩膜
    
    参数:
        qc_array (numpy.ndarray): QC波段数组
    
    返回:
        numpy.ndarray: 布尔掩膜数组，True表示有效像素，False表示需要过滤的像素
    """
    # 确保QC数组是整数类型
    qc_array = qc_array.astype(np.uint16)
    mask = np.ones_like(qc_array, dtype=bool)
    
    # 遍历每个像素
    for i in range(qc_array.shape[0]):
        for j in range(qc_array.shape[1]):
            qc_value = int(qc_array[i, j])  # 确保qc_value是整数
            
            # 检查各个条件
            mandatory_qa = extract_bits(qc_value, 0, 1)      # 检查像素是否生成
            cloud_flag = extract_bits(qc_value, 4, 5)        # 检查云标志
            emissivity_accuracy = extract_bits(qc_value, 12, 13)  # 检查发射率精度
            lst_accuracy = extract_bits(qc_value, 14, 15)    # 检查LST精度
            
            # 应用掩膜条件
            if (mandatory_qa > 1 or  # 像素未生成（值为2或3）
                cloud_flag > 1 or   # 非无云且非薄卷云（值为2或3）
                emissivity_accuracy < 1 or  # 发射率精度不够好（值为0）
                lst_accuracy < 1):   # LST精度不够好（值为0）
                mask[i, j] = False
    
    return mask

def apply_mask_to_band(band_data, mask):
    """
    将掩膜应用到波段数据
    
    参数:
        band_data (numpy.ndarray): 波段数据
        mask (numpy.ndarray): 布尔掩膜数组
    
    返回:
        numpy.ndarray: 掩膜后的数据，无效像素被设置为-9999
    """
    masked_data = band_data.copy()
    masked_data[~mask] = -9999  # 使用-9999作为无效值
    return masked_data

def process_file(input_file, output_dir):
    """
    处理单个多波段GeoTIFF文件
    
    参数:
        input_file (str): 输入文件路径
        output_dir (str): 输出目录路径
    """
    try:
        # 打开输入文件
        ds = gdal.Open(input_file)
        if ds is None:
            raise Exception(f"无法打开文件: {input_file}")

        # 获取文件信息
        geo_transform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        width = ds.RasterXSize
        height = ds.RasterYSize
        num_bands = ds.RasterCount

        # 读取QC波段（假设是最后一个波段）
        qc_band = ds.GetRasterBand(num_bands).ReadAsArray()
        
        # 创建掩膜
        mask = create_mask(qc_band)

        # 创建输出文件名（去掉'masked_'前缀，因为我们已经过滤了数据）
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, base_name)

        # 创建输出文件（不包含QC波段）
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            output_file,
            width,
            height,
            num_bands - 1,  # 不包含QC波段
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
        )

        if dst_ds is None:
            raise Exception(f"无法创建输出文件: {output_file}")

        # 设置地理信息
        dst_ds.SetGeoTransform(geo_transform)
        dst_ds.SetProjection(projection)

        # 处理每个波段（除了QC波段）
        for i in range(num_bands - 1):
            band_data = ds.GetRasterBand(i + 1).ReadAsArray()
            masked_data = apply_mask_to_band(band_data, mask)
            
            # 写入掩膜后的数据
            dst_ds.GetRasterBand(i + 1).WriteArray(masked_data)
            
            # 复制波段描述
            band_desc = ds.GetRasterBand(i + 1).GetDescription()
            dst_ds.GetRasterBand(i + 1).SetDescription(band_desc)

        # 清理
        dst_ds = None
        ds = None

        logger.info(f"成功处理文件: {output_file}")
        return True

    except Exception as e:
        logger.error(f"处理文件 {input_file} 时出错: {str(e)}")
        return False

def process_files_wrapper(args):
    """
    包装函数，用于多进程处理
    """
    input_file, output_dir = args
    return process_file(input_file, output_dir)

def main():
    # 设置输入输出目录
    input_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/MODIS_LST_29_31_32_QC/MOD21A1N_multiband'
    output_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/MODIS_LST_29_31_32_QC/MOD21A1N_filtered'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有输入文件
    input_files = glob.glob(os.path.join(input_dir, '*.tif'))
    
    if not input_files:
        logger.warning(f"在目录 {input_dir} 中没有找到GeoTIFF文件")
        return

    # 准备参数
    args = [(f, output_dir) for f in input_files]

    # 设置进程数为CPU内核数的45%
    num_processes = max(1, int(multiprocessing.cpu_count() * 0.45))
    logger.info(f"使用 {num_processes} 个进程进行并行处理")

    # 使用进程池处理文件
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度
        list(tqdm(
            pool.imap(process_files_wrapper, args),
            total=len(args),
            desc="处理文件"
        ))

    logger.info("所有文件处理完成")

if __name__ == '__main__':
    main() 