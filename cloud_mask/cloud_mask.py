#!/usr/bin/env python3
"""
QA掩膜生成器 - 基于cloud_mask.py的云掩膜逻辑
使用MODIS_Terra\\Aqua_QC_state_state_1km掩膜MYD09GA.061 Aqua Surface Reflectance Daily Global 1km and 500m的云
也可使用MODIS_Terra_band2021_QC_Coarse_Resolution_State掩膜MODIS_Terra_Aqua_B20_21_merged_resampled

该脚本根据MODIS QA数据的位信息生成掩膜：
- Cloud state (Bits 0-1): 云状态
- Cloud shadow (Bit 2): 云阴影
- Cirrus (Bits 8-9): 卷云
- Internal cloud algorithm flag (Bit 10): 内部云算法标志
- Adjacent to cloud (Bit 13): 邻接云
"""

import os
import glob
import numpy as np
from osgeo import gdal
import logging
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import time
import threading
from functools import lru_cache
import signal
import sys
import atexit

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置GDAL配置 - 激进的内存优化
gdal.SetConfigOption('GDAL_CACHEMAX', '256')  # 设置很小的缓存
gdal.SetConfigOption('GDAL_NUM_THREADS', '1')  # 禁用GDAL多线程
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
gdal.SetConfigOption('VSI_CACHE', 'FALSE')  # 禁用VSI缓存
gdal.SetConfigOption('GDAL_MAX_DATASET_POOL_SIZE', '1')  # 限制数据集池大小
gdal.UseExceptions()

# 设置很小的GDAL缓存
gdal.SetCacheMax(256)  # 256MB
logger.info("设置GDAL缓存大小为: 256MB")

# 改进的文件缓存机制
class FileCache:
    def __init__(self, max_size=50, max_memory_gb=10):
        self.cache = {}
        self.max_size = max_size
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.current_memory = 0
        self.lock = threading.Lock()
        self.access_order = []  # LRU顺序
        
        # 注册清理函数
        atexit.register(self._cleanup_on_exit)
    
    def _cleanup_on_exit(self):
        """程序退出时的清理函数"""
        try:
            self.clear()
        except Exception:
            pass
    
    def get_memory_usage(self, data):
        """估算数据内存使用量"""
        if data is None:
            return 0
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, tuple) and len(data) == 2:
            array_data, ds = data
            if isinstance(array_data, np.ndarray):
                return array_data.nbytes
        return 0
    
    def add(self, file_path, data_ds_tuple):
        """添加数据到缓存"""
        with self.lock:
            # 计算内存使用量
            memory_usage = self.get_memory_usage(data_ds_tuple)
            
            # 如果数据太大，直接跳过缓存
            if memory_usage > self.max_memory_bytes // 4:
                logger.debug(f"文件过大，跳过缓存: {file_path} ({memory_usage // (1024*1024)}MB)")
                return
            
            # 清理缓存直到有足够空间
            while (len(self.cache) >= self.max_size or 
                   self.current_memory + memory_usage > self.max_memory_bytes):
                if not self.cache:
                    return  # 缓存为空，无法添加
                
                # 移除最旧的条目
                oldest_key = self.access_order.pop(0)
                if oldest_key in self.cache:
                    old_data = self.cache[oldest_key]
                    old_memory = self.get_memory_usage(old_data)
                    self.current_memory -= old_memory
                    del self.cache[oldest_key]
                    
                    # 确保GDAL数据集被正确释放
                    if isinstance(old_data, tuple) and len(old_data) == 2:
                        _, ds = old_data
                        if ds is not None:
                            ds = None
            
            # 添加新数据
            self.cache[file_path] = data_ds_tuple
            self.current_memory += memory_usage
            self.access_order.append(file_path)
            
            logger.debug(f"缓存添加: {file_path} (内存: {memory_usage // (1024*1024)}MB, 总计: {self.current_memory // (1024*1024)}MB)")
    
    def get(self, file_path):
        """从缓存获取数据"""
        with self.lock:
            if file_path in self.cache:
                # 更新访问顺序
                if file_path in self.access_order:
                    self.access_order.remove(file_path)
                self.access_order.append(file_path)
                return self.cache[file_path]
            return None
    
    def clear(self):
        """清理缓存"""
        with self.lock:
            for data_ds_tuple in self.cache.values():
                # 确保GDAL数据集被正确释放
                if isinstance(data_ds_tuple, tuple) and len(data_ds_tuple) == 2:
                    _, ds = data_ds_tuple
                    if ds is not None:
                        ds = None
            
            self.cache.clear()
            self.access_order.clear()
            self.current_memory = 0
            logger.info("文件缓存已清理")
    
    def get_stats(self):
        """获取缓存统计信息"""
        with self.lock:
            return {
                'size': len(self.cache),
                'memory_mb': self.current_memory // (1024 * 1024),
                'max_size': self.max_size,
                'max_memory_gb': self.max_memory_bytes // (1024 * 1024 * 1024)
            }

# 禁用文件缓存，避免内存泄漏
file_cache = None  # 完全禁用缓存

# 全局变量用于跟踪程序状态
processing_active = True
executor = None

def cleanup_resources():
    """清理所有资源"""
    global processing_active, executor
    
    logger.info("开始清理资源...")
    
    # 停止处理
    processing_active = False
    
    # 关闭执行器
    if executor:
        try:
            executor.shutdown(wait=False)
            executor = None
        except Exception as e:
            logger.warning(f"关闭执行器时出错: {e}")
    
    # 强制垃圾回收
    try:
        import gc
        gc.collect()
    except Exception as e:
        logger.warning(f"垃圾回收时出错: {e}")
    
    # 清理GDAL缓存
    try:
        gdal.SetCacheMax(0)
        gdal.SetCacheMax(256)
    except Exception as e:
        logger.warning(f"清理GDAL缓存时出错: {e}")
    
    # 清理所有可能的文件句柄
    try:
        import resource
        # 获取当前进程的文件描述符限制
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info(f"文件描述符限制: {soft}/{hard}")
    except Exception:
        pass
    
    logger.info("资源清理完成")

def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"收到信号 {signum}，开始优雅退出...")
    global processing_active
    processing_active = False
    cleanup_resources()
    logger.info("资源清理完成，程序退出")
    sys.exit(0)

def clear_file_cache():
    """清理文件缓存"""
    # 缓存已禁用，无需清理
    pass

def add_to_cache(file_path, data_ds_tuple):
    """添加数据到缓存"""
    # 缓存已禁用，直接跳过
    pass

def get_cloud_mask(qa):
    """
    基于 state_1km 的中等标准云掩膜生成函数
    （保留火像元和雪像元，只去掉云相关像元）
    
    参数:
        qa : np.ndarray
            QA 数据数组 (uint32)
        allow_mixed : bool
            是否允许 cloud_state=2 (Mixed)
    
    返回:
        np.ndarray
            云相关像元的掩膜 (True = 应该被掩膜)
    """
    
    allow_mixed=True
    qa_uint32 = qa.astype(np.uint32)

    # 云状态 Bits 0-1
    cloud_state_val = qa_uint32 & 0b11  # 0: clear, 1: cloudy, 2: mixed, 3: assumed clear
    if allow_mixed:
        cloud_state_bad = (cloud_state_val == 1)  # 仅去掉Cloudy
    else:
        cloud_state_bad = ~((cloud_state_val == 0) | (cloud_state_val == 3))  # 只保留Clear/Not set

    # 云影 Bit 2
    # cloud_shadow_bad = ((qa_uint32 >> 2) & 0b1) == 1

    # # 内部云算法 Bit 10
    # internal_cloud_bad = ((qa_uint32 >> 10) & 0b1) == 1

    # 邻云 Bit 13
    # adjacent_cloud_bad = ((qa_uint32 >> 13) & 0b1) == 1

    # # 气溶膠 Bits 6-7（排除 High）
    # aerosol_val = (qa_uint32 >> 6) & 0b11
    # aerosol_bad = aerosol_val == 3

    # 卷云 Bits 8-9（排除 High）
    # cirrus_val = (qa_uint32 >> 9) & 0b11
    # cirrus_bad = cirrus_val >= 2

    # 最终云掩膜
    cloud_mask = (
        cloud_state_bad # |
        # cloud_shadow_bad |
        # internal_cloud_bad
        # adjacent_cloud_bad |
        # aerosol_bad |
        # cirrus_bad
    )

    return cloud_mask  # True = should be masked


def get_quality_mask(qa):
    """
    生成质量掩膜 - 基于MODLAND QA (Bits 0-1)
    
    参数:
        qa: QA数据数组 (uint32)
        
    返回:
        numpy.ndarray: 质量掩膜 (True = 应该被掩膜)
    """
    qa_uint32 = qa.astype(np.uint32)
    
    # MODLAND QA: Bits 0-1
    modland_qa = qa_uint32 & 0b11
    
    # 质量掩膜：只保留理想质量 (值为0)
    # 0 = 理想质量
    # 1 = 次优质量  
    # 2 = 产品未生成（云）
    # 3 = 产品未生成（其他原因）
    quality_mask = (modland_qa > 0)
    
    return quality_mask

def get_combined_mask(terra_qa, aqua_qa, mask_type='cloud'):
    """
    生成组合掩膜 - Terra和Aqua的联合掩膜
    
    参数:
        terra_qa: Terra QA数据 (可以是None)
        aqua_qa: Aqua QA数据 (可以是None)
        mask_type: 掩膜类型 ('cloud', 'quality', 'both')
        
    返回:
        numpy.ndarray: 组合掩膜 (True = 应该被掩膜)
    """
    # 检查QA数据是否存在
    terra_exists = terra_qa is not None
    aqua_exists = aqua_qa is not None
    
    if not terra_exists and not aqua_exists:
        # 两个QA文件都不存在，返回全False掩膜（不掩膜任何像素）
        logger.warning("两个QA文件都不存在，不进行掩膜处理")
        return np.zeros((1, 1), dtype=bool)  # 返回空掩膜，后续会被替换为正确尺寸
    
    if mask_type == 'cloud':
        if terra_exists and aqua_exists:
            # 两个QA文件都存在
            terra_mask = get_cloud_mask(terra_qa)
            aqua_mask = get_cloud_mask(aqua_qa)
            # Terra或Aqua任一有云就掩膜
            combined_mask = terra_mask | aqua_mask
        elif terra_exists:
            # 只有Terra QA文件存在
            logger.info("只使用Terra QA文件生成云掩膜")
            combined_mask = get_cloud_mask(terra_qa)
        else:
            # 只有Aqua QA文件存在
            logger.info("只使用Aqua QA文件生成云掩膜")
            combined_mask = get_cloud_mask(aqua_qa)
        
    elif mask_type == 'quality':
        if terra_exists and aqua_exists:
            # 两个QA文件都存在
            terra_mask = get_quality_mask(terra_qa)
            aqua_mask = get_quality_mask(aqua_qa)
            # Terra和Aqua都质量差才掩膜
            combined_mask = terra_mask & aqua_mask
        elif terra_exists:
            # 只有Terra QA文件存在
            logger.info("只使用Terra QA文件生成质量掩膜")
            combined_mask = get_quality_mask(terra_qa)
        else:
            # 只有Aqua QA文件存在
            logger.info("只使用Aqua QA文件生成质量掩膜")
            combined_mask = get_quality_mask(aqua_qa)
        
    elif mask_type == 'both':
        if terra_exists and aqua_exists:
            # 两个QA文件都存在
            terra_cloud_mask = get_cloud_mask(terra_qa)
            aqua_cloud_mask = get_cloud_mask(aqua_qa)
            terra_quality_mask = get_quality_mask(terra_qa)
            aqua_quality_mask = get_quality_mask(aqua_qa)
            
            # 云掩膜：任一有云就掩膜
            cloud_mask = terra_cloud_mask | aqua_cloud_mask
            # 质量掩膜：都质量差才掩膜
            quality_mask = terra_quality_mask & aqua_quality_mask
            
            # 组合掩膜：有云或质量差都掩膜
            combined_mask = cloud_mask | quality_mask
        elif terra_exists:
            # 只有Terra QA文件存在
            logger.info("只使用Terra QA文件生成组合掩膜")
            terra_cloud_mask = get_cloud_mask(terra_qa)
            terra_quality_mask = get_quality_mask(terra_qa)
            combined_mask = terra_cloud_mask | terra_quality_mask
        else:
            # 只有Aqua QA文件存在
            logger.info("只使用Aqua QA文件生成组合掩膜")
            aqua_cloud_mask = get_cloud_mask(aqua_qa)
            aqua_quality_mask = get_quality_mask(aqua_qa)
            combined_mask = aqua_cloud_mask | aqua_quality_mask
        
    else:
        raise ValueError(f"不支持的掩膜类型: {mask_type}")
    
    return combined_mask

@lru_cache(maxsize=100)  # 减少缓存大小
def get_file_info(file_path):
    """缓存文件信息"""
    ds = gdal.Open(file_path, gdal.GA_ReadOnly)
    if ds is None:
        return None
    
    info = {
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'bands': ds.RasterCount,
        'geotransform': ds.GetGeoTransform(),
        'projection': ds.GetProjection(),
        'datatype': ds.GetRasterBand(1).DataType
    }
    ds = None  # 确保数据集被释放
    return info

def read_gdal_data_fast(file_path):
    """快速读取GDAL数据，无缓存版本"""
    ds = gdal.Open(file_path, gdal.GA_ReadOnly)
    if ds is None:
        return None, None
    
    try:
        data = ds.ReadAsArray()
        # 立即释放数据集，只返回数据
        ds = None
        return data, None
    except Exception as e:
        logger.error(f"读取文件失败: {file_path}, 错误: {e}")
        if ds:
            ds = None  # 确保数据集被释放
        return None, None

def create_output_file(data, reference_ds, output_file, nodata_value):
    """创建输出文件"""
    if reference_ds is None:
        return
    
    geotransform = reference_ds.GetGeoTransform()
    projection = reference_ds.GetProjection()
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_file, data.shape[2], data.shape[1], data.shape[0], 
                          gdal.GDT_Float32, options=['COMPRESS=LZW', 'TILED=YES'])
    
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    for band in range(data.shape[0]):
        out_band = out_ds.GetRasterBand(band + 1)
        out_band.WriteArray(data[band])
        out_band.SetNoDataValue(nodata_value)
        out_band.FlushCache()
    
    out_ds = None  # 确保输出数据集被释放

def create_output_file_direct(data, geotransform, projection, output_file, nodata_value):
    """直接创建输出文件，不依赖reference_ds"""
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_file, data.shape[2], data.shape[1], data.shape[0], 
                          gdal.GDT_Float32, options=['COMPRESS=LZW', 'TILED=YES'])
    
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    for band in range(data.shape[0]):
        out_band = out_ds.GetRasterBand(band + 1)
        out_band.WriteArray(data[band])
        out_band.SetNoDataValue(nodata_value)
        out_band.FlushCache()
        out_band = None  # 释放波段引用
    
    out_ds = None  # 确保输出数据集被释放

def process_multispectral_file(multispectral_file, terra_qa_dir, aqua_qa_dir, output_dir, mask_type='cloud', strict_qa=False):
    """处理单个多光谱文件，激进内存管理版本"""
    global processing_active
    
    # 检查是否应该停止处理
    if not processing_active:
        logger.info("处理已停止")
        return False
    
    # 在每个文件处理前强制清理GDAL缓存
    gdal.SetCacheMax(0)
    gdal.SetCacheMax(256)
    
    try:
        filename = os.path.basename(multispectral_file)
        logger.info(f"处理文件: {filename}")
        
        # 提取日期
        date_str = extract_date_from_filename(filename)
        
        # 查找对应的QA文件
        terra_qa_file = os.path.join(terra_qa_dir, f"MODIS_Terra_band2021_QC_Coarse_Resolution_State_QA_{date_str}.tif")
        aqua_qa_file = os.path.join(aqua_qa_dir, f"MODIS_Aqua_band2021_QC_Coarse_Resolution_State_QA_{date_str}.tif")
        
        # terra_qa_file = os.path.join(terra_qa_dir, f"MODIS_Terra_QC_state_state_1km_{date_str}.tif")
        # aqua_qa_file = os.path.join(aqua_qa_dir, f"MODIS_Aqua_QC_state_state_1km_{date_str}.tif")
        
        # 检查QA文件是否存在
        terra_qa_exists = os.path.exists(terra_qa_file)
        aqua_qa_exists = os.path.exists(aqua_qa_file)
        
        # 严格模式：如果任一QA文件不存在就失败
        if strict_qa and (not terra_qa_exists or not aqua_qa_exists):
            missing_files = []
            if not terra_qa_exists:
                missing_files.append("Terra QA")
            if not aqua_qa_exists:
                missing_files.append("Aqua QA")
            logger.error(f"严格模式：QA文件缺失 {', '.join(missing_files)}: {date_str}")
            return False
        
        if not terra_qa_exists and not aqua_qa_exists:
            logger.warning(f"两个QA文件都不存在，直接复制原文件: {date_str}")
            # 直接复制原文件
            import shutil
            output_file = os.path.join(output_dir, f"{filename}")
            shutil.copy2(multispectral_file, output_file)
            logger.info(f"文件已复制: {output_file}")
            return True
        
        # 获取文件信息用于创建输出文件
        reference_ds = gdal.Open(multispectral_file, gdal.GA_ReadOnly)
        if reference_ds is None:
            logger.error(f"无法打开参考文件: {filename}")
            return False
        
        geotransform = reference_ds.GetGeoTransform()
        projection = reference_ds.GetProjection()
        nodata_band = reference_ds.GetRasterBand(1)
        nodata_value = nodata_band.GetNoDataValue()
        if nodata_value is None:
            nodata_value = -9999.0
        
        # 读取多光谱数据
        multispectral_data, _ = read_gdal_data_fast(multispectral_file)
        if multispectral_data is None:
            reference_ds = None
            logger.error(f"多光谱数据读取失败: {filename}")
            return False
        
        # 读取QA数据（如果存在）
        terra_qa_data = None
        aqua_qa_data = None
        
        if terra_qa_exists:
            terra_qa_data, _ = read_gdal_data_fast(terra_qa_file)
            if terra_qa_data is None:
                logger.warning(f"Terra QA数据读取失败: {terra_qa_file}")
                terra_qa_exists = False
        
        if aqua_qa_exists:
            aqua_qa_data, _ = read_gdal_data_fast(aqua_qa_file)
            if aqua_qa_data is None:
                logger.warning(f"Aqua QA数据读取失败: {aqua_qa_file}")
                aqua_qa_exists = False
        
        # 如果两个QA文件都读取失败，直接复制原文件
        if not terra_qa_exists and not aqua_qa_exists:
            reference_ds = None
            del multispectral_data
            logger.warning(f"所有QA文件读取失败，直接复制原文件: {date_str}")
            import shutil
            output_file = os.path.join(output_dir, f"{filename}")
            shutil.copy2(multispectral_file, output_file)
            logger.info(f"文件已复制: {output_file}")
            return True
        
        logger.info(f"数据形状: {multispectral_data.shape}, NoData: {nodata_value}")
        
        # 生成掩膜
        mask = get_combined_mask(terra_qa_data, aqua_qa_data, mask_type)
        
        # 立即删除QA数据释放内存
        if terra_qa_data is not None:
            del terra_qa_data
        if aqua_qa_data is not None:
            del aqua_qa_data
        
        # 如果掩膜尺寸不匹配，创建正确尺寸的空掩膜（不掩膜任何像素）
        if mask.shape != multispectral_data.shape[1:]:
            logger.warning(f"掩膜尺寸不匹配，创建空掩膜: 期望{mask.shape}, 实际{multispectral_data.shape[1:]}")
            mask = np.zeros(multispectral_data.shape[1:], dtype=bool)
        
        # 统计掩膜信息
        total_pixels = mask.size
        masked_pixels = np.sum(mask)
        masked_ratio = masked_pixels / total_pixels * 100
        
        logger.info(f"掩膜统计: {masked_pixels:,}/{total_pixels:,} ({masked_ratio:.1f}%)")
        
        # 应用掩膜（就地修改，避免复制）
        for band in range(multispectral_data.shape[0]):
            multispectral_data[band, mask] = nodata_value
        
        # 删除掩膜释放内存
        del mask
        
        # 创建输出文件（使用我们自己的函数，不依赖reference_ds）
        output_file = os.path.join(output_dir, f"{filename}")
        create_output_file_direct(multispectral_data, geotransform, projection, output_file, nodata_value)
        
        # 立即清理内存
        del multispectral_data
        reference_ds = None
        
        # 强制内存清理
        force_memory_cleanup()
        
        logger.info("处理完成")
        return True
        
    except Exception as e:
        logger.error(f"处理文件 {multispectral_file} 时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def process_single_file(args):
    """单个文件处理包装器"""
    if len(args) == 6:
        multispectral_file, terra_qa_dir, aqua_qa_dir, output_dir, mask_type, strict_qa = args
    else:
        multispectral_file, terra_qa_dir, aqua_qa_dir, output_dir, mask_type = args
        strict_qa = False
    
    try:
        return process_multispectral_file(multispectral_file, terra_qa_dir, aqua_qa_dir, output_dir, mask_type, strict_qa)
    except Exception as e:
        logger.error(f"处理文件 {multispectral_file} 时出错: {str(e)}")
        return False

def process_all_files(multispectral_files, terra_qa_dir, aqua_qa_dir, output_dir, mask_type='cloud', max_workers=None, strict_qa=False):
    """并行处理所有文件，激进内存管理版本"""
    global processing_active, executor
    
    # 大幅减少并行数，避免内存过载
    if max_workers is None:
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().available // (1024**3)
        max_workers = min(4, cpu_count // 2, memory_gb // 8)  # 更保守的并行数
        max_workers = max(1, max_workers)  # 至少1个进程
    
    logger.info(f"使用 {max_workers} 个进程进行并行处理")
    
    file_args = []
    for multispectral_file in multispectral_files:
        file_args.append((multispectral_file, terra_qa_dir, aqua_qa_dir, output_dir, mask_type, strict_qa))
    
    total_count = len(file_args)
    logger.info(f"开始处理 {total_count} 个文件")
    
    completed_count = 0
    success_count = 0
    start_time = time.time()
    last_memory_check = time.time()
    
    try:
        executor = ProcessPoolExecutor(max_workers=max_workers)
        
        # 检查是否应该停止处理
        if not processing_active:
            logger.info("处理已停止")
            return 0
            
        future_to_file = {executor.submit(process_single_file, args): args[0] for args in file_args}
        
        for future in as_completed(future_to_file):
            # 检查是否应该停止处理
            if not processing_active:
                logger.info("处理已停止")
                break
                
            completed_count += 1
            filename = os.path.basename(future_to_file[future])
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / completed_count if completed_count > 0 else 0
            
            # 每处理2个文件检查一次内存使用情况
            if completed_count % 2 == 0:
                current_time = time.time()
                if current_time - last_memory_check > 10:  # 缩短检查间隔
                    memory_info = psutil.virtual_memory()
                    process = psutil.Process()
                    process_memory = process.memory_info()
                    
                    logger.info(f"内存使用: {memory_info.percent:.1f}% ({memory_info.available // (1024**3)}GB可用)")
                    logger.info(f"进程内存: {process_memory.rss // (1024**2)}MB")
                    last_memory_check = current_time
                    
                    # 如果内存使用率超过70%，强制内存清理
                    if memory_info.percent > 70:
                        force_memory_cleanup()
                        logger.warning("内存使用率过高，已清理")
            
            print(f"\r处理进度: {completed_count}/{total_count} - {filename} (平均: {avg_time:.3f}s/文件)", end='', flush=True)
            
            try:
                result = future.result()
                if result:
                    success_count += 1
            except Exception as e:
                logger.error(f"处理文件时出错: {e}")
    
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止处理...")
        processing_active = False
        return success_count
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        return success_count
    finally:
        # 确保执行器被正确关闭
        if executor:
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass
        
        # 最终清理
        force_memory_cleanup()
        
        total_time = time.time() - start_time
        avg_time = total_time / total_count if total_count > 0 else 0
        
        print(f"\r处理完成: {success_count}/{total_count} 个文件成功，总时间: {total_time:.1f}s，平均: {avg_time:.3f}s/文件")
        return success_count

def extract_date_from_filename(filename):
    """从文件名提取日期字符串"""
    import re
    date_pattern = r'Shortwave_(\d{4})_(\d{1,2})_(\d{1,2})\.tif'  # Shortwave_ for B20 21
    match = re.search(date_pattern, filename)
    
    if match:
        year, month, day = match.groups()
        return f"{year}_{month.zfill(2)}_{day.zfill(2)}"
    else:
        raise ValueError(f"无法从文件名中提取日期: {filename}")

def monitor_memory_usage():
    """监控内存使用情况"""
    memory_info = psutil.virtual_memory()
    
    # 获取当前进程的内存使用情况
    process = psutil.Process()
    process_memory = process.memory_info()
    
    logger.info(f"内存使用情况: {memory_info.percent:.1f}% 使用中")
    logger.info(f"  总内存: {memory_info.total // (1024**3)}GB")
    logger.info(f"  可用内存: {memory_info.available // (1024**3)}GB")
    logger.info(f"  已使用: {memory_info.used // (1024**3)}GB")
    logger.info(f"  当前进程内存: {process_memory.rss // (1024**2)}MB")
    logger.info("  缓存状态: 已禁用")
    
    # 检查文件描述符使用情况
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info(f"  文件描述符限制: {soft}/{hard}")
    except Exception:
        pass
    
    return memory_info.percent

def force_memory_cleanup():
    """强制内存清理"""
    import gc
    import ctypes
    
    # 强制垃圾回收
    for _ in range(3):
        gc.collect()
    
    # 清理GDAL缓存
    gdal.SetCacheMax(0)
    gdal.SetCacheMax(256)
    
    # 尝试释放libc缓存（仅在Linux上有效）
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
    
    logger.info("强制内存清理完成")

def main():
    global processing_active
    
    # 注册信号处理器和退出清理函数
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_resources)
    
    parser = argparse.ArgumentParser(description='QA掩膜生成器 - 基于cloud_mask.py')
    parser.add_argument('--multispectral-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked/MODIS_Terra_Aqua_B20_21_merged_resampled', help='多光谱数据目录')
    parser.add_argument('--terra-qa-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/zips/Reflection_supplements_band2021_QC_aligned', help='Terra QA数据目录')
    parser.add_argument('--aqua-qa-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/zips/Reflection_supplements_band2021_QC_aligned', help='Aqua QA数据目录')
    parser.add_argument('--mask-type', choices=['cloud', 'quality', 'both'], default='cloud', help='掩膜类型')
    parser.add_argument('--pattern', default='Shortwave_*.tif', help='文件匹配模式')  # Shortwave_*.tif for B20 21
    parser.add_argument('--max-workers', type=int, default=None, help='最大并行进程数')
    parser.add_argument('--sequential', action='store_true', default=False, help='使用顺序处理而不是并行处理（推荐）')
    parser.add_argument('--parallel', action='store_true', help='使用并行处理（可能导致内存泄漏）')
    parser.add_argument('--monitor-memory', action='store_true', help='启用内存监控')
    parser.add_argument('--allow-missing-qa', action='store_true', default=True, help='允许QA文件缺失（默认启用）')
    parser.add_argument('--strict-qa', action='store_true', help='严格模式：要求所有QA文件都存在')
    
    # B20 21
    # parser.add_argument('--multispectral-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked/MODIS_Terra_Aqua_B20_21_merged_resampled', help='多光谱数据目录')
    # parser.add_argument('--terra-qa-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/zips/Reflection_supplements_band2021_QC_aligned', help='Terra QA数据目录')
    # parser.add_argument('--aqua-qa-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/zips/Reflection_supplements_band2021_QC_aligned', help='Aqua QA数据目录')
    
    args = parser.parse_args()
    
    # 显示初始内存状态
    if args.monitor_memory:
        logger.info("=== 初始内存状态 ===")
        monitor_memory_usage()
        logger.info("==================")

    # 检查输入目录
    if not os.path.exists(args.multispectral_dir):
        logger.error(f"多光谱数据目录不存在: {args.multispectral_dir}")
        return
    
    if not os.path.exists(args.terra_qa_dir):
        logger.error(f"Terra QA数据目录不存在: {args.terra_qa_dir}")
        return
    
    if not os.path.exists(args.aqua_qa_dir):
        logger.error(f"Aqua QA数据目录不存在: {args.aqua_qa_dir}")
        return
    
    # 创建输出目录
    output_dir = os.path.join(args.multispectral_dir, f'qa_masked_{args.mask_type}')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 查找多光谱文件
    pattern = os.path.join(args.multispectral_dir, args.pattern)
    multispectral_files = glob.glob(pattern)
    
    if not multispectral_files:
        logger.error(f"在目录 {args.multispectral_dir} 中未找到匹配模式 {args.pattern} 的文件")
        return
    
    logger.info(f"找到 {len(multispectral_files)} 个多光谱文件")
    
    # 处理文件
    try:
        if args.parallel:
            # 并行处理（可能有内存泄漏）
            process_all_files(
                multispectral_files, args.terra_qa_dir, args.aqua_qa_dir, 
                output_dir, args.mask_type, args.max_workers, args.strict_qa
            )
        else:
            # 顺序处理（推荐，内存安全）
            success_count = 0
            for i, multispectral_file in enumerate(multispectral_files):
                # 检查是否应该停止处理
                if not processing_active:
                    logger.info("处理已停止")
                    break
                    
                logger.info(f"处理文件 {i+1}/{len(multispectral_files)}: {os.path.basename(multispectral_file)}")
                try:
                    result = process_multispectral_file(
                        multispectral_file, args.terra_qa_dir, args.aqua_qa_dir, 
                        output_dir, args.mask_type, args.strict_qa
                    )
                    if result:
                        success_count += 1
                    
                    # 每处理10个文件强制清理内存
                    if (i + 1) % 10 == 0:
                        force_memory_cleanup()
                        logger.info(f"已处理 {i+1} 个文件，强制清理内存")
                        
                except Exception as e:
                    logger.error(f"处理文件 {multispectral_file} 时出错: {e}")
            
            logger.info(f"处理完成: {success_count}/{len(multispectral_files)} 个文件成功")
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止处理...")
        processing_active = False
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
    finally:
        # 确保资源被清理
        cleanup_resources()
    
    # 显示最终内存状态
    if args.monitor_memory:
        logger.info("=== 最终内存状态 ===")
        monitor_memory_usage()
        logger.info("==================")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        cleanup_resources()
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        cleanup_resources()
    finally:
        logger.info("程序退出") 