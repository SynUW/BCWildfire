# -*- coding: utf-8 -*-
"""
使用GDAL将相同日期的多个TIFF tile文件合并为单个文件：
- 输入：指定输入文件夹，包含按日期命名的tile文件（如 ERA5_2013_09_11_tile00.tif）
- 输出：指定输出文件夹，按日期生成合并后的TIFF文件（如 2013_09_11.tif）
- 支持跳过已处理的文件（检查文件是否存在且大小>0）
- 自动按日期分组，每个日期生成一个输出文件

使用方法：
    python merge_patches.py --input-dir /path/to/input --output-dir /path/to/output
    python merge_patches.py --input-dir /path/to/input --output-dir /path/to/output --overwrite
    python merge_patches.py --input-dir /path/to/input --output-dir /path/to/output --output-name single_file.tif

GDAL版本优势：
- 更快的I/O性能
- 更好的内存管理
- 内置的多线程支持
- 更高效的瓦片合并算法
"""

import os
import sys
import argparse
import logging
import time
import re
from pathlib import Path

from osgeo import gdal, gdalconst

# ================== 默认配置 ==================
DEFAULT_MAX_WORKERS = max(1, os.cpu_count() // 2)  # 合并并发度

# GDAL优化配置（针对大内存系统优化）
# 充分利用300GB内存：分配250GB给GDAL缓存，留50GB给系统和Python
GDAL_CACHE_SIZE    = 256000       # GDAL缓存大小 (MB) = 250GB，充分利用大内存
GDAL_NUM_THREADS   = "ALL_CPUS"   # GDAL线程数
COMPRESS           = "LZW"        # 输出压缩（可以设置为None禁用压缩以提升速度）
TILED              = True          # 输出块瓦片标志
BIGTIFF            = "IF_SAFER"   # 自动切换 BigTIFF
BLOCK_SIZE         = 512           # 输出文件块大小（增大以提升性能）
# 大内存优化：使用更大的处理块（充分利用内存）
LARGE_MEMORY_BLOCK_SIZE = 16384   # 大内存系统的处理块大小（像素）= 16Kx16K，充分利用内存
MEMORY_THRESHOLD   = 50 * 1024 * 1024 * 1024  # 50GB：小于此大小的文件直接读取到内存（更激进）

# ============ 日志（简洁） ============
logging.basicConfig(
    level=logging.WARNING,  # 只显示警告和错误
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mosaic_gdal")

# 日期匹配模式 - 匹配各种日期格式
DATE_PATTERNS = [
    re.compile(r'(\d{4})[_-](\d{2})[_-](\d{2})'),  # YYYY_MM_DD 或 YYYY-MM-DD
    re.compile(r'(\d{4})(\d{2})(\d{2})'),          # YYYYMMDD
    re.compile(r'(\d{4})[_-](\d{1,2})[_-](\d{1,2})'),  # YYYY_M_D 或 YYYY-M-D
]

# 初始化GDAL
gdal.UseExceptions()  # 启用GDAL异常处理，避免警告
gdal.SetCacheMax(GDAL_CACHE_SIZE * 1024 * 1024)  # 转换为字节
gdal.SetConfigOption('GDAL_NUM_THREADS', GDAL_NUM_THREADS)
gdal.SetConfigOption('GDAL_CACHEMAX', str(GDAL_CACHE_SIZE))
# 激进的I/O优化
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')  # 禁用目录扫描
gdal.SetConfigOption('CPL_VSIL_CURL_USE_HEAD', 'NO')  # 禁用HTTP头检查
gdal.SetConfigOption('GDAL_MAX_DATASET_POOL_SIZE', '1000')  # 增加数据集池大小
gdal.SetConfigOption('GDAL_SWATH_SIZE', '268435456')  # 256MB的swath大小


def format_time(seconds):
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def print_progress(current, total, start_time, last_print_time=[0]):
    """打印自定义格式的进度条 - 单行动态更新"""
    current_time = time.time()
    
    # 限制打印频率：每秒最多更新一次，或者每处理1%的项目时更新
    progress_percent = current / total * 100
    should_update = (
        current_time - last_print_time[0] >= 2.0 or  # 至少2秒间隔
        progress_percent - int(progress_percent) == 0 or  # 每1%进度
        current == total  # 完成时
    )
    
    if not should_update and current < total:
        return
    
    last_print_time[0] = current_time
    
    elapsed = current_time - start_time
    
    if current > 0:
        # 计算平均处理时间
        avg_time_per_item = elapsed / current
        remaining_items = total - current
        eta = remaining_items * avg_time_per_item
        eta_str = format_time(eta)
    else:
        eta_str = "?:??:??"
    
    elapsed_str = format_time(elapsed)
    total_estimated = elapsed / current * total if current > 0 else 0
    total_str = format_time(total_estimated)
    
    # 使用\r回到行首，在同一行更新进度
    print(f"\r[{current}/{total}], {elapsed_str}/{total_str}, ETA: {eta_str}", end='', flush=True)
    
    if current == total:
        print()  # 完成后换行


def extract_date_from_filename(filename: str):
    """从文件名中提取日期，返回YYYY_MM_DD格式的字符串"""
    for pattern in DATE_PATTERNS:
        match = pattern.search(filename)
        if match:
            year, month, day = match.groups()
            # 确保月份和日期是两位数
            month = month.zfill(2)
            day = day.zfill(2)
            return f"{year}_{month}_{day}"
    return None


def validate_geospatial_info(file_paths):
    """验证输入文件的地理空间信息一致性"""
    if not file_paths:
        return True
    
    # 获取第一个文件作为参考
    ref_ds = gdal.Open(str(file_paths[0]), gdal.GA_ReadOnly)
    if ref_ds is None:
        logger.error(f"无法打开参考文件: {file_paths[0]}")
        return False
    
    ref_gt = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()
    ref_pixel_x = abs(ref_gt[1])
    ref_pixel_y = abs(ref_gt[5])
    
    inconsistent_files = []
    
    for i, file_path in enumerate(file_paths[1:], 1):
        ds = gdal.Open(str(file_path), gdal.GA_ReadOnly)
        if ds is None:
            continue
            
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        pixel_x = abs(gt[1])
        pixel_y = abs(gt[5])
        
        # 检查像素大小是否一致
        if abs(pixel_x - ref_pixel_x) > 1e-8 or abs(pixel_y - ref_pixel_y) > 1e-8:
            inconsistent_files.append((file_path, f"像素大小不一致: {pixel_x:.8f} x {pixel_y:.8f}"))
        
        # 检查投影是否一致
        if proj != ref_proj:
            inconsistent_files.append((file_path, "投影系统不一致"))
        
        ds = None
    
    ref_ds = None
    
    if inconsistent_files:
        logger.error("发现地理空间信息不一致的文件:")
        for file_path, reason in inconsistent_files:
            logger.error(f"  {file_path.name}: {reason}")
        return False
    
    return True


def group_files_by_date(input_dir: str):
    """按日期分组TIFF文件，返回 {date: [file_list]}"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    all_files = []
    
    for pattern in patterns:
        all_files.extend(input_path.glob(pattern))
    
    # 按日期分组
    date_groups = {}
    for file_path in all_files:
        date_str = extract_date_from_filename(file_path.name)
        if date_str:
            if date_str not in date_groups:
                date_groups[date_str] = []
            date_groups[date_str].append(file_path)
        else:
            logger.warning(f"无法从文件名中提取日期: {file_path.name}")
    
    # 对每个日期的文件进行排序
    for date_str in date_groups:
        date_groups[date_str] = sorted(date_groups[date_str])
    
    return date_groups


def get_tiff_files(input_dir: str):
    """获取输入目录中所有TIFF文件（向后兼容）"""
    date_groups = group_files_by_date(input_dir)
    all_files = []
    for files in date_groups.values():
        all_files.extend(files)
    return sorted(all_files)


def get_file_info(file_path: str):
    """获取TIFF文件信息"""
    ds = gdal.Open(str(file_path), gdalconst.GA_ReadOnly)
    if ds is None:
        return None
    
    info = {
        'ds': ds,
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'bands': ds.RasterCount,
        'geotransform': ds.GetGeoTransform(),
        'projection': ds.GetProjection(),
        'nodata': ds.GetRasterBand(1).GetNoDataValue(),
        'datatype': ds.GetRasterBand(1).DataType
    }
    return info


def merge_files_gdal(file_paths, out_path, overwrite=False):
    """
    使用GDAL将多个TIFF文件合并为单个文件
    """
    if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    try:
        # 获取第一个文件的信息作为参考
        ref_info = get_file_info(file_paths[0])
        if ref_info is None:
            logger.error(f"无法打开参考文件: {file_paths[0]}")
            return False
        
        ref_bands = ref_info['bands']
        ref_nodata = ref_info['nodata']
        ref_datatype = ref_info['datatype']
        
        # 计算合并后的边界
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        file_infos = []
        for file_path in file_paths:
            info = get_file_info(file_path)
            if info is None:
                logger.error(f"无法打开文件: {file_path}")
                continue
            
            file_infos.append(info)
            gt = info['geotransform']
            
            # 计算文件的地理边界
            file_min_x = gt[0]
            file_max_x = gt[0] + info['width'] * gt[1]
            file_min_y = gt[3] + info['height'] * gt[5]
            file_max_y = gt[3]
            
            min_x = min(min_x, file_min_x)
            max_x = max(max_x, file_max_x)
            min_y = min(min_y, file_min_y)
            max_y = max(max_y, file_max_y)
        
        if not file_infos:
            logger.error(f"没有有效的文件: {out_path}")
            return False
        
        # 计算输出尺寸
        pixel_size_x = abs(ref_info['geotransform'][1])
        pixel_size_y = abs(ref_info['geotransform'][5])
        
        out_width = int((max_x - min_x) / pixel_size_x)
        out_height = int((max_y - min_y) / pixel_size_y)
        
        # 创建输出数据集
        driver = gdal.GetDriverByName('GTiff')
        out_geotransform = (min_x, pixel_size_x, 0, max_y, 0, -pixel_size_y)
        
        # 创建选项（优化以充分利用内存）
        creation_options = []
        if COMPRESS:
            creation_options.append(f'COMPRESS={COMPRESS}')
        creation_options.extend([
            'TILED=YES' if TILED else 'TILED=NO',
            f'BLOCKXSIZE={BLOCK_SIZE}',
            f'BLOCKYSIZE={BLOCK_SIZE}',
            f'BIGTIFF={BIGTIFF}',
            'NUM_THREADS=ALL_CPUS'
        ])
        
        # 创建输出文件
        out_ds = driver.Create(
            out_path,
            out_width,
            out_height,
            ref_bands,
            ref_datatype,
            options=creation_options
        )
        
        if out_ds is None:
            logger.error(f"无法创建输出文件: {out_path}")
            return False
        
        out_ds.SetGeoTransform(out_geotransform)
        out_ds.SetProjection(ref_info['projection'])
        
        # 设置NoData值
        if ref_nodata is not None:
            for i in range(1, ref_bands + 1):
                out_ds.GetRasterBand(i).SetNoDataValue(ref_nodata)
        
        # 智能选择处理策略：根据文件大小和可用内存
        # 对于小文件，直接读取整个文件到内存（更快）
        # 对于大文件，使用大块处理（充分利用内存）
        
        # 估算单个文件的内存占用（字节）
        # 假设每个像素2字节（Int16），多波段
        estimated_file_size = file_infos[0]['width'] * file_infos[0]['height'] * ref_bands * 2 if file_infos else 0
        
        # 估算总内存需求（所有文件）
        total_estimated_size = estimated_file_size * len(file_infos)
        
        # 更激进的策略：充分利用大内存
        # 如果总内存需求小于阈值，直接读取所有文件到内存
        if total_estimated_size < MEMORY_THRESHOLD:
            # 直接读取整个文件到内存（最快）
            block_size = max(out_width, out_height)  # 使用整个图像大小
        elif estimated_file_size < MEMORY_THRESHOLD // 4:  # 单个文件小于12.5GB
            # 单个文件较小，直接读取整个文件
            block_size = max(out_width, out_height)
        else:
            # 超大文件：使用超大块处理（充分利用内存）
            # 使用LARGE_MEMORY_BLOCK_SIZE的块（16Kx16K），充分利用大内存
            block_size = min(LARGE_MEMORY_BLOCK_SIZE, out_width, out_height)
        
        # 合并每个波段
        # 优化：对于可以直接读取的文件，一次性读取所有文件到内存
        use_direct_read = (block_size >= max([info['width'] for info in file_infos] + [info['height'] for info in file_infos]))
        
        for band_idx in range(1, ref_bands + 1):
            out_band = out_ds.GetRasterBand(band_idx)
            
            # 初始化输出波段为NoData值
            if ref_nodata is not None:
                out_band.Fill(ref_nodata)
            
            # 合并每个文件的该波段
            for file_info in file_infos:
                file_ds = file_info['ds']
                file_band = file_ds.GetRasterBand(band_idx)
                
                # 计算文件在输出中的位置
                file_gt = file_info['geotransform']
                file_x = int((file_gt[0] - min_x) / pixel_size_x)
                file_y = int((max_y - file_gt[3]) / pixel_size_y)
                
                file_width = file_info['width']
                file_height = file_info['height']
                
                # 如果可以直接读取整个文件，使用最快的方法
                if use_direct_read or block_size >= max(file_width, file_height):
                    # 直接读取整个文件（最快，充分利用内存）
                    file_data = file_band.ReadAsArray()
                    out_band.WriteArray(file_data, file_x, file_y)
                else:
                    # 分块读取和写入（使用超大块，充分利用内存）
                    # 使用更大的块以减少I/O次数
                    for y in range(0, file_height, block_size):
                        y_size = min(block_size, file_height - y)
                        for x in range(0, file_width, block_size):
                            x_size = min(block_size, file_width - x)
                            
                            # 读取块数据
                            file_data = file_band.ReadAsArray(x, y, x_size, y_size)
                            
                            # 写入到输出波段（考虑文件在输出中的位置）
                            out_x = file_x + x
                            out_y = file_y + y
                            out_band.WriteArray(file_data, out_x, out_y)
        
        # 统一刷新缓存（而不是每个波段都刷新）
        out_ds.FlushCache()
        
        # 清理资源
        out_ds = None
        for info in file_infos:
            info['ds'] = None
        
        return True

    except Exception as e:
        logger.error(f"GDAL合并失败：{out_path} - {e}")
        return False


def merge_files_gdal_warp(file_paths, out_path, overwrite=False):
    """
    使用GDAL的gdalwarp进行合并，采用最优化策略
    使用VRT作为中间格式以提升性能
    """
    if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    try:
        # 方法1：直接使用VRT + Warp（最快）
        # 创建VRT作为中间格式，可以显著提升性能
        vrt_path = '/vsimem/temp_mosaic.vrt'
        
        # 创建VRT（内存中，非常快）
        vrt_options = gdal.BuildVRTOptions(
            separate=False,
            addAlpha=False,
            resampleAlg=gdal.GRA_NearestNeighbour
        )
        
        vrt_ds = gdal.BuildVRT(vrt_path, [str(p) for p in file_paths], options=vrt_options)
        if vrt_ds is None:
            logger.error("无法创建VRT文件")
            return False
        
        # 优化选项：最小化所有不必要的操作
        creation_opts = []
        if COMPRESS:
            creation_opts.append(f'COMPRESS={COMPRESS}')
        creation_opts.extend([
            'TILED=YES',
            f'BLOCKXSIZE={BLOCK_SIZE}',
            f'BLOCKYSIZE={BLOCK_SIZE}',
            f'BIGTIFF={BIGTIFF}',
        ])
        
        # 最激进的Warp选项
        warp_options = gdal.WarpOptions(
            format='GTiff',
            creationOptions=creation_opts,
            resampleAlg=gdal.GRA_NearestNeighbour,
            multithread=True,
            # 禁用所有不必要的操作
            copyMetadata=False,
            dstNodata=None,
            # 使用最激进的性能选项
            options=[
                'NUM_THREADS=ALL_CPUS',
                'GDAL_CACHEMAX=256000',
                'GDAL_SWATH_SIZE=536870912',  # 512MB swath（更大）
            ],
        )
        
        # 执行合并（从VRT到最终文件）
        out_ds = gdal.Warp(out_path, vrt_ds, options=warp_options)
        
        # 清理VRT
        vrt_ds = None
        try:
            gdal.GetDriverByName('VRT').Delete(vrt_path)
        except Exception:
            pass
        
        if out_ds is None:
            logger.error(f"gdal.Warp合并失败: {out_path}")
            return False
        
        out_ds = None
        return True
        
    except Exception as e:
        logger.error(f"gdal.Warp合并失败：{out_path} - {e}")
        return False


def merge_files_precision(file_paths, out_path, overwrite=False):
    """
    高精度合并方法，确保像素完美对齐
    """
    if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    try:
        # 获取第一个文件作为参考
        ref_ds = gdal.Open(str(file_paths[0]), gdal.GA_ReadOnly)
        if ref_ds is None:
            logger.error(f"无法打开参考文件: {file_paths[0]}")
            return False
        
        ref_gt = ref_ds.GetGeoTransform()
        ref_proj = ref_ds.GetProjection()
        ref_pixel_x = abs(ref_gt[1])
        ref_pixel_y = abs(ref_gt[5])
        
        logger.info("使用高精度对齐方法")
        logger.info(f"参考像素大小: {ref_pixel_x:.10f} x {ref_pixel_y:.10f}")
        
        # 使用VRT进行精确合并
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.vrt', delete=False) as tmp:
            vrt_path = tmp.name
        
        # 创建VRT
        vrt_options = gdal.BuildVRTOptions(
            separate=False,
            addAlpha=False,
            resampleAlg=gdal.GRA_NearestNeighbour
        )
        
        vrt_ds = gdal.BuildVRT(vrt_path, [str(p) for p in file_paths], options=vrt_options)
        if vrt_ds is None:
            logger.error("无法创建VRT文件")
            return False
        
        # 使用精确的Warp参数
        warp_options = gdal.WarpOptions(
            format='GTiff',
            creationOptions=[
                f'COMPRESS={COMPRESS}',
                'TILED=YES' if TILED else 'TILED=NO',
                f'BLOCKXSIZE={BLOCK_SIZE}',
                f'BLOCKYSIZE={BLOCK_SIZE}',
                f'BIGTIFF={BIGTIFF}',
                'NUM_THREADS=ALL_CPUS'
            ],
            multithread=True,
            warpOptions=['NUM_THREADS=ALL_CPUS'],
            # 关键：使用精确的分辨率
            xRes=ref_pixel_x,
            yRes=ref_pixel_y,
            # 使用最邻近重采样
            resampleAlg=gdal.GRA_NearestNeighbour,
            # 确保目标CRS一致
            dstSRS=ref_proj if ref_proj else None,
            # 高精度控制
            targetAlignedPixels=True,
            errorThreshold=0.0,
            # 禁用自动调整
            noData=None,
            initDest=None
        )
        
        # 执行合并
        out_ds = gdal.Warp(out_path, vrt_ds, options=warp_options)
        
        # 清理
        vrt_ds = None
        ref_ds = None
        try:
            os.unlink(vrt_path)
        except Exception:
            pass
        
        if out_ds is None:
            logger.error(f"高精度合并失败: {out_path}")
            return False
        
        # 验证输出
        out_gt = out_ds.GetGeoTransform()
        logger.debug(f"输出地理变换: {out_gt}")
        logger.debug(f"输出像素大小: {abs(out_gt[1]):.10f} x {abs(out_gt[5]):.10f}")
        
        out_ds = None
        return True
        
    except Exception as e:
        logger.error(f"高精度合并失败：{out_path} - {e}")
        return False


def main():
    # 在函数开始就声明global，避免后续使用时的语法错误
    global GDAL_CACHE_SIZE, LARGE_MEMORY_BLOCK_SIZE
    
    parser = argparse.ArgumentParser(description='合并文件夹内的多个TIFF文件')
    
    parser.add_argument('--input-dir',default=r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/MYD09CMG',
                       help='包含TIFF文件的输入目录')
    parser.add_argument('--output-dir', default=r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers_processed/MYD09CMG',
                       help='输出目录')
    parser.add_argument('--output-name', default=None,
                       help='输出文件名 (默认: 从输入文件名自动提取日期)')
    parser.add_argument('--date', default=None,
                       help='手动指定日期 (格式: YYYY_MM_DD，用于文件名中无完整日期的情况)')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的输出文件')
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() // 2),
                       help=f'并行工作线程数 (默认: {max(1, os.cpu_count() // 2)})')
    parser.add_argument('--method', default='warp',
                       choices=['gdal', 'warp', 'precision'],
                       help='合并方法: gdal=手动合并, warp=gdal.Warp, precision=高精度对齐 (默认: warp)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='跳过地理空间信息验证以提升速度（不推荐，但可以显著提速）')
    # 保存默认值用于参数定义
    default_gdal_cache = GDAL_CACHE_SIZE
    default_block_size = LARGE_MEMORY_BLOCK_SIZE
    
    parser.add_argument('--gdal-cache', type=int, default=default_gdal_cache,
                       help=f'GDAL缓存大小 (MB, 默认: {default_gdal_cache}MB = {default_gdal_cache//1024}GB)')
    parser.add_argument('--block-size', type=int, default=default_block_size,
                       help=f'处理块大小 (像素, 默认: {default_block_size}x{default_block_size})')
    parser.add_argument('--no-compress', action='store_true',
                       help='禁用压缩以提升处理速度（文件会更大但处理更快）')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细的地理信息（调试用）')
    
    args = parser.parse_args()
    
    # 应用用户自定义的GDAL缓存大小
    if args.gdal_cache != GDAL_CACHE_SIZE:
        GDAL_CACHE_SIZE = args.gdal_cache
        gdal.SetCacheMax(GDAL_CACHE_SIZE * 1024 * 1024)
        gdal.SetConfigOption('GDAL_CACHEMAX', str(GDAL_CACHE_SIZE))
        logger.info(f"GDAL缓存大小设置为: {GDAL_CACHE_SIZE}MB ({GDAL_CACHE_SIZE//1024}GB)")
    
    # 应用用户自定义的块大小
    if args.block_size != LARGE_MEMORY_BLOCK_SIZE:
        LARGE_MEMORY_BLOCK_SIZE = args.block_size
        logger.info(f"处理块大小设置为: {LARGE_MEMORY_BLOCK_SIZE}x{LARGE_MEMORY_BLOCK_SIZE}")
    
    # 应用压缩设置
    global COMPRESS
    if args.no_compress:
        COMPRESS = None
        logger.info("已禁用压缩以提升处理速度")
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证输入目录
    if not os.path.isdir(args.input_dir):
        logger.error(f"输入目录不存在：{args.input_dir}")
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 按日期分组TIFF文件
    try:
        date_groups = group_files_by_date(args.input_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not date_groups:
        logger.error(f"在目录 {args.input_dir} 中未找到有效的TIFF文件")
        sys.exit(1)

    total_dates = len(date_groups)
    total_files = sum(len(files) for files in date_groups.values())
    print(f"找到 {total_dates} 个日期的数据，共 {total_files} 个文件")
    
    # 选择合并方法
    if args.method == 'gdal':
        merge_func = merge_files_gdal
    elif args.method == 'precision':
        merge_func = merge_files_precision
    else:
        merge_func = merge_files_gdal_warp
    
    # 处理每个日期的文件（串行处理，避免I/O竞争）
    # 注意：并行处理可能导致I/O竞争，反而变慢
    total_success = 0
    total_failed = 0
    start_time = time.time()
    
    for current_idx, (date_str, tiff_files) in enumerate(sorted(date_groups.items()), 1):
        # 打印进度
        print_progress(current_idx, total_dates, start_time)
        
        # 确定输出文件名
        if args.output_name:
            # 如果指定了输出文件名，只处理第一个日期
            if date_str != list(date_groups.keys())[0]:
                continue
            output_filename = args.output_name
            if not output_filename.endswith('.tif'):
                output_filename += '.tif'
        else:
            output_filename = f"{date_str}.tif"
        
        # 确定输出路径
        output_path = output_dir / output_filename
        
        # 检查是否需要跳过
        if (not args.overwrite) and output_path.exists() and output_path.stat().st_size > 0:
            total_success += 1
            continue
        
        # 验证地理空间信息一致性（可选）
        if not args.skip_validation:
            if not validate_geospatial_info(tiff_files):
                logger.error(f"日期 {date_str} 地理空间验证失败")
                total_failed += 1
                continue
        
        # 执行合并
        success = merge_func(tiff_files, str(output_path), args.overwrite)
        
        if success:
            total_success += 1
        else:
            logger.error(f"日期 {date_str} 合并失败!")
            total_failed += 1
    
    elapsed_time = time.time() - start_time
    
    # 总结
    print(f"\n处理完成! 总耗时: {format_time(elapsed_time)}")
    print(f"成功: {total_success}/{total_dates} 个日期")
    if total_failed > 0:
        logger.error(f"失败: {total_failed} 个日期")
        sys.exit(1)


if __name__ == "__main__":
    main()
