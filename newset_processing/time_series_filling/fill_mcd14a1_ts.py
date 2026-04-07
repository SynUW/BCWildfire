#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCD14A1时间序列填充工具
填充缺失的时间点，使用前向填充、后向填充或均值填充
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from osgeo import gdal
import time

# 设置GDAL
gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

class MCD14A1TimeSeriesFiller:
    def __init__(self, input_folder: Path):
        self.input_folder = input_folder
        
    def parse_date_from_filename(self, filename: str) -> Optional[datetime]:
        """从文件名解析日期"""
        try:
            # 格式: yyyy_mm_dd.tif
            date_str = filename.replace('.tif', '').replace('.TIF', '')
            parts = date_str.split('_')
            if len(parts) == 3:
                year, month, day = map(int, parts)
                return datetime(year, month, day)
        except:
            pass
        return None
    
    def get_existing_files(self) -> dict:
        """获取现有文件及其日期"""
        existing_files = {}
        
        for file_path in self.input_folder.glob("*.tif"):
            date = self.parse_date_from_filename(file_path.name)
            if date:
                existing_files[date] = file_path
        
        for file_path in self.input_folder.glob("*.TIF"):
            date = self.parse_date_from_filename(file_path.name)
            if date:
                existing_files[date] = file_path
                
        return existing_files
    
    def create_zero_tiff(self, template_file: Path, output_file: Path) -> bool:
        """创建全0像素的TIFF文件"""
        try:
            # 打开模板文件获取结构信息
            src_ds = gdal.Open(str(template_file))
            if src_ds is None:
                return False
            
            # 获取文件信息
            width = src_ds.RasterXSize
            height = src_ds.RasterYSize
            bands = src_ds.RasterCount
            geotransform = src_ds.GetGeoTransform()
            projection = src_ds.GetProjection()
            dtype = src_ds.GetRasterBand(1).DataType
            
            # 创建输出文件
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                str(output_file),
                width, height, bands,
                dtype,
                options=['TILED=YES', 'COMPRESS=LZW']
            )
            
            if dst_ds is None:
                src_ds = None
                return False
            
            # 复制地理信息
            dst_ds.SetGeoTransform(geotransform)
            dst_ds.SetProjection(projection)
            
            # 写入所有波段
            for b in range(bands):
                src_band = src_ds.GetRasterBand(b + 1)
                dst_band = dst_ds.GetRasterBand(b + 1)
                
                # 获取原始nodata值
                original_nodata = src_band.GetNoDataValue()
                
                # 创建数据：读取原始数据，然后将非nodata像素设为0
                if original_nodata is not None:
                    # 读取原始数据
                    original_data = src_band.ReadAsArray()
                    # 创建新的数据：保持nodata像素为nodata，其他像素设为0
                    zero_data = np.where(original_data == original_nodata, original_nodata, 0)
                else:
                    # 如果没有nodata值，直接创建全0数组
                    if dtype == gdal.GDT_Byte:
                        zero_data = np.zeros((height, width), dtype=np.uint8)
                    elif dtype == gdal.GDT_Int16:
                        zero_data = np.zeros((height, width), dtype=np.int16)
                    elif dtype == gdal.GDT_UInt16:
                        zero_data = np.zeros((height, width), dtype=np.uint16)
                    elif dtype == gdal.GDT_Int32:
                        zero_data = np.zeros((height, width), dtype=np.int32)
                    elif dtype == gdal.GDT_UInt32:
                        zero_data = np.zeros((height, width), dtype=np.uint32)
                    elif dtype == gdal.GDT_Float32:
                        zero_data = np.zeros((height, width), dtype=np.float32)
                    elif dtype == gdal.GDT_Float64:
                        zero_data = np.zeros((height, width), dtype=np.float64)
                    else:
                        # 默认使用float32
                        zero_data = np.zeros((height, width), dtype=np.float32)
                
                dst_band.WriteArray(zero_data)
                
                # 保留原始的nodata值
                if original_nodata is not None:
                    dst_band.SetNoDataValue(original_nodata)
                
                # 复制波段描述
                description = src_band.GetDescription()
                if description:
                    dst_band.SetDescription(description)
            
            # 关闭文件
            dst_ds.FlushCache()
            dst_ds = None
            src_ds = None
            
            return True
            
        except Exception as e:
            print(f"❌ 创建0值文件失败: {e}")
            return False
    
    def create_mean_tiff(self, file1: Path, file2: Path, output_file: Path) -> bool:
        """创建两个文件的像素级均值TIFF"""
        try:
            # 打开两个输入文件
            ds1 = gdal.Open(str(file1))
            ds2 = gdal.Open(str(file2))
            
            if ds1 is None or ds2 is None:
                return False
            
            # 检查文件尺寸是否一致
            if (ds1.RasterXSize != ds2.RasterXSize or 
                ds1.RasterYSize != ds2.RasterYSize or
                ds1.RasterCount != ds2.RasterCount):
                print(f"⚠️  文件尺寸不匹配: {file1.name} vs {file2.name}")
                ds1 = None
                ds2 = None
                return False
            
            # 获取文件信息
            width = ds1.RasterXSize
            height = ds1.RasterYSize
            bands = ds1.RasterCount
            geotransform = ds1.GetGeoTransform()
            projection = ds1.GetProjection()
            dtype = ds1.GetRasterBand(1).DataType
            
            # 创建输出文件
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                str(output_file),
                width, height, bands,
                dtype,
                options=['TILED=YES', 'COMPRESS=LZW']
            )
            
            if dst_ds is None:
                ds1 = None
                ds2 = None
                return False
            
            # 复制地理信息
            dst_ds.SetGeoTransform(geotransform)
            dst_ds.SetProjection(projection)
            
            # 处理每个波段
            for b in range(bands):
                band1 = ds1.GetRasterBand(b + 1)
                band2 = ds2.GetRasterBand(b + 1)
                
                # 获取原始nodata值
                nodata1 = band1.GetNoDataValue()
                nodata2 = band2.GetNoDataValue()
                
                # 读取数据
                data1 = band1.ReadAsArray().astype(np.float64)
                data2 = band2.ReadAsArray().astype(np.float64)
                
                # 处理nodata值
                if nodata1 is not None:
                    data1 = np.where(data1 == nodata1, np.nan, data1)
                if nodata2 is not None:
                    data2 = np.where(data2 == nodata2, np.nan, data2)
                
                # 计算均值，处理nan值
                with np.errstate(invalid='ignore'):
                    mean_data = np.round((data1 + data2) / 2.0)
                
                # 如果两个像素中有一个是nodata，结果也设为nodata
                if nodata1 is not None and nodata2 is not None:
                    mask = np.isnan(data1) | np.isnan(data2)
                    mean_data = np.where(mask, nodata1, mean_data)
                elif nodata1 is not None:
                    mask = np.isnan(data1)
                    mean_data = np.where(mask, nodata1, mean_data)
                elif nodata2 is not None:
                    mask = np.isnan(data2)
                    mean_data = np.where(mask, nodata2, mean_data)
                
                # 根据原始数据类型进行转换
                if dtype == gdal.GDT_Byte:
                    mean_data = mean_data.astype(np.uint8)
                elif dtype == gdal.GDT_Int16:
                    mean_data = mean_data.astype(np.int16)
                elif dtype == gdal.GDT_UInt16:
                    mean_data = mean_data.astype(np.uint16)
                elif dtype == gdal.GDT_Int32:
                    mean_data = mean_data.astype(np.int32)
                elif dtype == gdal.GDT_UInt32:
                    mean_data = mean_data.astype(np.uint32)
                elif dtype == gdal.GDT_Float32:
                    mean_data = mean_data.astype(np.float32)
                elif dtype == gdal.GDT_Float64:
                    mean_data = mean_data.astype(np.float64)
                else:
                    # 默认使用float32
                    mean_data = mean_data.astype(np.float32)
                
                # 写入数据
                out_band = dst_ds.GetRasterBand(b + 1)
                out_band.WriteArray(mean_data)
                
                # 保留原始的nodata值
                if nodata1 is not None:
                    out_band.SetNoDataValue(nodata1)
                elif nodata2 is not None:
                    out_band.SetNoDataValue(nodata2)
                
                # 复制波段描述
                description = band1.GetDescription()
                if description:
                    out_band.SetDescription(description)
            
            # 关闭文件
            dst_ds.FlushCache()
            dst_ds = None
            ds1 = None
            ds2 = None
            
            return True
            
        except Exception as e:
            print(f"❌ 创建均值文件失败: {e}")
            return False
    
    def fill_time_series(self, start_date: str, end_date: str) -> bool:
        """填充时间序列"""
        try:
            # 解析日期
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            
            print(f"📅 时间范围: {start_date} - {end_date}")
            print(f"📁 操作文件夹: {self.input_folder}")
            
            # 获取现有文件
            existing_files = self.get_existing_files()
            print(f"📦 找到 {len(existing_files)} 个现有文件")
            
            # 生成完整日期序列
            current_date = start_dt
            all_dates = []
            while current_date <= end_dt:
                all_dates.append(current_date)
                current_date += timedelta(days=1)
            
            print(f"📊 需要填充 {len(all_dates)-len(existing_files)} 个日期")
            
            # 找到合适的模板文件（优先使用较新的文件）
            if not existing_files:
                print("❌ 没有找到任何现有文件作为模板")
                return False
            
            # 优先使用较新的文件作为模板（更可能有正确的nodata值）
            sorted_dates = sorted(existing_files.keys(), reverse=True)
            template_file = None
            
            # 尝试找到nodata值不是0的文件作为模板
            for date in sorted_dates[:10]:  # 检查前10个最新文件
                test_file = existing_files[date]
                try:
                    ds = gdal.Open(str(test_file))
                    if ds:
                        band = ds.GetRasterBand(1)
                        nodata = band.GetNoDataValue()
                        ds = None
                        if nodata is not None and nodata != 0:
                            template_file = test_file
                            print(f"📋 使用模板文件: {template_file.name} (nodata={nodata})")
                            break
                except:
                    continue
            
            # 如果没有找到合适的模板，使用最新的文件
            if template_file is None:
                template_file = existing_files[sorted_dates[0]]
                print(f"📋 使用模板文件: {template_file.name} (默认)")
            
            # 开始填充
            start_time = time.time()
            filled_count = 0
            zero_filled_count = 0
            mean_filled_count = 0
            
            for i, target_date in enumerate(all_dates):
                output_filename = target_date.strftime("%Y_%m_%d.tif")
                output_file = self.input_folder / output_filename
                
                # 如果文件已存在，跳过
                if output_file.exists():
                    continue
                
                # 如果该日期已有文件，跳过（不需要复制）
                if target_date in existing_files:
                    continue
                
                # 需要填充的情况
                first_existing_date = min(existing_files.keys())
                if target_date < first_existing_date:
                    # 早期缺失：填充0值
                    if self.create_zero_tiff(template_file, output_file):
                        zero_filled_count += 1
                        filled_count += 1
                else:
                    # 找到前后两个最近的文件
                    prev_file = None
                    next_file = None
                    
                    # 寻找前一个文件
                    for date in sorted(existing_files.keys(), reverse=True):
                        if date < target_date:
                            prev_file = existing_files[date]
                            break
                    
                    # 寻找后一个文件
                    for date in sorted(existing_files.keys()):
                        if date > target_date:
                            next_file = existing_files[date]
                            break
                    
                    if prev_file and next_file:
                        # 使用前后文件的均值
                        if self.create_mean_tiff(prev_file, next_file, output_file):
                            mean_filled_count += 1
                            filled_count += 1
                    elif prev_file:
                        # 只有前一个文件，复制前一个
                        import shutil
                        shutil.copy2(prev_file, output_file)
                        filled_count += 1
                    elif next_file:
                        # 只有后一个文件，复制后一个
                        import shutil
                        shutil.copy2(next_file, output_file)
                        filled_count += 1
                
                # 更新进度
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = (len(all_dates) - i - 1) * avg_time
                    eta_str = f"{remaining/60:.0f}:{remaining%60:02.0f}" if remaining > 0 else "0:00"
                    elapsed_str = f"{elapsed/60:.0f}:{elapsed%60:02.0f}"
                    
                    print(f"\r[{i+1}/{len(all_dates)}] | 已填充:{filled_count} | 0值:{zero_filled_count} 均值:{mean_filled_count} | 已用:{elapsed_str} 预计:{eta_str}", 
                          end='', flush=True)
            
            # 最终统计
            total_time = time.time() - start_time
            print(f"\n✅ 填充完成!")
            print(f"📊 统计信息:")
            print(f"   总处理: {len(all_dates)} 个日期")
            print(f"   新填充: {filled_count} 个文件")
            print(f"   0值填充: {zero_filled_count} 个文件")
            print(f"   均值填充: {mean_filled_count} 个文件")
            print(f"   总耗时: {total_time/60:.1f} 分钟")
            
            return True
            
        except Exception as e:
            print(f"❌ 填充失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description="MCD14A1时间序列填充工具")
    parser.add_argument("--start_date", default='20000101', help="开始日期 (格式: YYYYMMDD)")
    parser.add_argument("--end_date", default='20241231', help="结束日期 (格式: YYYYMMDD)")
    parser.add_argument("--folder", type=Path, default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/MCD14A1_mosaic_binary_downsampled_masked', help="操作文件夹路径（直接在此文件夹内填充）")
    
    args = parser.parse_args()
    
    # 创建填充器
    filler = MCD14A1TimeSeriesFiller(args.folder)
    
    # 开始填充
    success = filler.fill_time_series(args.start_date, args.end_date)
    
    if success:
        print("🎉 时间序列填充完成!")
    else:
        print("❌ 时间序列填充失败!")
        exit(1)

if __name__ == "__main__":
    main()
