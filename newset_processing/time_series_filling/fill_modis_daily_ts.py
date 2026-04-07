#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODIS时间序列前向填充工具
使用前向填充策略填充缺失的时间点：
1. 对于时间序列中的gap，使用最近的之前日期数据进行填充
2. 对于时间序列开始前的缺失日期，使用最近的未来日期数据进行填充

只要是daily数据都使用这个填充方式，包括MCD09GA， NDVI-EVI，MCD11A1，MCD09CMG
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from osgeo import gdal
import time
import shutil

# 设置GDAL
gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

class MODISTimeSeriesFiller:
    def __init__(self, input_folder: Path):
        self.input_folder = input_folder
        
    def parse_date_from_filename(self, filename: str) -> Optional[datetime]:
        """从文件名解析日期"""
        try:
            # 格式: yyyy_mm_dd.tif 或 yyyy_mm_dd.TIF
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
    
    def copy_tiff_with_metadata(self, source_file: Path, target_file: Path) -> bool:
        """复制TIFF文件并保留所有元数据"""
        try:
            shutil.copy2(source_file, target_file)
            return True
        except Exception as e:
            print(f"❌ 复制文件失败 {source_file.name} -> {target_file.name}: {e}")
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
            
            # 检查是否有现有文件
            if not existing_files:
                print("❌ 没有找到任何现有文件")
                return False
            
            # 获取现有文件的最早和最晚日期
            existing_dates = sorted(existing_files.keys())
            earliest_date = existing_dates[0]
            latest_date = existing_dates[-1]
            
            print(f"📋 现有数据范围: {earliest_date.strftime('%Y_%m_%d')} - {latest_date.strftime('%Y_%m_%d')}")
            
            # 开始填充
            start_time = time.time()
            filled_count = 0
            forward_filled_count = 0  # 使用未来数据填充早期缺失
            gap_filled_count = 0      # 填充时间序列中的gap
            
            for i, target_date in enumerate(all_dates):
                output_filename = target_date.strftime("%Y_%m_%d.tif")
                output_file = self.input_folder / output_filename
                
                # 如果文件已存在，跳过
                if output_file.exists():
                    continue
                
                # 如果该日期已有文件，跳过
                if target_date in existing_files:
                    continue
                
                # 需要填充的情况
                source_file = None
                
                if target_date < earliest_date:
                    # 早期缺失：使用最早的现有文件（未来数据）
                    source_file = existing_files[earliest_date]
                    forward_filled_count += 1
                else:
                    # 在时间序列范围内：使用最近的之前日期
                    # 找到最近的之前日期
                    for date in sorted(existing_dates, reverse=True):
                        if date < target_date:
                            source_file = existing_files[date]
                            gap_filled_count += 1
                            break
                    
                    # 如果没有找到之前的日期，使用最早的日期
                    if source_file is None:
                        source_file = existing_files[earliest_date]
                        gap_filled_count += 1
                
                # 复制文件
                if source_file and self.copy_tiff_with_metadata(source_file, output_file):
                    filled_count += 1
                
                # 更新进度
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = (len(all_dates) - i - 1) * avg_time
                    eta_str = f"{remaining/60:.0f}:{remaining%60:02.0f}" if remaining > 0 else "0:00"
                    elapsed_str = f"{elapsed/60:.0f}:{elapsed%60:02.0f}"
                    
                    print(f"\r[{i+1}/{len(all_dates)}] | 已填充:{filled_count} | 前向填充:{forward_filled_count} gap填充:{gap_filled_count} | 已用:{elapsed_str} 预计:{eta_str}", 
                          end='', flush=True)
            
            # 最终统计
            total_time = time.time() - start_time
            print(f"\n✅ 填充完成!")
            print(f"📊 统计信息:")
            print(f"   总处理: {len(all_dates)} 个日期")
            print(f"   新填充: {filled_count} 个文件")
            print(f"   前向填充: {forward_filled_count} 个文件 (使用未来数据填充早期缺失)")
            print(f"   Gap填充: {gap_filled_count} 个文件 (使用之前数据填充gap)")
            print(f"   总耗时: {total_time/60:.1f} 分钟")
            
            return True
            
        except Exception as e:
            print(f"❌ 填充失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description="MODIS时间序列前向填充工具")
    parser.add_argument("--start_date", default='20000101', help="开始日期 (格式: YYYYMMDD)")
    parser.add_argument("--end_date", default='20241231', help="结束日期 (格式: YYYYMMDD)")
    parser.add_argument("--folder", type=Path, 
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/MCD09CMG_mosaic_QAapplied_filled_downsampled', 
                       help="操作文件夹路径（直接在此文件夹内填充）")
    
    args = parser.parse_args()
    
    # 创建填充器
    filler = MODISTimeSeriesFiller(args.folder)
    
    # 开始填充
    success = filler.fill_time_series(args.start_date, args.end_date)
    
    if success:
        print("🎉 时间序列填充完成!")
    else:
        print("❌ 时间序列填充失败!")
        exit(1)

if __name__ == "__main__":
    main()
