#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ERA5数据掩膜处理工具
- 随机选择一个TIFF文件的第一波段建立掩膜（非0值区域为有效区域）
- 对所有TIFF文件应用掩膜
- 将掩膜外区域设为-32768（NoData值）
- 如果原始数据中存在-32768值，会先将其替换为-32767，避免与掩膜NoData值混淆
  （掩膜内的原始-32768 -> -32767，掩膜外的区域 -> -32768）
"""

import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置GDAL
gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

class ERA5MaskProcessor:
    def __init__(self, input_path: Path, output_path: Path, workers: int = 8, replace_existing: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.workers = workers
        self.replace_existing = replace_existing
        self.mask = None
        self.mask_info = None
        
    def find_tiff_files(self) -> List[Path]:
        """查找所有TIFF文件"""
        tiff_files = list(self.input_path.glob("*.tif")) + list(self.input_path.glob("*.TIF"))
        return sorted(tiff_files)
    
    def create_mask_from_sample(self, sample_file: Path) -> Tuple[np.ndarray, dict]:
        """从样本文件的第一波段创建掩膜"""
        print(f"🔍 从样本文件创建掩膜: {sample_file.name}")
        
        # 打开样本文件
        ds = gdal.Open(str(sample_file))
        if ds is None:
            raise RuntimeError(f"无法打开样本文件: {sample_file}")
        
        # 获取第一波段
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        
        # 获取地理信息
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        width = ds.RasterXSize
        height = ds.RasterYSize
        
        # 创建掩膜：非0值区域为True
        mask = (data != 0)
        
        # 确保掩膜形状正确 (height, width)
        if mask.shape != (height, width):
            print(f"⚠️  掩膜形状需要调整: {mask.shape} -> ({height}, {width})")
            # 如果形状是 (width, height)，需要转置
            if mask.shape == (width, height):
                mask = mask.T
                print(f"✅ 掩膜已转置: {mask.shape}")
        
        mask_info = {
            'geotransform': geotransform,
            'projection': projection,
            'width': width,
            'height': height
        }
        
        ds = None
        
        # 统计掩膜信息
        total_pixels = mask.size
        valid_pixels = np.sum(mask)
        valid_ratio = valid_pixels / total_pixels * 100
        
        print(f"   掩膜信息:")
        print(f"   尺寸: {width} x {height}")
        print(f"   总像素: {total_pixels:,}")
        print(f"   有效像素: {valid_pixels:,}")
        print(f"   有效比例: {valid_ratio:.2f}%")
        
        return mask, mask_info
    
    def process_single_file(self, input_file: Path) -> bool:
        """处理单个TIFF文件"""
        try:
            # 构建输出文件路径
            output_file = self.output_path / input_file.name
            
            # 如果输出文件已存在且不需要替换，跳过
            if output_file.exists() and not self.replace_existing:
                return True
            
            # 打开输入文件
            src_ds = gdal.Open(str(input_file))
            if src_ds is None:
                print(f"\n❌ 无法打开文件: {input_file.name}")
                return False
            
            # 获取文件信息
            width = src_ds.RasterXSize
            height = src_ds.RasterYSize
            bands = src_ds.RasterCount
            geotransform = src_ds.GetGeoTransform()
            projection = src_ds.GetProjection()
            dtype = src_ds.GetRasterBand(1).DataType
            
            # 检查掩膜尺寸是否匹配
            if self.mask is not None:
                expected_mask_shape = (height, width)
                if self.mask.shape != expected_mask_shape:
                    print(f"\n⚠️  掩膜尺寸不匹配: {input_file.name}")
                    print(f"   文件尺寸: {width}x{height}")
                    print(f"   期望掩膜尺寸: {expected_mask_shape}")
                    print(f"   实际掩膜尺寸: {self.mask.shape}")
                    
                    # 尝试转置掩膜（创建副本，不修改原始掩膜）
                    if self.mask.shape == (width, height):
                        print(f"   尝试转置掩膜...")
                        mask_to_use = self.mask.T
                        print(f"   转置后掩膜尺寸: {mask_to_use.shape}")
                    else:
                        print(f"   无法修复尺寸不匹配，跳过文件")
                        src_ds = None
                        return False
                else:
                    mask_to_use = self.mask
            else:
                mask_to_use = None
            
            # 创建输出文件
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                str(output_file),
                width, height, bands,
                dtype,
                options=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=IF_SAFER']
            )
            
            if dst_ds is None:
                print(f"\n❌ 无法创建输出文件: {output_file}")
                src_ds = None
                return False
            
            # 复制地理信息
            dst_ds.SetGeoTransform(geotransform)
            dst_ds.SetProjection(projection)
            
            # 处理每个波段
            for b in range(bands):
                try:
                    src_band = src_ds.GetRasterBand(b + 1)
                    dst_band = dst_ds.GetRasterBand(b + 1)
                    
                    # 读取数据
                    data = src_band.ReadAsArray()
                    
                    # 应用掩膜：掩膜外区域设为-32768
                    if mask_to_use is not None:
                        # 检查原始数据中是否存在-32768值
                        has_32768 = np.any(data == -32768)
                        
                        # 如果原始数据中存在-32768，需要先替换为其他值，避免与掩膜NoData值混淆
                        if has_32768:
                            # 在掩膜内且值为-32768的像素，替换为-32767
                            # 这样掩膜外的-32768和掩膜内的原始-32768就能区分开
                            data = np.where((mask_to_use) & (data == -32768), -32767, data)
                        
                        # 应用掩膜：掩膜外区域设为-32768
                        data = np.where(mask_to_use, data, -32768)
                    
                    # 写入数据
                    dst_band.WriteArray(data)
                    dst_band.SetNoDataValue(-32768)
                    
                    # 复制波段描述
                    description = src_band.GetDescription()
                    if description:
                        dst_band.SetDescription(description)
                        
                except Exception as band_error:
                    print(f"\n❌ 处理波段{b+1}失败 {input_file.name}: {band_error}")
                    dst_ds = None
                    src_ds = None
                    return False
            
            # 关闭文件
            dst_ds.FlushCache()
            dst_ds = None
            src_ds = None
            
            return True
            
        except Exception as e:
            print(f"\n❌ 处理文件失败 {input_file.name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_all_files(self, tiff_files: List[Path]):
        """处理所有TIFF文件"""
        print(f"\n🚀 开始处理 {len(tiff_files)} 个TIFF文件...")
        
        # 准备输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 记录开始时间
        start_time = time.time()
        success_count = 0
        
        # 使用线程池处理文件
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path 
                for file_path in tiff_files
            }
            
            # 处理完成的任务
            processed_count = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                processed_count += 1
                
                try:
                    success = future.result(timeout=300)  # 5分钟超时
                    if success:
                        success_count += 1
                    
                    # 更新进度 - 修复计算逻辑
                    elapsed_time = time.time() - start_time
                    avg_time = elapsed_time / processed_count if processed_count > 0 else 0
                    remaining = (len(tiff_files) - processed_count) * avg_time
                    
                    eta_str = f"{remaining/60:.0f}:{remaining%60:02.0f}" if remaining > 0 else "0:00"
                    elapsed_str = f"{elapsed_time/60:.0f}:{elapsed_time%60:02.0f}"
                    
                    print(f"\r[{processed_count}/{len(tiff_files)}] | 成功:{success_count} 失败:{processed_count-success_count} | 平均:{avg_time:.1f}s/文件 | 已用:{elapsed_str} 预计:{eta_str}", 
                          end='', flush=True)
                    
                    # 每100个文件显示一次详细状态
                    # if processed_count % 100 == 0:
                    #     print(f"\n📊 状态更新: 已处理{processed_count}, 成功{success_count}, 失败{processed_count-success_count}")
                    
                except Exception as e:
                    print(f"\n❌ 处理文件异常 {file_path.name}: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print()  # 换行
        
        # 显示最终统计
        total_time = time.time() - start_time
        print(f"\n✅ 处理完成!")
        print(f"   成功: {success_count}/{len(tiff_files)}")
        print(f"   失败: {len(tiff_files) - success_count}")
        print(f"   总耗时: {total_time/60:.1f} 分钟")
        print(f"   平均速度: {total_time/len(tiff_files):.1f} 秒/文件")
    
    def run(self):
        """运行主处理流程"""
        print("="*70)
        print("🔧 ERA5数据掩膜处理工具")
        print("="*70)
        
        # 查找TIFF文件
        tiff_files = self.find_tiff_files()
        if not tiff_files:
            print(f"❌ 未找到TIFF文件: {self.input_path}")
            return
        
        print(f"📁 输入目录: {self.input_path}")
        print(f"📁 输出目录: {self.output_path}")
        print(f"📦 找到 {len(tiff_files)} 个TIFF文件")
        print(f"🧵 工作线程: {self.workers}")
        print(f"🔄 替换模式: {'启用' if self.replace_existing else '禁用（跳过已存在文件）'}")
        
        # 随机选择样本文件
        sample_file = random.choice(tiff_files)
        print(f"🎲 随机选择样本文件: {sample_file.name}")
        
        # 创建掩膜
        self.mask, self.mask_info = self.create_mask_from_sample(sample_file)
        
        # 处理所有文件
        self.process_all_files(tiff_files)


def main():
    parser = argparse.ArgumentParser(description="ERA5数据掩膜处理工具")
    parser.add_argument("input_path", type=Path, nargs='?',
                        default = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/ERA5_consistent_mosaic',
                        help="输入TIFF文件目录")
    parser.add_argument("output_path", type=Path, nargs='?',
                        default = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/ERA5_consistent_mosaic_withnodata',
                        help="输出目录")
    parser.add_argument("--workers", type=int, default=8, help="工作线程数 (默认8)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (可选)")
    parser.add_argument("--replace", action="store_true", help="替换已存在的输出文件 (默认跳过)")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # 检查输入目录
    if not args.input_path.exists():
        print(f"❌ 输入目录不存在: {args.input_path}")
        return
    
    if not args.input_path.is_dir():
        print(f"❌ 输入路径不是目录: {args.input_path}")
        return
    
    # 创建处理器并运行
    processor = ERA5MaskProcessor(
        input_path=args.input_path,
        output_path=args.output_path,
        workers=args.workers,
        replace_existing=args.replace
    )
    
    processor.run()


if __name__ == "__main__":
    main()
