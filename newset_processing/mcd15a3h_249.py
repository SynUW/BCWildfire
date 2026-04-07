#!/usr/bin/env python3
"""
MCD15A3H 数据处理脚本
功能：
1. 将像元值 249 改为 0
2. 将像元值 0 改为 -32768
3. 设置 NoData = -32768
4. 数据类型转换为 Int16（支持负值）
5. 输出到新文件夹
"""

import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from osgeo import gdal

# 禁用 GDAL 警告信息
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.SetConfigOption('CPL_LOG', '/dev/null')

gdal.UseExceptions()

def _fmt_time(s: float) -> str:
    """格式化时间显示"""
    s = int(s)
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"

def process_single_file(args):
    """处理单个TIFF文件"""
    input_path, output_path, overwrite = args
    
    try:
        # 检查输出文件是否已存在
        if output_path.exists() and not overwrite:
            return ('skipped', f"已存在: {output_path.name}")
        
        # 打开源文件
        src_ds = gdal.Open(str(input_path), gdal.GA_ReadOnly)
        if src_ds is None:
            return ('error', f"无法打开: {input_path.name}")
        
        # 获取元数据
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        nbands = src_ds.RasterCount
        geotransform = src_ds.GetGeoTransform()
        projection = src_ds.GetProjection()
        
        # 创建输出文件（使用 Int16 以支持负值）
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            str(output_path),
            width, height, nbands,
            gdal.GDT_Int16,  # 使用 Int16 支持 -32768
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
        )
        
        if dst_ds is None:
            src_ds = None
            return ('error', f"无法创建输出: {output_path.name}")
        
        # 设置地理信息
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        
        # 处理每个波段
        for band_idx in range(1, nbands + 1):
            src_band = src_ds.GetRasterBand(band_idx)
            dst_band = dst_ds.GetRasterBand(band_idx)
            
            # 设置 NoData 为 -32768
            dst_band.SetNoDataValue(-32768)
            
            # 读取数据
            data = src_band.ReadAsArray()
            if data is None:
                continue
            
            # 转换为 Int16 类型以支持负值
            data = data.astype(np.int16)
            
            # 处理像元值：
            # 1. 先将 0 改为 -32768（NoData）
            # 2. 再将 249 改为 0
            mask_zero = (data == 0)
            mask_249 = (data == 249)
            
            data[mask_zero] = -32768
            data[mask_249] = 0
            
            # 写入数据
            dst_band.WriteArray(data)
            dst_band.FlushCache()
            
            # 复制波段描述
            desc = src_band.GetDescription()
            if desc:
                dst_band.SetDescription(desc)
        
        # 关闭文件
        dst_ds.FlushCache()
        src_ds = None
        dst_ds = None
        
        return ('success', output_path.name)
        
    except Exception as e:
        return ('error', f"{input_path.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description='MCD15A3H 数据处理：249→0, 0→-32768(NoData)'
    )
    parser.add_argument('--input', type=str, default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MCD15A3H_mosaic',
                        help='输入文件夹路径')
    parser.add_argument('--output', type=str, default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MCD15A3H_mosaic_249',
                        help='输出文件夹路径')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='文件匹配模式（默认: *.tif）')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行线程数（默认: 8）')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的文件')
    parser.add_argument('--recursive', action='store_true',
                        help='递归处理子文件夹')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"❌ 输入文件夹不存在: {input_dir}")
        return
    
    # 创建输出文件夹
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有TIFF文件
    print("🔍 扫描TIFF文件...")
    if args.recursive:
        all_tifs = list(input_dir.rglob(args.pattern))
    else:
        all_tifs = list(input_dir.glob(args.pattern))
    
    total = len(all_tifs)
    
    if total == 0:
        print(f"❌ 未找到匹配的文件: {args.pattern}")
        return
    
    print(f"📊 找到 {total} 个文件")
    print(f"📁 输入: {input_dir}")
    print(f"📁 输出: {output_dir}")
    print(f"⚙️  使用 {args.workers} 个线程")
    print("=" * 60)
    
    # 准备任务
    tasks = []
    for tif_path in all_tifs:
        # 计算相对路径
        if args.recursive:
            rel_path = tif_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = output_dir / tif_path.name
        
        tasks.append((tif_path, out_path, args.overwrite))
    
    # 处理文件
    t0 = time.time()
    done = 0
    success = 0
    skipped = 0
    error = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        for future in as_completed(futures):
            status, msg = future.result()
            done += 1
            
            if status == 'success':
                success += 1
            elif status == 'skipped':
                skipped += 1
            elif status == 'error':
                error += 1
                print(f"\n❌ 错误: {msg}")
            
            # 更新进度
            elapsed = time.time() - t0
            eta = elapsed * (total - done) / max(1, done)
            avg_time = elapsed / done
            
            print(f"\r进度 [{done}/{total}] | "
                  f"成功:{success} 跳过:{skipped} 错误:{error} | "
                  f"平均:{avg_time:.3f}秒/文件 | "
                  f"用时:{_fmt_time(elapsed)} "
                  f"预计:{_fmt_time(eta)}", end='', flush=True)
    
    print(f"\n\n{'='*60}")
    print(f"✅ 处理完成！")
    print(f"   ├─ 总文件数: {total}")
    print(f"   ├─ 成功: {success}")
    print(f"   ├─ 跳过: {skipped}")
    print(f"   └─ 错误: {error}")
    print(f"\n⏱️  总用时: {_fmt_time(time.time() - t0)}")
    print(f"⚡ 平均速度: {total / (time.time() - t0):.1f} 文件/秒")
    print("=" * 60)
    
    print(f"\n📝 处理说明：")
    print(f"   1. 像元值 249 → 0")
    print(f"   2. 像元值 0 → -32768")
    print(f"   3. NoData 设置：-32768")
    print(f"   4. 数据类型：Int16")
    print(f"   5. 压缩方式：LZW")

if __name__ == '__main__':
    main()

