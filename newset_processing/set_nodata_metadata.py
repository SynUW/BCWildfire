#!/usr/bin/env python3
"""
TIFF NoData 删除脚本
功能：
1. 直接删除TIFF中的NoData设定
2. 不改变任何像元值
3. 原位操作，直接修改文件
4. 支持多线程并行处理
5. 支持递归处理子文件夹
"""

import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from osgeo import gdal
import numpy as np

# 禁用 GDAL 警告信息
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.SetConfigOption('CPL_LOG', '/dev/null')

gdal.UseExceptions()

def _fmt_time(s: float) -> str:
    """格式化时间显示"""
    s = int(s)
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"

def process_single_file(args):
    """原位处理单个TIFF文件：删除NoData设定，不改变像元值"""
    input_path, overwrite = args
    
    try:
        # 打开文件进行更新
        src_ds = gdal.Open(str(input_path), gdal.GA_Update)
        if src_ds is None:
            return ('error', f"无法打开: {input_path.name}")
        
        nbands = src_ds.RasterCount
        processed_bands = 0
        removed_nodata_bands = 0
        
        # 处理每个波段
        for band_idx in range(1, nbands + 1):
            band = src_ds.GetRasterBand(band_idx)
            
            # 获取当前NoData值
            current_nodata = band.GetNoDataValue()
            
            if current_nodata is not None:
                # 删除NoData设定
                band.DeleteNoDataValue()
                removed_nodata_bands += 1
            
            processed_bands += 1
            band.FlushCache()
        
        # 关闭文件
        src_ds.FlushCache()
        src_ds = None
        
        return ('success', f"{input_path.name} (总波段:{nbands}, 处理:{processed_bands}, 删除NoData:{removed_nodata_bands})")
        
    except Exception as e:
        return ('error', f"{input_path.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description='原位修改TIFF文件：删除NoData设定，不改变像元值'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='输入文件夹路径')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='文件匹配模式（默认: *.tif）')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行线程数（默认: 8）')
    parser.add_argument('--recursive', action='store_true',
                        help='递归处理子文件夹')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        print(f"❌ 输入文件夹不存在: {input_dir}")
        return
    
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
    print(f"📁 处理目录: {input_dir}")
    print(f"⚙️  使用 {args.workers} 个线程")
    print("=" * 60)
    
    # 准备任务（原位操作，不需要输出路径）
    tasks = []
    for tif_path in all_tifs:
        tasks.append((tif_path, True))  # overwrite参数保留但不使用
    
    # 处理文件
    t0 = time.time()
    done = 0
    success = 0
    error = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        for future in as_completed(futures):
            status, msg = future.result()
            done += 1
            
            if status == 'success':
                success += 1
            elif status == 'error':
                error += 1
                print(f"\n❌ 错误: {msg}")
            
            # 更新进度
            elapsed = time.time() - t0
            eta = elapsed * (total - done) / max(1, done)
            avg_time = elapsed / done
            
            print(f"\r进度 [{done}/{total}] | "
                  f"成功:{success} 错误:{error} | "
                  f"平均:{avg_time:.3f}秒/文件 | "
                  f"用时:{_fmt_time(elapsed)} "
                  f"预计:{_fmt_time(eta)}", end='', flush=True)
    
    print(f"\n\n{'='*60}")
    print(f"✅ 处理完成！")
    print(f"   ├─ 总文件数: {total}")
    print(f"   ├─ 成功: {success}")
    print(f"   └─ 错误: {error}")
    print(f"\n⏱️  总用时: {_fmt_time(time.time() - t0)}")
    print(f"⚡ 平均速度: {total / (time.time() - t0):.1f} 文件/秒")
    print("=" * 60)
    
    print(f"\n📝 处理说明：")
    print(f"   1. 原位操作：直接修改原文件")
    print(f"   2. 删除所有NoData设定")
    print(f"   3. 不改变任何像元值")
    print(f"   4. 保持原数据类型和压缩方式")

if __name__ == '__main__':
    main()
