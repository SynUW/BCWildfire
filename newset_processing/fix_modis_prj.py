#!/usr/bin/env python3
"""
检查并修复MODIS的投影
SRC 是 Sinusoidal，但被写成了 +datum=WGS84 的“伪 Sinusoidal”，缺少 +R=6371007.181（MODIS 用的球半径）。这会在转到 WGS84 时产生系统性错位。
"""
from osgeo import gdal, osr
from pathlib import Path
import time
import sys
from multiprocessing import Pool, Manager
from functools import partial

# 禁用SWIG内存泄漏警告
if hasattr(sys, 'stderr'):
    _stderr = sys.stderr
    sys.stderr = open('/dev/null', 'w')
    import warnings
    warnings.filterwarnings('ignore')
    sys.stderr = _stderr

ROOT = Path("/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/trash")
FIX_WKT_STR = None

def _fmt_time(s: float) -> str:
    """格式化时间显示"""
    s = int(s)
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"

def need_fix(proj4: str) -> bool:
    if not proj4: return False
    p = proj4.lower()
    return ("+proj=sinu" in p) and (("+datum=wgs84" in p) or ("+r=" not in p))

def process_single_file(tif_path, fix_wkt_str):
    """
    处理单个TIFF文件
    返回: (status, message)
        status: 'fixed', 'normal', 'skipped', 'error'
    """
    try:
        ds = gdal.Open(str(tif_path), gdal.GA_Update)
        if ds is None:
            return ('error', f"无法打开: {tif_path.name}")
        
        s = osr.SpatialReference()
        try:
            s.ImportFromWkt(ds.GetProjection() or "")
            p4 = s.ExportToProj4() or ""
        except Exception as e:
            return ('error', f"读取投影失败: {tif_path.name}")
        finally:
            del s
        
        if need_fix(p4):
            ds.SetProjection(fix_wkt_str)
            ds.FlushCache()
            ds = None
            return ('fixed', tif_path.name)
        else:
            ds = None
            return ('normal', tif_path.name)
            
    except Exception as e:
        return ('error', f"处理异常: {tif_path.name} - {str(e)}")

def main():
    global FIX_WKT_STR
    
    # 准备投影字符串
    FIX_WKT = osr.SpatialReference()
    FIX_WKT.ImportFromProj4('+proj=sinu +R=6371007.181 +lon_0=0 +x_0=0 +y_0=0 +no_defs')
    FIX_WKT_STR = FIX_WKT.ExportToWkt()
    
    # 收集所有TIFF文件
    print("🔍 扫描TIFF文件...")
    print(f"📁 根目录: {ROOT}")
    
    subfolders = [d for d in ROOT.iterdir() if d.is_dir()]
    print(f"📂 找到 {len(subfolders)} 个子文件夹")
    
    all_tifs = list(ROOT.rglob("*.tif"))
    total = len(all_tifs)
    
    if total == 0:
        print("❌ 未找到TIFF文件")
        exit(1)
    
    print(f"📊 总计 {total} 个TIFF文件")
    
    # 确定进程数
    import os
    n_workers = min(os.cpu_count() or 4, 16)  # 最多16个进程
    print(f"⚙️  使用 {n_workers} 个进程并行处理")
    print("=" * 60)
    
    # 处理文件
    t0 = time.time()
    done = 0
    fixed = 0
    normal = 0
    error = 0
    
    # 使用偏函数固定 fix_wkt_str 参数
    process_func = partial(process_single_file, fix_wkt_str=FIX_WKT_STR)
    
    with Pool(processes=n_workers) as pool:
        for status, msg in pool.imap_unordered(process_func, all_tifs):
            done += 1
            
            if status == 'fixed':
                fixed += 1
            elif status == 'normal':
                normal += 1
            elif status == 'error':
                error += 1
            
            # 更新进度
            elapsed = time.time() - t0
            eta = elapsed * (total - done) / max(1, done)
            avg_time = elapsed / done
            
            print(f"\r进度 [{done}/{total}] | "
                  f"修复:{fixed} 正常:{normal} 错误:{error} | "
                  f"平均:{avg_time:.3f}秒/文件 | "
                  f"用时:{_fmt_time(elapsed)} "
                  f"预计:{_fmt_time(eta)}", end='', flush=True)
    
    print(f"\n\n{'='*60}")
    print(f"✅ 处理完成！")
    print(f"   ├─ 子文件夹: {len(subfolders)}")
    print(f"   ├─ 总文件数: {total}")
    print(f"   ├─ 修复: {fixed} (有问题并已修复)")
    print(f"   ├─ 正常: {normal} (投影正确)")
    print(f"   └─ 错误: {error} (无法处理)")
    print(f"\n⏱️  总用时: {_fmt_time(time.time() - t0)}")
    print(f"⚡ 平均速度: {total / (time.time() - t0):.1f} 文件/秒")
    print("=" * 60)

if __name__ == '__main__':
    main()
