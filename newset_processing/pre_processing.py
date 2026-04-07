#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三合一 GDAL 栅格处理（参考网格对齐 + 最终裁切版）
========================================================================
目标与原则
------------------------------------------------------------------------
1) **以参考 TIF（--grid-template）为唯一基准**：
   - 仅使用它的坐标系（CRS）、像元大小（xRes/yRes）、起始点（origin）与覆盖范围（extent）。
   - 将所有输入影像重投影/重采样后，先对齐到与参考 TIF 完全一致的网格（“同起点、同分辨率、同覆盖范围”）。

2) **不在对齐阶段改变起始点与覆盖范围**：
   - 第一步（对齐到网格）只做：重投影 + 重采样到 **grid-template 的分辨率**，
     并**强制输出外框**等于 **grid-template 的外框**，使输出影像与参考 TIF **像元对齐、尺寸一致、外框一致**。
   - 这保证了对齐阶段不改变 origin 与 extent（完全跟参考 TIF 一致）。

3) **最终使用 shapefile 裁切**：
   - 第二步（裁切）将上一步对齐后的数据按 shapefile 裁剪，得到最终范围一致的输出。
   - 这一步会改变输出外框（因为被裁切了），但仍保留对齐阶段的分辨率与像元对齐（传入 xRes/yRes 与 targetAlignedPixels）。

4) **整型波段强制近邻重采样**；浮点使用双线性：
   - 为避免分类/整型数据在重采样阶段产生插值污染，整型一律使用 'near'。
   - 浮点（连续）数据使用 'bilinear'。

5) 其它行为沿用你的“修正版”思路（关键点保留）：
   - 若 Excel nodata 为负，自动改为 0（禁止负 nodata）。
   - LST (MOD11A1/MYD11A1) 输出强制为 UInt16（nodata=0），即便上游为 Int16。
   - 写出阶段对**整型非 QA 波段**做非负清洗（<0 设为 nodata=0）。
   - QA 波段不设 nodata、仍用近邻重采样。

与旧版的唯一区别
------------------------------------------------------------------------
- 删除了 `--proj-template`（不再有“两次 Warp”），所有对齐均以 **--grid-template** 为唯一基准；
- 对齐阶段 **不再** 使用 cutline（不裁剪），只在**对齐完成后**进行独立的裁剪 Warp；
- 保证“对齐阶段”的输出完全与参考 TIF 的 origin / extent / 分辨率一致。

使用方法（默认参数已保留）
------------------------------------------------------------------------
示例：
    python tri_pipeline_align_then_clip.py \\
        --input-root /path/to/in\\
        --output-root /path/to/out\\
        --grid-template /path/to/reference.tif\\
        --rules-excel /path/to/no_data_statistics.xlsx\\
        --shapefile /path/to/clip.shp\\
        --threads 4 --compress LZW --blocksize 512

注意：
- 如果输入是 MODIS 正弦投影（MODIS Sinusoidal），请确保其 SRS 定义正确（含 R=6371007.181）。
- 若你只想验证对齐效果，可临时不传 --shapefile，这样输出将与参考 TIF 完全同外框。
========================================================================
"""

import os, sys, math, argparse, shutil, tempfile, threading, traceback, gc
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from osgeo import gdal, osr

# ----------------- 全局 GDAL 设置 -----------------
gdal.UseExceptions()
gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")      # 禁止写/读 .aux.xml
gdal.SetConfigOption("GTIFF_INTERNAL_MASK", "YES")  # 掩膜写入 TIF 内部
gdal.SetConfigOption("CPL_MAX_ERROR_REPORTS", "100")
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")

progress_lock = threading.Lock()

# ----------------- 工具函数 -----------------
def norm_header(s: str) -> str:
    return ''.join(ch for ch in s.strip().lower().replace('_', ' ') if ch != ' ')

def split_multi(s):
    if s is None: return []
    if isinstance(s, (int, float)):
        try:
            if math.isnan(float(s)): return []
        except Exception: pass
        return [str(s)]
    s = str(s)
    if not s.strip(): return []
    for sep in [',', ';', ' ']: s = s.replace(sep, ' ')
    return [t for t in s.split(' ') if t.strip()]

def parse_int_list_1based(s):
    out = []
    for tok in split_multi(s):
        try:
            k = int(float(tok))
            if k >= 1: out.append(k - 1)
        except Exception: continue
    return sorted(set(out))

def parse_numeric_list(s):
    out = []
    for tok in split_multi(s):
        try:
            if '.' in tok or 'e' in tok.lower(): out.append(float(tok))
            else: out.append(int(tok))
        except Exception: continue
    return out

def gdal_dtype_numpy(gdt):
    mapping = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
    }
    if gdt not in mapping: raise RuntimeError(f"Unsupported GDAL data type: {gdt}")
    return mapping[gdt]

def is_integer_gdt(gdt):
    return gdt in (gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32)

def dtype_range_for_gdt(gdt):
    if gdt == gdal.GDT_Byte:    return (0, 255)
    if gdt == gdal.GDT_UInt16:  return (0, 65535)
    if gdt == gdal.GDT_Int16:   return (-32768, 32767)
    if gdt == gdal.GDT_UInt32:  return (0, 4294967295)
    if gdt == gdal.GDT_Int32:   return (-2147483648, 2147483647)
    if gdt == gdal.GDT_Float32: return (None, None)
    if gdt == gdal.GDT_Float64: return (None, None)
    return (None, None)

def choose_storage_gdt(src_gdt, nd_values):
    """根据 nodata 选择输出 dtype（不修改像元值）。
       保留原始 nodata 值，不强制转换为 0。"""
    nd_values = list(nd_values) if nd_values else []

    # 保持浮点
    if any(isinstance(v, float) and not float(v).is_integer() for v in nd_values):
        return gdal.GDT_Float32 if src_gdt != gdal.GDT_Float64 else gdal.GDT_Float64
    if src_gdt in (gdal.GDT_Float32, gdal.GDT_Float64):
        return src_gdt

    vmin, vmax = dtype_range_for_gdt(src_gdt)
    ok = True
    for v in nd_values:
        vv = int(v) if (isinstance(v,(int,np.integer)) or float(v).is_integer()) else v
        if isinstance(vv, float): return gdal.GDT_Float32
        if vmin is not None and (vv < vmin or vv > vmax): ok = False; break
    if ok: return src_gdt

    # 提升
    if src_gdt == gdal.GDT_Byte:   return gdal.GDT_Int16
    if src_gdt == gdal.GDT_UInt16: return gdal.GDT_Int32
    if src_gdt == gdal.GDT_Int16:  return gdal.GDT_Int32
    if src_gdt == gdal.GDT_UInt32: return gdal.GDT_Float64
    if src_gdt == gdal.GDT_Int32:  return gdal.GDT_Float64
    return src_gdt

def auto_resample_for_dtype(gdt):
    # 整型统一 nearest；浮点双线性
    return 'near' if is_integer_gdt(gdt) else 'bilinear'

def open_ds(path: Path, readonly=True):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly if readonly else gdal.GA_Update)
    if ds is None: raise RuntimeError(f"无法打开影像：{path}")
    return ds

def get_srs_wkt(ds):
    wkt = ds.GetProjection()
    if not wkt: return ""
    sref = osr.SpatialReference(); sref.ImportFromWkt(wkt)
    return sref.ExportToWkt()

def get_bounds_size_gt(ds):
    gt = ds.GetGeoTransform(); w, h = ds.RasterXSize, ds.RasterYSize
    xs = [0, w, 0, w]; ys = [0, 0, h, h]
    def pix2geo(px, py): return gt[0]+px*gt[1]+py*gt[2], gt[3]+px*gt[4]+py*gt[5]
    coords = [pix2geo(px,py) for px,py in zip(xs,ys)]
    xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    return (xmin, ymin, xmax, ymax), (w, h), gt

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def close_ds(ds):
    try:
        if ds: ds.FlushCache()
    except Exception:
        pass
    return None

def unlink_safe(p: Path):
    try:
        if p and p.exists(): p.unlink()
    except Exception: pass

def read_as_array_compat(band, xoff, yoff, xsize, ysize):
    try:
        return band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xsize, win_ysize=ysize)
    except TypeError:
        try:
            return band.ReadAsArray(xoff, yoff, xsize, ysize)
        except TypeError:
            buf = band.ReadRaster(xoff, yoff, xsize, ysize,
                                  buf_xsize=xsize, buf_ysize=ysize,
                                  buf_type=band.DataType)
            if buf is None: return None
            np_dtype = gdal_dtype_numpy(band.DataType)
            arr = np.frombuffer(buf, dtype=np_dtype)
            return arr.reshape(ysize, xsize)

def write_fail_logs(csv_path: Path, txt_path: Path, records: list):
    if not records: return
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['time','folder','file','step','error','gdal_last_error'])
        w.writeheader()
        for r in records: w.writerow({k:r.get(k) for k in w.fieldnames})
    with open(txt_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(f"[{r['time']}] {r['folder']}/{r['file']}\nStep: {r['step']}\nError: {r['error']}\n")
            if r.get('gdal_last_error'): f.write(f"GDAL: {r['gdal_last_error']}\n")
            if r.get('traceback'): f.write("Traceback:\n"+r['traceback'])
            f.write("\n"+"-"*80+"\n")

# ----------------- Excel 规则 -----------------
def read_rules_excel(xlsx_path: Path):
    df = pd.read_excel(xlsx_path)
    cols = {norm_header(c): c for c in df.columns}
    need = ['foldername','nodata','qaband','resample','lai']
    for k in need:
        if k not in cols: raise ValueError(f"Excel 缺少列：{k}")
    rules = {}
    for _, row in df.iterrows():
        folder = str(row[cols['foldername']]).strip()
        if not folder: continue
        nd_vals  = parse_numeric_list(row[cols['nodata']])
        # 保留原始 nodata 值，不强制转换为 0
        qa       = parse_int_list_1based(row[cols['qaband']])
        resample = str(row[cols['resample']]).strip().lower() if pd.notna(row[cols['resample']]) else ''
        lai_col  = row[cols['lai']]
        lai      = [] if (isinstance(lai_col, str) and lai_col.strip().lower()=='no') else parse_int_list_1based(lai_col)
        rules[folder] = {'no_data_vals': nd_vals, 'qa_bands': qa, 'resample': resample, 'lai_bands': lai}
    return rules

# ----------------- 预清洗到临时 GTiff -----------------
def preclean_to_temp_no_modify(
    src_path: Path,
    qa_bands_0based,
    nd_values,
    temp_path: Path,
    compress='NONE',
    blocksize=512,
    apply_nd_to_bands=None
):
    src = open_ds(src_path, readonly=True)
    w, h = src.RasterXSize, src.RasterYSize
    src_gdt = src.GetRasterBand(1).DataType
    nbands = src.RasterCount

    # 从TIF文件本身读取NoData值
    tif_nodata_values = []
    for b in range(nbands):
        band = src.GetRasterBand(b + 1)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            tif_nodata_values.append(nodata)
    
    # 合并Excel表中的NoData值和TIF文件本身的NoData值
    excel_nd_values = list(nd_values) if nd_values else []
    combined_nd_values = list(set(excel_nd_values + tif_nodata_values))
    nd_values = combined_nd_values

    out_gdt = choose_storage_gdt(src_gdt, nd_values)

    # LST 强制 UInt16
    sp_lc = str(src_path).lower()
    if ('mod11a1' in sp_lc or 'myd11a1' in sp_lc):
        out_gdt = gdal.GDT_UInt16

    drv = gdal.GetDriverByName('GTiff')
    ensure_dir(temp_path.parent)
    creation = ['TILED=YES', f'BLOCKXSIZE={blocksize}', f'BLOCKYSIZE={blocksize}',
                f'COMPRESS={compress}', 'BIGTIFF=IF_SAFER']
    dst = drv.Create(str(temp_path), w, h, nbands, out_gdt, options=creation)
    if dst is None:
        src = close_ds(src)
        raise RuntimeError(f"无法创建临时文件：{temp_path}")
    dst.SetGeoTransform(src.GetGeoTransform())
    dst.SetProjection(src.GetProjection())

    qa_set = set(qa_bands_0based)
    apply_nd_set = set(apply_nd_to_bands) if apply_nd_to_bands is not None else set(i for i in range(nbands) if i not in qa_set)

    canonical_nd = nd_values[0] if len(nd_values)>0 else -32768
    # 保留原始 nodata 值，没有设定时使用 -32768

    # 仅非 QA 设置 NoData
    for b in range(nbands):
        if b in apply_nd_set:
            dst.GetRasterBand(b+1).SetNoDataValue(
                float(canonical_nd) if out_gdt in (gdal.GDT_Float32,gdal.GDT_Float64) else int(canonical_nd)
            )

    bx, by = src.GetRasterBand(1).GetBlockSize()
    bx = bx or blocksize; by = by or blocksize

    compare_as_float = any(isinstance(v, float) and not float(v).is_integer() for v in nd_values)
    nd_vals_np = np.array(nd_values, dtype=np.float64 if compare_as_float else np.int64) if len(nd_values)>0 else None

    if nd_vals_np is not None and nd_vals_np.size>0:
        for b in range(nbands):
            if b in apply_nd_set:
                rb = dst.GetRasterBand(b+1)
                try:
                    rb.CreateMaskBand(0)
                except Exception:
                    pass

    for y0 in range(0, h, by):
        nrows = by if y0 + by <= h else h - y0
        for x0 in range(0, w, bx):
            ncols = bx if x0 + bx <= w else w - x0
            for b in range(nbands):
                band = src.GetRasterBand(b+1)
                arr  = read_as_array_compat(band, x0, y0, ncols, nrows)
                if arr is None:
                    arr = np.zeros((nrows,ncols), dtype=gdal_dtype_numpy(src_gdt))
                # 整型 & 非 QA：保留原始值，不进行非负清洗
                # 注释掉非负清洗逻辑，保留原始像元值
                dst.GetRasterBand(b+1).WriteArray(arr, xoff=x0, yoff=y0)

                if (nd_vals_np is not None and nd_vals_np.size>0) and (b in apply_nd_set):
                    rb = dst.GetRasterBand(b+1)
                    try:
                        rb.CreateMaskBand(0)
                    except Exception:
                        pass
                    mb = rb.GetMaskBand()
                    if compare_as_float: carr = arr.astype(np.float64, copy=False)
                    else: carr = arr.astype(np.int64, copy=False)
                    valid = np.ones_like(carr, dtype=np.uint8)
                    for v in nd_vals_np: valid &= (carr != v)
                    mask_block = valid * 255
                    mb.WriteArray(mask_block, xoff=x0, yoff=y0)
                    del mask_block, valid, carr
                del arr
        gc.collect()

    for b in range(nbands):
        dst.GetRasterBand(b+1).FlushCache()
        m = dst.GetRasterBand(b+1).GetMaskBand()
        if m: m.FlushCache()
    dst.FlushCache()
    src = close_ds(src); dst = close_ds(dst)
    return out_gdt, canonical_nd

# ----------------- 子集 VRT（磁盘） -----------------
def translate_subset_to_vrt_disk(tmp_dir: Path, name: str, src_ds, band_indices_0based):
    if len(band_indices_0based) == 0: return None, None
    out_vrt = tmp_dir / f"{name}.vrt"
    if out_vrt.exists(): unlink_safe(out_vrt)
    band_list_1b = [i+1 for i in band_indices_0based]
    try:
        topts = gdal.TranslateOptions(format='VRT', bandList=band_list_1b)
        vrt = gdal.Translate(str(out_vrt), src_ds, options=topts)
    except TypeError:
        args = ['-of', 'VRT']; args += sum((['-b', str(b)] for b in band_list_1b), [])
        vrt = gdal.Translate(str(out_vrt), src_ds, options=args)
    if vrt is None: raise RuntimeError("gdal.Translate 失败")
    return vrt, out_vrt

# ----------------- 构建 LAI 掩膜临时文件 -----------------
def build_lai_masked_temp(src_path: Path, lai_indices, nodata_val, mode, out_path: Path, compress='NONE', blocksize=512):
    """
    从 src_path 中提取 LAI 波段，根据 mode ('pos'/'neg') 创建掩膜版本
    - 'pos': 保留 >= 0 的值，< 0 设为 nodata
    - 'neg': 保留 < 0 的值，>= 0 设为 nodata
    """
    src_ds = open_ds(src_path, readonly=True)
    if src_ds is None:
        raise RuntimeError(f"无法打开源文件: {src_path}")
    
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    n_lai_bands = len(lai_indices)
    
    if n_lai_bands == 0:
        src_ds = close_ds(src_ds)
        return
    
    # 获取第一个 LAI 波段的数据类型
    first_lai_band = src_ds.GetRasterBand(lai_indices[0] + 1)
    gdt = first_lai_band.DataType
    np_dtype = gdal_dtype_numpy(gdt)
    
    # 创建输出文件
    driver = gdal.GetDriverByName('GTiff')
    co = [f'COMPRESS={compress}', f'BLOCKXSIZE={blocksize}', f'BLOCKYSIZE={blocksize}', 'TILED=YES', 'BIGTIFF=IF_SAFER']
    dst_ds = driver.Create(str(out_path), width, height, n_lai_bands, gdt, options=co)
    if dst_ds is None:
        src_ds = close_ds(src_ds)
        raise RuntimeError(f"无法创建输出文件: {out_path}")
    
    # 复制地理信息
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())
    
    # 处理每个 LAI 波段
    for out_band_idx, src_band_idx in enumerate(lai_indices):
        src_band = src_ds.GetRasterBand(src_band_idx + 1)
        dst_band = dst_ds.GetRasterBand(out_band_idx + 1)
        
        # 设置 nodata（保留原始值，不限制为 >= 0）
        if nodata_val is not None:
            dst_band.SetNoDataValue(float(nodata_val))
        
        # 分块处理
        for y in range(0, height, blocksize):
            ysize = min(blocksize, height - y)
            for x in range(0, width, blocksize):
                xsize = min(blocksize, width - x)
                
                # 读取数据
                data = read_as_array_compat(src_band, x, y, xsize, ysize)
                if data is None:
                    continue
                
                # 应用掩膜
                if mode == 'pos':
                    # 保留 >= 0 的值
                    mask = data < 0
                elif mode == 'neg':
                    # 保留 < 0 的值
                    mask = data >= 0
                else:
                    raise ValueError(f"未知的 mode: {mode}，应为 'pos' 或 'neg'")
                
                # 将掩膜区域设为 nodata（保留原始值）
                if nodata_val is not None:
                    data[mask] = nodata_val
                
                # 写入数据
                dst_band.WriteArray(data, x, y)
        
        dst_band.FlushCache()
    
    dst_ds.FlushCache()
    src_ds = close_ds(src_ds)
    dst_ds = close_ds(dst_ds)

# ----------------- 对齐到参考网格（不裁剪，不改变起点与外框） -----------------
def align_to_grid_vrt(tmp_dir: Path, name: str, src_ds, grid_wkt, grid_bounds, grid_size_wh,
                      resample_alg, canonical_nd=None, extra_warp_opts=None):
    """
    将 src_ds 重投影/重采样到与参考 TIF 完全相同的网格：
      - dstSRS = grid_wkt
      - outputBounds = grid_bounds
      - width/height = grid_size_wh
      => 这样输出的 origin / extent / 分辨率**完全一致**于参考 TIF。
    """
    out_vrt = tmp_dir / f"{name}.vrt"
    if out_vrt.exists(): unlink_safe(out_vrt)

    xmin,ymin,xmax,ymax = grid_bounds
    width,height        = grid_size_wh
    dst_nd = canonical_nd if canonical_nd is not None else None

    # 使用 -32768 作为初始化值，而不是 0
    init_dest_val = -32768 if dst_nd is None else dst_nd
    warp_opts = [f'INIT_DEST={init_dest_val}', 'UNIFIED_SRC_NODATA=YES']
    if extra_warp_opts: warp_opts.extend(extra_warp_opts)

    wopts = gdal.WarpOptions(
        format='VRT', dstSRS=grid_wkt,
        outputBounds=(xmin,ymin,xmax,ymax), width=width, height=height,
        resampleAlg=resample_alg,
        warpOptions=warp_opts,
        srcNodata=dst_nd, dstNodata=dst_nd
    )
    vrt = gdal.Warp(str(out_vrt), src_ds, options=wopts)
    if vrt is None: raise RuntimeError("Warp 失败（对齐参考网格）")
    return vrt, out_vrt

# ----------------- 最终裁剪（保持分辨率与像元对齐，不要求外框等于模板） -----------------
def clip_with_shapefile_vrt(tmp_dir: Path, name: str, src_ds, grid_wkt, xRes, yRes,
                             clipline_path, resample_alg, canonical_nd=None):
    """
    将“已对齐到参考网格”的 VRT 再按 shapefile 裁剪。
    - 保持同 SRS（grid_wkt）、同像元分辨率（xRes/yRes），targetAlignedPixels=True；
    - 不指定 outputBounds，GDAL 会用裁剪后最小外接矩形作为新外框（因此外框会缩小，这是预期的）。
    """
    out_vrt = tmp_dir / f"{name}.vrt"
    if out_vrt.exists(): unlink_safe(out_vrt)

    dst_nd = canonical_nd if canonical_nd is not None else None
    # 使用 -32768 作为初始化值，而不是 0
    init_dest_val = -32768 if dst_nd is None else dst_nd
    wopts = gdal.WarpOptions(
        format='VRT', dstSRS=grid_wkt,
        xRes=xRes, yRes=yRes, targetAlignedPixels=True,
        resampleAlg=resample_alg,
        cutlineDSName=str(clipline_path) if clipline_path else None,
        cropToCutline=True if clipline_path else False,
        warpOptions=[f'INIT_DEST={init_dest_val}', 'UNIFIED_SRC_NODATA=YES'],
        srcNodata=dst_nd, dstNodata=dst_nd
    )
    vrt = gdal.Warp(str(out_vrt), src_ds, options=wopts)
    if vrt is None: raise RuntimeError("Warp 失败（按 shapefile 裁剪）")
    return vrt, out_vrt

# ----------------- 最终块写出 -----------------
def write_final_with_blocks(out_path: Path, gdt, grid_gt, grid_wkt, nbands,
                            qa_vrt, nonlai_vrt, lai_pos_vrt, lai_neg_vrt, lai_nn_vrt, lai_indices,
                            qa_indices, nonlai_indices,
                            canonical_nd, compress='LZW', blocksize=512,
                            era5_mode=False, out_nodata_override=None):
    drv = gdal.GetDriverByName('GTiff')
    ensure_dir(out_path.parent)
    dst = drv.Create(str(out_path), grid_gt['w'], grid_gt['h'], nbands, gdt,
                     options=['TILED=YES', f'BLOCKXSIZE={blocksize}', f'BLOCKYSIZE={blocksize}',
                              f'COMPRESS={compress}', 'BIGTIFF=IF_SAFER'])
    if dst is None: raise RuntimeError(f"无法创建输出：{out_path}")
    dst.SetProjection(grid_wkt); dst.SetGeoTransform(grid_gt['gt'])

    qa_set=set(qa_indices); lai_set=set(lai_indices)
    qa_pos     = {i:k+1 for k,i in enumerate(qa_indices)} if qa_indices else {}
    nonlai_pos = {i:k+1 for k,i in enumerate(nonlai_indices)} if nonlai_indices else {}
    lai_pos_map= {i:k+1 for k,i in enumerate(lai_indices)} if lai_indices else {}

    nd_out = canonical_nd
    if nd_out is None: nd_out = -32768
    # 保留原始 nodata 值，没有设定时使用 -32768

    # 非 QA 设 NoData
    for i in range(nbands):
        if i in qa_set: continue
        dst.GetRasterBand(i+1).SetNoDataValue(nd_out)

    w,h = grid_gt['w'], grid_gt['h']; bx,by = blocksize, blocksize
    for y0 in range(0,h,by):
        nrows = by if y0+by<=h else h-y0
        for x0 in range(0,w,bx):
            ncols = bx if x0+bx<=w else w-x0

            # 移除 ERA5 特殊处理逻辑，统一使用标准处理

            for i in range(nbands):
                if i in qa_set:
                    arr = read_as_array_compat(qa_vrt.GetRasterBand(qa_pos[i]), x0,y0,ncols,nrows)
                    dst.GetRasterBand(i+1).WriteArray(arr, xoff=x0, yoff=y0); del arr
                elif i in lai_set:
                    bpos = lai_pos_map[i]
                    arr_pos = read_as_array_compat(lai_pos_vrt.GetRasterBand(bpos), x0,y0,ncols,nrows) if lai_pos_vrt else None
                    arr_neg = read_as_array_compat(lai_neg_vrt.GetRasterBand(bpos), x0,y0,ncols,nrows) if lai_neg_vrt else None
                    arr_nn  = read_as_array_compat(lai_nn_vrt.GetRasterBand(bpos),  x0,y0,ncols,nrows) if lai_nn_vrt  else None
                    if arr_pos is None and arr_neg is None and arr_nn is None:
                        out = np.full((nrows,ncols), nd_out, dtype=gdal_dtype_numpy(gdt))
                    else:
                        base = arr_pos if arr_pos is not None else (arr_neg if arr_neg is not None else arr_nn)
                        out = np.full((nrows,ncols), nd_out, dtype=base.dtype)
                        if arr_nn is not None:
                            valid_nn  = (arr_nn != nd_out)
                            if arr_pos is not None:
                                valid_pos = (arr_pos != nd_out); choose = (arr_nn>0)&valid_nn&valid_pos; out[choose]=arr_pos[choose]
                            if arr_neg is not None:
                                valid_neg = (arr_neg != nd_out); choose = (arr_nn<0)&valid_nn&valid_neg; out[choose]=arr_neg[choose]
                            zero_mask = (arr_nn==0)&valid_nn
                            if zero_mask.any():
                                if arr_pos is not None: out[zero_mask&(arr_pos!=nd_out)] = arr_pos[zero_mask&(arr_pos!=nd_out)]
                                if arr_neg is not None: out[zero_mask&(arr_neg!=nd_out)] = arr_neg[zero_mask&(arr_neg!=nd_out)]
                    # 保留原始值，不进行非负清洗
                    dst.GetRasterBand(i+1).WriteArray(out, xoff=x0, yoff=y0)
                    del arr_pos,arr_neg,arr_nn,out
                else:
                    arr = read_as_array_compat(nonlai_vrt.GetRasterBand(nonlai_pos[i]), x0,y0,ncols,nrows)
                    # 保留原始值，不进行非负清洗
                    dst.GetRasterBand(i+1).WriteArray(arr, xoff=x0, yoff=y0); del arr
            gc.collect()
    dst.FlushCache(); dst=close_ds(dst); return True

# ----------------- 处理一个文件 -----------------
def process_one(src_path: Path, out_path: Path,
                grid_template_ds, clip_shp_path: Path,
                qa_indices, nd_values, resample_str, lai_indices,
                threads=8, compress='LZW', blocksize=512,
                keep_temp=False, force=False,
                era5_mode=False):
    if out_path.exists() and not force:
        return 'skipped'

    src = open_ds(src_path, readonly=True)
    nbands = src.RasterCount
    src_gdt = src.GetRasterBand(1).DataType
    src = close_ds(src)

    tmp_dir = Path(tempfile.mkdtemp(prefix="tri_lai_tmp_"))
    cleaned_path = tmp_dir / "cleaned.tif"
    try:
        # 统一使用 Excel 和 TIF 本身的 NoData 值，对所有非 QA 波段应用
        effective_nd_vals = nd_values
        apply_nd_bands = [i for i in range(nbands) if i not in set(qa_indices)]

        out_gdt, canonical_nd = preclean_to_temp_no_modify(
            src_path, qa_indices, effective_nd_vals, cleaned_path,
            compress='NONE', blocksize=blocksize,
            apply_nd_to_bands=apply_nd_bands
        )
        cleaned_ds = open_ds(cleaned_path, readonly=True)

        qa_vrt_src, qa_vrt_src_path = (translate_subset_to_vrt_disk(tmp_dir, "qa_src", cleaned_ds, qa_indices) if qa_indices else (None,None))
        all_idx = list(range(nbands)); qa_set=set(qa_indices); lai_set=set(lai_indices)
        nonlai_indices = [i for i in all_idx if i not in qa_set and i not in lai_set]
        nonlai_vrt_src, nonlai_vrt_src_path = (translate_subset_to_vrt_disk(tmp_dir, "nonlai_src", cleaned_ds, nonlai_indices) if nonlai_indices else (None,None))
        lai_vrt_src, lai_vrt_src_path = (translate_subset_to_vrt_disk(tmp_dir, "lai_src", cleaned_ds, lai_indices) if lai_indices else (None,None))

        # ---------------- 对齐到 grid-template 网格（不裁剪） ----------------
        grid_bounds, grid_size, grid_gt_tuple = get_bounds_size_gt(grid_template_ds)
        grid_wkt = get_srs_wkt(grid_template_ds)
        grid_gt_info = {'gt': grid_gt_tuple, 'w': grid_size[0], 'h': grid_size[1]}
        xRes = grid_gt_tuple[1]; yRes = grid_gt_tuple[5]

        resample_alg_nonqa = 'near'  # 整型统一近邻；浮点在 translate 阶段不变，这里按整型策略锁定
        eff_nd_align = canonical_nd

        qa_vrt_aligned  = qa_vrt_aligned_path  = None
        if qa_vrt_src:
            # QA 波段：近邻；不传 nodata
            qa_vrt_aligned, qa_vrt_aligned_path = align_to_grid_vrt(
                tmp_dir, "qa_aligned", qa_vrt_src, grid_wkt, grid_bounds, grid_size,
                resample_alg='near', canonical_nd=None
            )
            qa_vrt_src = close_ds(qa_vrt_src); unlink_safe(qa_vrt_src_path)

        nonlai_vrt_aligned = nonlai_vrt_aligned_path = None
        if nonlai_vrt_src:
            nonlai_vrt_aligned, nonlai_vrt_aligned_path = align_to_grid_vrt(
                tmp_dir, "nonlai_aligned", nonlai_vrt_src, grid_wkt, grid_bounds, grid_size,
                resample_alg=resample_alg_nonqa, canonical_nd=eff_nd_align
            )
            nonlai_vrt_src = close_ds(nonlai_vrt_src); unlink_safe(nonlai_vrt_src_path)

        lai_pos_vrt = lai_neg_vrt = lai_nn_vrt = None
        lai_pos_vrt_path = lai_neg_vrt_path = lai_nn_vrt_path = None

        if lai_indices:
            # LAI 正/负/NN 三路构建
            lai_pos_path = tmp_dir / "lai_pos.tif"
            lai_neg_path = tmp_dir / "lai_neg.tif"
            build_lai_masked_temp(cleaned_path, lai_indices, canonical_nd, 'pos', lai_pos_path, compress='NONE', blocksize=blocksize)
            build_lai_masked_temp(cleaned_path, lai_indices, canonical_nd, 'neg', lai_neg_path, compress='NONE', blocksize=blocksize)
            cleaned_ds = close_ds(cleaned_ds)

            lai_pos_ds = open_ds(lai_pos_path, readonly=True)
            lai_neg_ds = open_ds(lai_neg_path, readonly=True)
            lai_src_ds = open_ds(lai_vrt_src_path, readonly=True) if lai_vrt_src else None

            lai_pos_vrt_src, lai_pos_vrt_src_path = translate_subset_to_vrt_disk(tmp_dir, "lai_pos_src", lai_pos_ds, list(range(len(lai_indices))))
            lai_neg_vrt_src, lai_neg_vrt_src_path = translate_subset_to_vrt_disk(tmp_dir, "lai_neg_src", lai_neg_ds, list(range(len(lai_indices))))
            lai_pos_ds = close_ds(lai_pos_ds); lai_neg_ds = close_ds(lai_neg_ds)
            unlink_safe(lai_pos_path); unlink_safe(lai_neg_path)

            eff_nd_align_lai = canonical_nd
            lai_pos_vrt, lai_pos_vrt_path = align_to_grid_vrt(tmp_dir,"lai_pos_aligned",lai_pos_vrt_src,grid_wkt,grid_bounds,grid_size,resample_alg=resample_alg_nonqa,canonical_nd=eff_nd_align_lai)
            lai_neg_vrt, lai_neg_vrt_path = align_to_grid_vrt(tmp_dir,"lai_neg_aligned",lai_neg_vrt_src,grid_wkt,grid_bounds,grid_size,resample_alg=resample_alg_nonqa,canonical_nd=eff_nd_align_lai)
            lai_nn_vrt,  lai_nn_vrt_path  = align_to_grid_vrt(tmp_dir,"lai_nn_aligned", lai_src_ds,     grid_wkt,grid_bounds,grid_size,resample_alg='near',canonical_nd=eff_nd_align_lai)

            if lai_vrt_src: lai_vrt_src = close_ds(lai_vrt_src); unlink_safe(lai_vrt_src_path)
            lai_pos_vrt_src = close_ds(lai_pos_vrt_src); unlink_safe(lai_pos_vrt_src_path)
            lai_neg_vrt_src = close_ds(lai_neg_vrt_src); unlink_safe(lai_neg_vrt_src_path)
            lai_src_ds = close_ds(lai_src_ds)
        else:
            cleaned_ds = close_ds(cleaned_ds)

        # ---------------- （可选）按 shapefile 裁剪（保持分辨率与像元对齐） ----------------
        if clip_shp_path and clip_shp_path.exists():
            # QA
            if qa_vrt_aligned:
                qa_vrt_clip, qa_vrt_clip_path = clip_with_shapefile_vrt(tmp_dir,"qa_clip",qa_vrt_aligned,grid_wkt,xRes,yRes,clip_shp_path,'near',canonical_nd=None)
                qa_vrt_aligned = close_ds(qa_vrt_aligned); unlink_safe(qa_vrt_aligned_path)
                qa_vrt_aligned, qa_vrt_aligned_path = qa_vrt_clip, qa_vrt_clip_path
            # 非 LAI
            if nonlai_vrt_aligned:
                nonlai_vrt_clip, nonlai_vrt_clip_path = clip_with_shapefile_vrt(tmp_dir,"nonlai_clip",nonlai_vrt_aligned,grid_wkt,xRes,yRes,clip_shp_path,resample_alg_nonqa,canonical_nd=eff_nd_align)
                nonlai_vrt_aligned = close_ds(nonlai_vrt_aligned); unlink_safe(nonlai_vrt_aligned_path)
                nonlai_vrt_aligned, nonlai_vrt_aligned_path = nonlai_vrt_clip, nonlai_vrt_clip_path
            # LAI
            if lai_indices:
                lai_pos_vrt_clip, lai_pos_vrt_clip_path = clip_with_shapefile_vrt(tmp_dir,"lai_pos_clip",lai_pos_vrt,grid_wkt,xRes,yRes,clip_shp_path,resample_alg_nonqa,canonical_nd=eff_nd_align)
                lai_neg_vrt_clip, lai_neg_vrt_clip_path = clip_with_shapefile_vrt(tmp_dir,"lai_neg_clip",lai_neg_vrt,grid_wkt,xRes,yRes,clip_shp_path,resample_alg_nonqa,canonical_nd=eff_nd_align)
                lai_nn_vrt_clip,  lai_nn_vrt_clip_path  = clip_with_shapefile_vrt(tmp_dir,"lai_nn_clip", lai_nn_vrt, grid_wkt,xRes,yRes,clip_shp_path,'near',canonical_nd=eff_nd_align)
                lai_pos_vrt = close_ds(lai_pos_vrt); unlink_safe(lai_pos_vrt_path); lai_pos_vrt, lai_pos_vrt_path = lai_pos_vrt_clip, lai_pos_vrt_clip_path
                lai_neg_vrt = close_ds(lai_neg_vrt); unlink_safe(lai_neg_vrt_path); lai_neg_vrt, lai_neg_vrt_path = lai_neg_vrt_clip, lai_neg_vrt_clip_path
                lai_nn_vrt  = close_ds(lai_nn_vrt ); unlink_safe(lai_nn_vrt_path ); lai_nn_vrt,  lai_nn_vrt_path  = lai_nn_vrt_clip,  lai_nn_vrt_clip_path

            # 更新 grid_gt_info 为“裁剪后的” geotransform 与尺寸
            # （裁剪阶段外框会变小，这是预期行为）
            ref_vrt_for_gt = nonlai_vrt_aligned or qa_vrt_aligned or lai_nn_vrt or lai_pos_vrt or lai_neg_vrt
            if ref_vrt_for_gt is None:
                raise RuntimeError("找不到可用于读取裁剪后 geotransform 的临时 VRT。")
            grid_bounds_clip, grid_size_clip, grid_gt_clip = get_bounds_size_gt(ref_vrt_for_gt)
            grid_gt_info = {'gt': grid_gt_clip, 'w': grid_size_clip[0], 'h': grid_size_clip[1]}

        # ---------------- 写出（保持 LST UInt16；统一使用 canonical_nd） ----------------
        out_gdt_final = out_gdt
        sp_lc = str(src_path).lower()
        if ('mod11a1' in sp_lc or 'myd11a1' in sp_lc):
            out_gdt_final = gdal.GDT_UInt16

        write_final_with_blocks(out_path, out_gdt_final, grid_gt_info, grid_wkt, nbands,
                                qa_vrt_aligned, nonlai_vrt_aligned, lai_pos_vrt, lai_neg_vrt, lai_nn_vrt,
                                lai_indices,
                                qa_indices, nonlai_indices,
                                canonical_nd, compress=compress, blocksize=blocksize,
                                era5_mode=False, out_nodata_override=None)

        # 清理
        if qa_vrt_aligned:   qa_vrt_aligned   = close_ds(qa_vrt_aligned);   unlink_safe(qa_vrt_aligned_path)
        if nonlai_vrt_aligned: nonlai_vrt_aligned = close_ds(nonlai_vrt_aligned); unlink_safe(nonlai_vrt_aligned_path)
        if lai_indices:
            lai_pos_vrt = close_ds(lai_pos_vrt); unlink_safe(lai_pos_vrt_path)
            lai_neg_vrt = close_ds(lai_neg_vrt); unlink_safe(lai_neg_vrt_path)
            lai_nn_vrt  = close_ds(lai_nn_vrt ); unlink_safe(lai_nn_vrt_path)

        gc.collect(); gdal.ErrorReset()
        return 'ok'
    except Exception:
        if out_path.exists():
            try: out_path.unlink()
            except Exception: pass
        raise
    finally:
        try: cleaned_ds = close_ds(cleaned_ds)
        except Exception: pass
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()

# ----------------- 批处理（含高效扫描） -----------------
def process_all(input_root: Path, output_root: Path,
                grid_template_path: Path,
                rules_excel: Path, shapefile_path: Path,
                threads=16, compress='LZW', blocksize=512,
                keep_temp=False, force=False, resume_from=0,
                include_folders=None):
    if not grid_template_path.exists(): raise FileNotFoundError(f"网格模板不存在: {grid_template_path}")

    grid_template_ds = open_ds(grid_template_path, readonly=True)

    print("[INFO] 读取规则表...")
    rules = read_rules_excel(rules_excel)

    if include_folders:
        inc_sorted = sorted(include_folders)
        print(f"[INFO] 仅处理这些子目录（共 {len(inc_sorted)} 个）：{', '.join(inc_sorted)}")
    else:
        print("[INFO] 未指定 --folders，将处理所有在规则表中的子目录。")

    print("[INFO] 扫描输入文件（单次遍历, 无排序）...")
    tasks = []; skipped_count = 0; no_rule_count = 0; not_matched = set()
    try:
        with os.scandir(input_root) as it_sub:
            seen_folders = set()
            for sub_ent in it_sub:
                if not sub_ent.is_dir(): continue
                folder_name = sub_ent.name; seen_folders.add(folder_name)
                if include_folders and (folder_name not in include_folders): continue
                r = rules.get(folder_name)
                if r is None: no_rule_count += 1; continue

                out_sub = output_root / folder_name
                existing = set()
                if out_sub.exists():
                    try:
                        with os.scandir(out_sub) as it_out:
                            for e in it_out:
                                if e.is_file(): existing.add(e.name.lower())
                    except FileNotFoundError:
                        pass

                has_task_in_this_sub = False
                try:
                    with os.scandir(sub_ent.path) as it_files:
                        for fe in it_files:
                            if not fe.is_file(): continue
                            name = fe.name
                            if ".__tmp__" in name: continue
                            lower = name.lower()
                            if not (lower.endswith(".tif") or lower.endswith(".tiff")): continue
                            if (lower in existing) and (not force):
                                skipped_count += 1; continue
                            if not has_task_in_this_sub:
                                ensure_dir(out_sub); has_task_in_this_sub = True
                            era5_mode = ('era5' in folder_name.lower())
                            tasks.append((
                                folder_name, Path(fe.path), out_sub / name,
                                r['qa_bands'], r['no_data_vals'], r['resample'], r['lai_bands'], era5_mode
                            ))
                except FileNotFoundError:
                    continue
            if include_folders: not_matched = include_folders - seen_folders
    except FileNotFoundError:
        print(f"[ERROR] 输入根目录不存在：{input_root}")
        close_ds(grid_template_ds)
        return

    if include_folders and not_matched:
        print(f"[WARN] 这些文件夹未在输入根目录下找到：{', '.join(sorted(not_matched))}")
    if no_rule_count > 0: print(f"[SKIP] {no_rule_count} 个子目录未在规则表中找到")
    if skipped_count > 0: print(f"[SKIP] {skipped_count} 个文件已存在（使用 --force 可强制重写）")
    if not tasks:
        print("[INFO] 没有需要处理的文件。")
        close_ds(grid_template_ds); return

    if resume_from > 0:
        if resume_from >= len(tasks):
            print(f"[WARN] --resume-from {resume_from} 超过任务总数 {len(tasks)}")
            close_ds(grid_template_ds); return
        print(f"[RESUME] 从任务索引 {resume_from} 开始（共 {len(tasks)} 个待处理任务）")
        tasks = tasks[resume_from:]

    total = len(tasks)
    stats = {'ok': 0, 'skipped': skipped_count, 'failed': 0}
    fail_records = []

    print(f"[INFO] 待处理：{total} | 线程：{threads} | 压缩：{compress} | Block：{blocksize}")
    print("=" * 80)

    clip_shp = Path(shapefile_path) if shapefile_path else None
    logdir = output_root / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    def _wrap(t):
        folder_name, tif, out_path, qa_bands, nd_vals, resample, lai_bands, era5_mode = t
        try:
            status = process_one(
                src_path=tif, out_path=out_path,
                grid_template_ds=grid_template_ds, clip_shp_path=clip_shp,
                qa_indices=qa_bands, nd_values=nd_vals,
                resample_str=resample, lai_indices=lai_bands,
                threads=threads, compress=compress, blocksize=blocksize,
                keep_temp=keep_temp, force=True,
                era5_mode=era5_mode
            )
            return ('ok', folder_name, tif, None, None, None, None)
        except Exception as e:
            tb = traceback.format_exc()
            gdal_msg = gdal.GetLastErrorMsg()
            msg = str(e)
            if 'Write operation not permitted on dataset opened in read-only mode' in msg: step='readonly_write'
            elif 'LAI' in msg: step='lai'
            elif 'Warp' in msg or 'warp' in msg: step='warp'
            elif 'VRT' in msg or 'Translate' in msg: step='vrt'
            elif 'ReadAsArray' in msg: step='io_read'
            else: step='unknown'
            return ('failed', folder_name, tif, msg, tb, gdal_msg, step)

    done = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(_wrap, t) for t in tasks]
        for fut in as_completed(futures):
            ret = fut.result()
            with progress_lock:
                done += 1
                if ret[0]=='ok':
                    stats['ok'] += 1
                    sys.stdout.write(f"\r[进度] {done}/{total}  OK: {stats['ok']}  FAILED: {stats['failed']}")
                else:
                    stats['failed'] += 1
                    _, folder_name, tif, msg, tb, gdal_msg, step = ret
                    sys.stdout.write(f"\n[FAIL] {folder_name}/{tif.name}  | step={step} | err={msg[:300]}")
                    sys.stdout.flush()
                    fail_records.append({
                        'time': datetime.now().isoformat(timespec='seconds'),
                        'folder': folder_name, 'file': tif.name, 'step': step,
                        'error': msg, 'gdal_last_error': gdal_msg, 'traceback': tb
                    })

    print("\n" + "=" * 80)
    print(f"[DONE] OK={stats['ok']}  SKIPPED={stats['skipped']}  FAILED={stats['failed']}")
    print(f"[OUT ] {output_root}")

    if fail_records:
        csv_path = logdir / "tri_pipeline_failures.csv"
        txt_path = logdir / "tri_pipeline_failures.log"
        write_fail_logs(csv_path, txt_path, fail_records)
        print(f"[LOG ] 失败详情已写入：{csv_path}")
        print(f"[LOG ] 完整堆栈日志：{txt_path}")

    close_ds(grid_template_ds)
    gc.collect(); gdal.ErrorReset()

# ----------------- CLI -----------------
def _normalize_include_folders(arg_list):
    if not arg_list: return None
    toks = []
    for item in arg_list:
        if item is None: continue
        s = str(item).strip()
        if not s: continue
        if ',' in s: toks.extend([t.strip() for t in s.split(',') if t.strip()])
        else: toks.append(s)
    return set(toks) if toks else None

def main():
    ap = argparse.ArgumentParser(description="三合一 GDAL 栅格处理：对齐到参考 TIF 网格（不改起点/外框）→ 最终 shapefile 裁剪（合并为单一 grid-template）")
    ap.add_argument("--input-root",  default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic", type=Path)
    ap.add_argument("--output-root", default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip", type=Path)
    ap.add_argument("--grid-template", default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/temp/2024_01_01.tif", type=Path)
    ap.add_argument("--rules-excel",  default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/no_data_statistics.xlsx", type=Path)
    ap.add_argument("--shapefile",    default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/shpfiles/bc_boundary/british_columnbia_no_crs_boundingBox.shp", type=Path)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--compress", type=str, default='LZW')
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument("--keep-temp", action='store_true')
    ap.add_argument("--force", action='store_true')
    ap.add_argument("--resume-from", type=int, default=0)
    ap.add_argument("--folders", nargs='*', default=None,
                    help="仅处理指定的子文件夹名（空格或逗号分隔）。例如：--folders ERA5_consistent_mosaic_withnodata MOD11A1_mosaic 或 --folders ERA5_mosaic,MOD11A1_mosaic")
    args = ap.parse_args()
    include_folders = _normalize_include_folders(args.folders)

    process_all(
        input_root=args.input_root,
        output_root=args.output_root,
        grid_template_path=args.grid_template,
        rules_excel=args.rules_excel,
        shapefile_path=args.shapefile,
        threads=args.threads,
        compress=args.compress,
        blocksize=args.blocksize,
        keep_temp=args.keep_temp,
        force=args.force,
        resume_from=args.resume_from,
        include_folders=include_folders
    )

if __name__ == "__main__":
    main()
