#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODIS时序数据填充脚本 - GDAL版本
（高性能 + 快速路径优化，按“最近历史已处理影像”回退 & 首日未来多天合并）

本版（v5.4 兼容修复）
- 解决 “-tap option cannot be used without using -tr.”：
  在 WarpOptions 中显式提供像元大小 xRes/yRes（由目标 geotransform 推出），
  同时保留 outputBounds，去掉 width/height，让 GDAL 自行推导尺寸；保持 targetAlignedPixels=True。
- 继续保留 v5.2/5.3 的形状对齐与“公共窗口切片”防护策略，以及你要求的全部参数与注释。
"""

import argparse
import sys
import re
import time
import signal
import atexit
import random
import os
import numpy as np
from osgeo import gdal, ogr
from pathlib import Path
from datetime import timedelta, date as Date
from typing import Dict, List, Optional, Tuple

# ===================== 基础设置 =====================

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

DATE_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2}).tif$")

_global_executor = None
def _cleanup_executor():
    global _global_executor
    if _global_executor is not None:
        try:
            _global_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _global_executor = None

def _signal_handler(signum, frame):
    _cleanup_executor()
    sys.exit(1)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(_cleanup_executor)

def _fmt_time(s: float) -> str:
    s = int(s)
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"

def _parse_date(p: Path) -> Optional[Date]:
    m = DATE_RE.match(p.name)
    if not m:
        return None
    y, M, d = map(int, m.groups())
    return Date(y, M, d)

def _build_index(folder: Path) -> Dict[Date, Path]:
    idx = {}
    for p in sorted(folder.iterdir()):
        if p.is_file():
            d = _parse_date(p)
            if d:
                idx[d] = p
    return idx

def _build_output_index(folder: Path) -> Dict[Date, Path]:
    if not folder.exists():
        return {}
    idx = {}
    for p in sorted(folder.iterdir()):
        if p.is_file():
            d = _parse_date(p)
            if d:
                idx[d] = p
    return idx

# ===================== AOI / 掩膜 / 有效性判断 =====================

def load_aoi_mask(shp_path: Optional[Path], width: int, height: int,
                   geotransform: Tuple, projection: str) -> np.ndarray:
    """
    从shapefile加载AOI mask
    Returns:
        np.ndarray: 布尔数组，True表示在AOI内
    """
    if shp_path is None or not Path(shp_path).exists():
        return np.ones((height, width), dtype=bool)
    mem_driver = gdal.GetDriverByName('MEM')
    target_ds = mem_driver.Create('', width, height, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(projection)
    shp_ds = ogr.Open(str(shp_path))
    if shp_ds is None:
        raise RuntimeError(f"无法打开shapefile: {shp_path}")
    layer = shp_ds.GetLayer()
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[1])
    mask = target_ds.GetRasterBand(1).ReadAsArray()
    target_ds = None
    shp_ds = None
    return mask.astype(bool)

def is_nodata(arr: np.ndarray,
              builtin_nodata,
              custom_nodata_list=None,
              valid_min=None,
              valid_max=None) -> np.ndarray:
    """返回 True=无效"""
    mask = np.zeros(arr.shape, dtype=bool)
    if builtin_nodata is not None:
        mask |= (arr == builtin_nodata)
    if custom_nodata_list:
        for nodata_val in custom_nodata_list:
            mask |= (arr == nodata_val)
    if arr.dtype.kind == 'f':
        mask |= np.isnan(arr)
    if valid_min is not None:
        mask |= (arr < valid_min)
    if valid_max is not None:
        mask |= (arr > valid_max)
    return mask

def spatial_fill(data: np.ndarray, need_fill: np.ndarray,
                 max_distance: int = 100) -> np.ndarray:
    """GDAL FillNodata 兜底"""
    if not np.any(need_fill):
        return data
    h, w = data.shape
    mem_driver = gdal.GetDriverByName('MEM')
    temp = mem_driver.Create('', w, h, 1, gdal.GDT_Float32)
    work = data.astype(np.float32, copy=True)
    work[need_fill] = np.nan
    band = temp.GetRasterBand(1)
    band.SetNoDataValue(np.nan)
    band.WriteArray(work)
    gdal.FillNodata(targetBand=band, maskBand=None,
                    maxSearchDist=max_distance, smoothingIterations=0)
    out = band.ReadAsArray()
    temp = None
    return out

# ===================== 重采样工具（含对齐） =====================

def _force_align_to_target_shape(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """把任意二维数组裁剪/填充为 (target_h, target_w)。"""
    h, w = arr.shape
    if (h, w) == (target_h, target_w):
        return arr.astype(np.float32, copy=False)
    out = np.full((target_h, target_w), np.nan, dtype=np.float32)
    hh = min(h, target_h)
    ww = min(w, target_w)
    out[:hh, :ww] = arr[:hh, :ww]
    return out

def _read_aligned_bands_stack(data_path: Path,
                              bands_to_process: List[int],
                              target_width: int, target_height: int,
                              target_geotransform: Tuple, target_projection: str
                              ) -> Tuple[List[np.ndarray], Optional[float]]:
    """
    若与目标网格不同，一次性 Warp 到目标网格（兼容旧 GDAL）：
      - 提供 outputBounds 与 xRes/yRes（由目标 geotransform 推出）；
      - targetAlignedPixels=True；
      - 不再传 width/height，由 GDAL 自行推导尺寸（但我们传入的 dst_ds 已经具备目标尺寸）。
    然后仅按需读取所请求的波段；
    返回各波段数组（float32，且强制对齐为目标尺寸）及源 band1 的 NoData。
    """
    src_ds = gdal.Open(str(data_path), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"无法打开文件: {data_path}")
    src_w, src_h = src_ds.RasterXSize, src_ds.RasterYSize
    src_gt = src_ds.GetGeoTransform()
    src_prj = src_ds.GetProjection()

    # 允许轻微浮点误差
    def _gt_equal(a, b, eps=1e-9):
        return all(abs(a[i]-b[i]) <= eps for i in range(6))

    need_resample = (
        (src_w != target_width) or
        (src_h != target_height) or
        (src_prj != target_projection) or
        (not _gt_equal(src_gt, target_geotransform))
    )

    builtin_nodata = src_ds.GetRasterBand(1).GetNoDataValue()

    if not need_resample:
        arrays: List[np.ndarray] = []
        for b in bands_to_process:
            a = src_ds.GetRasterBand(b + 1).ReadAsArray().astype(np.float32)
            arrays.append(_force_align_to_target_shape(a, target_height, target_width))
        src_ds = None
        return arrays, builtin_nodata

    # 需要重采样：warp 到目标网格（所有波段），再读取指定波段
    mem_driver = gdal.GetDriverByName('MEM')
    dst_ds = mem_driver.Create('', target_width, target_height,
                               src_ds.RasterCount, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(target_geotransform)
    dst_ds.SetProjection(target_projection)

    xRes = target_geotransform[1]
    yRes = abs(target_geotransform[5])

    opts = gdal.WarpOptions(
        format='MEM',
        dstSRS=target_projection,
        resampleAlg=gdal.GRA_NearestNeighbour,
        outputBounds=(
            target_geotransform[0],
            target_geotransform[3] + target_height * target_geotransform[5],
            target_geotransform[0] + target_width * target_geotransform[1],
            target_geotransform[3]
        ),
        xRes=xRes, yRes=yRes,
        multithread=True,
        targetAlignedPixels=True
    )
    gdal.Warp(dst_ds, src_ds, options=opts)

    arrays: List[np.ndarray] = []
    for b in bands_to_process:
        a = dst_ds.GetRasterBand(b + 1).ReadAsArray().astype(np.float32)
        arrays.append(_force_align_to_target_shape(a, target_height, target_width))

    dst_ds = None
    src_ds = None
    return arrays, builtin_nodata

# ===================== 永久无效像元（可选） =====================

def _check_single_date_nodata(args):
    date, file_path, builtin_nodata, custom_nodata_values, valid_ranges, bands_to_check = args
    try:
        ds = gdal.Open(str(file_path), gdal.GA_ReadOnly)
        if ds is None:
            return None
        band_masks = []
        for band_idx in bands_to_check:
            band = ds.GetRasterBand(band_idx + 1)
            arr = band.ReadAsArray().astype(np.float32)
            if valid_ranges and band_idx < len(valid_ranges) and valid_ranges[band_idx] is not None:
                vmin, vmax = valid_ranges[band_idx]
            else:
                vmin, vmax = None, None
            band_masks.append(is_nodata(arr, builtin_nodata, custom_nodata_values, vmin, vmax))
        ds = None
        return band_masks
    except Exception:
        return None

def detect_permanently_invalid_pixels(index: Dict[Date, Path],
                                      sample_size: int = 1000,
                                      custom_nodata_values: Optional[List[float]] = None,
                                      valid_ranges: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
                                      top_k_bands: Optional[int] = None,
                                      workers: int = 8) -> Optional[np.ndarray]:
    if not index:
        return None
    all_dates = sorted(index.keys())
    n_sample = min(sample_size, len(all_dates))
    random.seed(42)
    sampled_dates = sorted(random.sample(all_dates, n_sample))
    first_ds = gdal.Open(str(index[sampled_dates[0]]), gdal.GA_ReadOnly)
    if first_ds is None:
        return None
    width, height = first_ds.RasterXSize, first_ds.RasterYSize
    total_bands = first_ds.RasterCount
    builtin_nodata = first_ds.GetRasterBand(1).GetNoDataValue()
    first_ds = None
    if top_k_bands is not None:
        bands_to_check = list(range(min(top_k_bands, total_bands)))
    else:
        bands_to_check = list(range(total_bands))
    permanently_invalid = {bi: np.ones((height, width), dtype=bool) for bi in range(len(bands_to_check))}
    from concurrent.futures import ProcessPoolExecutor, as_completed
    global _global_executor
    _global_executor = ProcessPoolExecutor(max_workers=workers)
    try:
        tasks = []
        for d in sampled_dates:
            tasks.append((d, index[d], builtin_nodata, custom_nodata_values, valid_ranges, bands_to_check))
        futures = {_global_executor.submit(_check_single_date_nodata, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res is None:
                    continue
                for idx, nod_m in enumerate(res):
                    permanently_invalid[idx] &= nod_m
            except Exception:
                continue
    finally:
        _global_executor.shutdown(wait=True)
        _global_executor = None
    combined = np.zeros((height, width), dtype=bool)
    for k in permanently_invalid:
        combined |= permanently_invalid[k]
    return combined

# ===================== 核心填补逻辑 =====================

def _find_latest_processed_before(date: Date, output_folder: Path,
                                  max_past_days: int) -> Optional[Date]:
    """在 output_folder 中查找“比 date 更早且已输出”的最近日期（逐日回退）"""
    for offset in range(1, max_past_days + 1):
        cand = date - timedelta(days=offset)
        outp = output_folder / f"{cand:%Y_%m_%d}.tif"
        if outp.exists():
            return cand
    return None

def _iter_future_inputs(date: Date, input_index: Dict[Date, Path],
                        max_future_days: int) -> List[Date]:
    """返回从 date+1 起、最多 max_future_days 天内存在于输入索引中的日期列表"""
    futs = []
    for offset in range(1, max_future_days + 1):
        cand = date + timedelta(days=offset)
        if cand in input_index:
            futs.append(cand)
    return futs

def _remove_if_exists(path: Path):
    """确保 GDAL Create 能覆写：若存在则先删除"""
    try:
        if path.exists():
            os.remove(path)
    except Exception:
        pass

def fill_single_date(
    target_date: Date,
    input_index: Dict[Date, Path],
    output_path: Path,
    output_folder: Path,
    max_past_days: int,
    max_future_days: int,
    mode: str,  # 'overwrite_first' | 'resume' | 'resume_first_future'
    is_first_of_run: bool,
    aoi_shp: Optional[Path],
    enable_spatial_interp: bool,
    spatial_interp_distance: int,
    compress: str,
    blocksize: int,
    custom_nodata_values: Optional[List[float]] = None,
    valid_ranges: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    keep_bands: Optional[List[int]] = None,
    permanently_invalid_mask: Optional[np.ndarray] = None
) -> bool:
    """
    填补 target_date：
      - mode='overwrite_first'：本次运行的首日强制用未来多天合并；之后仅用历史已处理回退；
      - mode='resume'：仅用历史已处理回退，不使用未来数据；
      - mode='resume_first_future'：当且仅当首日用未来多天合并，其后按历史回退。
    """
    if target_date not in input_index:
        return False

    # 打开当天
    src_ds = gdal.Open(str(input_index[target_date]), gdal.GA_ReadOnly)
    if src_ds is None:
        return False
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    total_bands = src_ds.RasterCount
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()

    bands_to_process = list(range(total_bands)) if keep_bands is None else [b for b in keep_bands if 0 <= b < total_bands]
    n_bands = len(bands_to_process)
    if n_bands == 0:
        src_ds = None
        return False

    builtin_nodata = src_ds.GetRasterBand(bands_to_process[0] + 1).GetNoDataValue()
    aoi_mask = load_aoi_mask(aoi_shp, width, height, geotransform, projection)
    fillable_mask = aoi_mask.copy()
    if permanently_invalid_mask is not None:
        ph, pw = permanently_invalid_mask.shape
        hh = min(height, ph)
        ww = min(width, pw)
        temp_mask = np.zeros((height, width), dtype=bool)
        temp_mask[:hh, :ww] = permanently_invalid_mask[:hh, :ww]
        fillable_mask &= ~temp_mask

    data = np.zeros((n_bands, height, width), dtype=np.float32)
    dtypes = []
    for i, bidx in enumerate(bands_to_process):
        band = src_ds.GetRasterBand(bidx + 1)
        dtypes.append(band.DataType)
        a = band.ReadAsArray().astype(np.float32)
        ah, aw = a.shape
        th, tw = height, width
        hh = min(ah, th)
        ww = min(aw, tw)
        out = np.full((th, tw), np.nan, dtype=np.float32)
        out[:hh, :ww] = a[:hh, :ww]
        data[i] = out
    src_ds = None

    # ===== 选择来源 =====
    source_arrays_list = []

    use_future_for_first = (mode in ('overwrite_first', 'resume_first_future')) and is_first_of_run
    if use_future_for_first:
        # 首日：逐日向未来合并
        future_dates = _iter_future_inputs(target_date, input_index, max_future_days)
        for fd in future_dates:
            arrays, s_nd = _read_aligned_bands_stack(
                input_index[fd], bands_to_process, width, height, geotransform, projection
            )
            source_arrays_list.append((arrays, s_nd))
    else:
        # 历史已处理回退（优先前一天）
        back = _find_latest_processed_before(target_date, output_folder, max_past_days)
        if back is not None:
            source_path_out = output_folder / f"{back:%Y_%m_%d}.tif"
            arrays, s_nd = _read_aligned_bands_stack(
                source_path_out, bands_to_process, width, height, geotransform, projection
            )
            source_arrays_list.append((arrays, s_nd))

    # ===== 逐波段填补（多来源级联，使用公共窗口切片） =====
    for bi in range(n_bands):
        if valid_ranges and bi < len(valid_ranges) and valid_ranges[bi] is not None:
            vmin, vmax = valid_ranges[bi]
        else:
            vmin, vmax = None, None

        nodata_mask = is_nodata(data[bi], builtin_nodata, custom_nodata_values, vmin, vmax)
        need_fill = nodata_mask & fillable_mask

        if np.any(need_fill) and source_arrays_list:
            th, tw = data[bi].shape
            for arrays, s_nd in source_arrays_list:
                src_arr = arrays[bi].astype(np.float32)
                sh, sw = src_arr.shape
                hh = min(th, sh)
                ww = min(tw, sw)
                if hh <= 0 or ww <= 0:
                    continue
                tgt_need = need_fill[:hh, :ww]
                src_sub  = src_arr[:hh, :ww]
                src_valid = ~is_nodata(src_sub, s_nd, custom_nodata_values, vmin, vmax)
                fill_pos = tgt_need & src_valid
                if np.any(fill_pos):
                    data[bi][:hh, :ww][fill_pos] = src_sub[fill_pos]
                    nodata_mask = is_nodata(data[bi], builtin_nodata, custom_nodata_values, vmin, vmax)
                    need_fill = nodata_mask & fillable_mask
                if not np.any(need_fill):
                    break

        if enable_spatial_interp and np.any(need_fill):
            try:
                data[bi] = spatial_fill(data[bi], need_fill, spatial_interp_distance)
            except Exception:
                pass

    # ===== 将仍无效的像元强制写为 NoData（包含 valid_ranges 范围外） =====
    for bi in range(n_bands):
        if valid_ranges and bi < len(valid_ranges) and valid_ranges[bi] is not None:
            vmin, vmax = valid_ranges[bi]
        else:
            vmin, vmax = None, None
        nodata_mask_final = is_nodata(data[bi], builtin_nodata, custom_nodata_values, vmin, vmax)
        if builtin_nodata is not None:
            data[bi][nodata_mask_final] = float(builtin_nodata)
        else:
            data[bi][nodata_mask_final] = np.nan

    # ===== 写出结果（覆盖/新建） =====
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output_path.exists():
            os.remove(output_path)
    except Exception:
        pass

    driver = gdal.GetDriverByName('GTiff')
    dst = driver.Create(
        str(output_path), width, height, n_bands, gdal.GDT_Float32,
        options=[
            f'COMPRESS={compress}', 'TILED=YES',
            f'BLOCKXSIZE={blocksize}', f'BLOCKYSIZE={blocksize}', 'BIGTIFF=IF_SAFER'
        ]
    )
    dst.SetGeoTransform(geotransform)
    dst.SetProjection(projection)
    for i in range(n_bands):
        band = dst.GetRasterBand(i + 1)
        if builtin_nodata is not None:
            band.SetNoDataValue(float(builtin_nodata))
        if dtypes[i] in [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32]:
            out = np.rint(data[i])
            if dtypes[i] == gdal.GDT_Byte:
                out = np.clip(out, 0, 255).astype(np.uint8)
            else:
                out = np.clip(out, 0, 65535).astype(np.uint16)
            band.WriteArray(out)
        else:
            band.WriteArray(data[i])
    dst.FlushCache()
    dst = None
    return True

# ===================== 主程序 =====================

def main():
    ap = argparse.ArgumentParser(
        description="MODIS填充：GDAL高性能版本（快速路径 + 最近历史已处理影像回退 & 首日未来多天合并）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
处理逻辑说明：
  - 使用 --overwrite：从最早日期开始重算；数据集最开始一天强制使用未来多天合并（最多 --max_future_days）；
  - 未使用 --overwrite：继续处理。
      • 若已有输出：从最后完成日的下一天开始，仅使用历史已处理回退；
      • 若输出为空：从最早日期开始，且首日使用未来多天合并。
"""
    )

    # —— 以下参数与注释必须保留 ——
    ap.add_argument("--input_folder", type=Path, required=True,
                   help="输入文件夹，包含YYYY_MM_DD.tif格式的文件")
    ap.add_argument("--output_folder", type=Path, required=True,
                   help="输出文件夹")
    ap.add_argument("--aoi_shp", type=Path, 
                    default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/shpfiles/bc_boundary/bc_boundary_without_sea.shp',
                   help="AOI shapefile路径（可选），只填充AOI内的NoData")
    
    # 填充参数
    ap.add_argument("--max_past_days", type=int, default=60,
                   help="最大向前查找天数（默认：365）")
    ap.add_argument("--max_future_days", type=int, default=10240,
                   help="最大向后查找天数，仅对第一个时间点生效（默认：120）")
    
    # 空间插值参数
    ap.add_argument("--enable_spatial_interp", action="store_true",
                   help="启用空间插值兜底（默认：禁用）")
    # ap.add_argument("--disable_spatial_interp", action="store_true",
    #                help="禁用空间插值兜底（默认：禁用）")
    ap.add_argument("--spatial_interp_distance", type=int, default=100,
                   help="空间插值最大搜索距离（像素，默认：100）")
    
    # NoData和数值范围参数
    ap.add_argument("--custom_nodata", type=str, default=None,
                   help="自定义NoData值列表，逗号分隔，如='32768,-32768,1,0'，注意这里要使用等号='-32768,-9999'（默认：仅使用TIFF内置NoData）")
    # MCD11A1 很奇怪，这里需要用的无效值为'32768,-32768,1,255'
    ap.add_argument("--valid_ranges", type=str, default=None,
                   help="每个波段的有效值范围，用分号分隔，每个范围为min,max或None。例如：0,10000;0,10000;None")
    # MCD11A1也需要使用有效值范围'7500,65535;7500,65535;1,255;1,255'
    # 最好给所有需要填充的数据都加上有效值范围
    ap.add_argument("--keep_bands", type=str, default=None,
                   help="要保留的波段索引（从0开始），逗号分隔，如'0,1,2'表示只保留前3个波段，丢弃QA/QC波段（默认：保留所有波段）")
    
    # 永久无效像素检测（与AOI互斥）
    # 永久无效像元检测可能失效，这时会对所有观测填充，而且不考虑AOI限制
    ap.add_argument("--detect_permanent_invalid", action="store_true", default=False,
                   help="启用永久无效像素检测并跳过填充。注意：启用后将忽略AOI限制（默认：禁用）")
    ap.add_argument("--detect_top_k_bands", type=int, default=None,
                   help="只检测前k个波段（0到k-1），例如 --detect_top_k_bands 4 检测波段0,1,2,3（默认：检测所有波段）")
    ap.add_argument("--permanent_invalid_sample_size", type=int, default=1000,
                   help="永久无效像素检测的采样数量（默认：1000）")
    ap.add_argument("--permanent_invalid_workers", type=int, default=16,
                   help="永久无效像素检测的并行进程数，仅用于检测阶段（默认：16）")
    
    # 输出控制
    ap.add_argument("--overwrite", action="store_true",
                   help="覆盖已存在的输出文件（默认：跳过）")
    
    # 压缩设置
    ap.add_argument("--compress", type=str, default="LZW",
                   help="压缩方法（默认：LZW）")
    ap.add_argument("--blocksize", type=int, default=512,
                   help="块大小（默认：512）")

    args = ap.parse_args()

    enable_spatial = args.enable_spatial_interp and not getattr(args, "disable_spatial_interp", False)

    custom_nodata_values = None
    if args.custom_nodata:
        try:
            custom_nodata_values = [float(x.strip()) for x in args.custom_nodata.split(',')]
        except ValueError:
            print(f"❌ 无效的custom_nodata格式: {args.custom_nodata}", file=sys.stderr)
            sys.exit(1)

    valid_ranges = None
    if args.valid_ranges:
        try:
            ranges = []
            for band_range in args.valid_ranges.split(';'):
                band_range = band_range.strip()
                if band_range.lower() == 'none':
                    ranges.append(None)
                else:
                    parts = band_range.split(',')
                    if len(parts) != 2:
                        raise ValueError(f"波段范围应为'min,max'格式: {band_range}")
                    vmin = float(parts[0].strip()) if parts[0].strip().lower() != 'none' else None
                    vmax = float(parts[1].strip()) if parts[1].strip().lower() != 'none' else None
                    ranges.append((vmin, vmax))
            valid_ranges = ranges
        except ValueError as e:
            print(f"❌ 无效的valid_ranges格式: {e}", file=sys.stderr)
            sys.exit(1)

    keep_bands = None
    if args.keep_bands:
        try:
            keep_bands = [int(x.strip()) for x in args.keep_bands.split(',')]
        except ValueError:
            print(f"❌ 无效的keep_bands格式: {args.keep_bands}", file=sys.stderr)
            sys.exit(1)

    # —— 构建输入/输出索引 ——
    print("构建输入文件索引...")
    input_idx = _build_index(args.input_folder)
    if not input_idx:
        print(f"❌ 在 {args.input_folder} 中未找到 YYYY_MM_DD.tif 格式的文件", file=sys.stderr)
        sys.exit(1)
    dates_all = sorted(input_idx.keys())
    out_idx = _build_output_index(args.output_folder)

    # —— 永久无效像元（可选） ——
    permanently_invalid_mask = None
    if args.detect_permanent_invalid:
        print(f"\n⚠️  启用永久无效像素检测模式（忽略AOI，使用检测结果屏蔽像元）")
        permanently_invalid_mask = detect_permanently_invalid_pixels(
            input_idx,
            sample_size=args.permanent_invalid_sample_size,
            custom_nodata_values=custom_nodata_values,
            valid_ranges=valid_ranges,
            top_k_bands=args.detect_top_k_bands,
            workers=args.permanent_invalid_workers
        )
        if permanently_invalid_mask is None:
            print("   ⚠️ 检测失败，将继续正常流程（不使用永久无效屏蔽）")
        else:
            print("   ✅ 永久无效像元检测完成")
            args.aoi_shp = None  # 忽略 AOI

    # —— 确定起始与模式 ——
    if args.overwrite:
        start_idx = 0
        mode = 'overwrite_first'
        print("▶️  模式：--overwrite（从头重算；首日强制用未来多天合并）")
    else:
        if out_idx:
            last_done = max(out_idx.keys())
            start_date = None
            for d in dates_all:
                if d > last_done:
                    start_date = d
                    break
            if start_date is None:
                print("✅ 输出目录中的日期已与输入同步，无需继续。")
                return
            start_idx = dates_all.index(start_date)
            mode = 'resume'
            print(f"▶️  模式：默认继续处理（从 {start_date} 开始；上一已完成：{last_done}；不使用未来数据）")
        else:
            start_idx = 0
            mode = 'resume_first_future'
            print("▶️  模式：默认继续处理（输出为空，从最早日期开始；首日使用未来多天合并）")

    # —— 主循环 ——
    n_total = len(dates_all) - start_idx
    if n_total <= 0:
        print("✅ 没有需要处理的日期。")
        return

    print(f"📊 需要处理 {n_total} 天")
    print(f"📅 日期范围: {dates_all[start_idx]} 至 {dates_all[-1]}")
    print(f"🔙 最大向前天数: {args.max_past_days}")
    print(f"🔜 最大向后天数: {args.max_future_days}（仅用于首日未来合并）")
    print(f"🗺️  AOI限制: {'是 (' + str(args.aoi_shp.name) + ')' if args.aoi_shp else '否（处理整图或永久无效模式）'}")
    print(f"🌐 空间插值兜底: {'启用' if enable_spatial else '禁用'}")
    print(f"📝 输出目录: {args.output_folder}")

    t0 = time.time()
    first_of_run_idx = start_idx

    for i in range(start_idx, len(dates_all)):
        d = dates_all[i]
        outp = args.output_folder / f"{d:%Y_%m_%d}.tif"

        is_first_of_run = (i == first_of_run_idx)

        # resume 模式：若已存在则跳过；其他模式会覆写
        if mode == 'resume' and outp.exists():
            el = time.time() - t0
            finished = (i - start_idx + 1)
            eta = el * (len(dates_all) - finished) / max(1, finished)
            print(f"\r进度 [{finished}/{n_total}] {d} | 已存在(跳过) | 用时:{_fmt_time(el)} 预计:{_fmt_time(eta)}", end='')
            continue

        try:
            ok = fill_single_date(
                d, input_idx, outp, args.output_folder,
                args.max_past_days, args.max_future_days,
                mode, is_first_of_run,
                args.aoi_shp, enable_spatial, args.spatial_interp_distance,
                args.compress, args.blocksize,
                custom_nodata_values, valid_ranges, keep_bands,
                permanently_invalid_mask
            )
        except Exception as e:
            ok = False
            print(f"\n❌ 处理失败 {d}: {e}")

        el = time.time() - t0
        finished = (i - start_idx + 1)
        eta = el * (len(dates_all) - finished) / max(1, finished)
        stat = "OK" if ok else "FAIL"
        print(f"\r进度 [{finished}/{n_total}] {d} | {stat} | 用时:{_fmt_time(el)} 预计:{_fmt_time(eta)}", end='')

    print("\n✅ 全部完成")
    print("\n" + "=" * 60)
    print("[完成] 所有处理完成")
    print(f"[输出] {args.output_folder}")

if __name__ == "__main__":
    main()
