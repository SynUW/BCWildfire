#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODIS GeoTIFF -> True-color MP4 (标准真彩色RGB合成)
--------------------------------------------------
• 假设每个 GeoTIFF 文件包含 MCD09GA 的 4 个波段：
  - Band 1: 红色 (620-670nm)
  - Band 4: 绿色 (545-565nm) ⭐ 真实绿色波段
  - Band 3: 蓝色 (459-479nm)
  - Band 2: 近红外 (841-876nm) - 可选用于增强植被

• 对每个日期文件：
  1) 重投影到 WGS84 (EPSG:4326)，对齐到首帧的公共网格；
  2) 标准真彩色合成：R=Band1, G=Band4, B=Band3
  3) 5%线性拉伸（5%-95%百分位）+ 可选伽马校正
  4) 写入视频帧，并用白字（带黑边）叠加日期（来自文件名）。

• 视频参数：默认 2 秒/帧 (fps=0.5)，H.264/MP4 或 mp4v 作为降级备选。

依赖:
  pip install gdal numpy opencv-python tqdm

用法示例：
  # 标准真彩色 + 5%线性拉伸（默认配置）
  python video_gen.py \
      --input-dir /path/to/tiffs \
      --output out.mp4 \
      --fps 0.5
  
  # 快速测试（4帧）
  python video_gen.py \
      --input-dir /path/to/tiffs \
      --output test.mp4 \
      --max-frames 4

修复说明:
  - 使用GDAL替代rasterio避免NumPy版本冲突
  - 自动跳过无法打开的TIFF文件
  - 使用真实绿色波段（Band 4）而非合成绿色

文件名要求能解析日期：
  例如 2003_07_15.tif / 20030715.tif / 2003-07-15.tif
"""

import re
import os
import math
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from osgeo import gdal, osr
import cv2
from tqdm import tqdm

# 禁用GDAL错误输出到stderr
# 注意：不使用UseExceptions()，因为它会触发gdal_array导入，导致NumPy版本冲突
gdal.PushErrorHandler('CPLQuietErrorHandler')

DATE_PATTERNS = [
    re.compile(r".*?(\d{4})[-_]?(\d{2})[-_]?(\d{2}).*?$"),  # YYYY[-_]MM[-_]DD or YYYYMMDD
]

def parse_date_from_name(name: str) -> Optional[str]:
    for pat in DATE_PATTERNS:
        m = pat.match(name)
        if m:
            y, mth, d = m.groups()
            return f"{y}-{mth}-{d}"
    return None

def percent_clip_stretch(arr: np.ndarray, p_low=2.0, p_high=98.0, eps=1e-6) -> np.ndarray:
    """对单通道做百分位拉伸，输出 0..1"""
    lo = np.nanpercentile(arr, p_low)
    hi = np.nanpercentile(arr, p_high)
    if not np.isfinite(lo): lo = np.nanmin(arr)
    if not np.isfinite(hi): hi = np.nanmax(arr)
    if hi - lo < eps:
        out = np.zeros_like(arr, dtype=np.float32)
    else:
        out = (arr - lo) / (hi - lo)
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)

def gamma_correct(arr01: np.ndarray, gamma=1.0) -> np.ndarray:
    """对 0..1 图像做伽马校正"""
    if abs(gamma - 1.0) < 1e-6:
        return arr01
    arr = np.clip(arr01, 0.0, 1.0)
    return np.power(arr, 1.0 / gamma)

def build_true_color(b1_red, b4_green, b3_blue, b2_nir=None,
                     use_nir_enhance: bool = False,
                     nir_weight: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    标准真彩色合成 (True Color RGB):
      R = Band 1 (红色, 620-670nm)
      G = Band 4 (绿色, 545-565nm)  ⭐ 使用真实绿色波段
      B = Band 3 (蓝色, 459-479nm)
    
    可选：用NIR增强绿色通道（适用于植被区域）
      G_enhanced = G + nir_weight * NIR
    """
    R = b1_red.astype(np.float32)
    G = b4_green.astype(np.float32)
    B = b3_blue.astype(np.float32)
    
    # 可选：NIR增强（突出植被）
    if use_nir_enhance and b2_nir is not None:
        G = G + nir_weight * b2_nir.astype(np.float32)
        G = np.clip(G, 0, None)  # 避免负值

    return R, G, B

def to_uint8_rgb(R01, G01, B01) -> np.ndarray:
    rgb = np.stack([R01, G01, B01], axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb

def compute_common_grid(first_ds, dst_crs="EPSG:4326", dst_res: Optional[Tuple[float, float]] = None):
    """
    用第一幅影像决定视频公共网格（WGS84），返回 (transform, width, height)
    使用GDAL
    """
    src_width = first_ds.RasterXSize
    src_height = first_ds.RasterYSize
    src_geotransform = first_ds.GetGeoTransform()
    src_projection = first_ds.GetProjection()
    
    # 创建目标坐标系
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_projection)
    
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)
    
    # 使用GDAL AutoCreateWarpedVRT估算目标尺寸
    vrt = gdal.AutoCreateWarpedVRT(first_ds, src_projection, dst_srs.ExportToWkt())
    if vrt is None:
        raise RuntimeError("无法创建WarpedVRT")
    
    width = vrt.RasterXSize
    height = vrt.RasterYSize
    transform = vrt.GetGeoTransform()
    
    # 如果指定了分辨率，重新计算
    if dst_res is not None:
        # 获取边界
        minx = transform[0]
        maxy = transform[3]
        maxx = minx + transform[1] * width
        miny = maxy + transform[5] * height
        
        # 重新计算尺寸
        width = int((maxx - minx) / dst_res[0])
        height = int((miny - maxy) / abs(dst_res[1]))
        transform = (minx, dst_res[0], 0, maxy, 0, -abs(dst_res[1]))
    
    vrt = None
    return transform, width, height

def reproject_to_grid(src_ds, indexes, dst_crs_epsg, dst_transform, dst_shape, resampling=gdal.GRA_Bilinear) -> np.ndarray:
    """
    将给定波段索引的数组重投影到公共网格
    使用GDAL Warp API（避免ReadAsArray的NumPy版本问题）
    性能优化：一次性处理所有波段
    """
    bands = len(indexes)
    dst_height, dst_width = dst_shape
    out = np.zeros((bands, dst_height, dst_width), dtype=np.float32)
    
    # 创建目标坐标系
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(dst_crs_epsg)
    dst_wkt = dst_srs.ExportToWkt()
    
    # 创建VRT选择所需波段
    vrt_options = gdal.BuildVRTOptions(bandList=indexes)
    vrt = gdal.BuildVRT('', src_ds, options=vrt_options)
    
    # 一次性Warp所有波段（比逐个波段快很多）
    warp_options = gdal.WarpOptions(
        format='MEM',
        dstSRS=dst_wkt,
        resampleAlg=resampling,
        dstNodata=np.nan,
        multithread=True,  # 启用多线程
        width=dst_width,
        height=dst_height
    )
    # Warp直接创建目标数据集
    result_ds = gdal.Warp('', vrt, options=warp_options)
    result_ds.SetGeoTransform(dst_transform)
    
    # 读取所有波段数据（使用RasterIO，不使用ReadAsArray）
    for i in range(bands):
        band = result_ds.GetRasterBand(i + 1)
        buf = band.ReadRaster(0, 0, dst_width, dst_height, dst_width, dst_height, gdal.GDT_Float32)
        out[i] = np.frombuffer(buf, dtype=np.float32).reshape((dst_height, dst_width))
    
    # 清理
    vrt = None
    result_ds = None
    
    return out

def put_date_text(frame_bgr: np.ndarray, date_text: str, font_scale=1.2, thickness=2) -> np.ndarray:
    """
    左下角叠加白字日期，黑色描边
    """
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    org = (int(0.02 * w), int(h - 0.02 * h))  # 左下角稍微上移
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 黑色描边
    cv2.putText(img, date_text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # 白字
    cv2.putText(img, date_text, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser(description="Generate true-color MP4 from MODIS GeoTIFFs (b1,b2,b3,b7).")
    ap.add_argument("--input-dir", type=str, required=True, help="包含 GeoTIFF 的文件夹")
    ap.add_argument("--output", type=str, required=True, help="输出 MP4 文件路径，如 out.mp4")
    ap.add_argument("--fps", type=float, default=0.5, help="视频帧率（帧/秒）；2 秒/帧即 0.5 fps")
    ap.add_argument("--codec", type=str, default="h264", choices=["h264", "mp4v"], help="视频编码器")
    ap.add_argument("--band-order", type=str, default="1,4,3,2",
                    help="输入 GeoTIFF 波段顺序 R,G,B,NIR，逗号分隔，默认 1=红,4=绿,3=蓝,2=NIR (MCD09GA标准)")
    ap.add_argument("--dst-res", type=str, default=None,
                    help="目标分辨率（经纬度度值，例如 '0.0045,0.0045'），默认自动按首帧估计")
    ap.add_argument("--use-nir-enhance", action="store_true", help="用NIR增强绿色通道（突出植被）")
    ap.add_argument("--nir-weight", type=float, default=0.1, help="NIR增强权重，典型 0.05~0.15")
    ap.add_argument("--stretch-plow", type=float, default=5.0, help="百分位拉伸下界（默认5%线性拉伸）")
    ap.add_argument("--stretch-phigh", type=float, default=95.0, help="百分位拉伸上界（默认5%线性拉伸）")
    ap.add_argument("--gamma", type=float, default=1.0, help="伽马校正，>1 提升亮度，<1 增强对比")
    ap.add_argument("--global-stretch", action="store_true",
                    help="使用全局（全时序）百分位拉伸而非逐帧拉伸")
    ap.add_argument("--font-scale", type=float, default=4, help="日期文字大小")
    ap.add_argument("--thickness", type=int, default=2, help="日期文字线宽")
    ap.add_argument("--max-frames", type=int, default=None, help="最大帧数限制（用于快速测试，如 --max-frames 4）")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    assert in_dir.is_dir(), f"输入目录不存在: {in_dir}"

    tiffs = sorted([p for p in in_dir.glob("*.tif*")])
    if not tiffs:
        raise FileNotFoundError("输入目录下未找到任何 .tif/.tiff 文件")

    # 解析日期并排序（若无日期解析将放到最后）
    items = []
    for p in tiffs:
        ds = parse_date_from_name(p.stem)
        items.append((p, ds if ds else "9999-99-99"))
    items.sort(key=lambda x: x[1])
    
    # 限制帧数（用于快速测试）
    if args.max_frames and args.max_frames > 0:
        items = items[:args.max_frames]
        print(f"⚠️  限制帧数模式：只处理前 {len(items)} 帧")

    # 解析波段映射 (R, G, B, NIR)
    try:
        r_idx, g_idx, b_idx, nir_idx = [int(x.strip()) for x in args.band_order.split(",")]
    except Exception:
        raise ValueError("--band-order 格式应为 'R,G,B,NIR'，例如 '1,4,3,2' (MCD09GA标准)")

    # 打开首帧，建立公共网格
    # 尝试打开第一个有效的TIFF文件
    src0 = None
    first_valid_idx = 0
    for idx, (p, _) in enumerate(items):
        try:
            src0 = gdal.Open(str(p), gdal.GA_ReadOnly)
            if src0 is not None:
                first_valid_idx = idx
                break
        except Exception as e:
            print(f"⚠️  跳过无法打开的文件: {p.name} ({e})")
            continue
    
    if src0 is None:
        raise RuntimeError("所有TIFF文件都无法打开！")
    
    dst_crs = "EPSG:4326"
    dst_res = None
    if args.dst_res:
        a, b = args.dst_res.split(",")
        dst_res = (float(a), float(b))
    dst_transform, dst_w, dst_h = compute_common_grid(src0, dst_crs=dst_crs, dst_res=dst_res)
    src0 = None  # 关闭

    # 如果需要全局拉伸，先一遍统计全局百分位
    global_stats = None
    if args.global_stretch:
        vals_R, vals_G, vals_B = [], [], []
        # 性能优化：只采样部分帧进行全局统计（最多50帧或全部）
        sample_items = items if len(items) <= 50 else items[::max(1, len(items)//50)]
        print(f"全局统计：采样 {len(sample_items)}/{len(items)} 帧")
        for p, _ in tqdm(sample_items, desc="扫描全局百分位", ncols=100):
            try:
                src = gdal.Open(str(p), gdal.GA_ReadOnly)
                if src is None:
                    print(f"⚠️  跳过: {p.name}")
                    continue
                
                arr = reproject_to_grid(
                    src, [r_idx, g_idx, b_idx, nir_idx], 4326, dst_transform, (dst_h, dst_w),
                    resampling=gdal.GRA_Bilinear
                )
                src = None
                
                band_r, band_g, band_b, band_nir = arr
                R, G, B = build_true_color(band_r, band_g, band_b, band_nir,
                                           use_nir_enhance=args.use_nir_enhance,
                                           nir_weight=args.nir_weight)
                # 收集样本（为节省内存，随机抽样 10% 像素）
                for ch, buf in zip([R, G, B], [vals_R, vals_G, vals_B]):
                    flat = ch.ravel()
                    n = flat.size
                    step = max(1, n // (n // 10 if n >= 1_000_000 else 10))  # 约 10% 或更少
                    buf.append(flat[::step])
            except Exception as e:
                print(f"⚠️  全局统计跳过 {p.name}: {e}")
                continue
        
        R_all = np.concatenate(vals_R) if vals_R else np.array([0.0])
        G_all = np.concatenate(vals_G) if vals_G else np.array([0.0])
        B_all = np.concatenate(vals_B) if vals_B else np.array([0.0])
        def perc(a):
            lo = np.nanpercentile(a, args.stretch_plow)
            hi = np.nanpercentile(a, args.stretch_phigh)
            return lo, hi
        global_stats = (perc(R_all), perc(G_all), perc(B_all))

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*('avc1' if args.codec == 'h264' else 'mp4v'))
    out_video = cv2.VideoWriter(str(args.output), fourcc, args.fps, (dst_w, dst_h))
    if not out_video.isOpened():
        # 某些系统 avc1/h264 不可用，降级到 mp4v
        fourcc_fallback = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(args.output), fourcc_fallback, args.fps, (dst_w, dst_h))
        if not out_video.isOpened():
            raise RuntimeError("无法打开视频写入器，请尝试更换 --codec 或检查 OpenCV/FFmpeg 支持。")

    # 写帧
    skipped_count = 0
    for p, ds in tqdm(items, desc="生成视频帧", ncols=100):
        try:
            src = gdal.Open(str(p), gdal.GA_ReadOnly)
            if src is None:
                print(f"⚠️  跳过无法打开的文件: {p.name}")
                skipped_count += 1
                continue
            
            arr = reproject_to_grid(
                src, [r_idx, g_idx, b_idx, nir_idx], 4326, dst_transform, (dst_h, dst_w),
                resampling=gdal.GRA_Bilinear
            )
            src = None
            
            band_r, band_g, band_b, band_nir = arr

            # 标准真彩色合成
            R, G, B = build_true_color(
                band_r, band_g, band_b, band_nir,
                use_nir_enhance=args.use_nir_enhance,
                nir_weight=args.nir_weight
            )

            # 拉伸 & 伽马
            if args.global_stretch and global_stats is not None:
                (Rlo, Rhi), (Glo, Ghi), (Blo, Bhi) = global_stats
                def stretch_fix(ch, lo, hi, eps=1e-6):
                    if hi - lo < eps:
                        return np.zeros_like(ch, dtype=np.float32)
                    x = (ch - lo) / (hi - lo)
                    return np.clip(x, 0.0, 1.0).astype(np.float32)
                R01 = stretch_fix(R, Rlo, Rhi)
                G01 = stretch_fix(G, Glo, Ghi)
                B01 = stretch_fix(B, Blo, Bhi)
            else:
                R01 = percent_clip_stretch(R, args.stretch_plow, args.stretch_phigh)
                G01 = percent_clip_stretch(G, args.stretch_plow, args.stretch_phigh)
                B01 = percent_clip_stretch(B, args.stretch_plow, args.stretch_phigh)

            if args.gamma and abs(args.gamma - 1.0) > 1e-6:
                R01 = gamma_correct(R01, args.gamma)
                G01 = gamma_correct(G01, args.gamma)
                B01 = gamma_correct(B01, args.gamma)

            rgb = to_uint8_rgb(R01, G01, B01)
            # OpenCV 用 BGR
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # 叠加日期（若无解析，显示文件名）
            date_txt = ds if ds != "9999-99-99" else p.stem
            bgr = put_date_text(bgr, date_txt, font_scale=args.font_scale, thickness=args.thickness)

            out_video.write(bgr)
        
        except Exception as e:
            print(f"⚠️  处理失败，跳过 {p.name}: {e}")
            skipped_count += 1
            continue

    out_video.release()
    print(f"\n✅ 视频生成完成：{args.output}")
    print(f"📊 总文件数：{len(items)}")
    print(f"✓  成功处理：{len(items) - skipped_count}")
    print(f"⚠  跳过文件：{skipped_count}")

if __name__ == "__main__":
    main()
