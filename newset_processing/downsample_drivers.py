#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级下采样脚本（高性能版，内置你的 create_sample_config）
- average/sum：积分图（Summed Area Table）O(1) 窗口运算；滑窗也很快
- mode：非重叠走整块重排 + bincount；滑窗走“按类别分批 + 积分图”的滑动直方图
- nearest/bilinear：stride==factor 时直接用 GDAL 重采样（rasterio.warp.reproject）
- 智能无效值：只用有效像元统计，低于阈值输出无效
- QA 波段自动跳过
- 多进程 imap_unordered 流式进度
- 保持原始 dtype，做安全裁剪

用法示例
--------
# 非重叠（推荐，最快）
python run_downsample_fast.py --downsample-factor 10 --stride 10 --max-workers 8 --skip-existing --output-dir /mnt/raid/zhengsen/downsample_out

# 滑窗重叠（例如 stride=5）
python run_downsample_fast.py --downsample-factor 10 --stride 5 --max-workers 8 --skip-existing --output-dir /mnt/raid/zhengsen/downsample_out

# 针对 sum（FIRMS 计数）但想保持与其它 stride 一致的输出尺寸
python run_downsample_fast.py --downsample-factor 10 --stride 5 --sum-nonoverlap-same-size --output-dir /mnt/raid/zhengsen/downsample_out
"""

import os
import sys
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from affine import Affine

# 环境与日志
os.environ.setdefault('GDAL_CACHEMAX', '10240')       # 10GB
os.environ.setdefault('GDAL_NUM_THREADS', 'ALL_CPUS')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Advanced_Downsampler_Fast")


# ============== 你的 create_sample_config（原样内置） ==============
def create_sample_config() -> List[Dict]:
    return [
        # {
        #     # MCD09GA b1/2/3/7（表观反射率，连续量）→ 平均
        #     # QA 波段（例：Surface_Reflectance_QA、State QA 等）需要先掩膜后再聚合
        #     'path': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/MCD09GA_b1237_mosaic_withQA_QAapplied_filled',
        #     'downsample_method': 'average',
        #     'invalid_value': 0,
        #     'qa_bands': []  # 合成后的无云数据没有QA波段
        # },
        
        # {
        #     # MCD11A1（LST，连续量）→ 平均；先用 QA 掩膜
        #     'path': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/MCD11A1_mosaic_QAapplied_filled',
        #     'downsample_method': 'average',
        #     'invalid_value': 0,
        #     'qa_bands': []  # 原来 [1] 改成 0 起始
        # },
        
        # {
        #   # MCD09CMG
        #   'path': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/MCD09CMG_mosaic_QAapplied_filled',
        #   'downsample_method': 'average',
        #   'invalid_value': -32765,  # 之前处理的时候设定错了，所以这个无效值是-32765，不是-32768
        #   'qa_bands': []
        # },

        # {
        #     # MCD11A2（LST，连续量）→ 平均；先用 QA 掩膜
        #     'path': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/MCD11A2_mosaic_filled',
        #     'downsample_method': 'average',
        #     'invalid_value': 0,
        #     'qa_bands': [4, 5, 6]  # 同理，按 0 起始（原配置 [5,6]）
        # },

        {
            # ERA5（连续变量：温度、风分量、湿度、气压等）→ 面积平均
            # *若其中含“累计量”（如 total_precipitation），建议拆文件夹并对那部分用 sum（见下一个可选块）
            'path': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/ERA5_consistent_mosaic_withnodata',
            'downsample_method': 'average',
            'invalid_value': -32768,  # 为避免0值参与插值，将数据使用第一波段掩膜后将无效值设为-32768
            'qa_bands': []  # ERA5 无 QA 波段
        },
        # {
        #     # MCD12Q1（年度土地覆盖，离散分类）→ 众数
        #     'path': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/MCD12Q1_mosaic_right',
        #     'downsample_method': 'mode',
        #     'invalid_value': 0,
        #     'qa_bands': []  # 原来 [1] 改成 0 起始
        # },

        # {
        #     # MCD15A3H（LAI/FPAR，连续量）→ 平均；先 QA 掩膜
        #     'path': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/MCD15A3H_mosaic_249_filled',
        #     'downsample_method': 'average',
        #     'invalid_value': -32768, # 没有无效值，占位用
        #     'qa_bands': []  # 合成后没有QA波段
        # },
        
        # {
        #     # MCD14A1 二值火点/检测栅格：使用 sum（计数），无 QA 波段
        #     'path': '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip/MCD14A1_mosaic_binary',
        #     'downsample_method': 'sum',
        #     'invalid_value': 255,  # 二值化数据的 NoData 值
        #     'qa_bands': []
        # }
    ]


# =================== 核心数学工具（积分图等） ===================
def _integral_image(arr: np.ndarray) -> np.ndarray:
    sat = arr.cumsum(axis=0).cumsum(axis=1)
    pad = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=sat.dtype)
    pad[1:, 1:] = sat
    return pad

def _window_sum(sat: np.ndarray, y0: np.ndarray, x0: np.ndarray, h: int, w: int) -> np.ndarray:
    y1 = y0 + h
    x1 = x0 + w
    return sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]

def _clip_cast_float_to_int(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    out = np.rint(arr)
    if dtype == np.uint8:
        out = np.clip(out, 0, 255)
    elif dtype == np.uint16:
        out = np.clip(out, 0, 65535)
    elif dtype == np.uint32:
        out = np.clip(out, 0, 4294967295)
    return out.astype(dtype, copy=False)


# =================== average / sum：快速路径 ===================
def _downsample_avg_sum_integral(
    band: np.ndarray,
                         method: str, 
                         invalid_value: float,
    dtype: np.dtype,
    factor: int,
    stride: int,
    invalid_threshold: float,
    mirror_pad: int,
    sum_nonoverlap_same_size: bool
) -> np.ndarray:
    # 有效掩膜 + 无效置零
    if np.isnan(invalid_value) and np.issubdtype(band.dtype, np.floating):
        valid = ~np.isnan(band)
        data = np.where(valid, band, 0.0)
    else:
        valid = (band != invalid_value)
        data = np.where(valid, band, 0.0)

    # 一次性镜像 padding
    if mirror_pad > 0:
        data = np.pad(data, ((mirror_pad, mirror_pad), (mirror_pad, mirror_pad)), mode='reflect')
        valid = np.pad(valid.astype(np.uint8), ((mirror_pad, mirror_pad), (mirror_pad, mirror_pad)), mode='reflect')
    else:
        valid = valid.astype(np.uint8)

    H, W = band.shape
    Hp, Wp = data.shape
    out_h = (H + stride - 1) // stride
    out_w = (W + stride - 1) // stride

    yy, xx = np.meshgrid(np.arange(out_h, dtype=np.int64),
                         np.arange(out_w, dtype=np.int64),
                         indexing='ij')
    y_nom = yy * stride
    x_nom = xx * stride

    if method == 'sum' and sum_nonoverlap_same_size:
        y0 = (y_nom // factor) * factor
        x0 = (x_nom // factor) * factor
    else:
        y0 = y_nom
        x0 = x_nom

    y0 = np.clip(y0, 0, Hp - factor)
    x0 = np.clip(x0, 0, Wp - factor)

    S = _integral_image(data.astype(np.float64, copy=False))
    C = _integral_image(valid.astype(np.int64, copy=False))

    win_sum = _window_sum(S, y0, x0, factor, factor)
    win_cnt = _window_sum(C, y0, x0, factor, factor)

    area = float(factor * factor)
    valid_ratio = win_cnt.astype(np.float64) / area

    # 使用 NaN 初始化，避免计算结果恰好等于 invalid_value 时的冲突
    out = np.full((out_h, out_w), np.nan, dtype=np.float64)
    m = valid_ratio >= (1.0 - invalid_threshold)
    if method == 'average':
        out[m] = win_sum[m] / np.maximum(win_cnt[m], 1)
    else:  # sum
        out[m] = win_sum[m]
    
    # 如果计算结果恰好等于 invalid_value，加1避免冲突
    # 注意：即使是float64，部分波段可能以整数形式计算，结果可能精确等于invalid_value
    if not np.isnan(invalid_value):
        # m 标记有效位置，在这些位置中，如果计算结果等于 invalid_value，需要调整
        computed_invalid = m & (out == invalid_value)
        if np.any(computed_invalid):
            out[computed_invalid] = invalid_value + 1.0
    
    # 将未处理的像素（仍为 NaN）设为 invalid_value
    out[np.isnan(out)] = invalid_value

    if np.issubdtype(dtype, np.integer):
        return _clip_cast_float_to_int(out, dtype)
    else:
        return out.astype(dtype, copy=False)


# =================== mode：非重叠/滑窗路径 ===================
def _downsample_mode_nonoverlap_fast(
    band: np.ndarray,
    invalid_value: int,
    dtype: np.dtype,
    factor: int,
    invalid_threshold: float
) -> np.ndarray:
    H, W = band.shape
    out_h = (H + factor - 1) // factor
    out_w = (W + factor - 1) // factor

    pad_h = out_h * factor - H
    pad_w = out_w * factor - W
    if pad_h or pad_w:
        band_pad = np.pad(band, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        band_pad = band

    view = band_pad.reshape(out_h, factor, out_w, factor).transpose(0, 2, 1, 3).reshape(out_h, out_w, factor * factor)
    valid_mask = (view != invalid_value)
    valid_cnt = valid_mask.sum(axis=2).astype(np.int32)
    need_valid = (valid_cnt.astype(np.float32) / (factor * factor)) >= (1.0 - invalid_threshold)

    out = np.full((out_h, out_w), invalid_value, dtype=band.dtype)
    bins = 256 if band.dtype == np.uint8 else None

    for i in range(out_h):
        row = view[i]
        row_valid = valid_mask[i]
        for j in range(out_w):
            if not need_valid[i, j]:
                continue
            vals = row[j][row_valid[j]]
            if vals.size == 0:
                continue
            if bins is None:
                m = int(vals.max()) + 1
                hist = np.bincount(vals, minlength=m)
            else:
                hist = np.bincount(vals, minlength=bins)
            out[i, j] = hist.argmax()

    return out.astype(dtype, copy=False)


def _downsample_mode_sliding_histogram(
    band: np.ndarray,
    invalid_value: int,
    dtype: np.dtype,
    factor: int,
    stride: int,
    invalid_threshold: float,
    mirror_pad: int,
    max_class_value: int = 255,
    class_batch: int = 32
) -> np.ndarray:
    assert stride < factor, "滑窗众数仅在 stride < factor 时使用；非重叠请走 fast 路径。"

    H, W = band.shape
    if mirror_pad > 0:
        band_p = np.pad(band, ((mirror_pad, mirror_pad), (mirror_pad, mirror_pad)), mode='reflect')
    else:
        band_p = band
    Hp, Wp = band_p.shape

    out_h = (H + stride - 1) // stride
    out_w = (W + stride - 1) // stride
    yy, xx = np.meshgrid(np.arange(out_h, dtype=np.int64),
                         np.arange(out_w, dtype=np.int64),
                         indexing='ij')
    y0 = np.clip(yy * stride, 0, Hp - factor)
    x0 = np.clip(xx * stride, 0, Wp - factor)

    valid = (band_p != invalid_value).astype(np.uint8)
    C = _integral_image(valid)
    win_cnt = _window_sum(C, y0, x0, factor, factor)
    valid_ratio = win_cnt.astype(np.float64) / float(factor * factor)
    need = valid_ratio >= (1.0 - invalid_threshold)

    best_counts = np.full((out_h, out_w), -1, dtype=np.int32)
    best_labels = np.full((out_h, out_w), invalid_value, dtype=np.int32)

    for start in range(0, max_class_value + 1, class_batch):
        end = min(start + class_batch - 1, max_class_value)
        num = end - start + 1
        cls_vals = np.arange(start, end + 1, dtype=band_p.dtype).reshape(num, 1, 1)
        masks = (band_p[None, :, :] == cls_vals) & (band_p[None, :, :] != invalid_value)
        masks = masks.astype(np.uint8)

        for k in range(num):
            Sk = _integral_image(masks[k])
            cnt = _window_sum(Sk, y0, x0, factor, factor).astype(np.int32)
            better = (cnt > best_counts) & need
            best_counts[better] = cnt[better]
            best_labels[better] = start + k

        del masks

    out = np.where(need, best_labels, invalid_value).astype(dtype, copy=False)
    return out


# =================== GDAL 重采样（stride == factor） ===================
def _downsample_by_reproject(band: np.ndarray, method: str, dtype: np.dtype, factor: int) -> np.ndarray:
    H, W = band.shape
    out_h = (H + factor - 1) // factor
    out_w = (W + factor - 1) // factor
    dst = np.zeros((out_h, out_w), dtype=band.dtype)
    res_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'average': Resampling.average,
    }
    r = res_map[method]
    src_transform = Affine.identity()
    dst_transform = Affine.scale(factor, factor)
    reproject(
        source=band,
        destination=dst,
        src_transform=src_transform,
        src_crs=None,
        dst_transform=dst_transform,
        dst_crs=None,
        resampling=r,
        num_threads=os.environ.get('GDAL_NUM_THREADS', 'ALL_CPUS')
    )
    return dst.astype(dtype, copy=False)


# =================== 主类 ===================
class AdvancedDownsampler:
    def __init__(
        self,
        downsample_factor: int = 10,
        invalid_threshold: float = 0.5,
        stride: Optional[int] = None,
        max_workers: Optional[int] = None,
        skip_existing: bool = True,
        sum_nonoverlap_same_size: bool = True,
        mode_max_class_value: int = 255,
        mode_class_batch: int = 32
    ):
        self.factor = int(downsample_factor)
        self.stride = int(stride) if stride is not None else max(1, self.factor // 2)
        self.invalid_threshold = float(invalid_threshold)
        self.max_workers = int(max_workers) if (max_workers and max_workers > 0) else min(8, os.cpu_count() or 4)
        self.skip_existing = bool(skip_existing)
        self.sum_nonoverlap_same_size = bool(sum_nonoverlap_same_size)
        self.mode_max_class_value = int(mode_max_class_value)
        self.mode_class_batch = int(mode_class_batch)

        logger.info(
            f"init: factor={self.factor}, stride={self.stride}, invalid_th={self.invalid_threshold}, "
            f"workers={self.max_workers}, skip_existing={self.skip_existing}, "
            f"sum_nonoverlap_same_size={self.sum_nonoverlap_same_size}, "
            f"mode_max_class={self.mode_max_class_value}, mode_class_batch={self.mode_class_batch}"
        )

    @staticmethod
    def _band_names(src: rasterio.io.DatasetReader) -> List[str]:
        return [src.descriptions[i] if src.descriptions[i] else f"Band_{i+1}" for i in range(src.count)]

    def _process_one_band(self, band: np.ndarray, method: str, invalid_value: float, dtype: np.dtype) -> np.ndarray:
        method = method.lower()
        f, s, pad = self.factor, self.stride, self.factor // 2

        if method in ('average', 'sum'):
            return _downsample_avg_sum_integral(
                band, method, invalid_value, dtype,
                factor=f, stride=s,
                invalid_threshold=self.invalid_threshold,
                mirror_pad=pad,
                sum_nonoverlap_same_size=self.sum_nonoverlap_same_size
            )

        if method in ('nearest', 'bilinear'):
            if s == f:
                return _downsample_by_reproject(band, method, dtype, factor=f)
            # 简化滑窗近邻/双线性（不建议大规模使用）
            h, w = band.shape
            out_h = (h + s - 1) // s
            out_w = (w + s - 1) // s
            pad_w = f // 2
            band_p = np.pad(band, ((pad_w, pad_w), (pad_w, pad_w)), mode='reflect')
            out = np.empty((out_h, out_w), dtype=np.float64)
            ys, xs = np.indices((f, f))
            cy, cx = (f - 1) / 2.0, (f - 1) / 2.0
            dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
            wgt = 1.0 / (1.0 + dist)
            for i in range(out_h):
                y0 = min(i * s, h - 1)
                for j in range(out_w):
                    x0 = min(j * s, w - 1)
                    win = band_p[y0:y0 + f, x0:x0 + f]
                    if method == 'nearest':
                        out[i, j] = float(win[f // 2, f // 2])
                    else:
                        out[i, j] = float(np.average(win.astype(np.float64), weights=wgt))
            if np.issubdtype(dtype, np.integer):
                return _clip_cast_float_to_int(out, dtype)
            return out.astype(dtype, copy=False)

        if method == 'mode':
            if s == f:
                return _downsample_mode_nonoverlap_fast(
                band=band, invalid_value=int(invalid_value), dtype=dtype,
                factor=f, invalid_threshold=self.invalid_threshold
            )
            return _downsample_mode_sliding_histogram(
                band=band, invalid_value=int(invalid_value), dtype=dtype,
                factor=f, stride=s, invalid_threshold=self.invalid_threshold,
                mirror_pad=pad, max_class_value=self.mode_max_class_value,
                class_batch=self.mode_class_batch
            )

        raise ValueError(f"不支持的下采样方式: {method}")

    def process_tif_file(self, input_file: str, output_file: str, config: Dict) -> bool:
        try:
            if self.skip_existing and os.path.exists(output_file):
                return True

            with rasterio.open(input_file) as src:
                names = self._band_names(src)
                qa_bands = set(config.get('qa_bands', []))
                proc_bands = [i for i in range(src.count) if i not in qa_bands]
                if not proc_bands:
                    logger.warning(f"[跳过] 无可处理波段: {input_file}")
                    return False
                
                data = src.read()  # (count, H, W)
                profile = src.profile.copy()
                h, w = profile['height'], profile['width']
                out_h = (h + self.stride - 1) // self.stride
                out_w = (w + self.stride - 1) // self.stride

                new_transform = src.transform * Affine.scale(self.stride, self.stride)
                out_prof = profile.copy()
                out_prof.update({
                    'count': len(proc_bands),
                    'width': out_w,
                    'height': out_h,
                    'transform': new_transform
                })

                method = config['downsample_method'].lower()
                invalid_value = config['invalid_value']

                out_bands, out_desc = [], []
                for b in proc_bands:
                    band = data[b]
                    dtype = np.dtype(src.dtypes[b])
                    out_band = self._process_one_band(band, method, invalid_value, dtype)
                    out_bands.append(out_band.astype(dtype, copy=False))
                    out_desc.append(names[b])

                out_arr = np.stack(out_bands, axis=0)
                with rasterio.open(output_file, 'w', **out_prof) as dst:
                    dst.write(out_arr)
                    dst.descriptions = out_desc
                
                return True
                
        except Exception as e:
            logger.error(f"[失败] {input_file} -> {output_file}: {e}", exc_info=False)
            return False
    
    def process_directory(self, input_dir: str, output_dir: str, config: Dict) -> Tuple[int, int]:
        inp = Path(input_dir)
        outp = Path(output_dir)
        if not inp.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            return 0, 0
        outp.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(list(inp.glob("*.tif")) + list(inp.glob("*.TIF")))
        if not tif_files:
            logger.warning(f"未找到 TIF：{input_dir}")
            return 0, 0
        
        logger.info(f"开始处理目录: {input_dir} | 共 {len(tif_files)} 个文件")
        logger.info(f"方法={config['downsample_method']}, invalid_value={config['invalid_value']}, QA={config.get('qa_bands', [])}")

        # 准备参数列表，避免使用lambda函数
        args = [(str(f), str(outp / f.name), config) for f in tif_files]
        
        # 记录开始时间
        import time
        start_time = time.time()

        success = 0
        with mp.Pool(processes=self.max_workers) as pool:
            # 使用apply_async获得实时进度更新
            results = []
            for arg in args:
                result = pool.apply_async(self.process_tif_file, arg)
                results.append(result)
            
            # 自定义数字进度条
            total = len(args)
            for i, result in enumerate(results, 1):
                try:
                    ok = result.get(timeout=3600)  # 1小时超时
                    if ok:
                        success += 1
                except Exception as e:
                    print(f"\n❌ 处理失败: {e}")
                
                # 实时更新数字进度条
                elapsed_time = time.time() - start_time
                if i > 0:
                    avg_time = elapsed_time / i
                    remaining = (total - i) * avg_time
                    eta_str = f"{remaining/60:.0f}:{remaining%60:02.0f}"
                    elapsed_str = f"{elapsed_time/60:.0f}:{elapsed_time%60:02.0f}"
                    print(f"\r[{i}/{total}] | 成功:{success} 失败:{i-success} | 平均:{avg_time:.1f}s/文件 | 已用:{elapsed_str} 预计:{eta_str}", end='', flush=True)
            
            print()  # 换行

        return success, len(tif_files)


# =================== CLI ===================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="高级下采样（高性能版，内置配置）")
    parser.add_argument('--downsample-factor', type=int, default=5, # 对于非mcd14a1，是5；mcd14a1是2
                        help='窗口大小/下采样倍数，默认 10. 下采样倍数是downsample-factor / 2')
    parser.add_argument('--stride', type=int, default=None,
                        help='输出网格步长；默认 sum=factor，其它=factor//2')
    parser.add_argument('--invalid-threshold', type=float, default=0.1,
                        help='窗口内有效比例阈值；低于则输出 invalid，默认 0.5')
    parser.add_argument('--max-workers', type=int, default=8,
                        help='并发进程数，默认 8')
    parser.add_argument('--skip-existing', action='store_true', 
                        help='若输出已存在则跳过')
    parser.add_argument('--sum-nonoverlap-same-size', action='store_true',
                        help='sum 使用非重叠桶计数，但输出尺寸随 stride（与其它方式一致）')
    parser.add_argument('--mode-max-class', type=int, default=255,
                        help='滑窗众数的类别上限（uint8 通常为 255）')
    parser.add_argument('--mode-class-batch', type=int, default=32,
                        help='滑窗众数“按类别分批”大小')
    parser.add_argument('--output-dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x',
                        help='总输出目录（每个数据集建 *_downsampled 子目录）')
    args = parser.parse_args()
    
    cfg_list = create_sample_config()

    total_ok, total_n = 0, 0
    for cfg in cfg_list:
        method = cfg['downsample_method'].lower()
        # stride 默认：sum=factor，其它=factor//2
        if args.stride is not None:
            stride = int(args.stride)
        else:
            stride = args.downsample_factor if method == 'sum' else max(1, args.downsample_factor // 2)

        inp = cfg['path']
        if args.output_dir:
            base_out = Path(args.output_dir)
            base_out.mkdir(parents=True, exist_ok=True)
            out_dir = base_out / f"{Path(inp).name}_downsampled"
        else:
            out_dir = Path(inp).parent / f"{Path(inp).name}_downsampled"

        logger.info(f"\n=== 处理数据集 ===")
        logger.info(f"in : {inp}")
        logger.info(f"out: {out_dir}")
        logger.info(f"method={method}, factor={args.downsample_factor}, stride={stride}, invalid_th={args.invalid_threshold}")

        ds = AdvancedDownsampler(
            downsample_factor=args.downsample_factor,
            invalid_threshold=args.invalid_threshold,
            stride=stride,
        max_workers=args.max_workers,
            skip_existing=args.skip_existing,
            sum_nonoverlap_same_size=args.sum_nonoverlap_same_size,
            mode_max_class_value=args.mode_max_class,
            mode_class_batch=args.mode_class_batch
        )

        ok, n = ds.process_directory(inp, str(out_dir), cfg)
        total_ok += ok
        total_n += n
        logger.info(f"[完成] {ok}/{n}")

    logger.info(f"\n全部完成: {total_ok}/{total_n}")


if __name__ == "__main__":
    main()
