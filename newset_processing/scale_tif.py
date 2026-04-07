# -*- coding: utf-8 -*-
"""
对指定波段进行缩放+归一化（不裁剪到[0,1]）并统一 NoData=-32768 的通用脚本
- 输入：包含TIF文件的文件夹（单一目录，不递归）
- 输出：处理后的TIF文件保存到新文件夹
- 功能：
    * 对指定波段进行 scale（乘/加）和 min-max 归一化（(x - min)/(max-min)）
    * 避免对 NoData 像元进行任何数值变换；输出统一把 NoData 写成 -32768，并设置 nodata=-32768
    * QA 波段基于波段描述关键词检测，跳过缩放/归一化（仅统一 NoData 值）
    * 在写出TIF之前检查像素值是否在指定min-max范围内，超出范围的设为NoData
- 顺序：先 scale&offset，再按 min/max 归一化（不裁剪到[0,1]），最后进行范围检查和NoData设置
- dtype：
    * 若任意波段需要归一化或小数 scale/offset → 输出统一 float32
    * 否则保持原 dtype；但若原 dtype 不能表示 -32768（如无符号整型），自动升级为 int16

新增：
- --band_config 既可以是内联配置字符串，也可以是文件路径。
- 若 **--input_folder** 的路径字符串包含 "13Q1"（不区分大小写），
  则在缩放过程中对 Excel 指定的 LAI 波段：原像元值为 0 的位置保持 0（不受 scale/offset/归一化影响）。
  注意：LAI 波段索引依然从 Excel 读取，且计数从 1 开始（与用户口径一致）。
- 范围检查：对于配置了min/max的波段，在写出前检查像素值是否在[vmin, vmax]范围内，可选择设为NoData或clip到范围边界。
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# ================== 配置 ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_QA_KEYWORDS = ['qa', 'cloud', 'quality', 'mask']  # 不区分大小写
UNIFIED_NODATA = -32768
BLOCK = 1024  # 窗口大小（用于分块读写，降低内存占用）


# ================== Excel 规则读取（NoData + LAI列） ==================
def _norm_header(s: str) -> str:
    return ''.join(ch for ch in s.strip().lower().replace('_', ' ') if ch != ' ')

def _split_multi(s) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    # 同时支持中英文分隔符
    for sep in [',', '，', ';', '；', '、', ' ']:
        s = s.replace(sep, ' ')
    return [t for t in s.split(' ') if t.strip()]

def _parse_numeric_list(s) -> List[float]:
    out = []
    for tok in _split_multi(s):
        try:
            if tok.lower() in ('nan', 'none'):
                out.append(np.nan)
            elif '.' in tok or 'e' in tok.lower():
                out.append(float(tok))
            else:
                out.append(int(tok))
        except Exception:
            continue
    return out

def _parse_int_list_1based(s) -> List[int]:
    ints = []
    for tok in _split_multi(s):
        try:
            v = int(tok)
            if v >= 1:
                ints.append(v)
        except Exception:
            continue
    return ints

def read_rules_from_excel(xlsx_path: Path, folder_name: str) -> Tuple[List[float], List[int]]:
    """
    从规则 Excel（含列：folder name / no data / qa band / resample / lai）中取出：
      - NoData 列表（数值）
      - LAI 波段索引（1-based）
    若未命中该目录或文件不存在，返回 ([], [])。
    """
    if not xlsx_path or not Path(xlsx_path).exists():
        return [], []
    df = pd.read_excel(xlsx_path)
    cols = { _norm_header(c): c for c in df.columns }
    if 'foldername' not in cols:
        return [], []
    # 处理下采样文件夹：移除 _downsampled 后缀来查找对应的规则
    search_folder_name = str(folder_name).strip()
    if search_folder_name.endswith('_downsampled'):
        search_folder_name = search_folder_name[:-len('_downsampled')]
        logger.debug(f"检测到下采样文件夹，使用原始文件夹名查找规则: {search_folder_name}")
    
    rows = df[df[cols['foldername']].astype(str).str.strip() == search_folder_name]
    if rows.empty:
        return [], []
    # NoData
    nd_vals = []
    if 'nodata' in cols:
        nd_vals = _parse_numeric_list(rows.iloc[0][cols['nodata']])
    # LAI（1-based）
    lai_idxs_1b = []
    if 'lai' in cols:
        lai_idxs_1b = _parse_int_list_1based(rows.iloc[0][cols['lai']])
    return nd_vals, lai_idxs_1b


# ================== QA 波段检测 ==================
def detect_qa_bands(band_names: List[str], qa_keywords: Optional[List[str]] = None) -> List[int]:
    if qa_keywords is None:
        qa_keywords = DEFAULT_QA_KEYWORDS
    qa_keys = [k.lower() for k in qa_keywords]
    qa_indices = []
    for i, name in enumerate(band_names):
        nm = (name or f'band_{i+1}').lower()
        if any(k in nm for k in qa_keys):
            qa_indices.append(i)
            logger.info(f"检测到QA波段: {name or f'Band_{i+1}'} (索引: {i})")
    return qa_indices


# ================== 读取/解析 band_config（支持文件路径 + 内联字符串） ==================
def _load_band_config_text(arg_value: str) -> str:
    """
    返回配置文本：
    - 若 arg_value 指向存在的文件，则读入文件内容（允许换行、空格、逗号混合分隔）
    - 否则按原始字符串返回
    """
    p = Path(arg_value)
    if p.exists() and p.is_file():
        return p.read_text(encoding='utf-8')
    return arg_value

def _normalize_config_text(text: str) -> str:
    """
    将配置文本规范化为逗号分隔的一行字符串。
    支持换行、中文逗号分号等。
    """
    if not text:
        return ""
    # 统一分隔符为空格
    for sep in [',', '，', ';', '；', '、', '\n', '\r', '\t']:
        text = text.replace(sep, ' ')
    # 压缩多空格为单空格，再用逗号连接
    parts = [t for t in text.split(' ') if t.strip()]
    return ','.join(parts)

def parse_band_config(config_str_or_path: str) -> Dict[int, Tuple[float, float, Optional[float], Optional[float]]]:
    """
    配置格式（内联或文件）：
        "band_index:scale_factor:offset:min:max,band_index:scale_factor:offset:min:max,..."
    例：
        "0:0.02:0:240:320,1:0.02:0:240:320,2:0.002:0:nan:nan,3:0.002:0:nan:nan"

    返回：
        mapping: { band_idx(0-based): (scale, offset, min, max) }
                 其中 min/max 若为 None → 不做归一化
    """
    raw_text = _load_band_config_text(config_str_or_path)
    normalized = _normalize_config_text(raw_text)
    mapping: Dict[int, Tuple[float, float, Optional[float], Optional[float]]] = {}
    if not normalized.strip():
        return mapping

    for cfg in normalized.split(','):
        parts = cfg.strip().split(':')
        if len(parts) != 5:
            raise ValueError(f"无效配置：{cfg}（需要5段：band:scale:offset:min:max）")
        try:
            bidx = int(parts[0])
            scale = float(parts[1])
            offset = float(parts[2])

            def _to_opt_float(x):
                s = str(x).strip().lower()
                return None if s in ('nan', 'none', '') else float(x)

            vmin = _to_opt_float(parts[3])
            vmax = _to_opt_float(parts[4])
            if vmin is not None and vmax is not None and vmax <= vmin:
                raise ValueError(f"归一化区间必须满足 max>min：{cfg}")
            mapping[bidx] = (scale, offset, vmin, vmax)
        except Exception as e:
            raise ValueError(f"无效配置：{cfg}；错误：{e}")

    return mapping


# ================== NoData 掩膜 ==================
def build_nodata_mask(arr: np.ndarray, nd_values: List[float]) -> np.ndarray:
    """
    返回 True 表示像元为 NoData
    支持多 NoData 值；浮点数组会额外处理 NaN。
    """
    if not nd_values:
        # 若未提供列表，不把任何值当成 NoData（回退由 src.nodata 决定）
        return np.zeros(arr.shape, dtype=bool)
    mask = np.zeros(arr.shape, dtype=bool)
    if np.issubdtype(arr.dtype, np.floating):
        for v in nd_values:
            if isinstance(v, float) and np.isnan(v):
                mask |= np.isnan(arr)
            else:
                mask |= (arr == v)
    else:
        for v in nd_values:
            if isinstance(v, float) and np.isnan(v):
                continue
            mask |= (arr == int(v))
    return mask


# ================== dtype 决策 ==================
def _dtype_can_hold_unified_nodata(np_dtype_str: str) -> bool:
    try:
        info = np.iinfo(np_dtype_str)
        return (info.min <= UNIFIED_NODATA <= info.max)
    except ValueError:
        # 浮点类型都可表示 -32768
        return True

def decide_output_dtype(src_dtype: str, band_ops: Dict[int, Tuple[float, float, Optional[float], Optional[float]]]) -> str:
    """
    规则：
      - 若任意波段需要归一化或小数 scale/offset → float32
      - 否则尽量保持原 dtype；但若不能表示 -32768，则升级到 int16
    """
    # 任意需要浮点计算？
    for _, (sf, off, vmin, vmax) in band_ops.items():
        if (vmin is not None and vmax is not None) or (abs(sf - round(sf)) > 1e-12) or (abs(off - round(off)) > 1e-12):
            return 'float32'
    # 尝试保持原 dtype
    if _dtype_can_hold_unified_nodata(src_dtype):
        return src_dtype
    # 升级到 int16（可表示 -32768）
    return 'int16'


# ================== 单文件处理（分块） ==================
def process_tif_file(
    input_path: Path,
    output_path: Path,
    band_ops: Dict[int, Tuple[float, float, Optional[float], Optional[float]]],
    qa_keywords: Optional[List[str]],
    excel_nd_values: List[float],
    block_size: int = BLOCK,
    lai_bands_0b: Optional[Set[int]] = None,
    is_mcd15_context: bool = False,
    is_13q1_mode: bool = False,
    out_of_range_action: str = "nodata"
) -> bool:
    """
    band_ops: {band_idx: (scale, offset, vmin, vmax)}（0-based）
    excel_nd_values: 来自 Excel 的 NoData 值列表；若为空，将使用 src.nodata 回退
    lai_bands_0b: 为 LAI 波段的 0-based 索引集合（由 Excel 的 1-based 索引变换而来）。
    is_mcd15_context: 若处于 MCD15A3H_mosaic 语境，执行 249→0 且不视作 NoData 的特殊规则。
    is_13q1_mode: 若 True（由 --input_folder 路径名检测到包含 13Q1），
                  则在"LAI 波段"缩放过程中，原值为 0 的像元保持 0（不受缩放/归一化影响）。
    out_of_range_action: 超出范围的像素处理方式，"nodata"设为NoData，"clip"裁剪到范围边界。
    """
    try:
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(input_path) as src:
                src_profile = src.profile.copy()
                band_names = [src.descriptions[i] if src.descriptions[i] else f'Band_{i+1}' for i in range(src.count)]
                qa_indices = detect_qa_bands(band_names, qa_keywords)

                # 输出 dtype 决策
                out_dtype = decide_output_dtype(src_profile['dtype'], band_ops)

                # nodata 元数据统一设置为 -32768
                profile = src_profile.copy()
                profile.update(dtype=out_dtype, nodata=UNIFIED_NODATA)
                profile.pop('compress', None)  # 如需压缩，可在外层另行添加
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # 建立 NoData 基准列表（Excel 优先；否则回退到影像 nodata；两者都无则空）
                nd_vals = excel_nd_values[:]
                if not nd_vals and src_profile.get('nodata') is not None:
                    nd_vals = [src_profile['nodata']]

                # 输出
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.descriptions = tuple(band_names)

                    height, width = src.height, src.width
                    for b in range(1, src.count + 1):
                        # 是否需要处理该波段（非 QA 且在 band_ops 里才做缩放/归一化）
                        do_math = ((b - 1) in band_ops) and ((b - 1) not in qa_indices)

                        # 分块处理
                        for y in range(0, height, block_size):
                            h = min(block_size, height - y)
                            for x in range(0, width, block_size):
                                w = min(block_size, width - x)
                                win = Window(x, y, w, h)
                                data = src.read(b, window=win)

                                # 构建 NoData 掩膜（先按常规）
                                mask_nd = build_nodata_mask(data, nd_vals) if nd_vals else (
                                    np.isnan(data) if np.issubdtype(data.dtype, np.floating) else np.zeros(data.shape, dtype=bool)
                                )

                                # ====== 特殊规则：MCD15A3H 的 LAI 波段，把值==249改成0，且这些位置不视作 NoData ======
                                if is_mcd15_context and (lai_bands_0b is not None) and ((b - 1) in lai_bands_0b):
                                    to_zero = (data == 249)
                                    if np.any(to_zero):
                                        # 这些位置不应该作为 NoData（即使 Excel nodata 中包含 249）
                                        mask_nd[to_zero] = False
                                        # 把 249 改写成 0
                                        if out_dtype.startswith('float'):
                                            data = data.astype(np.float32, copy=False)
                                        elif out_dtype.startswith('int'):
                                            data = data.astype(np.int16 if out_dtype == 'int16' else np.int32, copy=False)
                                        data[to_zero] = 0

                                if do_math:
                                    scale, offset, vmin, vmax = band_ops[b - 1]
                                    # 只对有效像元做变换
                                    valid = ~mask_nd
                                    if np.any(valid):
                                        # ====== 新增：MOD13Q1 模式 + LAI 波段：原值为 0 的像元保持为 0 ======
                                        keep_zero_mask = None
                                        if is_13q1_mode and (lai_bands_0b is not None) and ((b - 1) in lai_bands_0b):
                                            # 仅对有效像元中的“原始值 == 0”位置标记
                                            keep_zero_mask = (valid & (data == 0))

                                        dv = data[valid].astype(np.float64)
                                        # scale + offset
                                        dv = dv * scale + offset
                                        
                                        # 检查范围（在归一化之前）
                                        if vmin is not None and vmax is not None:
                                            out_of_range_mask = (dv < vmin) | (dv > vmax)
                                            if np.any(out_of_range_mask):
                                                if out_of_range_action == "nodata":
                                                    # 将超出范围的像素标记为无效
                                                    valid[valid] = ~out_of_range_mask
                                                    dv = dv[~out_of_range_mask]
                                                elif out_of_range_action == "clip":
                                                    # 将超出范围的像素裁剪到范围边界
                                                    dv = np.clip(dv, vmin, vmax)
                                            
                                            # 归一化（将数据缩放到[0,1]）
                                            rng = (vmax - vmin)
                                            if rng != 0:
                                                dv = (dv - vmin) / rng

                                        # 写回（先转换到目标 dtype 的计算友好类型）
                                        data = data.astype(np.float32 if out_dtype.startswith('float') else data.dtype, copy=False)
                                        if np.any(valid):
                                            data[valid] = dv.astype(data.dtype, copy=False)

                                        # 应用“保持 0”规则：将 keep_zero_mask 位置强制置 0
                                        if keep_zero_mask is not None and np.any(keep_zero_mask):
                                            if out_dtype.startswith('float'):
                                                data[keep_zero_mask] = 0.0
                                            else:
                                                data = data.astype(np.int16 if out_dtype == 'int16' else np.int32, copy=False)
                                                data[keep_zero_mask] = 0

                                        del dv

                                # 将 NoData 像元统一写成 -32768
                                if np.any(mask_nd):
                                    # 为确保类型安全，先转换到目标 dtype 再赋值
                                    if out_dtype.startswith('float'):
                                        data = data.astype(np.float32, copy=False)
                                    elif out_dtype.startswith('int'):
                                        data = data.astype(np.int16 if out_dtype == 'int16' else np.int32, copy=False)
                                    data[mask_nd] = UNIFIED_NODATA


                                # 写块
                                dst.write(data.astype(out_dtype, copy=False), b, window=win)

                return True

    except Exception as e:
        logger.error(f"处理文件 {input_path} 失败: {e}")
        return False


# ================== 处理文件夹 ==================
def process_folder(
    input_folder: Path,
    output_folder: Path,
    band_ops: Dict[int, Tuple[float, float, Optional[float], Optional[float]]],
    qa_keywords: Optional[List[str]],
    rules_excel: Optional[Path],
    max_workers: Optional[int] = None,
    out_of_range_action: str = "nodata",
    skip_existing: bool = False,
    verbose: bool = False
) -> Tuple[int, int]:
    if not input_folder.exists():
        logger.error(f"输入文件夹不存在: {input_folder}")
        return 0, 0

    output_folder.mkdir(parents=True, exist_ok=True)

    tif_files = list(input_folder.glob("*.tif")) + list(input_folder.glob("*.TIF"))
    if not tif_files:
        logger.warning(f"在 {input_folder} 中未找到TIF文件")
        return 0, 0

    folder_name = input_folder.name
    excel_nd_values: List[float] = []
    lai_idxs_1b: List[int] = []

    if rules_excel:
        excel_nd_values, lai_idxs_1b = read_rules_from_excel(Path(rules_excel), folder_name)
        if excel_nd_values:
            logger.info(f"从规则表读取到 NoData 值（{folder_name}）：{excel_nd_values}")
        else:
            logger.info(f"规则表未提供 {folder_name} 的 NoData，回退到影像 nodata 或不使用 NoData 掩膜")
        if lai_idxs_1b:
            logger.info(f"从规则表读取到 LAI 波段（1-based，{folder_name}）：{lai_idxs_1b}")

    # 处于 MCD15A3H 语境？
    is_mcd15_context = ('mcd15a3h_mosaic' in str(input_folder.as_posix()).lower())

    # ===== 处于 MOD13Q1 模式？根据 input_folder 路径判断 =====
    is_13q1_mode = ('13q1' in str(input_folder.as_posix()).lower())
    if is_13q1_mode:
        logger.info("检测到输入路径包含 '13Q1'：启用 MOD13Q1 模式（LAI 波段原始 0 值保持为 0）")

    # 准备 LAI 波段的 0-based 集合（仅当 Excel 提供时才使用）
    lai_bands_0b: Optional[Set[int]] = None
    if lai_idxs_1b:
        lai_bands_0b = set([i - 1 for i in lai_idxs_1b if i >= 1])
        logger.info(f"LAI 波段(0-based)={sorted(list(lai_bands_0b))} 已加载。"
                    f"{'启用 MOD13Q1 保持0规则；' if is_13q1_mode else ''}"
                    f"{'启用 MCD15A3H 249→0规则；' if is_mcd15_context else ''}")

    logger.info(f"找到 {len(tif_files)} 个TIF文件")

    # 过滤已存在的文件（如果启用跳过）
    files_to_process = []
    skipped_count = 0
    
    for tif in tif_files:
        out_file = output_folder / tif.name
        
        if skip_existing and out_file.exists() and out_file.stat().st_size > 0:
            skipped_count += 1
            if verbose:
                logger.info(f"跳过已存在的文件: {out_file}")
        else:
            files_to_process.append(tif)
    
    if skipped_count > 0:
        logger.info(f"跳过 {skipped_count} 个已存在的文件")
    
    logger.info(f"需要处理 {len(files_to_process)} 个文件")

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(files_to_process))

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for tif in files_to_process:
            out_file = output_folder / tif.name
            fut = ex.submit(
                process_tif_file,
                tif,
                out_file,
                band_ops,
                qa_keywords,
                excel_nd_values,
                BLOCK,
                lai_bands_0b=lai_bands_0b,
                is_mcd15_context=is_mcd15_context,
                is_13q1_mode=is_13q1_mode,
                out_of_range_action=out_of_range_action
            )
            futures[fut] = tif

        with tqdm(total=len(files_to_process), desc="处理TIF文件") as pbar:
            for fut in as_completed(futures):
                ok = False
                try:
                    ok = fut.result()
                except Exception as e:
                    logger.error(f"处理文件 {futures[fut]} 时出错: {e}")
                if ok:
                    success_count += 1
                pbar.update(1)

    return success_count, len(files_to_process)


# ================== CLI ==================
def main():
    parser = argparse.ArgumentParser(description='对TIF文件的指定波段进行缩放+归一化处理，并统一 NoData=-32768（不裁剪到[0,1]）')
    parser.add_argument('--input_folder',
                        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/ERA5_consistent_mosaic_withnodata_downsampled',
                        type=Path, help='输入文件夹路径（单层，不递归）')
    parser.add_argument('--output_folder',
                        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/ERA5_consistent_mosaic_withnodata_downsampled',
                        type=Path, help='输出文件夹路径')
    parser.add_argument('--band_config',
                        # band:scale:offset:min:max   —— min/max 用 nan 表示不归一化
                        # MCD09GA_b1237_mosaic
                        # default='0:0.0001:0:-0.01:1.6, 1:0.0001:0:-0.01:1.6, 2:0.0001:0:-0.01:1.6, 3:0.0001:0:-0.01:1.11',
                        # after mosaic downsampling
                        # default='0:0.0001:0:0:1.54, 1:0.0001:0:0:1.53, 2:0.0001:0:0:1.41, 3:0.0001:0:0:0.92',
                        # MCD11A1_mosaic
                        # default='0:0.02:0:223:333, 1:0.02:0:221:320, 2:0.002:0.49:0.97:1, 3:0.002:0.49:0.967:1',
                        # downsampled
                        # default='0:0.02:0:5.1:333.7, 1:0.02:0:5.1:333.7, 2:0.002:0.49:0.75:1, 3:0.002:0.49:0.75:1',
                        # MCD13Q1_mosaic
                        # just for clip
                        # default='0:1:0:0:1, 1:1:0:0:1, 2:1:0:0:1, 3:1:0:0:1, 4:1:0:0:1, 5:1:0:0:1',
                        # real norm
                        # default='0:0.0001:0:-0.2:1, 1:0.0001:0:-0.2:1, 2:0.0001:0:0:1, 3:0.0001:0:0:1, 4:0.0001:0:0:1, 5:0.0001:0:0:1',
                        # MCD14A1_mosaic
                        # default='0:1:0:nan:nan, 1:0.1:0:0:18000',
                        # MCD15A3H_mosaic
                        # default='0:0.1:0:0:7, 1:0.01:0:0:1',
                        # ERA5_mosaic
                        # default='0:1:0:224:309.5, 1:1:0:-17:16.4, 2:1:0:-18.6:19.241, 3:1:0:0:100, 4:1:0:0:0.21, 5:1:0:-35157552:9870987, 6:1:0:220:295, 7:1:0:72353:104455, 8:1:0:0:0.7661, 9:1:0:0:0.7661,10:1:0:0:0.7661, 11:1:0:0:0.7661',
                        default='0:1:0:224:310, 1:1:0:-17:16.4, 2:1:0:-18.6:19.241, 3:1:0:-3:100, 4:1:0:-6.4:0.21, 5:1:0:-39971252:9905048, 6:1:0:219.9:296, 7:1:0:71048.5:104608, 8:1:0:0:0.7661, 9:1:0:0:0.7661,10:1:0:0:0.7661, 11:1:0:0:0.7661',
                        # MCD09CMG_mosaic
                        # default='0:1:0:0.01:400, 1:1:0:0.01:400, 2:1:0:0.01:400, 3:1:0:0.01:400',
                        # MCD09CMG_mosaic filled downsampled
                        # default='0:1:0:219:334.4, 1:1:0:193:400, 2:1:0:204:356, 3:1:0:202:386.3',
                        help='波段配置，格式: "band:scale:offset:min:max"，多项用逗号分隔；min/max 用 nan 表示不归一化')
    parser.add_argument('--rules-excel',
                        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/no_data_statistics.xlsx',
                        type=Path, help='规则 Excel（用于获取该目录的 NoData 列表和 LAI 列（1-based））')
    parser.add_argument('--qa-keywords', nargs='+', default=DEFAULT_QA_KEYWORDS,
                        help='QA波段关键词列表（不区分大小写）')
    parser.add_argument('--max-workers', type=int, default=8,
                        help='最大并发数 (默认: CPU核心数)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细输出')
    parser.add_argument('--out-of-range-action', choices=['nodata', 'clip'], default='clip',
                        help='超出范围像素的处理方式：nodata设为NoData，clip裁剪到范围边界（默认：clip）')
    parser.add_argument('--skip-existing', action='store_true', default=False,
                        help='跳过已存在的输出文件（默认：False，重新处理所有文件）')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        band_ops = parse_band_config(args.band_config)
        if not band_ops:
            logger.warning("band_config 为空，本次不会对任何波段做缩放/归一化（只会把 NoData 统一写成 -32768，且设置 nodata=-32768）")
        else:
            logger.info("波段配置: " + ", ".join(
                f"{b}:{sf}:{off}:{'nan' if vmn is None else vmn}:{'nan' if vmx is None else vmx}"
                for b,(sf,off,vmn,vmx) in band_ops.items()
            ))
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    logger.info(f"开始处理文件夹: {args.input_folder}")
    logger.info(f"输出文件夹: {args.output_folder}")

    ok, total = process_folder(
        args.input_folder,
        args.output_folder,
        band_ops,
        args.qa_keywords,
        args.rules_excel,
        args.max_workers,
        args.out_of_range_action,
        args.skip_existing,
        args.verbose
    )
    logger.info(f"处理完成: {ok}/{total} 个文件成功")


if __name__ == "__main__":
    main()
