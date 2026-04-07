#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整数据集生成器 - 年份范围版本（保持原 H5 结构；两阶段筛选+流式写入）
全局有效像素：所有驱动因素（跨全部日期，AND 后对日期做 OR 聚合）

更新：仅使用图像内置NoData值进行像素筛选
- 从TIFF图像中获取内置NoData值
- 通过所有驱动因素随机采样100张TIF计算全局有效值范围交集
- 数据存储：直接保存原始值，不进行任何缩放或转换
"""

import os
import h5py
import numpy as np
import glob
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
from functools import lru_cache
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import ast
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YearOnlyDatasetGenerator:
    def __init__(self, data_dir, output_dir, target_years=[2024], 
                 max_workers=12, batch_size=10000, max_aggregation_dates=10, force_retrain=False):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.target_years = target_years if isinstance(target_years, list) else [target_years]
        self.max_workers = max(1, int(max_workers))
        self.batch_size = batch_size  # 兼容参数
        self.max_aggregation_dates = max_aggregation_dates  # 聚合时的最大日期数
        self.force_retrain = force_retrain  # 是否强制重新处理

        # 与既有顺序保持一致
        self.driver_order = [
            'MCD14A1_mosaic_binary_downsampled_masked', # checked and fixed
            'ERA5_consistent_mosaic_withnodata_downsampled', # checked and fixed
            'distance_maps',  # checked
            'NDVI_EVI_withQA',
            'MCD15A3H_mosaic_249_filled_downsampled', # checked
            'MCD09GA_b1237_mosaic_withQA_QAapplied_filled_downsampled',
            'MCD11A1_mosaic_QAapplied_filled_downsampled_filled',
            'MCD09CMG_mosaic_QAapplied_filled_downsampled',
            'MCD12Q1_mosaic_downsampled'
        ]

        os.makedirs(output_dir, exist_ok=True)
        self.driver_dirs = self._get_driver_directories()
        self.total_channels, self.channel_mapping = self._calculate_channels_and_mapping()

        # 运行期缓存
        self._global_date_mapping = None
        self._global_valid_pixels = None
        self._global_geo_reference = None
        self._global_value_range = None  # (vmin, vmax)，来自所有驱动、每驱动随机采样100张后的交集

        logger.info(f"数据目录: {data_dir}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"目标年份: {self.target_years}")
        logger.info(f"找到 {len(self.driver_dirs)} 个驱动因素")
        logger.info(f"总通道数: {self.total_channels}")
        logger.info(f"并行线程: {self.max_workers}")
        logger.info(f"强制重新处理: {self.force_retrain}")

    # ---------------- 基础工具 ----------------
    def _get_driver_directories(self):
        driver_dirs = {}
        for root, _, files in os.walk(self.data_dir):
            if any(f.endswith('.tif') for f in files):
                rel_path = os.path.relpath(root, self.data_dir)
                if rel_path == '.':
                    continue
                rel_path = rel_path.replace(os.sep, '/')
                driver_dirs[rel_path] = root
        logger.info("找到驱动因素目录:")
        for name, path in driver_dirs.items():
            logger.info(f"  {name}: {path}")
        return driver_dirs

    def _safe_sample_file(self, driver_dir):
        files = glob.glob(os.path.join(driver_dir, '*.tif'))
        return files[0] if files else None

    def _calculate_channels_and_mapping(self):
        """计算总通道数并创建通道映射"""
        total_channels = 0
        channel_mapping = {}

        for driver_name in self.driver_order:
            if driver_name not in self.driver_dirs:
                logger.warning(f"缺少驱动目录: {driver_name}（将以 NaN 填充该段通道）")
                continue

            sample_file = self._safe_sample_file(self.driver_dirs[driver_name])
            if not sample_file:
                logger.warning(f"驱动目录无样本: {driver_name}（将以 NaN 填充该段通道）")
                continue

            data, _, _ = self._load_single_file_with_gdal(sample_file)
            if data is None:
                logger.warning(f"无法读取样本: {driver_name}（将以 NaN 填充该段通道）")
                continue

            channels = data.shape[0] if data.ndim == 3 else 1
            start_idx = total_channels
            end_idx_exclusive = start_idx + channels

            channel_mapping[driver_name] = {
                'start_idx': start_idx,
                'channels' : channels,
                'end_idx'  : end_idx_exclusive
            }

            logger.info(
                f"  {driver_name}: {channels} 个通道 (索引 {start_idx}-{end_idx_exclusive - 1})"
            )

            total_channels = end_idx_exclusive

        return total_channels, channel_mapping

    def _check_existing_years(self):
        """检查输出目录中已存在的年份数据"""
        existing_years = set()
        
        if not os.path.exists(self.output_dir):
            logger.info("输出目录不存在，将创建新目录")
            return existing_years
        
        # 扫描输出目录中的H5文件
        for filename in os.listdir(self.output_dir):
            if filename.endswith('_year_dataset.h5'):
                try:
                    # 提取年份 (格式: YYYY_year_dataset.h5)
                    year_str = filename.split('_')[0]
                    year = int(year_str)
                    if 2000 <= year <= 2024:  # 合理的年份范围
                        existing_years.add(year)
                except (ValueError, IndexError):
                    continue
        
        if existing_years:
            logger.info(f"发现已存在的年份数据: {sorted(existing_years)}")
        else:
            logger.info("未发现已存在的年份数据")
        
        return existing_years

    def _filter_target_years(self):
        """根据已存在数据过滤目标年份"""
        if self.force_retrain:
            logger.info("强制重新处理模式，将处理所有目标年份")
            return self.target_years
        
        existing_years = self._check_existing_years()
        remaining_years = [year for year in self.target_years if year not in existing_years]
        
        if not remaining_years:
            logger.info("所有目标年份的数据都已存在，无需处理")
            return []
        
        skipped_years = [year for year in self.target_years if year in existing_years]
        if skipped_years:
            logger.info(f"跳过已存在的年份: {skipped_years}")
        
        logger.info(f"需要处理的年份: {remaining_years}")
        return remaining_years

    @lru_cache(maxsize=1000)
    def _get_date_from_filename(self, filename):
        try:
            import re
            m = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
            if m:
                y, M, d = m.groups()
                return datetime(int(y), int(M), int(d))
        except Exception as e:
            logger.debug(f"从文件名提取日期失败 {filename}: {e}")
        return None

    def _load_single_file_with_gdal(self, file_path, block_size=2048):
        """
        优化的文件读取函数，支持分块读取以减少内存占用
        同时获取图像的NoData值信息
        """
        try:
            from osgeo import gdal
            ds = gdal.Open(file_path, gdal.GA_ReadOnly)
            if ds is None:
                return None, None, None
            
            bands = ds.RasterCount
            width, height = ds.RasterXSize, ds.RasterYSize
            
            # 获取NoData值信息
            nodata_values = []
            for i in range(1, bands + 1):
                band = ds.GetRasterBand(i)
                nodata_value = band.GetNoDataValue()
                nodata_values.append(nodata_value)
            
            # 对于大文件，使用分块读取
            if width * height > 10000000:  # 大于10M像素
                logger.debug(f"使用分块读取大文件: {file_path}")
                
                if bands == 1:
                    # 单波段分块读取
                    band = ds.GetRasterBand(1)
                    arr = np.zeros((height, width), dtype=band.ReadAsArray(0, 0, 1, 1).dtype)
                    
                    for y in range(0, height, block_size):
                        y_end = min(y + block_size, height)
                        for x in range(0, width, block_size):
                            x_end = min(x + block_size, width)
                            block = band.ReadAsArray(x, y, x_end - x, y_end - y)
                            arr[y:y_end, x:x_end] = block
                else:
                    # 多波段分块读取
                    arr = np.zeros((bands, height, width), dtype=np.float32)
                    for i in range(1, bands + 1):
                        band = ds.GetRasterBand(i)
                        for y in range(0, height, block_size):
                            y_end = min(y + block_size, height)
                            for x in range(0, width, block_size):
                                x_end = min(x + block_size, width)
                                block = band.ReadAsArray(x, y, x_end - x, y_end - y)
                                arr[i-1, y:y_end, x:x_end] = block
            else:
                # 小文件直接读取
                if bands == 1:
                    arr = ds.GetRasterBand(1).ReadAsArray()
                else:
                    arr = np.array([ds.GetRasterBand(i).ReadAsArray() for i in range(1, bands+1)])
            
            geo_info = {
                'geotransform': ds.GetGeoTransform(),
                'projection': ds.GetProjection(),
                'raster_size': (width, height),
                'raster_count': bands,
                'nodata_values': nodata_values
            }
            ds = None
            return arr, geo_info, nodata_values
            
        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {e}")
            return None, None, None

    def _get_all_required_dates(self, target_year):
        start = datetime(target_year, 1, 1)
        end   = datetime(target_year, 12, 31)
        dates = []
        cur = start
        while cur <= end:
            dates.append(cur)
            cur += timedelta(days=1)
        logger.info(f"年份 {target_year} 日期范围: {start:%Y-%m-%d} 到 {end:%Y-%m-%d}")
        logger.info(f"总天数: {len(dates)}")
        return dates

    def _build_global_date_mapping(self):
        logger.info("构建全局日期到文件映射（全部驱动、全部日期）...")
        mp = {}
        with tqdm(total=len(self.driver_dirs), desc="扫描驱动因素") as pbar:
            for driver_name, driver_dir in self.driver_dirs.items():
                pbar.set_description(f"扫描: {driver_name}")
                for fp in glob.glob(os.path.join(driver_dir, '*.tif')):
                    dt = self._get_date_from_filename(os.path.basename(fp))
                    if dt is None:
                        continue
                    mp.setdefault(dt, {})[driver_name] = fp
                pbar.update(1)
        logger.info(f"映射了 {len(mp)} 个日期")
        return mp

    # =========== 有效掩膜工具 ===========
    @staticmethod
    def _valid_mask_from_array(arr, invalid_values=(-9999, 255, -32768), image_nodata_values=None):
        """
        仅使用图像内置 NoData（不再使用任何预设无效值）。
        若设置了全局有效值范围（由所有驱动抽样100张TIF求交集得到），则再按照该范围进行约束。
        多波段：按首通道判断（与原风格一致）。
        """
        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr[0]
            # 如果有图像NoData值，使用第一个波段（索引0）的NoData值
            image_nodata = image_nodata_values[0] if image_nodata_values and len(image_nodata_values) > 0 else None
        else:
            # 单波段图像，使用第一个（也是唯一的）NoData值
            image_nodata = image_nodata_values[0] if image_nodata_values and len(image_nodata_values) > 0 else None
            
        # 基于图像内置 NoData
        if np.issubdtype(arr.dtype, np.floating):
            mask = ~np.isnan(arr)
            if image_nodata is not None and not np.isnan(image_nodata):
                mask &= (arr != image_nodata)
        else:
            if image_nodata is not None:
                mask = (arr != image_nodata)
            else:
                mask = np.ones_like(arr, dtype=bool)

        # 叠加全局有效值范围（若存在）
        try:
            from inspect import currentframe
            # 通过闭包外部 self 访问全局范围（静态方法内没有 self，改用调用方注入的全局变量方案不安全，这里通过属性查找）
        except Exception:
            pass
        global_range = getattr(YearOnlyDatasetGenerator, "_GLOBAL_VALUE_RANGE_IN_EFFECT", None)
        if global_range is not None:
            vmin, vmax = global_range
            if vmin is not None:
                mask &= (arr >= vmin)
            if vmax is not None:
                mask &= (arr <= vmax)
        return mask

    # ---------------- 全局有效像素（跨全部日期） ----------------
    def _get_global_valid_pixels(self, all_date_to_file):
        """
        全局有效像素：对所有驱动因素同时存在的日期：
            per_day_mask = valid(所有驱动因素) AND
        然后对日期维做 OR 聚合：
            global_mask = OR_over_days(per_day_mask)
        返回 global_mask 中的像素坐标列表；所有年份共享。
        """
        logger.info("使用所有驱动因素构建有效掩膜（仅使用图像内置NoData + 全局有效值范围）并跨日期 OR 聚合…")
        driver_names = [dn for dn in self.driver_order if dn in self.driver_dirs]
        if not driver_names:
            raise RuntimeError("未找到任何驱动目录")

        # 选择一个样本以确定尺寸和地理参考
        sample_fp = None
        for d, m in all_date_to_file.items():
            ok = all(dn in m for dn in driver_names)
            if ok:
                # 用首个驱动作为参考
                sample_fp = m[driver_names[0]]
                break
        if sample_fp is None:
            raise RuntimeError("未找到同时具备所有驱动因素的任一日期，无法构建全局有效掩膜。")

        _, geo_reference, _ = self._load_single_file_with_gdal(sample_fp)
        width, height = geo_reference['raster_size']
        height, width = int(height), int(width)

        global_mask = np.zeros((height, width), dtype=bool)
        pair_dates = [d for d in sorted(all_date_to_file.keys()) if all(dn in all_date_to_file[d] for dn in driver_names)]
        logger.info(f"用于聚合的日期数（所有驱动均存在）：{len(pair_dates)}")

        # 性能优化：采样策略 - 如果日期太多，只处理一部分
        max_dates = self.max_aggregation_dates
        if len(pair_dates) > max_dates:
            # 均匀采样
            step = len(pair_dates) // max_dates
            pair_dates = pair_dates[::step][:max_dates]
            logger.info(f"采样后日期数：{len(pair_dates)}")
        
        # 进一步优化：如果日期仍然很多，使用更激进的采样
        # if len(pair_dates) > 500:
        #     # 只处理每10个日期中的1个
        #     pair_dates = pair_dates[::10]
        #     logger.info(f"激进采样后日期数：{len(pair_dates)}")

        # 并行处理优化
        def process_single_date(date_info):
            """处理单个日期，返回该日期的掩膜（所有驱动 AND）"""
            date, driver_to_file = date_info
            
            try:
                day_mask = None
                for dn in driver_names:
                    fp = driver_to_file.get(dn)
                    if not fp:
                        return None
                    arr, _, nodata_vals = self._load_single_file_with_gdal(fp)
                    if arr is None:
                        return None
                    mask = YearOnlyDatasetGenerator._valid_mask_from_array(arr, image_nodata_values=nodata_vals)
                    day_mask = mask if day_mask is None else (day_mask & mask)
                return day_mask
                
            except Exception as e:
                logger.warning(f"处理日期 {date} 失败: {e}")
                return None

        # 准备并行处理的数据
        date_tasks = []
        for d in pair_dates:
            date_tasks.append((d, all_date_to_file[d]))

        # 并行处理
        logger.info(f"开始并行处理 {len(date_tasks)} 个日期，使用 {self.max_workers} 个线程")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_date = {
                executor.submit(process_single_date, task): task[0] 
                for task in date_tasks
            }
            
            # 收集结果并聚合 - 使用批量处理优化
            processed_count = 0
            batch_size = 10  # 批量处理掩膜
            mask_batch = []
            
            for future in tqdm(as_completed(future_to_date), 
                             total=len(future_to_date), 
                             desc="聚合全局掩膜"):
                day_mask = future.result()
                if day_mask is not None:
                    mask_batch.append(day_mask)
                
                processed_count += 1
                
                # 批量聚合掩膜
                if len(mask_batch) >= batch_size or processed_count == len(future_to_date):
                    if mask_batch:
                        # 使用numpy的高效位运算批量聚合
                        batch_array = np.array(mask_batch)
                        batch_or = np.any(batch_array, axis=0)
                        global_mask |= batch_or
                        mask_batch = []  # 清空批次
                
                # 定期垃圾回收
                if processed_count % 50 == 0:
                    gc.collect()

        elapsed_time = time.time() - start_time
        logger.info(f"聚合完成，耗时: {elapsed_time:.2f}秒")

        rows, cols = np.where(global_mask)
        valid_pixels = [(int(r), int(c)) for r, c in zip(rows, cols)]
        logger.info(f"【全局】有效像素数：{len(valid_pixels)}")
        return valid_pixels, geo_reference

    # ---------------- 采样各驱动求“全局有效值范围交集” ----------------
    def _compute_global_value_range(self, max_samples_per_driver=100, seed=42):
        """
        对每个驱动目录随机抽取至多100张 TIF：
          - 仅使用图像内置 NoData 识别有效像元；
          - 计算每张影像的数值最小/最大（多波段按首通道，与掩膜保持一致）；
        跨所有驱动、所有样本，取“最小值的最大值”与“最大值的最小值”作为全局有效值范围（交集）。
        返回 (vmin, vmax) 或 None。
        """
        import random, glob, math
        random.seed(seed)
        per_file_mins = []
        per_file_maxs = []
        for dn in [dn for dn in self.driver_order if dn in self.driver_dirs]:
            folder = self.driver_dirs[dn]
            files = glob.glob(os.path.join(folder, '*.tif'))
            if not files:
                continue
            if len(files) > max_samples_per_driver:
                files = random.sample(files, max_samples_per_driver)
            for fp in files:
                try:
                    arr, _, nodata_vals = self._load_single_file_with_gdal(fp)
                    if arr is None:
                        continue
                    mask = YearOnlyDatasetGenerator._valid_mask_from_array(arr, image_nodata_values=nodata_vals)
                    if mask is None or not np.any(mask):
                        continue
                    if arr.ndim == 3:
                        arr_use = arr[0]
                    else:
                        arr_use = arr
                    vals = arr_use[mask]
                    if vals.size == 0:
                        continue
                    # 忽略极端 NaN（防御）
                    if np.issubdtype(vals.dtype, np.floating):
                        vals = vals[~np.isnan(vals)]
                        if vals.size == 0:
                            continue
                    per_file_mins.append(np.nanmin(vals))
                    per_file_maxs.append(np.nanmax(vals))
                except Exception as e:
                    logger.debug(f"采样 {fp} 失败: {e}")
                    continue
        if not per_file_mins or not per_file_maxs:
            logger.warning("未能从样本中计算有效值范围，将不启用全局范围约束。")
            return None
        vmin = max(per_file_mins)
        vmax = min(per_file_maxs)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            logger.warning(f"采样得到的全局范围异常：vmin={vmin}, vmax={vmax}，将不启用全局范围约束。")
            return None
        logger.info(f"全局有效值范围（交集）：[{vmin}, {vmax}]（基于所有驱动×每驱动≤100样本）")
        return (float(vmin), float(vmax))

    # ---------------- Pass-1: 统计“通道×时间”的非NaN比例，筛选像素 ----------------
    def _count_valid_channels_for_day(self, date_files, rows, cols):
        P = rows.shape[0]
        per_day_counts = np.zeros(P, dtype=np.uint16)

        def load_one(driver_name):
            fp = date_files.get(driver_name, None)
            if fp is None:
                return driver_name, None
            arr, _, nodata_vals = self._load_single_file_with_gdal(fp)
            return driver_name, arr, nodata_vals

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(load_one, dn): dn for dn in self.driver_order}
            for fut in as_completed(futs):
                dn = futs[fut]
                try:
                    _, arr, nodata_vals = fut.result()
                except Exception as e:
                    logger.error(f"[{dn}] 读取失败: {e}")
                    arr = None
                    nodata_vals = None
                if arr is None:
                    continue

                # 使用与全局有效像素相同的掩膜逻辑
                if arr.ndim == 2:
                    a = arr[rows, cols]
                    mask = YearOnlyDatasetGenerator._valid_mask_from_array(a, image_nodata_values=nodata_vals)
                    per_day_counts += mask.astype(np.uint16)
                elif arr.ndim == 3:
                    sub = arr[:, rows, cols]
                    valid_ch = np.zeros((sub.shape[0], sub.shape[1]), dtype=np.uint16)
                    for i in range(sub.shape[0]):
                        band_nodata = nodata_vals[i] if i < len(nodata_vals) else None
                        band_mask = YearOnlyDatasetGenerator._valid_mask_from_array(sub[i], image_nodata_values=[band_nodata])
                        valid_ch[i] = band_mask.astype(np.uint16)
                    per_day_counts += valid_ch.sum(axis=0)
                del arr
        return per_day_counts

    def _select_pixels_by_non_nan_ratio(self, all_dates, date_map, rows, cols, C):
        T = len(all_dates)
        total_needed = C * T
        total_counts = np.zeros(rows.shape[0], dtype=np.uint32)

        for d in tqdm(all_dates, desc="Pass-1 统计有效比例"):
            date_files = date_map.get(d, {})
            per_day = self._count_valid_channels_for_day(date_files, rows, cols)
            total_counts += per_day.astype(np.uint32)
            del per_day
            if (d.timetuple().tm_yday % 3) == 0:
                gc.collect()

        ratio = total_counts / float(total_needed)
        keep_mask = ratio > 0.1
        kept = int(keep_mask.sum())
        logger.info(f"筛选后像素数: {kept}/{rows.shape[0]}（non_nan_ratio > 0.1）")
        return keep_mask, ratio

    # ---------------- Pass-2: 写入（仅保留像素） ----------------
    class _DSCache(OrderedDict):
        def __init__(self, h5, capacity=4096):
            super().__init__()
            self.h5 = h5
            self.capacity = capacity
        def get_ds(self, name):
            ds = super().get(name)
            if ds is not None:
                self.move_to_end(name)
                return ds
            ds = self.h5[name]
            super().__setitem__(name, ds)
            if len(self) > self.capacity:
                self.popitem(last=False)
            return ds

    def _gather_vals_for_day(self, date_files, rows_keep, cols_keep):
        C = self.total_channels
        Pk = rows_keep.shape[0]
        vals = np.full((C, Pk), 0, dtype=np.float32)

        def load_one(driver_name):
            fp = date_files.get(driver_name, None)
            if fp is None:
                return driver_name, None
            arr, _, _ = self._load_single_file_with_gdal(fp)
            return driver_name, arr

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(load_one, dn): dn for dn in self.driver_order}
            for fut in as_completed(futs):
                dn = futs[fut]
                info = self.channel_mapping.get(dn)
                if info is None:
                    continue
                s, e = info['start_idx'], info['end_idx']
                try:
                    _, arr = fut.result()
                except Exception as e:
                    logger.error(f"[{dn}] 读取失败: {e}")
                    arr = None
                if arr is None:
                    continue

                if arr.ndim == 2:
                    # 直接保存原始值
                    processed_data = arr[rows_keep, cols_keep]
                    vals[s, :] = processed_data
                elif arr.ndim == 3:
                    sub = arr[:, rows_keep, cols_keep]
                    # 直接保存原始值
                    processed_sub = sub
                    vals[s:e, :] = processed_sub
                del arr
        return vals

    def generate_dataset_for_year(self, target_year):
        logger.info(f"开始生成年份 {target_year} 的完整数据集（保持原 H5 结构）...")

        # 检查输出文件是否已存在
        output_file = os.path.join(self.output_dir, f'{target_year}_year_dataset.h5')
        if os.path.exists(output_file) and not self.force_retrain:
            logger.info(f"年份 {target_year} 的数据集已存在，跳过: {output_file}")
            return

        if self._global_date_mapping is None:
            self._global_date_mapping = self._build_global_date_mapping()
        date_map = self._global_date_mapping

        # —— 全局有效像素（仅计算一次）：
        if self._global_valid_pixels is None or self._global_geo_reference is None:
            valid_pixels, geo_reference = self._get_global_valid_pixels(date_map)
            if not valid_pixels:
                logger.error("全局有效像素为空，终止。")
                return
            self._global_valid_pixels = valid_pixels
            self._global_geo_reference = geo_reference
        else:
            valid_pixels = self._global_valid_pixels
            geo_reference = self._global_geo_reference

        all_dates = self._get_all_required_dates(target_year)

        # 影像尺寸
        if not geo_reference:
            logger.error("无法确定影像尺寸信息")
            return
        width, height = geo_reference['raster_size']
        height, width = int(height), int(width)

        rows_all = np.array([r for r, _ in valid_pixels], dtype=np.int32)
        cols_all = np.array([c for _, c in valid_pixels], dtype=np.int32)
        P_all = rows_all.shape[0]
        T = len(all_dates)
        C = self.total_channels

        # -------- Pass-1：按“通道×时间”统计非NaN比例，筛选像素 --------
        keep_mask, ratio = self._select_pixels_by_non_nan_ratio(all_dates, date_map, rows_all, cols_all, C)
        rows = rows_all[keep_mask]
        cols = cols_all[keep_mask]
        P = rows.shape[0]
        if P == 0:
            logger.error("筛选后无像素，终止")
            return

        # 输出文件路径（已在前面检查过存在性）
        output_file = os.path.join(self.output_dir, f'{target_year}_year_dataset.h5')

        logger.info(f"创建输出文件: {output_file}")
        with h5py.File(output_file, 'w', libver='latest') as h5_file:
            # 文件级属性
            h5_file.attrs['year'] = str(target_year)
            h5_file.attrs['total_time_steps'] = T
            h5_file.attrs['total_channels'] = C
            h5_file.attrs['data_format'] = 'channels_x_time'
            h5_file.attrs['data_type'] = 'year_pixel_time_series'
            h5_file.attrs['driver_names'] = self.driver_order
            h5_file.attrs['total_pixels'] = int(P_all)
            h5_file.attrs['processed_pixels'] = int(P)
            h5_file.attrs['start_date'] = all_dates[0].strftime('%Y-%m-%d')
            h5_file.attrs['end_date'] = all_dates[-1].strftime('%Y-%m-%d')
            for driver_name, channel_info in self.channel_mapping.items():
                h5_file.attrs[f'channel_mapping_{driver_name}'] = f"{channel_info['start_idx']}-{channel_info['end_idx']-1}"
            if geo_reference:
                h5_file.attrs['geotransform'] = geo_reference['geotransform']
                h5_file.attrs['projection'] = geo_reference['projection']
                h5_file.attrs['raster_size'] = geo_reference['raster_size']
                h5_file.attrs['raster_count'] = geo_reference['raster_count']

            logger.info("创建每像素 dataset（仅保留像素）...")
            for r, c in tqdm(zip(rows, cols), total=P, desc="创建像素 datasets"):
                name = f"{int(r)}_{int(c)}"
                ds = h5_file.create_dataset(
                    name,
                    shape=(C, T),
                    dtype=np.float32,
                    chunks=(C, 1),  # 按天写
                    compression='gzip',
                    compression_opts=6,
                    shuffle=True
                )
                ds.attrs['pixel_coord'] = (int(r), int(c))
                ds.attrs['data_shape'] = f"{C}x{T}"
            h5_file.flush()

            # tile 写入
            tile_h, tile_w = 512, 512
            tiles = []
            for y0, y1, x0, x1 in self._iter_tiles(height, width, tile_h, tile_w):
                mask = (rows >= y0) & (rows < y1) & (cols >= x0) & (cols < x1)
                idxs = np.where(mask)[0]
                if idxs.size == 0:
                    continue
                local_rows = (rows[idxs] - y0).astype(np.int32, copy=False)
                local_cols = (cols[idxs] - x0).astype(np.int32, copy=False)
                tiles.append((y0, y1, x0, x1, idxs, local_rows, local_cols))

            logger.info(f"需要处理的 tile 数：{len(tiles)}")

            for (y0, y1, x0, x1, idxs, lrs, lcs) in tqdm(tiles, desc=f"{target_year} Tile 写入"):
                P_tile = idxs.size
                buf = np.empty((C, T, P_tile), dtype=np.float32)

                for t, d in enumerate(all_dates):
                    date_files = self._global_date_mapping.get(d, {})
                    day_tile = self._build_day_tile_stack_window(date_files, y0, y1, x0, x1)  # [C, h, w]
                    buf[:, t, :] = day_tile[:, lrs, lcs]
                    del day_tile
                    if (t + 1) % 8 == 0:
                        gc.collect()

                for k, idx in enumerate(idxs):
                    r = int(rows[idx]); c = int(cols[idx])
                    name = f"{r}_{c}"
                    ds = h5_file[name]
                    ds[:, :] = buf[:, :, k]

                del buf
                gc.collect()

        logger.info(f"年份 {target_year} 完整数据集生成完成: {output_file}")
        logger.info(f"成功处理 {P}/{P_all} 个像素")
        self._print_dataset_info(output_file)

    def generate_all_datasets(self):
        logger.info(f"开始生成 {len(self.target_years)} 个年份的完整数据集...")
        
        # 过滤需要处理的年份
        years_to_process = self._filter_target_years()
        if not years_to_process:
            logger.info("所有目标年份的数据都已存在，程序结束")
            return
        
        logger.info(f"实际需要处理的年份: {years_to_process} (共{len(years_to_process)}个)")
        
        # 先构建一次全局日期映射与全局有效像素
        if self._global_date_mapping is None:
            self._global_date_mapping = self._build_global_date_mapping()
        # 计算全局有效值范围（仅一次）
        if self._global_value_range is None:
            self._global_value_range = self._compute_global_value_range()
        # 将范围注入静态方法可访问的类级变量（在本次进程内生效）
        YearOnlyDatasetGenerator._GLOBAL_VALUE_RANGE_IN_EFFECT = self._global_value_range

        if self._global_valid_pixels is None or self._global_geo_reference is None:
            vp, gr = self._get_global_valid_pixels(self._global_date_mapping)
            if not vp:
                raise RuntimeError("全局有效像素为空，无法继续。")
            self._global_valid_pixels = vp
            self._global_geo_reference = gr

        for year_idx, target_year in enumerate(years_to_process):
            logger.info(f"\n{'='*60}")
            logger.info(f"处理年份 {target_year} ({year_idx + 1}/{len(years_to_process)})")
            logger.info(f"{'='*60}")
            start_time = time.time()
            self.generate_dataset_for_year(target_year)
            end_time = time.time()
            logger.info(f"年份 {target_year} 完成，耗时: {end_time - start_time:.2f} 秒")
        logger.info(f"\n所有年份数据集生成完成！")

    def _print_dataset_info(self, output_file):
        try:
            with h5py.File(output_file, 'r') as f:
                print(f"\n数据集信息:")
                print(f"="*50)
                print(f"文件: {output_file}")
                print(f"年份: {f.attrs.get('year', 'N/A')}")
                print(f"日期范围: {f.attrs.get('start_date', 'N/A')} ~ {f.attrs.get('end_date', 'N/A')}")
                print(f"数据集数量(像素数): {len(f.keys())}")
                print(f"总通道数: {f.attrs.get('total_channels', 'N/A')}")
                print(f"总时间步数: {f.attrs.get('total_time_steps', 'N/A')}")
                print(f"数据格式: {f.attrs.get('data_format', 'N/A')}")
                print(f"\n通道映射:")
                for key in f.attrs.keys():
                    if key.startswith('channel_mapping_'):
                        driver_name = key.replace('channel_mapping_', '')
                        mapping = f.attrs[key]
                        print(f"  {driver_name}: {mapping}")
                if len(f.keys()) > 0:
                    first_key = next(iter(f.keys()))
                    first_dataset = f[first_key]
                    print(f"\n样本数据集 ({first_key}): 形状={first_dataset.shape}, dtype={first_dataset.dtype}")
                    arr = first_dataset[:, :]
                    print(f"  数据范围: {np.nanmin(arr)} ~ {np.nanmax(arr)}")
                print(f"="*50)
        except Exception as e:
            logger.error(f"打印数据集信息失败: {e}")
            
    def _iter_tiles(self, height, width, tile_h=96, tile_w=96):
        y = 0
        while y < height:
            x = 0
            y1 = min(y + tile_h, height)
            while x < width:
                x1 = min(x + tile_w, width)
                yield y, y1, x, x1
                x = x1
            y = y1

    def _build_day_tile_stack_window(self, date_files, y0, y1, x0, x1):
        """
        读取"某一天"的所有驱动在 [y0:y1, x0:x1] 窗口内的数据，合成为 [C, h, w]。
        存储时：已归一化数据(除MCD14A1和MCD12Q1)乘以255取整，MCD14A1和MCD12Q1保留原值。
        """
        h = y0.__class__(y1 - y0) if isinstance(y0, np.generic) else (y1 - y0)
        w = x0.__class__(x1 - x0) if isinstance(x0, np.generic) else (x1 - x0)
        h = int(h); w = int(w)
        C = self.total_channels
        tile = np.full((C, h, w), 0, dtype=np.float32)

        def read_window(fp):
            from osgeo import gdal
            ds = gdal.Open(fp, gdal.GA_ReadOnly)
            if ds is None:
                return None
            bands = ds.RasterCount
            if bands == 1:
                rb = ds.GetRasterBand(1)
                arr = rb.ReadAsArray(int(x0), int(y0), int(w), int(h))
            else:
                arrs = []
                for i in range(1, bands + 1):
                    rb = ds.GetRasterBand(i)
                    arrs.append(rb.ReadAsArray(int(x0), int(y0), int(w), int(h)))
                arr = np.array(arrs)
            ds = None
            return arr

        for dn in self.driver_order:
            info = self.channel_mapping.get(dn)
            if info is None:
                continue
            s, e = info['start_idx'], info['end_idx']
            fp = date_files.get(dn)
            if fp is None:
                continue
            arr = read_window(fp)
            if arr is None:
                continue
            if arr.ndim == 2:
                if arr.shape != (h, w):
                    continue
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                if arr.shape[1:] != (h, w):
                    continue
                if arr.shape[0] != (e - s):
                    arr = arr[:(e - s), ...]
            
            # 直接保存原始值
            processed_arr = arr
            
            tile[s:s+processed_arr.shape[0], :, :] = processed_arr
            del arr
        return tile


# ---------------- CLI ----------------
def parse_year_list(year_string):
    try:
        return ast.literal_eval(year_string)
    except:
        try:
            return [int(x.strip()) for x in year_string.split(',')]
        except:
            return [int(year_string)]

def main():
    import argparse
    parser = argparse.ArgumentParser(description='生成年份范围数据集 - 两阶段筛选+流式写入（保持原 H5 结构）')
    parser.add_argument('--data_dir', 
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm',
                       help='数据根目录')
    parser.add_argument('--output_dir',
                       default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/newset_dataset_v7',
                       help='输出目录')
    parser.add_argument('--years', type=str, default='[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]',
                       help='目标年份列表，格式：[2000,2021,...] 或 2000,2021,...')
    parser.add_argument('--max_workers', type=int, default=12,
                       help='单日驱动读取的并行线程数')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='兼容参数，无实际用途')
    parser.add_argument('--force-retrain', action='store_true',
                       help='强制重新处理所有年份，即使数据已存在')
    args = parser.parse_args()

    target_years = parse_year_list(args.years)
    generator = YearOnlyDatasetGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_years=target_years,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        force_retrain=args.force_retrain
    )
    generator.generate_all_datasets()

if __name__ == "__main__":
    main()
