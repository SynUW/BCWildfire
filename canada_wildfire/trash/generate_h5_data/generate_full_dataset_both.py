#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified HDF5 Generator (Pixel-level & Patch-level) with NoData Normalization
- Choose mode: pixel or patch
- Pixel mode: X(N, C, T), coords(N,2)
- Patch mode: X(N_patches, C, T, P, P), patch_coords(N_patches, 4)=(r0,c0,r1,c1)
- Streamed writing by time to control memory
- NoData normalization during writing:
    * Replace any -999 -> -9999
    * FIRMS (channel 0): 255 -> -9999
    * LAI (last channel): if LAI==0 AND all other channels are invalid (-9999) at the same pixel,
      set LAI to -9999; otherwise keep 0.
    * After rules, NaN -> -9999
- Driver order defaults to *_10x 版本（与现有数据匹配）

Last update: 2025-08-26

整合了pixel和patch两种模式的数据集生成并且加入了NoData规范化，统一为-9999。

逐点生成：
python unified_generator.py \
  pixel \
  --data_dir /mnt/raid/.../all_data_masked_10x_masked \
  --output_dir /mnt/raid/.../all_data_masked_result_10x_QAapplied_unified \
  --years [2019,2020,2021,2022,2023,2024]

patch生成，包括非重叠+重叠：
python unified_generator.py \
  patch \
  --data_dir /mnt/raid/.../all_data_masked_10x_masked \
  --output_dir /mnt/raid/.../all_data_masked_result_10x_QAapplied_patch_unified \
  --years_nonoverlap [2019,2020,2021,2022,2023,2024] \
  --years_overlap 2024 \
  --patch_size 140 --overlap_stride 70
  
注意，FIRMS 为第 0 通道，LAI 为最后一通道：无效值规范化依赖该假设。

"""

import os
import sys
import glob
import ast
import gc
import h5py
import json
import math
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

# =================== Tunables & Env ===================
# HDF5 compression
H5_COMPRESSION = 'gzip'   # 'gzip' | 'lzf' | None
H5_GZIP_LEVEL = 6
H5_PIXEL_CHUNK_ROWS = 8192   # pixel mode: chunk on N
H5_PATCH_CHUNK_N    = 64     # patch mode: chunk on N

# GDAL cache & threads
os.environ.setdefault("GDAL_CACHEMAX", "512")     # MB
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
# ======================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UnifiedGen")


# ----------------- Common helpers -----------------

def parse_year_list(year_string):
    """Parse '[2000,2001]' or '2000,2001' or '2000' to list[int]"""
    try:
        v = ast.literal_eval(year_string)
        if isinstance(v, int):
            return [v]
        return list(v)
    except Exception:
        try:
            return [int(x.strip()) for x in str(year_string).split(',') if x.strip()]
        except Exception:
            return [int(year_string)]

def get_date_from_filename(name):
    """Extract date YYYY_MM_DD from filename"""
    import re
    m = re.search(r'(\d{4})_(\d{2})_(\d{2})', name)
    if not m:
        return None
    y, M, d = map(int, m.groups())
    try:
        return datetime(y, M, d)
    except Exception:
        return None

def date_list_for_year(year:int):
    """All dates in a year"""
    start = datetime(year,1,1)
    end = datetime(year,12,31)
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out

def find_tifs(root_dir):
    """Walk and collect directories that contain .tif; return {rel_path: abs_path}"""
    dirs = {}
    for r, _, fnames in os.walk(root_dir):
        if any(f.endswith('.tif') for f in fnames):
            rel = os.path.relpath(r, root_dir)
            if rel == '.':
                continue
            rel = rel.replace(os.sep, '/')
            dirs[rel] = r
    return dirs

def build_date_map(driver_dirs):
    """Return {date: {driver_rel_path: file_path}} by scanning tif files"""
    dm = {}
    for drv_name, drv_dir in tqdm(driver_dirs.items(), desc="Scan drivers"):
        for fp in glob.glob(os.path.join(drv_dir, '*.tif')):
            dt = get_date_from_filename(os.path.basename(fp))
            if dt is None:
                continue
            dm.setdefault(dt, {})[drv_name] = fp
    logger.info(f"Mapped {len(dm)} dates.")
    return dm

def infer_channels_and_size(driver_order, driver_dirs):
    """Probe first tif in each present driver to get channel counts; also return (H,W) from first found."""
    from osgeo import gdal
    total = 0
    mapping = {}
    H = W = None
    for drv in driver_order:
        if drv not in driver_dirs:
            logger.warning(f"Missing driver dir: {drv}")
            continue
        files = glob.glob(os.path.join(driver_dirs[drv], '*.tif'))
        if not files:
            logger.warning(f"No tif in driver: {drv}")
            continue
        sample = files[0]
        ds = gdal.Open(sample, gdal.GA_ReadOnly)
        if ds is None:
            logger.warning(f"GDAL open failed: {sample}")
            continue
        bands = max(1, ds.RasterCount)
        if H is None or W is None:
            H, W = ds.RasterYSize, ds.RasterXSize
        ds = None
        mapping[drv] = {'start_idx': total, 'channels': bands, 'end_idx': total+bands}
        logger.info(f"{drv}: {bands} bands (ch idx {total}-{total+bands-1})")
        total += bands
    if H is None or W is None:
        raise RuntimeError("Failed to infer raster size; no readable sample tif.")
    return total, mapping, (H, W)

# ----------------- NoData normalization logic -----------------

def normalize_nodata_pixel_daybuf(day_buf):
    """
    day_buf: (C, N) float32
    Rules:
      - Any -999 -> -9999
      - FIRMS channel 0: 255 -> -9999
      - LAI last channel: if LAI==0 and all other channels == -9999 at the same column, set LAI=-9999
      - NaN -> -9999
    """
    C, N = day_buf.shape
    if C == 0 or N == 0:
        return day_buf

    # Replace -999 -> -9999
    day_buf[day_buf == -999] = -9999

    # FIRMS at 0
    firms = day_buf[0]
    firms[firms == 255] = -9999

    # LAI last channel
    lai = day_buf[C-1]
    # others invalid mask per column
    others = day_buf[0:C-1, :]
    # after FIRMS rule and -999 mapped
    others_invalid_all = np.all(others == -9999, axis=0)
    lai_zero = (lai == 0)
    lai[(lai_zero) & (others_invalid_all)] = -9999

    # NaN -> -9999 (apply at the end)
    np.nan_to_num(day_buf, copy=False, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
    return day_buf

def normalize_nodata_patch_buf(buf):
    """
    buf: (C, P, P) float32
    Same rules per pixel:
      - -999 -> -9999
      - FIRMS (0): 255 -> -9999
      - LAI (C-1): if LAI==0 and all other channels == -9999 at the same (r,c), set LAI=-9999
      - NaN -> -9999
    """
    C, P, _ = buf.shape
    if C == 0 or P == 0:
        return buf

    # -999 -> -9999
    buf[buf == -999] = -9999

    # FIRMS
    firms = buf[0]
    firms[firms == 255] = -9999

    # LAI
    lai = buf[C-1]
    # others per-pixel invalid mask
    others = buf[0:C-1]  # (C-1, P, P)
    others_invalid_all = np.all(others == -9999, axis=0)  # (P, P)
    lai_zero = (lai == 0)
    lai[(lai_zero) & (others_invalid_all)] = -9999

    # NaN -> -9999
    np.nan_to_num(buf, copy=False, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
    return buf

# ============================================================
#                        PIXEL MODE
# (基于你之前的逐点流式生成器，按日写列，加入NoData规范化)  :contentReference[oaicite:2]{index=2}
# ============================================================

class PixelGenerator:
    def __init__(self, data_dir, output_dir, years, driver_order):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.years = years
        self.driver_order = driver_order

        self.driver_dirs = find_tifs(self.data_dir)
        self.C_total, self.chan_map, (self.H, self.W) = infer_channels_and_size(self.driver_order, self.driver_dirs)
        self.date_map = build_date_map(self.driver_dirs)

    def _read_raster_values_at(self, path, rows, cols):
        """read full then index -> (C_d, N)"""
        from osgeo import gdal
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"GDAL open failed: {path}")
        B = max(1, ds.RasterCount)
        if B == 1:
            arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32, copy=False)  # (H,W)
            vals = arr[rows, cols][None, :]
        else:
            H, W = ds.RasterYSize, ds.RasterXSize
            stack = np.empty((B, H, W), dtype=np.float32)
            for b in range(B):
                stack[b] = ds.GetRasterBand(b+1).ReadAsArray().astype(np.float32, copy=False)
            vals = stack[:, rows, cols]
        ds = None
        return vals

    def _valid_pixels_from_midday(self, year):
        from osgeo import gdal
        target_date = datetime(year, 6, 15)
        closest, mdiff = None, 1e9
        for d in self.date_map.keys():
            if d.year == year:
                df = abs((d - target_date).days)
                if df < mdiff:
                    mdiff = df
                    closest = d
        if closest is None:
            logger.error(f"Year {year}: no date found to build valid mask.")
            return [], None

        # Prefer FIRMS; else first available
        valid_driver = None
        for drv in self.driver_order:
            if drv in self.date_map.get(closest, {}):
                valid_driver = drv
                break
        if valid_driver is None:
            logger.error("No driver to build valid mask.")
            return [], None

        fp = self.date_map[closest][valid_driver]
        ds = gdal.Open(fp, gdal.GA_ReadOnly)
        if ds is None:
            logger.error(f"GDAL open failed: {fp}")
            return [], None
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32, copy=False)
        geo = {
            'geotransform': ds.GetGeoTransform(),
            'projection': ds.GetProjection(),
            'raster_size': (ds.RasterXSize, ds.RasterYSize),
            'raster_count': ds.RasterCount
        }
        ds = None
        valid_mask = (~np.isnan(arr)) & (arr != 255) & (arr != -9999)
        rows, cols = np.where(valid_mask)
        pixels = [(int(r), int(c)) for r,c in zip(rows, cols)]
        logger.info(f"Year {year}: valid pixels = {len(pixels)}")
        return pixels, geo

    def _write_year(self, year):
        dates = date_list_for_year(year)
        T = len(dates)
        pixels, geo = self._valid_pixels_from_midday(year)
        if not pixels:
            return
        rows = np.array([r for r,c in pixels], dtype=np.int32)
        cols = np.array([c for r,c in pixels], dtype=np.int32)
        N = rows.shape[0]
        C = self.C_total

        out_path = os.path.join(self.output_dir, f"{year}_year_dataset.h5")
        if os.path.exists(out_path):
            logger.info(f"Exists, remove: {out_path}")
            os.remove(out_path)

        with h5py.File(out_path, 'w') as f:
            # attrs
            f.attrs['year'] = str(year)
            f.attrs['total_time_steps'] = int(T)
            f.attrs['total_channels'] = int(C)
            f.attrs['data_format'] = 'N_by_C_by_T'
            f.attrs['data_type'] = 'year_pixel_time_series'
            f.attrs['driver_names'] = json.dumps(self.driver_order)
            f.attrs['start_date'] = dates[0].strftime('%Y-%m-%d')
            f.attrs['end_date'] = dates[-1].strftime('%Y-%m-%d')
            if geo:
                f.attrs['geotransform'] = geo['geotransform']
                f.attrs['projection'] = geo['projection']
                f.attrs['raster_size'] = geo['raster_size']
                f.attrs['raster_count'] = geo['raster_count']
            for drv, info in self.chan_map.items():
                f.attrs[f'channel_mapping_{drv}'] = f"{info['start_idx']}-{info['end_idx']-1}"

            f.create_dataset('coords', data=np.stack([rows, cols], axis=1), dtype=np.int32)
            chunks = (min(H5_PIXEL_CHUNK_ROWS, N), C, 1)
            create_args = dict(shape=(N, C, T), dtype=np.float32, chunks=chunks, shuffle=True)
            if H5_COMPRESSION == 'gzip':
                create_args.update(dict(compression='gzip', compression_opts=int(H5_GZIP_LEVEL)))
            elif H5_COMPRESSION == 'lzf':
                create_args.update(dict(compression='lzf'))
            X = f.create_dataset('X', **create_args)

            # write by day
            for t, d in enumerate(tqdm(dates, desc=f"Year {year} [pixel] writing by day")):
                day_buf = np.full((C, N), -9999.0, dtype=np.float32)  # prefill -9999
                ch_off = 0
                day_files = self.date_map.get(d, {})
                for drv in self.driver_order:
                    ch = self.chan_map.get(drv, {}).get('channels', 0)
                    if ch <= 0:
                        continue
                    fp = day_files.get(drv)
                    if not fp:
                        ch_off += ch
                        continue
                    vals = self._read_raster_values_at(fp, rows, cols)  # (C_d, N)
                    # place
                    place_ch = min(ch, vals.shape[0])
                    day_buf[ch_off:ch_off+place_ch, :] = vals[:place_ch, :]
                    ch_off += ch

                # NoData normalization
                day_buf = normalize_nodata_pixel_daybuf(day_buf)
                # write (N,C) slice for day t
                X[:, :, t] = day_buf.T
                if (t+1) % 16 == 0:
                    f.flush()
            f.flush()
        logger.info(f"Pixel H5 done: {out_path}")

    def run(self):
        for y in self.years:
            self._write_year(y)


# ============================================================
#                        PATCH MODE
# (基于你之前的patch生成器：非重叠/重叠，加入NoData规范化)  :contentReference[oaicite:3]{index=3}
# ============================================================

def make_nonoverlap_grid(H, W, P, edge_min_frac=2.0/3.0):
    starts_r = list(range(0, max(0, H - P + 1), P))
    rem_h = H - (starts_r[-1] + P) if starts_r else H
    if rem_h >= edge_min_frac * P:
        starts_r.append(starts_r[-1] + P if starts_r else 0)
    starts_c = list(range(0, max(0, W - P + 1), P))
    rem_w = W - (starts_c[-1] + P) if starts_c else W
    if rem_w >= edge_min_frac * P:
        starts_c.append(starts_c[-1] + P if starts_c else 0)
    return [(r0, c0) for r0 in starts_r for c0 in starts_c]

def make_overlap_grid(H, W, P, S):
    def axis_starts(L):
        xs = []
        pos = 0
        while pos <= max(L - P, 0):
            xs.append(pos)
            pos += S
        if not xs or xs[-1] < max(L - P, 0):
            xs.append(max(L - P, 0))
        return xs
    rows = axis_starts(H)
    cols = axis_starts(W)
    return [(r0, c0) for r0 in rows for c0 in cols]

def pad_window(arr2d, r0, c0, P, mode='reflect', constant_value=0):
    H, W = arr2d.shape
    r1 = r0 + P
    c1 = c0 + P
    r0c = max(0, r0); c0c = max(0, c0)
    r1c = min(H, r1); c1c = min(W, c1)
    sub = arr2d[r0c:r1c, c0c:c1c]
    out = np.zeros((P, P), dtype=arr2d.dtype)
    off_r = r0c - r0
    off_c = c0c - c0
    out[off_r:off_r + sub.shape[0], off_c:off_c + sub.shape[1]] = sub
    # pad top
    if off_r > 0:
        if mode == 'reflect':
            out[0:off_r, :] = out[off_r:2*off_r, :][::-1, :]
        elif mode == 'replicate':
            out[0:off_r, :] = out[off_r:off_r+1, :]
        else:
            out[0:off_r, :] = constant_value
    # pad bottom
    extra_r = P - (off_r + sub.shape[0])
    if extra_r > 0:
        if mode == 'reflect':
            out[P-extra_r:P, :] = out[P-extra_r- extra_r:P-extra_r, :][::-1, :]
        elif mode == 'replicate':
            out[P-extra_r:P, :] = out[P-extra_r-1:P-extra_r, :]
        else:
            out[P-extra_r:P, :] = constant_value
    # pad left
    if off_c > 0:
        if mode == 'reflect':
            out[:, 0:off_c] = out[:, off_c:2*off_c][:, ::-1]
        elif mode == 'replicate':
            out[:, 0:off_c] = out[:, off_c:off_c+1]
        else:
            out[:, 0:off_c] = constant_value
    # pad right
    extra_c = P - (off_c + sub.shape[1])
    if extra_c > 0:
        if mode == 'reflect':
            out[:, P-extra_c:P] = out[:, P-extra_c- extra_c:P-extra_c][:, ::-1]
        elif mode == 'replicate':
            out[:, P-extra_c:P] = out[:, P-extra_c-1:P-extra_c]
        else:
            out[:, P-extra_c:P] = constant_value
    return out

def read_driver_full(path):
    from osgeo import gdal
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"GDAL open failed: {path}")
    B = max(1, ds.RasterCount)
    H, W = ds.RasterYSize, ds.RasterXSize
    if B == 1:
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32, copy=False)
        out = arr[None, ...]
    else:
        out = np.empty((B, H, W), dtype=np.float32)
        for b in range(B):
            out[b] = ds.GetRasterBand(b+1).ReadAsArray().astype(np.float32, copy=False)
    ds = None
    return out

class PatchGenerator:
    def __init__(self, data_dir, output_dir,
                 years_nonoverlap, years_overlap,
                 driver_order,
                 patch_size=256,
                 edge_min_frac=2.0/3.0,
                 overlap_stride=None,
                 padding_mode='reflect', padding_constant=0.0,
                 use_window_read=False,
                 batch_patches=2048):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.years_nonoverlap = years_nonoverlap or []
        self.years_overlap = years_overlap or []
        self.P = int(patch_size)
        self.edge_min_frac = float(edge_min_frac)
        self.S_overlap = int(overlap_stride) if overlap_stride is not None else max(1, self.P // 2)
        self.padding_mode = padding_mode
        self.padding_constant = float(padding_constant)
        self.use_window_read = bool(use_window_read)
        self.batch_patches = int(batch_patches)

        self.driver_order = driver_order
        self.driver_dirs = find_tifs(self.data_dir)
        self.C_total, self.chan_map, (self.H, self.W) = infer_channels_and_size(self.driver_order, self.driver_dirs)
        self.date_map = build_date_map(self.driver_dirs)

    def _anchors_and_coords(self, kind='nonoverlap'):
        if kind == 'nonoverlap':
            anchors = make_nonoverlap_grid(self.H, self.W, self.P, self.edge_min_frac)
        else:
            anchors = make_overlap_grid(self.H, self.W, self.P, self.S_overlap)
        coords = []
        for (r0, c0) in anchors:
            r1 = r0 + self.P - 1
            c1 = c0 + self.P - 1
            coords.append((r0, c0, r1, c1))
        coords = np.asarray(coords, dtype=np.int32)
        return anchors, coords

    def _open_h5(self, out_path, N, T, kind, year, dates):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            os.remove(out_path)
        f = h5py.File(out_path, 'w')
        chunks = (min(H5_PATCH_CHUNK_N, N), self.C_total, 1, self.P, self.P)
        create_args = dict(shape=(N, self.C_total, T, self.P, self.P),
                           dtype=np.float32,
                           chunks=chunks,
                           shuffle=True)
        if H5_COMPRESSION == 'gzip':
            create_args.update(dict(compression='gzip', compression_opts=int(H5_GZIP_LEVEL)))
        elif H5_COMPRESSION == 'lzf':
            create_args.update(dict(compression='lzf'))
        X = f.create_dataset('X', **create_args)

        f.attrs['year'] = str(year)
        f.attrs['total_time_steps'] = int(T)
        f.attrs['total_channels'] = int(self.C_total)
        f.attrs['patch_size'] = int(self.P)
        f.attrs['stride_type'] = 'non_overlap' if kind == 'nonoverlap' else 'overlap'
        if kind == 'overlap':
            f.attrs['overlap_stride'] = int(self.S_overlap)
        f.attrs['edge_min_frac'] = float(self.edge_min_frac)
        f.attrs['raster_size'] = (int(self.W), int(self.H))
        f.attrs['start_date'] = dates[0].strftime('%Y-%m-%d') if T>0 else ''
        f.attrs['end_date']   = dates[-1].strftime('%Y-%m-%d') if T>0 else ''
        f.attrs['padding_mode'] = self.padding_mode
        f.attrs['padding_constant'] = float(self.padding_constant)
        f.attrs['driver_order'] = json.dumps(self.driver_order)
        for drv, info in self.chan_map.items():
            f.attrs[f'channel_mapping_{drv}'] = f"{info['start_idx']}-{info['end_idx']-1}"
        return f, X

    def _assemble_patch_from_full(self, drv_arrays, r0, c0):
        P = self.P
        out = np.full((self.C_total, P, P), -9999.0, dtype=np.float32)
        ch_off = 0
        for drv in self.driver_order:
            arr = drv_arrays.get(drv, None)
            ch = self.chan_map.get(drv, {}).get('channels', 0)
            if ch <= 0:
                continue
            if arr is None:
                ch_off += ch
                continue
            for b in range(min(ch, arr.shape[0])):
                patch = pad_window(arr[b], r0, c0, P, mode=self.padding_mode, constant_value=self.padding_constant)
                out[ch_off + b] = patch
            ch_off += ch
        # NoData normalization for this patch
        out = normalize_nodata_patch_buf(out)
        return out

    def _write_year(self, year:int, kind:str):
        assert kind in ('nonoverlap', 'overlap')
        P = self.P
        anchors, coords = self._anchors_and_coords(kind=kind)
        N = len(anchors)
        dates = date_list_for_year(year)
        T = len(dates)
        if N == 0 or T == 0:
            logger.warning(f"Year {year}: nothing to write (N={N}, T={T}).")
            return

        subdir = 'nonoverlap' if kind=='nonoverlap' else f'overlap_P{P}_S{self.S_overlap}'
        out_path = os.path.join(self.output_dir, subdir, f"{year}_patch_dataset.h5")
        f, X = self._open_h5(out_path, N, T, kind, year, dates)
        f.create_dataset('patch_coords', data=coords, dtype=np.int32)

        for t, d in enumerate(tqdm(dates, desc=f"Year {year} [patch {kind}] write by day")):
            # per-day arrays
            drv_full = {}
            for drv in self.driver_order:
                fp = self.date_map.get(d, {}).get(drv)
                if not fp:
                    drv_full[drv] = None
                    continue
                drv_full[drv] = read_driver_full(fp)

            # batch patches
            for i in range(0, N, 2048):
                j = min(i + 2048, N)
                batch = j - i
                buf = np.empty((batch, self.C_total, P, P), dtype=np.float32)
                for k, idx in enumerate(range(i, j)):
                    r0, c0 = anchors[idx]
                    buf[k] = self._assemble_patch_from_full(drv_full, r0, c0)
                X[i:j, :, t, :, :] = buf
            f.flush()
            drv_full = None
            gc.collect()

        f.flush()
        f.close()
        logger.info(f"Patch H5 done: {out_path}")
        try:
            with h5py.File(out_path, 'r') as rf:
                Xd = rf['X']
                logger.info(f"X shape: {Xd.shape}, patch_coords: {rf['patch_coords'].shape}")
        except Exception as e:
            logger.warning(f"Info read failed: {e}")

    def run(self):
        for y in (self.years_nonoverlap or []):
            self._write_year(y, 'nonoverlap')
        for y in (self.years_overlap or []):
            self._write_year(y, 'overlap')


# ----------------- CLI -----------------

def main():
    global H5_COMPRESSION, H5_GZIP_LEVEL
    import argparse
    parser = argparse.ArgumentParser(description="Unified H5 generator (pixel or patch) with NoData normalization.")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Common defaults for driver order (10x)
    default_driver_order = [
        'Firms_Detection_resampled_10x',  # 1 (FIRMS)
        'ERA5_multi_bands_10x',           # 2-13
        'LULC_BCBoundingbox_resampled_10x',  # 14
        'DEM_and_distance_map_interpolated_10x',  # 15-21 (根据数据)
        'NDVI_EVI_10x',          # ...
        'Reflection_500_merge_TerraAquaWGS84_clip_scaled_10x',
        'MODIS_Terra_Aqua_B20_21_merged_resampled_10x',
        'MOD21A1DN_multibands_filtered_resampled_10x',
        'LAI_BCBoundingbox_resampled_right_interpolation_clip_10x'  # LAST (LAI)
    ]

    # ---- pixel ----
    p_pixel = subparsers.add_parser('pixel', help='Generate pixel-level H5: X(N,C,T), coords(N,2)')
    p_pixel.add_argument('--data_dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_10x_without_qa')
    p_pixel.add_argument('--output_dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_result_10x_without_qa_pixel')
    p_pixel.add_argument('--years', type=str, default='[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]')
    p_pixel.add_argument('--driver_order', type=str, default=json.dumps(default_driver_order),
                         help='JSON list of driver folder names in fixed order (FIRMS first, LAI last).')

    # ---- patch ----
    p_patch = subparsers.add_parser('patch', help='Generate patch-level H5: X(N,C,T,P,P), patch_coords')
    p_patch.add_argument('--data_dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_10x_without_qa')
    p_patch.add_argument('--output_dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_result_10x_without_qa_patch')
    p_patch.add_argument('--years_nonoverlap', type=str, default='[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]')
    p_patch.add_argument('--years_overlap', type=str, default='2023,2024')
    p_patch.add_argument('--patch_size', type=int, default=140)
    p_patch.add_argument('--edge_min_frac', type=float, default=2.0/3.0)
    p_patch.add_argument('--overlap_stride', type=int, default=None, help='Stride for overlap; default P//2')
    p_patch.add_argument('--padding_mode', type=str, default='reflect', choices=['reflect', 'replicate', 'constant'])
    p_patch.add_argument('--padding_constant', type=float, default=0.0)
    p_patch.add_argument('--use_window_read', action='store_true',
                         help='(reserved, currently using full-read per day)')
    p_patch.add_argument('--batch_patches', type=int, default=2048)
    p_patch.add_argument('--driver_order', type=str, default=json.dumps(default_driver_order),
                         help='JSON list of driver folder names in fixed order (FIRMS first, LAI last).')

    # compression options
    parser.add_argument('--h5_compression', type=str, default='gzip', choices=['gzip', 'lzf', 'none'])
    parser.add_argument('--gzip_level', type=int, default=H5_GZIP_LEVEL)

    args = parser.parse_args()

    # Compression switches
    if args.h5_compression == 'none':
        H5_COMPRESSION = None
    else:
        H5_COMPRESSION = args.h5_compression
    H5_GZIP_LEVEL = int(args.gzip_level)

    # driver order
    try:
        driver_order = json.loads(getattr(args, 'driver_order'))
        assert isinstance(driver_order, list) and len(driver_order) >= 2
    except Exception as e:
        raise ValueError(f"Invalid --driver_order: {e}")

    if args.mode == 'pixel':
        years = parse_year_list(args.years)
        gen = PixelGenerator(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            years=years,
            driver_order=driver_order
        )
        gen.run()

    elif args.mode == 'patch':
        years_non = parse_year_list(args.years_nonoverlap) if args.years_nonoverlap else []
        years_ovl = parse_year_list(args.years_overlap) if args.years_overlap else []
        gen = PatchGenerator(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            years_nonoverlap=years_non,
            years_overlap=years_ovl,
            driver_order=driver_order,
            patch_size=args.patch_size,
            edge_min_frac=args.edge_min_frac,
            overlap_stride=args.overlap_stride,
            padding_mode=args.padding_mode,
            padding_constant=args.padding_constant,
            use_window_read=args.use_window_read,
            batch_patches=args.batch_patches
        )
        gen.run()

if __name__ == '__main__':
    main()
