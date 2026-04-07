#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patch Dataset Generator (Non-overlap for train/val/test, Overlap for seamless inference)
- Single HDF5 per year
- X: (N_patches, C_total, T, P, P)
- patch_coords: (N_patches, 4) with (r0, c0, r1, c1) in original image coords (before padding)
- Streamed by time axis (write one day at a time)
- Edge rule for non-overlap: if remainder >= edge_min_frac * P, add a padded edge patch (no overlap)
- Overlap set (optional): stride < P, always cover to edges (padding as needed), for inference fusion

Last update: 2025-08-26
"""

import os
import sys
import glob
import json
import math
import ast
import gc
import h5py
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

# =================== Tunables & Env ===================
# HDF5 compression
H5_COMPRESSION = 'gzip'   # 'gzip' | 'lzf' | None
H5_GZIP_LEVEL = 6
H5_CHUNK_N = 64           # N-dim chunk size for X
# GDAL cache & threads
os.environ.setdefault("GDAL_CACHEMAX", "512")     # MB
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
# ======================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PatchDatasetGen")

# ----------------- Helpers -----------------

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

def find_tifs(root_dir):
    """Walk and collect directories that contain .tif; return {rel_path: abs_path}"""
    dirs = {}
    for r, dnames, fnames in os.walk(root_dir):
        if any(f.endswith('.tif') for f in fnames):
            rel = os.path.relpath(r, root_dir)
            if rel == '.':
                continue
            rel = rel.replace(os.sep, '/')
            dirs[rel] = r
    return dirs

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

def make_nonoverlap_grid(H, W, P, edge_min_frac=2.0/3.0):
    """
    Non-overlap starts: step = P.
    If remainder >= edge_min_frac*P, add an extra edge patch (no overlap) whose window extends beyond the image and requires padding.
    Return list of (r0,c0) anchor positions in image coordinates (r0,c0 may be at image edge; width/height < P, padding will be applied).
    """
    starts_r = list(range(0, max(0, H - P + 1), P))
    rem_h = H - (starts_r[-1] + P) if starts_r else H
    if rem_h >= edge_min_frac * P:
        # Add a final anchor at last_full_end (no overlap); this window crosses the bottom edge and needs padding
        starts_r.append(starts_r[-1] + P if starts_r else 0)
    starts_c = list(range(0, max(0, W - P + 1), P))
    rem_w = W - (starts_c[-1] + P) if starts_c else W
    if rem_w >= edge_min_frac * P:
        starts_c.append(starts_c[-1] + P if starts_c else 0)

    anchors = [(r0, c0) for r0 in starts_r for c0 in starts_c]
    return anchors

def make_overlap_grid(H, W, P, S):
    """
    Overlap grid for inference. Step S (<P).
    Ensure coverage to edges by including last anchor at max(H-P, 0) / max(W-P, 0).
    """
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
    anchors = [(r0, c0) for r0 in rows for c0 in cols]
    return anchors

def pad_window(arr2d, r0, c0, P, mode='reflect', constant_value=0):
    """
    Extract window (P,P) centered at [r0:r0+P, c0:c0+P] with padding if out of bounds.
    arr2d shape (H,W). Return (P,P).
    """
    H, W = arr2d.shape
    r1 = r0 + P
    c1 = c0 + P
    r0_clip = max(0, r0)
    c0_clip = max(0, c0)
    r1_clip = min(H, r1)
    c1_clip = min(W, c1)
    sub = arr2d[r0_clip:r1_clip, c0_clip:c1_clip]
    out = np.empty((P, P), dtype=arr2d.dtype)
    out[:, :] = 0

    # place sub into out at the proper offset
    off_r = r0_clip - r0
    off_c = c0_clip - c0
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
    """Read a raster into (C,H,W) float32 using GDAL (entire image)."""
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

def read_driver_window(path, r0, c0, h, w):
    """Read a window from raster; returns (C,h,w) float32 (clipped at borders; caller pads to P)."""
    from osgeo import gdal
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"GDAL open failed: {path}")
    B = max(1, ds.RasterCount)
    H, W = ds.RasterYSize, ds.RasterXSize
    r0c = max(0, r0); c0c = max(0, c0)
    h_c = max(0, min(h, H - r0c))
    w_c = max(0, min(w, W - c0c))
    if B == 1:
        sub = ds.GetRasterBand(1).ReadAsArray(c0c, r0c, w_c, h_c).astype(np.float32, copy=False) if (h_c>0 and w_c>0) else np.zeros((0,0), np.float32)
        out = sub[None, ...]
    else:
        out = np.empty((B, h_c, w_c), dtype=np.float32)
        for b in range(B):
            if h_c>0 and w_c>0:
                out[b] = ds.GetRasterBand(b+1).ReadAsArray(c0c, r0c, w_c, h_c).astype(np.float32, copy=False)
            else:
                out[b] = np.zeros((h, w), dtype=np.float32)[:0,:0]
    ds = None
    return out

# ----------------- Core Generator -----------------

class PatchDatasetGenerator:
    def __init__(self, data_dir, output_dir,
                 years_nonoverlap, years_overlap=None,
                 driver_order=None,
                 patch_size=256,
                 edge_min_frac=2.0/3.0,
                 overlap_stride=None,
                 padding_mode='reflect', padding_constant=0.0,
                 use_window_read=False,
                 batch_patches=2048):
        """
        data_dir: root of drivers (each is a folder with daily GeoTIFFs)
        output_dir: where to write H5 files
        years_nonoverlap: list[int], years to generate non-overlap sets
        years_overlap: list[int] or None, years to generate overlap sets (inference)
        driver_order: list[str], fixed order; if None, will try to infer (not recommended)
        patch_size: P
        edge_min_frac: threshold for adding edge patch without overlap (needs padding)
        overlap_stride: stride S for overlap (if None and years_overlap given, default S = P//2)
        padding_mode: 'reflect' | 'replicate' | 'constant'
        use_window_read: False -> per day per driver full read; True -> per-patch window read (slower, lower memory)
        batch_patches: number of patches per write batch (controls memory)
        """
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

        # Fixed driver order (from your previous scripts)
        self.driver_order = driver_order or [
            'Firms_Detection_resampled_10x',  # 1
            'ERA5_multi_bands_10x',  # 2-13
            'LULC_BCBoundingbox_resampled_10x',  # 14
            'DEM_and_distance_map_interpolated_10x',  # 15-20
            'NDVI_EVI_with_qa_applied_10x', # 21-22
            'Reflection_500_merge_TerraAquaWGS84_clip_qa_masked_cloud_final_10x', # 23-26
            'MODIS_Terra_Aqua_B20_21_merged_resampled_qa_masked_cloud_nodata-unified_final_10x', # 27-30
            'MOD21A1DN_multibands_filtered_resampled_10x', # 31-38
            'LAI_BCBoundingbox_resampled_right_interpolation_clip_10x' # 39
        ]

        # Discover drivers
        self.driver_dirs = find_tifs(self.data_dir)
        # Channels & raster size
        self.C_total, self.chan_map, (self.H, self.W) = infer_channels_and_size(self.driver_order, self.driver_dirs)

        logger.info(f"Raster size: H={self.H}, W={self.W}, P={self.P}")
        logger.info(f"Total channels: {self.C_total}")
        logger.info(f"Use window read: {self.use_window_read}")

        # Build global date map once
        self.date_map = build_date_map(self.driver_dirs)

    def _anchors_and_coords(self, kind='nonoverlap'):
        """Return anchors list [(r0,c0), ...] and coords array (N,4) (r0,c0,r1,c1) in original image coords."""
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

    def _open_h5(self, out_path, N, T, kind, year, all_dates):
        """Create H5 structure and write attrs."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            logger.info(f"H5 exists, will overwrite: {out_path}")
            os.remove(out_path)
        f = h5py.File(out_path, 'w')

        # datasets
        chunks = (min(H5_CHUNK_N, N), self.C_total, 1, self.P, self.P)
        create_args = dict(shape=(N, self.C_total, T, self.P, self.P),
                           dtype=np.float32,
                           chunks=chunks,
                           shuffle=True)
        if H5_COMPRESSION == 'gzip':
            create_args.update(dict(compression='gzip', compression_opts=int(H5_GZIP_LEVEL)))
        elif H5_COMPRESSION == 'lzf':
            create_args.update(dict(compression='lzf'))
        X = f.create_dataset('X', **create_args)
        # coords will be created later by caller with known values
        # f.create_dataset('patch_coords', data=coords, dtype=np.int32)

        # attrs
        f.attrs['year'] = str(year)
        f.attrs['total_time_steps'] = int(T)
        f.attrs['total_channels'] = int(self.C_total)
        f.attrs['patch_size'] = int(self.P)
        f.attrs['stride_type'] = 'non_overlap' if kind == 'nonoverlap' else 'overlap'
        if kind == 'overlap':
            f.attrs['overlap_stride'] = int(self.S_overlap)
        f.attrs['edge_min_frac'] = float(self.edge_min_frac)
        f.attrs['raster_size'] = (int(self.W), int(self.H))
        f.attrs['start_date'] = all_dates[0].strftime('%Y-%m-%d') if T>0 else ''
        f.attrs['end_date']   = all_dates[-1].strftime('%Y-%m-%d') if T>0 else ''
        f.attrs['padding_mode'] = self.padding_mode
        f.attrs['padding_constant'] = float(self.padding_constant)
        f.attrs['driver_order'] = json.dumps(self.driver_order)

        # channel mapping
        for drv, info in self.chan_map.items():
            f.attrs[f'channel_mapping_{drv}'] = f"{info['start_idx']}-{info['end_idx']-1}"

        return f, X

    def _day_driver_arrays(self, date, use_full=True, window=None):
        """
        Load driver arrays for a given date.
        - use_full=True: read entire image per driver -> returns dict[drv]=(C_d,H,W)
        - use_full=False: read only a window (r0,c0,h,w), returns dict[drv]=(C_d,hc,wc) clipped; caller pads
        """
        arrays = {}
        for drv in self.driver_order:
            ch = self.chan_map.get(drv, {}).get('channels', 0)
            if ch <= 0:
                continue
            path = self.date_map.get(date, {}).get(drv)
            if not path:
                # Missing -> fill NaNs later when assembling
                arrays[drv] = None
                continue
            if use_full:
                arrays[drv] = read_driver_full(path)   # (C_d,H,W)
            else:
                r0, c0, h, w = window
                arrays[drv] = read_driver_window(path, r0, c0, h, w)  # (C_d,hc,wc)
        return arrays

    def _assemble_patch_from_full(self, drv_arrays, r0, c0):
        """
        Slice (C_total,P,P) from per-driver full arrays with padding if needed.
        """
        P = self.P
        out = np.full((self.C_total, P, P), np.nan, dtype=np.float32)
        chan_off = 0
        for drv in self.driver_order:
            arr = drv_arrays.get(drv, None)  # (C_d,H,W) or None
            ch = self.chan_map.get(drv, {}).get('channels', 0)
            if ch <= 0:
                continue
            if arr is None:
                chan_off += ch
                continue
            # for each band, pad window as needed
            for b in range(ch):
                patch = pad_window(arr[b], r0, c0, P, mode=self.padding_mode, constant_value=self.padding_constant)
                out[chan_off + b] = patch
            chan_off += ch
        # NaN -> 0
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _assemble_patch_from_windows(self, date, r0, c0):
        """
        Per-driver window read (clipped) + pad to (P,P); returns (C_total,P,P)
        """
        P = self.P
        out = np.full((self.C_total, P, P), np.nan, dtype=np.float32)
        chan_off = 0
        # read a clipped window box once to bound IO
        # we still call per-driver window read separately (keeps code simple)
        for drv in self.driver_order:
            ch = self.chan_map.get(drv, {}).get('channels', 0)
            if ch <= 0:
                continue
            path = self.date_map.get(date, {}).get(drv)
            if not path:
                chan_off += ch
                continue
            # Read clipped subwindow
            sub = read_driver_window(path, r0, c0, P, P)  # (C_d,hc,wc)
            # pad each band
            for b in range(sub.shape[0]):
                # make a full (P,P) padded canvas
                # figure offsets same as pad_window
                Hc, Wc = sub.shape[1], sub.shape[2]
                # compute clip offsets
                off_r = max(0, -r0)
                off_c = max(0, -c0)
                canvas = np.empty((P, P), dtype=np.float32)
                canvas[:, :] = 0.0
                if Hc>0 and Wc>0:
                    canvas[off_r:off_r+Hc, off_c:off_c+Wc] = sub[b]
                # pad borders (reuse pad_window logic by calling with an assembled temp; avoid double code)
                # simpler: we can run pad_window on a synthetic array by reconstructing original coords; but we already placed clipped content.
                # We'll emulate reflect/replicate by mirroring from the interior we have:
                # Use same branch as pad_window (copy here for performance clarity)
                # top
                if off_r > 0:
                    if self.padding_mode == 'reflect':
                        canvas[0:off_r, :] = canvas[off_r:2*off_r, :][::-1, :]
                    elif self.padding_mode == 'replicate':
                        canvas[0:off_r, :] = canvas[off_r:off_r+1, :]
                    else:
                        canvas[0:off_r, :] = self.padding_constant
                # bottom
                extra_r = P - (off_r + Hc)
                if extra_r > 0:
                    if self.padding_mode == 'reflect':
                        canvas[P-extra_r:P, :] = canvas[P-extra_r- extra_r:P-extra_r, :][::-1, :]
                    elif self.padding_mode == 'replicate':
                        canvas[P-extra_r:P, :] = canvas[P-extra_r-1:P-extra_r, :]
                    else:
                        canvas[P-extra_r:P, :] = self.padding_constant
                # left
                if off_c > 0:
                    if self.padding_mode == 'reflect':
                        canvas[:, 0:off_c] = canvas[:, off_c:2*off_c][:, ::-1]
                    elif self.padding_mode == 'replicate':
                        canvas[:, 0:off_c] = canvas[:, off_c:off_c+1]
                    else:
                        canvas[:, 0:off_c] = self.padding_constant
                # right
                extra_c = P - (off_c + Wc)
                if extra_c > 0:
                    if self.padding_mode == 'reflect':
                        canvas[:, P-extra_c:P] = canvas[:, P-extra_c- extra_c:P-extra_c][:, ::-1]
                    elif self.padding_mode == 'replicate':
                        canvas[:, P-extra_c:P] = canvas[:, P-extra_c-1:P-extra_c]
                    else:
                        canvas[:, P-extra_c:P] = self.padding_constant

                out[chan_off + b] = canvas
            # if sub bands less than declared ch (rare), skip the rest bands as NaN->0
            chan_off += ch
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _write_year(self, year:int, kind:str):
        """
        Generate one year's H5 for a given kind: 'nonoverlap' or 'overlap'
        """
        assert kind in ('nonoverlap', 'overlap')
        P = self.P
        anchors, coords = self._anchors_and_coords(kind=kind)
        N = len(anchors)
        dates = [d for d in date_list_for_year(year)]
        T = len(dates)

        if N == 0 or T == 0:
            logger.warning(f"Year {year}: nothing to write (N={N}, T={T}).")
            return

        subdir = 'nonoverlap' if kind=='nonoverlap' else f'overlap_P{P}_S{self.S_overlap}'
        out_path = os.path.join(self.output_dir, subdir, f"{year}_patch_dataset.h5")
        f, X = self._open_h5(out_path, N, T, kind, year, dates)
        # write coords
        f.create_dataset('patch_coords', data=coords, dtype=np.int32)

        # stream by day
        for t, date in enumerate(tqdm(dates, desc=f"Year {year} [{kind}] write by day")):
            # per-day driver arrays (full or none)
            drv_full = None
            if not self.use_window_read:
                drv_full = self._day_driver_arrays(date, use_full=True)
            # batch patches
            for i in range(0, N, self.batch_patches):
                j = min(i + self.batch_patches, N)
                batch = j - i
                buf = np.empty((batch, self.C_total, P, P), dtype=np.float32)
                for k, idx in enumerate(range(i, j)):
                    r0, c0 = anchors[idx]
                    if not self.use_window_read:
                        buf[k] = self._assemble_patch_from_full(drv_full, r0, c0)
                    else:
                        buf[k] = self._assemble_patch_from_windows(date, r0, c0)
                # write this day's column
                X[i:j, :, t, :, :] = buf
            f.flush()
            # free memory
            drv_full = None
            gc.collect()

        f.flush()
        f.close()
        logger.info(f"Finished year {year} [{kind}] -> {out_path}")
        # brief dataset info
        try:
            with h5py.File(out_path, 'r') as rf:
                Xd = rf['X']
                logger.info(f"X shape: {Xd.shape}, patch_coords: {rf['patch_coords'].shape}")
        except Exception as e:
            logger.warning(f"Info read failed: {e}")

    def run(self):
        # Non-overlap years
        for y in self.years_nonoverlap:
            self._write_year(y, kind='nonoverlap')
        # Overlap years
        for y in self.years_overlap:
            self._write_year(y, kind='overlap')

# ----------------- CLI -----------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate patch datasets (non-overlap for train/val/test; overlap for inference).")

    parser.add_argument('--data_dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_10x_masked', 
                        help='Root data directory containing driver folders with daily .tif')
    parser.add_argument('--output_dir', default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_result_10x_QAapplied_patch',
                        help='Output root directory to write H5 files')

    parser.add_argument('--years_nonoverlap', type=str, default='[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]',
                        help='Years for non-overlap patch datasets, e.g. "[2019,2020,2021]" or "2019,2020"')
    parser.add_argument('--years_overlap', type=str, default='2023, 2024',
                        help='Years for overlap patch datasets (inference), e.g. "2024" or "2023,2024"; leave empty to skip')

    parser.add_argument('--patch_size', type=int, default=140, help='Patch side length P')
    parser.add_argument('--edge_min_frac', type=float, default=2.0/3.0,
                        help='For non-overlap: if remainder >= edge_min_frac * P, add a padded edge patch (no overlap)')
    parser.add_argument('--overlap_stride', type=int, default=None, help='Stride S for overlap sets (default P//2)')

    parser.add_argument('--padding_mode', type=str, default='reflect', choices=['reflect', 'replicate', 'constant'],
                        help='Padding mode for out-of-bounds areas')
    parser.add_argument('--padding_constant', type=float, default=0.0, help='Constant padding value (if mode=constant)')

    parser.add_argument('--use_window_read', action='store_true',
                        help='Read per-patch windows from GDAL (lower memory, slower). Default reads full images per driver per day.')
    parser.add_argument('--batch_patches', type=int, default=2048, help='Patch mini-batch size per write (controls memory)')

    args = parser.parse_args()

    years_nonoverlap = parse_year_list(args.years_nonoverlap)
    years_overlap = parse_year_list(args.years_overlap) if args.years_overlap else []

    gen = PatchDatasetGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        years_nonoverlap=years_nonoverlap,
        years_overlap=years_overlap,
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
