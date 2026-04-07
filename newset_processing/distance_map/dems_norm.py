#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalize DEM / Slope / Aspect / Hillshade / Distance rasters in one folder.

File name patterns (case-insensitive):
- DEM:            ASTER_GDEM_elevation_1km.tif
- Slope(degree):  *_slope_degree.tif
- Aspect:         *_aspect.tif
- Hillshade:      *_hillshade.tif
- Distance:       distance_*.tif   (multiple)

Defaults (can be changed by CLI):
- DEM:       minmax_p2p98
- Slope:     mode=cap, cap=60 deg
- Aspect:    encode to cos/sin, mapped to [0,1]
- Hillshade: mode=div255   (use stretch_p1p99 to increase contrast)
- Distance:  transform=log1p  with k=median, cap_quantile=0.98

Outputs:
- *_norm.tif for DEM / Slope / Hillshade / Distance
- *_aspect_cos_norm.tif, *_aspect_sin_norm.tif for Aspect
- norm_params.json with all parameters used
"""

import argparse, json, math
from pathlib import Path
import numpy as np
import rasterio

# ------------- I/O helpers -------------

def read_band(path: Path, band=1):
    with rasterio.open(path) as src:
        arr = src.read(band).astype(np.float32, copy=False)
        profile = src.profile.copy()
        nodata = src.nodata  # may be None
        # treat nodata as NaN internally
        if nodata is not None:
            arr[np.isclose(arr, nodata, equal_nan=True)] = np.float32(np.nan)
        return arr, profile, nodata

def write_tif(path: Path, arr: np.ndarray, profile: dict,
              nodata=None, compress="ZSTD", blocksize=512, overwrite=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and overwrite:
        path.unlink()
    prof = profile.copy()
    prof.update(dtype="float32", count=1, tiled=True,
                blockxsize=blocksize, blockysize=blocksize)
    # keep original nodata if provided; if nodata=None we simply store NaNs
    if nodata is not None:
        prof.update(nodata=nodata)
    if compress != "NONE":
        prof.update(compress=compress)
        if compress == "ZSTD":
            prof.update(zstd_level=9)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)

def nanpercentile(a, q):
    v = a[~np.isnan(a)]
    return float(np.percentile(v, q)) if v.size else np.nan

# ------------- Normalizers -------------

# DEM
def normalize_dem(arr, params, mode="minmax_p2p98"):
    v = arr[~np.isnan(arr)]
    if v.size == 0:
        params.update({"type":"dem","mode":mode,"info":"empty"})
        return arr
    if mode == "minmax_p2p98":
        p2, p98 = float(np.percentile(v, 2)), float(np.percentile(v, 98))
        if p98 <= p2: p98 = p2 + 1e-6
        out = (np.clip(arr, p2, p98) - p2) / (p98 - p2)
        params.update({"type":"dem","mode":"minmax_p2p98","p2":p2,"p98":p98})
        return out
    elif mode == "robust_z":
        med = float(np.median(v))
        q25, q75 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
        iqr = max(q75 - q25, 1e-6)
        out = (arr - med) / iqr
        # map to ~[0,1] via sigmoid-like tanh, optional:
        out = 0.5 * (np.tanh(out) + 1.0)
        params.update({"type":"dem","mode":"robust_z","median":med,"iqr":iqr,"nonlinear":"tanh_to_01"})
        return out
    else:
        raise ValueError(f"Unknown DEM mode: {mode}")

# Slope (degrees)
def normalize_slope_deg(arr, params, mode="cap", cap_deg=60.0):
    v = arr[~np.isnan(arr)]
    if v.size == 0:
        params.update({"type":"slope_degree","mode":mode,"info":"empty"})
        return arr
    if mode == "cap":
        out = np.clip(arr, 0.0, float(cap_deg)) / float(cap_deg)
        params.update({"type":"slope_degree","mode":"cap","cap_deg":float(cap_deg)})
        return out
    elif mode == "auto_p98":
        cap = float(np.percentile(v, 98))
        cap = max(cap, 1e-6)
        out = np.clip(arr, 0.0, cap) / cap
        params.update({"type":"slope_degree","mode":"auto_p98","cap_p98":cap})
        return out
    elif mode == "minmax_p2p98":
        p2, p98 = float(np.percentile(v, 2)), float(np.percentile(v, 98))
        if p98 <= p2: p98 = p2 + 1e-6
        out = (np.clip(arr, p2, p98) - p2) / (p98 - p2)
        params.update({"type":"slope_degree","mode":"minmax_p2p98","p2":p2,"p98":p98})
        return out
    else:
        raise ValueError(f"Unknown slope mode: {mode}")

# Aspect → cos/sin channels mapped to [0,1]
def normalize_aspect(aspect_deg, params, map_to_01=True):
    theta = np.deg2rad(aspect_deg)
    cosv = np.cos(theta).astype(np.float32)
    sinv = np.sin(theta).astype(np.float32)
    if map_to_01:
        cosv = (cosv + 1.0) * 0.5
        sinv = (sinv + 1.0) * 0.5
        params.update({"type":"aspect","encode":"cos_sin","range":"[0,1]"})
    else:
        params.update({"type":"aspect","encode":"cos_sin","range":"[-1,1]"})
    return cosv, sinv

# Hillshade
def normalize_hillshade(hs, params, mode="div255"):
    """
    Hillshade 归一化：
      - 'div255' | '/255' | '255'  -> 直接 /255 到 [0,1]
      - 'stretch_p1p99' | 'p1p99' | 'stretch' | 'auto' -> P1–P99 线性拉伸到 [0,1]
    兼容大小写与前后空格。
    """
    m = (mode or "div255").strip().lower()
    v = hs[~np.isnan(hs)]
    if v.size == 0:
        params.update({"type": "hillshade", "mode": m, "info": "empty"})
        return hs

    if m in ("div255", "/255", "255"):
        out = np.clip(hs, 0.0, 255.0) / 255.0
        params.update({"type": "hillshade", "mode": "div255"})
        return out

    if m in ("stretch_p1p99", "p1p99", "stretch", "auto"):
        p1, p99 = float(np.percentile(v, 1)), float(np.percentile(v, 99))
        if p99 <= p1:
            p99 = p1 + 1e-6
        out = (np.clip(hs, p1, p99) - p1) / (p99 - p1)
        out = np.clip(out, 0.0, 1.0)
        params.update({"type": "hillshade", "mode": "stretch_p1p99", "p1": p1, "p99": p99})
        return out

    # 落到这里说明是未知拼法
    raise ValueError(f"Unknown hillshade mode: {mode}")


# Distance
def normalize_distance(arr, params,
                       transform="log1p",
                       k_mode="median", k_value=None,
                       cap_quantile=0.98):
    v = arr[~np.isnan(arr)]
    if v.size == 0:
        params.update({"type":"distance","mode":transform,"info":"empty"})
        return arr

    # cap for far tail
    q = float(cap_quantile)
    q = min(max(q, 0.5), 0.999)  # sanity
    cap = float(np.percentile(v, q*100.0))

    # k scale
    if k_value is not None:
        k = float(k_value)
    elif k_mode.lower() == "median":
        k = float(np.median(v))
    elif k_mode.lower() == "p75":
        k = float(np.percentile(v, 75))
    else:
        raise ValueError("k_mode must be 'median' or 'p75' (or set k_value).")
    k = max(k, 1e-6)
    cap = max(cap, k)

    if transform == "log1p":
        out = np.log1p(np.clip(arr, 0, cap) / k) / math.log1p(cap / k)
        params.update({"type":"distance","mode":"log1p","k":k,"cap_q":q,"cap":cap})
    elif transform == "exp":
        # proximity-style: exp(-d/k) then re-range by cap if you want contrast
        prox = np.exp(-np.clip(arr, 0, cap) / k)
        # optional min-max to [0,1] is already satisfied (0..1], so just pass
        out = prox.astype(np.float32)
        params.update({"type":"distance","mode":"exp","k":k,"cap_q":q,"cap":cap})
    else:
        raise ValueError("distance transform must be 'log1p' or 'exp'.")

    out = np.clip(out, 0.0, 1.0)
    return out

# ------------- Main -------------

def main():
    ap = argparse.ArgumentParser("Normalize DEM/Slope/Aspect/Hillshade/Distance rasters")
    ap.add_argument("--input-folder", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/distance_map_interpolated_margin_nodata', help="输入目录")
    ap.add_argument("--output-folder", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/distance_map_interpolated_margin_nodata_norm', help="输出目录")
    ap.add_argument("--band", type=int, default=1)

    # DEM
    ap.add_argument("--dem-mode", choices=["minmax_p2p98","robust_z"], default="minmax_p2p98")

    # Slope
    ap.add_argument("--slope-mode", choices=["cap","auto_p98","minmax_p2p98"], default="minmax_p2p98")
    ap.add_argument("--slope-cap", type=float, default=10.0)

    # Aspect
    ap.add_argument("--aspect-to-01", action="store_true", default=True,
                    help="map cos/sin from [-1,1] to [0,1] (default True)")
    ap.add_argument("--aspect-keep-neg1_1", dest="aspect_to_01", action="store_false")

    # Hillshade
    ap.add_argument("--hillshade-mode", choices=["div255","stretch_p1p99"], default="stretch_p1p99 ")

    # Distance
    ap.add_argument("--distance-transform", choices=["log1p","exp"], default="log1p")
    ap.add_argument("--distance-k-mode", choices=["median","p75"], default="median")
    ap.add_argument("--distance-k-value", type=float, default=None,
                    help="override k with a constant (same units as distance)")
    ap.add_argument("--distance-cap-quantile", type=float, default=0.98,
                    help="cap tail at this quantile before scaling (default 0.98)")

    # Write
    ap.add_argument("--compress", default="ZSTD", choices=["NONE","LZW","DEFLATE","ZSTD"])
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument("--save-meta", default="norm_params.json")

    args = ap.parse_args()
    in_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # gather files (case-insensitive by pattern)
    f_dem      = list(in_dir.glob("ASTER_GDEM_elevation_1km.tif"))
    f_slope    = list(in_dir.glob("*_slope_degree.tif"))
    f_aspect   = list(in_dir.glob("*_aspect.tif"))
    f_hill     = list(in_dir.glob("*_hillshade.tif"))
    f_dist     = sorted(list(in_dir.glob("distance_*.tif")))

    meta = {}

    # DEM
    for f in f_dem:
        arr, prof, nod = read_band(f, args.band)
        params = {}
        out = normalize_dem(arr, params, mode=args.dem_mode)
        out[np.isnan(arr)] = np.nan
        write_tif(out_dir/(f.stem+"_norm.tif"), out, prof, nodata=nod,
                  compress=args.compress, blocksize=args.blocksize)
        meta[f.name] = params
        print(f"[OK] DEM -> {f.stem}_norm.tif")

    # Slope
    for f in f_slope:
        arr, prof, nod = read_band(f, args.band)
        params = {}
        out = normalize_slope_deg(arr, params, mode=args.slope_mode, cap_deg=args.slope_cap)
        out[np.isnan(arr)] = np.nan
        write_tif(out_dir/(f.stem+"_norm.tif"), out, prof, nodata=nod,
                  compress=args.compress, blocksize=args.blocksize)
        meta[f.name] = params
        print(f"[OK] Slope -> {f.stem}_norm.tif")

    # Aspect
    for f in f_aspect:
        arr, prof, nod = read_band(f, args.band)
        params = {}
        cosv, sinv = normalize_aspect(arr, params, map_to_01=args.aspect_to_01)
        cosv[np.isnan(arr)] = np.nan
        sinv[np.isnan(arr)] = np.nan
        write_tif(out_dir/(f.stem+"_cos_norm.tif"), cosv, prof, nodata=nod,
                  compress=args.compress, blocksize=args.blocksize)
        write_tif(out_dir/(f.stem+"_sin_norm.tif"), sinv, prof, nodata=nod,
                  compress=args.compress, blocksize=args.blocksize)
        meta[f.name] = params
        print(f"[OK] Aspect -> {f.stem}_cos_norm.tif / _sin_norm.tif")

    # Hillshade
    for f in f_hill:
        arr, prof, nod = read_band(f, args.band)
        params = {}
        out = normalize_hillshade(arr, params, mode=args.hillshade_mode)
        out[np.isnan(arr)] = np.nan
        write_tif(out_dir/(f.stem+"_norm.tif"), out, prof, nodata=nod,
                  compress=args.compress, blocksize=args.blocksize)
        meta[f.name] = params
        print(f"[OK] Hillshade -> {f.stem}_norm.tif")

    # Distance (multiple)
    for f in f_dist:
        arr, prof, nod = read_band(f, args.band)
        params = {}
        out = normalize_distance(arr, params,
                                 transform=args.distance_transform,
                                 k_mode=args.distance_k_mode,
                                 k_value=args.distance_k_value,
                                 cap_quantile=args.distance_cap_quantile)
        out[np.isnan(arr)] = np.nan
        write_tif(out_dir/(f.stem+"_norm.tif"), out, prof, nodata=nod,
                  compress=args.compress, blocksize=args.blocksize)
        meta[f.name] = params
        print(f"[OK] Distance -> {f.stem}_norm.tif")

    # Save parameter log
    (out_dir/args.save_meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[DONE] Params saved to {out_dir/args.save_meta}")
    print("[ALL DONE]")
    
if __name__ == "__main__":
    main()
