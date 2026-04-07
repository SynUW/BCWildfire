#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unify invalid values in wildfire HDF5 datasets (pixel-based or patch-based).

Rules:
1) Globally: -999 -> -9999
2) FIRMS (channel 0): 255 -> -9999  (and -999 -> -9999 already covered)
3) LAI (last channel): if LAI==0 AND all other channels at that position == -9999,
   then LAI -> -9999; else keep 0.

Layouts supported:
- X(N,C,T)                 : non-patch year dataset (your pixel-packed form)
- X(N,C,T,P,P)             : patch dataset
- Per-pixel datasets (C,T) : keys like "row_col" when no X exists

In-place, chunked processing.
"""

import os
import sys
import re
import glob
import argparse
import h5py
import numpy as np
from tqdm import tqdm

INVALID_OLD = -999
INVALID_NEW = -9999
FIRMS_INVALID_255 = 255


def is_pixel_dataset_name(name: str) -> bool:
    if name in ('X', 'coords', 'patch_coords', 'valid_mask', 'split'):
        return False
    return bool(re.match(r'^\d+_\d+$', name))


def iter_h5_files(paths, recursive=False):
    for p in paths:
        if os.path.isfile(p) and p.endswith('.h5'):
            yield p
        elif os.path.isdir(p):
            if recursive:
                for root, _, files in os.walk(p):
                    for f in files:
                        if f.endswith('.h5'):
                            yield os.path.join(root, f)
            else:
                for f in glob.glob(os.path.join(p, '*.h5')):
                    yield f


def unify_neg999_to_neg9999(arr):
    """In-place replace -999 -> -9999."""
    np.place(arr, arr == INVALID_OLD, INVALID_NEW)


def fix_firms_c0(arr):
    """In-place fix FIRMS band: 255 -> -9999; also unify -999 -> -9999 just-in-case."""
    np.place(arr, arr == FIRMS_INVALID_255, INVALID_NEW)
    np.place(arr, arr == INVALID_OLD, INVALID_NEW)


def process_X_3d(X: h5py.Dataset, dry_run=False):
    """X shape: (N, C, T)."""
    N, C, T = X.shape
    c0 = 0
    cL = C - 1

    N_step = max(1, min(1024, N))
    T_step = max(1, min(64, T))

    for n0 in tqdm(range(0, N, N_step), desc="N blocks"):
        n1 = min(N, n0 + N_step)
        for t0 in range(0, T, T_step):
            t1 = min(T, t0 + T_step)

            block = X[n0:n1, :, t0:t1]  # (bn, C, bt)
            if dry_run:
                continue

            # 1) global unify
            unify_neg999_to_neg9999(block)

            # 2) FIRMS on channel 0
            fix_firms_c0(block[:, c0, :])  # (bn, bt)

            # 3) LAI conditional: others_all_invalid over channel axis=1
            lai = block[:, cL, :]                  # (bn, bt)
            others = block[:, 0:cL, :]             # (bn, C-1, bt)
            others_all_invalid = np.all(others == INVALID_NEW, axis=1)  # (bn, bt)
            mask = (lai == 0) & others_all_invalid
            lai[mask] = INVALID_NEW

            # write back
            X[n0:n1, :, t0:t1] = block


def process_X_5d(X: h5py.Dataset, dry_run=False):
    """X shape: (N, C, T, P, P)."""
    N, C, T, P, Q = X.shape
    assert P == Q, "Expected square patches (P,P)."
    c0 = 0
    cL = C - 1

    N_step = max(1, min(256, N))
    T_step = max(1, min(8, T))

    for n0 in tqdm(range(0, N, N_step), desc="N blocks"):
        n1 = min(N, n0 + N_step)
        for t0 in range(0, T, T_step):
            t1 = min(T, t0 + T_step)

            block = X[n0:n1, :, t0:t1, :, :]  # (bn, C, bt, P, P)
            if dry_run:
                continue

            # 1) global unify
            unify_neg999_to_neg9999(block)

            # 2) FIRMS
            fix_firms_c0(block[:, c0, :, :, :])  # (bn, bt, P, P)

            # 3) LAI conditional
            # others_all_invalid along channel axis=1 -> (bn, bt, P, P)
            others = block[:, 0:cL, :, :, :]                 # (bn, C-1, bt, P, P)
            others_all_invalid = np.all(others == INVALID_NEW, axis=1)  # (bn, bt, P, P)
            lai = block[:, cL, :, :, :]                       # (bn, bt, P, P)
            mask = (lai == 0) & others_all_invalid
            lai[mask] = INVALID_NEW

            X[n0:n1, :, t0:t1, :, :] = block


def process_per_pixel(f: h5py.File, dry_run=False):
    """Iterate datasets with name 'row_col' and shape (C,T)."""
    keys = [k for k in f.keys() if is_pixel_dataset_name(k)]
    if not keys:
        return

    # We’ll process in T chunks
    # Infer C,T from the first one
    C, T = f[keys[0]].shape
    c0 = 0
    cL = C - 1
    T_step = max(1, min(128, T))

    for name in tqdm(keys, desc="Pixel datasets"):
        dset = f[name]  # (C, T)
        for t0 in range(0, T, T_step):
            t1 = min(T, t0 + T_step)
            blk = dset[:, t0:t1]  # (C, tt)
            if dry_run:
                continue

            # 1) global unify
            unify_neg999_to_neg9999(blk)

            # 2) FIRMS
            fix_firms_c0(blk[c0, :])  # (tt,)

            # 3) LAI conditional: others_all_invalid along channel axis=0
            lai = blk[cL, :]                # (tt,)
            others = blk[0:cL, :]           # (C-1, tt)
            others_all_invalid = np.all(others == INVALID_NEW, axis=0)  # (tt,)
            mask = (lai == 0) & others_all_invalid
            lai[mask] = INVALID_NEW

            dset[:, t0:t1] = blk


def process_file(path, dry_run=False, verbose=True):
    mode = 'r' if dry_run else 'r+'
    with h5py.File(path, mode) as f:
        if verbose:
            print(f"\nProcessing: {path}")

        # Prefer unified X if present
        if 'X' in f and isinstance(f['X'], h5py.Dataset):
            X = f['X']
            if X.ndim == 3:
                if verbose: print(f"  Detected X shape {X.shape} (N,C,T)")
                process_X_3d(X, dry_run=dry_run)
            elif X.ndim == 5:
                if verbose: print(f"  Detected X shape {X.shape} (N,C,T,P,P)")
                process_X_5d(X, dry_run=dry_run)
            else:
                if verbose: print(f"  Unexpected X ndim={X.ndim}, fallback to per-dataset iteration")
                process_per_pixel(f, dry_run=dry_run)
        else:
            # per-pixel layout
            process_per_pixel(f, dry_run=dry_run)

        if verbose:
            print("  Done.")


def main():
    ap = argparse.ArgumentParser(description="Unify invalid values in HDF5 wildfire datasets.")
    ap.add_argument('--path',  default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_result_10x_QAapplied',
                    help='One or more .h5 files or directories.')
    ap.add_argument('--recursive', action='store_true',
                    help='If a directory is given, search .h5 recursively.')
    ap.add_argument('--dry_run', action='store_true',
                    help='Read without writing (sanity check).')
    args = ap.parse_args()

    files = list(iter_h5_files([args.path], recursive=args.recursive))
    if not files:
        print("No .h5 files found.")
        sys.exit(1)

    for fp in files:
        try:
            process_file(fp, dry_run=args.dry_run, verbose=True)
        except OSError as e:
            if "Resource temporarily unavailable" in str(e) or "unable to lock file" in str(e):
                print(f"SKIP {fp}: File is locked by another process")
            else:
                print(f"ERROR processing {fp}: {e}")
        except Exception as e:
            print(f"ERROR processing {fp}: {e}")


if __name__ == '__main__':
    main()
