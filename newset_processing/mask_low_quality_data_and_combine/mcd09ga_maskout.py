#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCD09GA 批量低置信度像元掩膜脚本（6波段版本：前4观测 + 2个QA）

新增规则：
- cloud_only：只按云相关标志过滤（默认仅 Cloudy），可选包含 Mixed、云阴影、邻云、内部云。
"""

import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
from tqdm import tqdm

warnings.simplefilter("ignore", NotGeoreferencedWarning)

def bitslice(arr, start, end):
    width = end - start + 1
    mask = (1 << width) - 1
    return (arr >> start) & mask

def get_bit(arr, bit):
    return (arr >> bit) & 1

def qa1_fields(qa1):
    return dict(
        modland=bitslice(qa1, 0, 1),
        dq_b1=bitslice(qa1, 2, 5),
        dq_b2=bitslice(qa1, 6, 9),
        dq_b3=bitslice(qa1, 10, 13),
        dq_b4=bitslice(qa1, 14, 17),
        dq_b5=bitslice(qa1, 18, 21),
        dq_b6=bitslice(qa1, 22, 25),
        dq_b7=bitslice(qa1, 26, 29),
        ac=get_bit(qa1, 30),
        adj=get_bit(qa1, 31),
    )

def qa2_fields(qa2):
    return dict(
        cloud_state=bitslice(qa2, 0, 1),   # 0 clear, 1 cloudy, 2 mixed, 3 assumed clear
        cloud_shadow=get_bit(qa2, 2),
        land_water=bitslice(qa2, 3, 5),
        aerosol=bitslice(qa2, 6, 7),
        cirrus=bitslice(qa2, 8, 9),
        internal_cloud=get_bit(qa2, 10),
        internal_fire=get_bit(qa2, 11),
        mod35_snow=get_bit(qa2, 12),
        adjacent_cloud=get_bit(qa2, 13),
        brdf_corrected=get_bit(qa2, 14),
        internal_snow=get_bit(qa2, 15),
    )

def build_invalid_mask(
    qa1, qa2,
    rule="strict",
    allow_assumed_clear=False,
    max_aerosol=2,
    max_cirrus=0,
    ignore_adjacent_cloud=False,
    check_bands=(1, 2, 3, 4),
    # ↓↓↓ 新增云专用选项 ↓↓↓
    include_mixed=False,
    also_mask_shadow=False,
    also_mask_adjacent=False,
    also_mask_internal_cloud=False,
):
    q1 = qa1_fields(qa1)
    q2 = qa2_fields(qa2)

    if rule == "cloud_only":
        # 仅依据云相关位来判定 invalid
        # 基础：Cloudy → invalid
        invalid = (q2["cloud_state"] == 1)
        # 可选：Mixed 也当云
        if include_mixed:
            invalid |= (q2["cloud_state"] == 2)
        # 可选：内部云算法标志
        if also_mask_internal_cloud:
            invalid |= (q2["internal_cloud"] == 1)
        # 可选：云阴影
        if also_mask_shadow:
            invalid |= (q2["cloud_shadow"] == 1)
        # 可选：邻云
        if also_mask_adjacent:
            invalid |= (q2["adjacent_cloud"] == 1)
        return invalid

    # ===== 原有 strict / moderate 规则 =====
    # 1) MODLAND
    if rule == "strict":
        ok_modland = (q1["modland"] == 0)
    elif rule == "moderate":
        ok_modland = np.isin(q1["modland"], [0, 1])
    else:
        raise ValueError(f"Unknown rule: {rule}")

    # 2) DQ for Band1..4
    dq_map = {1: q1["dq_b1"], 2: q1["dq_b2"], 3: q1["dq_b3"], 4: q1["dq_b4"],
              5: q1["dq_b5"], 6: q1["dq_b6"], 7: q1["dq_b7"]}
    if rule == "strict":
        ok_dq = np.ones(qa1.shape, dtype=bool)
        for b in check_bands:
            ok_dq &= (dq_map[b] == 0)
    else:  # moderate
        ok_dq = np.ones(qa1.shape, dtype=bool)
        for b in check_bands:
            ok_dq &= (dq_map[b] <= 1)

    # 3) QA2
    if allow_assumed_clear:
        ok_cloud_state = np.isin(q2["cloud_state"], [0, 3])
    else:
        ok_cloud_state = (q2["cloud_state"] == 0)

    ok_cloud_shadow = (q2["cloud_shadow"] == 0)
    ok_internal_cloud = (q2["internal_cloud"] == 0)
    ok_snow = (q2["mod35_snow"] == 0) & (q2["internal_snow"] == 0)
    ok_aerosol = (q2["aerosol"] <= max_aerosol)
    ok_cirrus = (q2["cirrus"] <= max_cirrus)
    ok_adjacent = (q2["adjacent_cloud"] == 0) if not ignore_adjacent_cloud else np.ones(qa2.shape, dtype=bool)

    ok_all = (ok_modland & ok_dq &
              ok_cloud_state & ok_cloud_shadow & ok_internal_cloud &
              ok_adjacent & ok_snow & ok_aerosol & ok_cirrus)

    return ~ok_all

def pick_nodata(dtype, existing_nodata):
    if existing_nodata is not None:
        return existing_nodata
    if np.issubdtype(dtype, np.floating):
        return np.nan
    return -32768

def process_one(src_path: Path, dst_path: Path, args):
    with rasterio.open(src_path) as src:
        if src.count < 6:
            raise ValueError(f"{src_path.name}: 需要至少6个波段（4观测 + 2 QA）。")

        obs = src.read(indexes=(1, 2, 3, 4))
        qa1 = src.read(5)
        qa2 = src.read(6)

        invalid = build_invalid_mask(
            qa1, qa2,
            rule=args.rule,
            allow_assumed_clear=args.allow_assumed_clear,
            max_aerosol=args.max_aerosol,
            max_cirrus=args.max_cirrus,
            ignore_adjacent_cloud=args.ignore_adjacent_cloud,
            check_bands=(1, 2, 3, 4),
            include_mixed=args.include_mixed,
            also_mask_shadow=args.also_mask_shadow,
            also_mask_adjacent=args.also_mask_adjacent,
            also_mask_internal_cloud=args.also_mask_internal_cloud,
        )

        profile = src.profile.copy()
        profile.update(
            compress=args.compress,
            tiled=True,
            blockxsize=min(profile.get("blockxsize", 512), 1024),
            blockysize=min(profile.get("blockysize", 512), 1024),
            bigtiff="IF_SAFER",
        )

        band_nodata = []
        all_nodata_vals = set()
        for b in range(1, src.count + 1):
            nd = src.nodatavals[b-1]
            if nd is None:
                dtype = np.dtype(src.dtypes[b-1])
                nd = pick_nodata(dtype, None)
            band_nodata.append(nd)
            if not (isinstance(nd, float) and np.isnan(nd)):
                all_nodata_vals.add(nd)

        unified_nodata = None
        if len(all_nodata_vals) == 1:
            unified_nodata = list(all_nodata_vals)[0]
            profile.update(nodata=unified_nodata)
        else:
            profile.pop("nodata", None)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            # 观测波段：应用 invalid → NoData
            for i in range(4):
                arr = obs[i].copy()
                nd = band_nodata[i]
                if np.issubdtype(arr.dtype, np.floating):
                    arr[invalid] = np.nan if (isinstance(nd, float) and np.isnan(nd)) else nd
                else:
                    arr[invalid] = np.int64(nd)
                dst.write(arr, i + 1)
                if unified_nodata is None and not (isinstance(nd, float) and np.isnan(nd)):
                    dst.update_tags(i + 1, nodata=str(nd))

            # QA 波段原样写回
            dst.write(qa1, 5)
            dst.write(qa2, 6)

    return True

def main():
    parser = argparse.ArgumentParser(description="Mask low-confidence pixels in MCD09GA (6-band: 4 obs + 2 QA).")
    parser.add_argument("--input_dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MCD09GA_b1237_mosaic_withQA', type=Path)
    parser.add_argument("--output_dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic/MCD09GA_b1237_mosaic_withQA_QAapplied', type=Path)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--compress", default="LZW")

    parser.add_argument("--rule", choices=["strict", "moderate", "cloud_only"], default="strict")

    # 旧规则细节（保留可用）
    parser.add_argument("--allow_assumed_clear", action="store_true")
    parser.add_argument("--max_aerosol", type=int, default=2)
    parser.add_argument("--max_cirrus", type=int, default=0)
    parser.add_argument("--ignore_adjacent_cloud", action="store_true")

    # 新增：云专用选项（仅 cloud_only 有意义）
    parser.add_argument("--include_mixed", action="store_true", help="cloud_state=Mixed 也视作云")
    parser.add_argument("--also_mask_shadow", action="store_true", help="同时屏蔽云阴影像元")
    parser.add_argument("--also_mask_adjacent", action="store_true", help="同时屏蔽邻云像元")
    parser.add_argument("--also_mask_internal_cloud", action="store_true", help="同时屏蔽 internal cloud 像元")

    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"输入目录不存在：{args.input_dir}", file=sys.stderr)
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted([p for p in args.input_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")])
    if not tifs:
        print("未在输入目录发现 TIF 文件。", file=sys.stderr)
        sys.exit(1)

    ok, fail = 0, 0
    for src in tqdm(tifs, desc="Processing"):
        dst = args.output_dir / (src.stem + args.suffix + src.suffix)
        if dst.exists() and not args.overwrite:
            ok += 1
            continue
        try:
            process_one(src, dst, args)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAILED] {src.name}: {e}", file=sys.stderr)

    print(f"Done. OK={ok}, FAILED={fail}")

if __name__ == "__main__":
    main()
