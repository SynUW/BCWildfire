#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOD11A1 批量低质量像元掩膜脚本（6波段：LST_Day, LST_Night, Emis_31, Emis_32, QC_Day, QC_Night）

默认策略（conservative）：“只屏蔽特别低质量，保留中等质量”
- Mandatory QA (bits 0-1): {2,3} -> mask
- LST error (bits 6-7): 3 (>3K) -> mask；0/1/2 (<=1/2/3K) -> keep
- Emissivity error (bits 4-5): 3 (>0.04) -> mask；0/1/2 (<=0.01/0.02/0.04) -> keep
- Data quality (bits 2-3): 0=good, 1=other -> keep；2/3=TBD -> mask（可通过参数放宽）

输出：
- 波段1 LST_Day_1km：按 QC_Day 掩膜
- 波段2 LST_Night_1km：按 QC_Night 掩膜
- 波段3 Emis_31：若对应 QC（Day 或 Night，见 --emis-qc-mode）判为“特别低质量”则掩膜
- 波段4 Emis_32：同上
- 波段5 QC_Day、波段6 QC_Night 原样写回

注意：
- 脚本会尽量复用源文件 nodata；若缺失则对整数用 -32768，对浮点用 NaN。
- 压缩/分块/BigTIFF 等输出参数可配置。
"""

import sys
import argparse
from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

warnings.simplefilter("ignore", NotGeoreferencedWarning)

# ---------- 位操作工具 ----------
def bitslice(arr, start, end):
    """Extract inclusive bit slice [start, end]."""
    width = end - start + 1
    mask = (1 << width) - 1
    return (arr >> start) & mask

# ---------- QC 字段解析 ----------
def qc_fields(qc_arr):
    """
    解析 MOD11A1 QC_Day / QC_Night 位段为字典：
    0-1: mandatory
    2-3: data_quality
    4-5: emis_error
    6-7: lst_error
    """
    return dict(
        mandatory = bitslice(qc_arr, 0, 1),
        data_quality = bitslice(qc_arr, 2, 3),
        emis_error = bitslice(qc_arr, 4, 5),
        lst_error = bitslice(qc_arr, 6, 7),
    )

# ---------- 掩膜判定 ----------
def build_bad_mask_from_qc(
    qc,
    mask_if_mandatory_23=True,
    mask_if_dq_tbd=True,
    dq_keep_1=True,
    max_keep_lst_error=2,     # 0..2 保留；3 (>3K) 掩膜
    max_keep_emis_error=2,    # 0..2 保留；3 (>0.04) 掩膜
):
    """
    返回 boolean bad_mask：True 表示“质量特别低，需要屏蔽”。
    """
    q = qc_fields(qc)

    bad = np.zeros(qc.shape, dtype=bool)

    # 1) mandatory QA：2/3 -> 一定屏蔽（未生产）
    if mask_if_mandatory_23:
        bad |= np.isin(q["mandatory"], [2, 3])

    # 2) data quality：0=好，1=其它质量（保留），2/3=TBD -> 默认屏蔽
    if mask_if_dq_tbd:
        bad |= np.isin(q["data_quality"], [2, 3])
    if not dq_keep_1:
        # 如需更严格，可选择屏蔽 data_quality==1
        bad |= (q["data_quality"] == 1)

    # 3) LST error：>3K（值=3） -> 屏蔽；<=3K（0/1/2） -> 保留
    bad |= (q["lst_error"] > max_keep_lst_error)

    # 4) Emissivity error：>0.04（值=3） -> 屏蔽；<=0.04（0/1/2） -> 保留
    bad |= (q["emis_error"] > max_keep_emis_error)

    return bad

def pick_nodata(dtype, existing_nodata):
    if existing_nodata is not None:
        return existing_nodata
    if np.issubdtype(dtype, np.floating):
        return np.nan
    return np.int64(-32768)

# ---------- 核心处理 ----------
def process_one(path_in: Path, path_out: Path, args) -> None:
    with rasterio.open(path_in) as src:
        if src.count < 6:
            raise ValueError(f"{path_in.name}: 需要 6 个波段（LST_Day, LST_Night, Emis_31, Emis_32, QC_Day, QC_Night）。")

        # 读取波段
        lst_day  = src.read(1)
        lst_nite = src.read(2)
        emis31   = src.read(3)
        emis32   = src.read(4)
        qc_day   = src.read(5)
        qc_nite  = src.read(6)

        # 基于 QC 生成“特别低质量”掩膜
        bad_day  = build_bad_mask_from_qc(
            qc_day,
            mask_if_mandatory_23=not args.allow_mandatory_23,
            mask_if_dq_tbd=not args.allow_dq_tbd,
            dq_keep_1=not args.mask_dq1,
            max_keep_lst_error=args.max_keep_lst_error,
            max_keep_emis_error=args.max_keep_emis_error,
        )
        bad_nite = build_bad_mask_from_qc(
            qc_nite,
            mask_if_mandatory_23=not args.allow_mandatory_23,
            mask_if_dq_tbd=not args.allow_dq_tbd,
            dq_keep_1=not args.mask_dq1,
            max_keep_lst_error=args.max_keep_lst_error,
            max_keep_emis_error=args.max_keep_emis_error,
        )

        profile = src.profile.copy()
        profile.update(
            compress=args.compress,
            tiled=True,
            blockxsize=min(profile.get("blockxsize", 512), 1024),
            blockysize=min(profile.get("blockysize", 512), 1024),
            bigtiff="IF_SAFER",
        )

        # 波段级 nodata（如源未设置则推断）
        band_nd = []
        for b in range(1, src.count + 1):
            nd = src.nodatavals[b-1]
            if nd is None:
                nd = pick_nodata(np.dtype(src.dtypes[b-1]), None)
            band_nd.append(nd)

        # 写出
        path_out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path_out, "w", **profile) as dst:
            # 1) LST_Day：按 bad_day 应用为 NoData
            arr = lst_day.copy()
            nd = band_nd[0]
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
                arr[bad_day] = np.nan if (isinstance(nd, float) and np.isnan(nd)) else nd
            else:
                arr[bad_day] = np.int64(nd)
            dst.write(arr, 1)

            # 2) LST_Night：按 bad_nite
            arr = lst_nite.copy()
            nd = band_nd[1]
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
                arr[bad_nite] = np.nan if (isinstance(nd, float) and np.isnan(nd)) else nd
            else:
                arr[bad_nite] = np.int64(nd)
            dst.write(arr, 2)

            # 3/4) Emis_31 / Emis_32：根据 --emis-qc-mode
            if args.emis_qc_mode == "day":
                bad_emis = bad_day
            elif args.emis_qc_mode == "night":
                bad_emis = bad_nite
            else:  # "union" (默认)：日/夜任一极差则屏蔽
                bad_emis = (bad_day | bad_nite)

            arr = emis31.copy()
            nd = band_nd[2]
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
                arr[bad_emis] = np.nan if (isinstance(nd, float) and np.isnan(nd)) else nd
            else:
                arr[bad_emis] = np.int64(nd)
            dst.write(arr, 3)

            arr = emis32.copy()
            nd = band_nd[3]
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
                arr[bad_emis] = np.nan if (isinstance(nd, float) and np.isnan(nd)) else nd
            else:
                arr[bad_emis] = np.int64(nd)
            dst.write(arr, 4)

            # 5/6) QA 原样写回
            dst.write(qc_day, 5)
            dst.write(qc_nite, 6)

            # 若原文件未统一设置 nodata，则逐波段写 tag 以兼容
            if profile.get("nodata", None) is None:
                for i, ndv in enumerate(band_nd, start=1):
                    if not (isinstance(ndv, float) and np.isnan(ndv)):
                        dst.update_tags(i, nodata=str(ndv))

def main():
    ap = argparse.ArgumentParser(
        description="Mask out特别低质量的 MOD11A1 像元（保留中等质量）。"
    )
    ap.add_argument("--input_dir", type=Path, required=True, help="输入目录（含 MOD11A1 六波段 GeoTIFF）")
    ap.add_argument("--output_dir", type=Path, required=True, help="输出目录")
    ap.add_argument("--suffix", default="", help="输出文件名追加后缀（可选）")
    ap.add_argument("--compress", default="LZW", help="输出压缩方式，默认 LZW")

    # 放宽/收紧选项（默认是保守屏蔽“特别低质量”）
    ap.add_argument("--allow-mandatory-23", action="store_true",
                    help="放宽：不因 mandatory=2/3（未生产）而屏蔽（一般不建议）")
    ap.add_argument("--allow-dq-tbd", action="store_true",
                    help="放宽：不屏蔽 DataQuality=2/3（TBD）")
    ap.add_argument("--mask-dq1", action="store_true",
                    help="收紧：连 DataQuality=1（Other quality）也屏蔽（默认不屏蔽以保留中等质量）")
    ap.add_argument("--max-keep-lst-error", type=int, default=3, choices=[0,1,2,3],
                    help="允许保留的最大 LST 误差级别（0..3）；默认 2（<=3K 保留，>3K 屏蔽）")
    ap.add_argument("--max-keep-emis-error", type=int, default=3, choices=[0,1,2,3],
                    help="允许保留的最大 Emissivity 误差级别（0..3）；默认 2（<=0.04 保留，>0.04 屏蔽）")

    ap.add_argument("--emis-qc-mode", choices=["union", "day", "night"], default="union",
                    help="发射率波段按哪个 QC 判定：union=日夜任一极差则屏蔽（默认）；day=仅用 QC_Day；night=仅用 QC_Night")

    ap.add_argument("--overwrite", action="store_true", help="已存在则覆盖")
    args = ap.parse_args()

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
