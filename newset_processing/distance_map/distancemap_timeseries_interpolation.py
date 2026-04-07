import argparse
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import sys
import time

DATE_FMT_IN = "%Y-%m-%d"       # 输入的起止日期格式
DATE_FMT_OUT = "%Y_%m_%d"      # 输出文件名中的日期格式

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"

def print_progress(current: int, total: int, start_time: float, prefix: str = ""):
    """打印进度条"""
    current_time = time.time()
    progress_percent = current / total * 100
    
    # 控制更新频率
    should_update = (
        current_time - start_time >= 2.0 or  # 至少2秒间隔
        progress_percent - int(progress_percent) == 0 or  # 每1%进度
        current == total  # 完成时
    )
    
    if not should_update and current < total:
        return
    
    elapsed = current_time - start_time
    if current > 0:
        avg_time_per_item = elapsed / current
        remaining_items = total - current
        eta = remaining_items * avg_time_per_item
        eta_str = format_time(eta)
    else:
        eta_str = "?:??:??"
    
    elapsed_str = format_time(elapsed)
    total_estimated = elapsed / current * total if current > 0 else 0
    total_str = format_time(total_estimated)
    
    print(f"\r{prefix}[{current}/{total}], {elapsed_str}/{total_str}, ETA: {eta_str}", end='', flush=True)
    
    if current == total:
        print()  # 完成后换行

def parse_args():
    p = argparse.ArgumentParser(
        description="将一个 .tif 模板按日复制为 yyyy_mm_dd.tif，日期含起止。"
    )
    p.add_argument("--src-dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x/distance_maps_materials', help="源目录，包含模板 .tif")
    p.add_argument("--template", default=None, help="模板文件名（可选）。不填则要求源目录仅含一个 .tif。")
    p.add_argument("--dst-dir", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches_mosaic_reproject_unifyResolution_clip_5x_norm/distance_maps', help="输出目录（不存在会自动创建）")
    p.add_argument("--start", default='2000-01-01', help="开始日期（含），格式 YYYY-MM-DD")
    p.add_argument("--end", default='2024-12-31', help="结束日期（含），格式 YYYY-MM-DD")
    p.add_argument("--overwrite", action="store_true", help="若目标存在则覆盖（默认为跳过）")
    return p.parse_args()

def find_template(src_dir: Path, template_name: str | None) -> Path:
    if template_name:
        tpl = src_dir / template_name
        if not tpl.exists():
            sys.exit(f"❌ 指定模板不存在：{tpl}")
        if tpl.suffix.lower() != ".tif" and tpl.suffix.lower() != ".tiff":
            sys.exit(f"❌ 指定模板不是 .tif/.tiff：{tpl.name}")
        return tpl

    # 未指定文件名时，从目录里找唯一的 .tif/.tiff
    tifs = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]
    if len(tifs) == 0:
        sys.exit("❌ 源目录未找到任何 .tif/.tiff 文件。请检查 --src-dir 或使用 --template 指定文件名。")
    if len(tifs) > 1:
        names = ", ".join(p.name for p in tifs[:5])
        more = "" if len(tifs) <= 5 else f" 等 {len(tifs)} 个"
        sys.exit(f"❌ 源目录包含多个 .tif：{names}{more}。\n请使用 --template 明确指定模板文件名。")
    return tifs[0]

def daterange_inclusive(start_date: datetime, end_date: datetime):
    if end_date < start_date:
        sys.exit("❌ 结束日期早于开始日期，请检查 --start / --end。")
    delta = (end_date - start_date).days
    for i in range(delta + 1):
        yield start_date + timedelta(days=i)

def main():
    args = parse_args()
    src_dir = Path(args.src_dir).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()

    if not src_dir.exists():
        sys.exit(f"❌ 源目录不存在：{src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    try:
        start_dt = datetime.strptime(args.start, DATE_FMT_IN)
        end_dt = datetime.strptime(args.end, DATE_FMT_IN)
    except ValueError as e:
        sys.exit(f"❌ 日期解析失败：{e}\n请使用 YYYY-MM-DD 格式。")

    template_path = find_template(src_dir, args.template)
    
    # 计算总日期数
    total_dates = (end_dt - start_dt).days + 1
    print(f"模板文件: {template_path.name}")
    print(f"日期范围: {start_dt.strftime(DATE_FMT_IN)} 到 {end_dt.strftime(DATE_FMT_IN)}，共 {total_dates} 个日期")
    print(f"输出目录: {dst_dir}")
    print()

    created, skipped, overwritten = 0, 0, 0
    start_time = time.time()
    
    for i, d in enumerate(daterange_inclusive(start_dt, end_dt), 1):
        print_progress(i, total_dates, start_time, prefix="复制 ")
        
        out_name = f"{d.strftime(DATE_FMT_OUT)}.tif"
        out_path = dst_dir / out_name

        if out_path.exists():
            if args.overwrite:
                shutil.copy2(template_path, out_path)
                overwritten += 1
            else:
                skipped += 1
        else:
            shutil.copy2(template_path, out_path)
            created += 1

    elapsed_time = time.time() - start_time
    print(f"\n处理完成! 总耗时: {format_time(elapsed_time)}")
    print(f"总计: {total_dates}, 新建: {created}, 跳过: {skipped}, 覆盖: {overwritten}")
    print(f"输出目录: {dst_dir}")

if __name__ == "__main__":
    main()
