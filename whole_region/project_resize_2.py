import os
from osgeo import gdal

# 配置参数
INPUT_DIR = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Terra_Aqua_Daily"  # 替换为输入文件夹路径
OUTPUT_DIR = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Terra_Aqua_Daily_resize"  # 替换为输出文件夹路径
TARGET_WIDTH, TARGET_HEIGHT = 3689, 3142  # 目标尺寸
TARGET_EXTENT = (  # 目标空间范围 (minX, minY, maxX, maxY)
    -1129826.0074664412532002,
    5350356.9789362540468574,
    719173.9925335587467998,
    6921356.9789362540468574
)

from multiprocessing import Pool, cpu_count


def process_file(args):
    """单文件处理函数（供多线程调用）"""
    filename, input_dir, output_dir = args
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"resized_{filename}")

    gdal.Warp(
        output_path,
        input_path,
        format="GTiff",
        outputBounds=TARGET_EXTENT,
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
        resampleAlg=gdal.GRA_NearestNeighbour,
        creationOptions=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"],
        # 启用GDAL内部多线程
        options=["NUM_THREADS=ALL_CPUS", "GDAL_CACHEMAX=512"]
    )
    return f"完成: {filename}"


def process_tifs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.tif')]

    # 根据CPU核心数设置线程数（留2个核心给系统）
    num_threads = max(1, cpu_count() - 2)
    with Pool(processes=num_threads) as pool:
        results = pool.imap_unordered(
            process_file,
            [(f, input_dir, output_dir) for f in files]
        )
        for res in results:
            print(res)


if __name__ == "__main__":
    print("===== 开始处理 =====")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"目标尺寸: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"目标范围: {TARGET_EXTENT}")

    process_tifs(INPUT_DIR, OUTPUT_DIR)

    print("===== 处理完成 =====")