import os
import numpy as np
from osgeo import gdal
from sklearn.decomposition import PCA
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

warnings.filterwarnings("ignore")


def process_single_file(tif_file, input_path, output_folder):
    input_file = os.path.join(input_path, tif_file)
    output_file = os.path.join(output_folder, f"pca_{tif_file}")

    try:
        # 读取原始GeoTIFF文件
        ds = gdal.Open(input_file)
        if ds is None:
            print(f"Failed to open {input_file}")
            return

        # 获取原始文件信息
        bands = ds.RasterCount
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        projection = ds.GetProjection()
        geotransform = ds.GetGeoTransform()

        # 分块读取数据以减少内存使用
        block_size = 1024  # 可以根据内存大小调整
        pca_bands = [np.zeros((rows, cols), dtype=np.float32) for _ in range(3)]

        for y in range(0, rows, block_size):
            y_end = min(y + block_size, rows)
            block_rows = y_end - y

            for x in range(0, cols, block_size):
                x_end = min(x + block_size, cols)
                block_cols = x_end - x

                # 读取当前块的所有波段数据
                block_data = np.zeros((block_rows * block_cols, bands))
                for b in range(bands):
                    band = ds.GetRasterBand(b + 1)
                    block = band.ReadAsArray(x, y, block_cols, block_rows)
                    block_data[:, b] = block.flatten()

                # 执行PCA变换
                pca = PCA(n_components=3)
                pca_data = pca.fit_transform(block_data)

                # 将结果填充到输出数组中
                for i in range(3):
                    pca_bands[i][y:y_end, x:x_end] = pca_data[:, i].reshape((block_rows, block_cols))

        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_file, cols, rows, 3, gdal.GDT_Float32)
        if out_ds is None:
            print(f"Failed to create output file {output_file}")
            return

        # 设置地理参考和投影信息
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)

        # 写入PCA结果
        for i in range(3):
            out_band = out_ds.GetRasterBand(i + 1)
            out_band.WriteArray(pca_bands[i])
            out_band.SetDescription(f"PCA Component {i + 1}")
            out_band.FlushCache()

        # 关闭数据集
        out_ds = None
        ds = None

        return f"Processed: {input_file} -> {output_file}"
    except Exception as e:
        return f"Error processing {input_file}: {str(e)}"


def process_geotiff_pca_parallel(input_path, output_folder, n_jobs=-1):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中所有tif文件
    tif_files = [f for f in os.listdir(input_path) if f.endswith(('.tif', '.tiff'))]

    # 设置并行任务数
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    print(f"Starting parallel processing with {n_jobs} workers...")

    # 并行处理文件
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_file)(tif_file, input_path, output_folder)
        for tif_file in tqdm(tif_files, desc="Processing files")
    )

    # 打印处理结果
    for result in results:
        if result:
            print(result)


if __name__ == "__main__":
    input_folder = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/merged_daily_data"
    output_folder = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/merged_daily_data_pca"

    # 使用所有可用的CPU核心
    process_geotiff_pca_parallel(input_folder, output_folder, n_jobs=-1)

    print("All files processed!")