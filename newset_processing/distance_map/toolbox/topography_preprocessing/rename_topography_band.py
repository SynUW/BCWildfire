"""
使用GEDL，重命名波段名
"""
import os
from osgeo import gdal


def rename_bands(input_dir, output_dir):
    """
    修改GeoTIFF文件的波段名称
    :param input_dir: 输入目录，包含要处理的TIFF文件
    :param output_dir: 输出目录，保存修改后的文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 定义新的波段名称
    band_names = ["Aspect", "DEM", "HillShade", "Slope"]

    # 遍历输入目录中的所有TIFF文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"正在处理: {filename}")

            try:
                # 打开原始文件
                ds = gdal.Open(input_path, gdal.GA_ReadOnly)
                if ds is None:
                    print(f"无法打开文件: {input_path}")
                    continue

                # 检查波段数量
                num_bands = ds.RasterCount
                if num_bands < 4:
                    print(f"文件 {filename} 只有 {num_bands} 个波段，需要至少4个波段")
                    ds = None
                    continue

                # 创建输出文件
                driver = gdal.GetDriverByName('GTiff')
                out_ds = driver.CreateCopy(output_path, ds, 0)

                # 设置波段名称
                for i in range(1, 5):  # 波段索引从1开始
                    band = out_ds.GetRasterBand(i)
                    band.SetDescription(band_names[i - 1])
                    band.FlushCache()

                print(f"成功处理: {filename}")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
            finally:
                # 确保关闭数据集
                if 'ds' in locals():
                    ds = None
                if 'out_ds' in locals():
                    out_ds = None


if __name__ == '__main__':
    # 配置路径 (根据实际情况修改)
    input_directory = r"D:\wildfire_dataset\self_built\drivers\Topography_everyday"  # 输入的TIFF文件目录
    output_directory = r"D:\wildfire_dataset\self_built\drivers\Topography_everyday_rename"  # 输出目录

    # 执行波段重命名
    rename_bands(input_directory, output_directory)