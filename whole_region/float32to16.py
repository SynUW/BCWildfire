import os
import numpy as np
from osgeo import gdal


def convert_float32_to_float16(input_path, output_path):
    """
    将GeoTIFF从Float32转换为Float16

    参数:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    # 打开输入文件
    src_ds = gdal.Open(input_path)
    if src_ds is None:
        raise RuntimeError(f"无法打开输入文件: {input_path}")

    try:
        # 获取输入文件信息
        band = src_ds.GetRasterBand(1)
        dtype = band.DataType
        nodata = band.GetNoDataValue()

        # 检查输入数据类型是否为Float32
        if gdal.GetDataTypeName(dtype) != 'Float32':
            print(
                f"警告: 输入文件 {os.path.basename(input_path)} 不是Float32类型(实际是{gdal.GetDataTypeName(dtype)})，跳过转换")
            return False

        # 读取数据
        data = band.ReadAsArray()

        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            output_path,
            src_ds.RasterXSize,
            src_ds.RasterYSize,
            1,  # 波段数
            gdal.GDT_Float32,  # 注意: GDAL没有直接的Float16类型，我们将使用Float32存储
            options=['COMPRESS=LZW', 'PREDICTOR=3']  # 使用LZW压缩
        )

        # 设置地理参考和投影
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())

        # 转换数据类型并写入
        output_band = dst_ds.GetRasterBand(1)

        # 将数据转换为float16(在内存中)
        data_float16 = data.astype(np.float16)

        # 写回float32(因为GDAL没有直接的float16支持)
        output_band.WriteArray(data_float16.astype(np.float32))

        # 设置NoData值
        if nodata is not None:
            output_band.SetNoDataValue(float(nodata))

        # 计算统计信息
        output_band.FlushCache()
        output_band.ComputeStatistics(False)

        print(f"成功转换并保存到: {output_path}")
        return True

    finally:
        # 确保关闭所有数据集
        if 'src_ds' in locals() and src_ds is not None:
            src_ds = None
        if 'dst_ds' in locals() and dst_ds is not None:
            dst_ds = None


def batch_convert(input_folder, output_folder):
    """
    批量转换文件夹中的所有GeoTIFF文件

    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹
    processed_count = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"float16_{filename}")

            # 检查输出文件是否已存在
            if os.path.exists(output_path):
                print(f"跳过: {filename} (输出文件已存在)")
                continue

            print(f"\n处理文件: {filename}")
            try:
                if convert_float32_to_float16(input_path, output_path):
                    processed_count += 1
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                # 删除可能已创建的部分输出文件
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass

    print(f"\n处理完成! 共转换了 {processed_count} 个文件")


# 使用示例
if __name__ == "__main__":
    # 设置路径 (根据实际情况修改)
    input_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Topo_Distance_WGS84_resize"
    output_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Topo_Distance_WGS84_resize_float16"

    # 调用批量转换函数
    batch_convert(input_folder, output_folder)