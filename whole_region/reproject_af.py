import os
from osgeo import gdal, osr


def batch_reproject_to_wgs84(input_folder, output_folder):
    """
    将文件夹内所有TIFF文件重投影到WGS84坐标系(EPSG:4326)，输出文件已存在则跳过

    参数:
        input_folder: 包含原始TIFF文件的文件夹路径
        output_folder: 重投影后TIFF文件的输出文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 设置GDAL配置选项
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')

    # 定义目标坐标系(WGS84)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)  # WGS84的EPSG代码

    # 遍历输入文件夹中的所有文件
    processed_count = 0
    skipped_count = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"wgs84_{filename}")

            # 检查输出文件是否已存在
            if os.path.exists(output_path):
                print(f"跳过: {filename} (输出文件已存在)")
                skipped_count += 1
                continue

            print(f"正在处理: {filename}")

            try:
                # 打开输入文件
                input_ds = gdal.Open(input_path)
                if input_ds is None:
                    print(f"无法打开文件: {input_path}")
                    continue

                # 检查是否已经是WGS84
                input_srs = input_ds.GetSpatialRef()
                if input_srs and input_srs.IsSame(target_srs):
                    print(f"文件 {filename} 已经是WGS84坐标系，跳过...")
                    input_ds = None
                    skipped_count += 1
                    continue

                # 设置重投影选项
                warp_options = gdal.WarpOptions(
                    dstSRS=target_srs,
                    resampleAlg=gdal.GRA_Bilinear,  # 双线性插值
                    multithread=True,  # 启用多线程
                    dstNodata=None  # 保留原始Nodata值
                )

                # 执行重投影
                gdal.Warp(
                    destNameOrDestDS=output_path,
                    srcDSOrSrcDSTab=input_ds,
                    options=warp_options
                )

                print(f"已保存重投影后的文件: {output_path}")
                processed_count += 1

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                # 如果处理失败，删除可能已创建的部分输出文件
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass
            finally:
                # 确保关闭数据集
                if 'input_ds' in locals() and input_ds is not None:
                    input_ds = None

    print(f"\n处理完成! 共处理了 {processed_count} 个文件，跳过了 {skipped_count} 个文件")
    print(f"输出目录: {output_folder}")


# 使用示例
if __name__ == "__main__":
    # 设置路径 (根据实际情况修改这些路径)
    input_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Firms_Aqua_500_Daily"  # 原始TIFF文件所在文件夹
    output_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Firms_Aqua_500_Daily_WGS84"  # 重投影后文件输出文件夹

    # 调用函数执行批量重投影
    batch_reproject_to_wgs84(input_folder, output_folder)