import os
from osgeo import gdal, ogr


def clip_and_resample(input_path, output_path, shapefile_path, target_width=5565, target_height=2602):
    """
    先裁剪后重采样栅格数据

    参数:
        input_path: 输入栅格文件路径
        output_path: 输出栅格文件路径
        shapefile_path: 用于裁剪的shapefile路径
        target_width: 目标宽度(像元数)
        target_height: 目标高度(像元数)
    """
    # 第一步：使用shapefile裁剪
    print("步骤1/2: 使用shapefile裁剪...")
    temp_clip_path = os.path.join(os.path.dirname(output_path), "temp_clipped.tif")

    # 裁剪选项
    clip_options = gdal.WarpOptions(
        cutlineDSName=shapefile_path,
        cropToCutline=True,
        dstNodata=None,
        multithread=True
    )

    gdal.Warp(
        destNameOrDestDS=temp_clip_path,
        srcDSOrSrcDSTab=input_path,
        options=clip_options
    )

    # 第二步：重采样到指定尺寸
    print("步骤2/2: 重采样到指定尺寸...")
    clipped_ds = gdal.Open(temp_clip_path)
    if clipped_ds is None:
        raise RuntimeError("裁剪后的临时文件创建失败")

    try:
        # 获取裁剪后的地理信息
        geo_transform = clipped_ds.GetGeoTransform()
        projection = clipped_ds.GetProjection()

        # 计算裁剪后的空间范围
        min_x = geo_transform[0]
        max_y = geo_transform[3]
        max_x = min_x + geo_transform[1] * clipped_ds.RasterXSize
        min_y = max_y + geo_transform[5] * clipped_ds.RasterYSize

        # 重采样选项
        resample_options = gdal.WarpOptions(
            format='GTiff',
            width=target_width,
            height=target_height,
            outputBounds=(min_x, min_y, max_x, max_y),  # 保持裁剪后的范围
            resampleAlg=gdal.GRA_Bilinear,  # 对于FIRMS，用的是nearst
            multithread=True
        )

        gdal.Warp(
            destNameOrDestDS=output_path,
            srcDSOrSrcDSTab=clipped_ds,
            options=resample_options
        )

    finally:
        # 确保关闭数据集并删除临时文件
        clipped_ds = None
        if os.path.exists(temp_clip_path):
            os.remove(temp_clip_path)

    print(f"处理完成! 结果保存到: {output_path}")


def batch_process(input_folder, output_folder, shapefile_path, target_width=5565, target_height=2602):
    """
    批量处理文件夹中的所有TIFF文件

    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        shapefile_path: 用于裁剪的shapefile路径
        target_width: 目标宽度(像元数)
        target_height: 目标高度(像元数)
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 设置GDAL多线程
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')

    # 遍历输入文件夹
    processed_count = 0
    filenames = os.listdir(input_folder)
    filenames.reverse()
    for filename in filenames:
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"processed_{filename}")

            # 检查输出文件是否已存在
            if os.path.exists(output_path):
                print(f"跳过: {filename} (输出文件已存在)")
                continue

            print(f"\n处理文件: {filename}")
            try:
                clip_and_resample(input_path, output_path, shapefile_path, target_width, target_height)
                processed_count += 1
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                # 删除可能已创建的部分输出文件
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass

    print(f"\n处理完成! 共处理了 {processed_count} 个文件")


# 使用示例
if __name__ == "__main__":
    # 设置路径 (根据实际情况修改)
    input_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500_merge_TerraAquaWGS84"
    output_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/Reflection_500_merge_TerraAquaWGS84_clip"

    shapefile_path = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_boundary/british_columnbia_no_crs_boundingBox.shp"

    # 调用批量处理函数
    batch_process(input_folder, output_folder, shapefile_path, 5565, 2602)