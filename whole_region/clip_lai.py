import os
from osgeo import gdal, ogr


def clip_tifs_with_shapefile(input_folder, output_folder, shapefile_path):
    """
    使用shapefile裁剪文件夹内所有的tif文件

    参数:
        input_folder: 包含待裁剪TIFF文件的文件夹路径
        output_folder: 裁剪后TIFF文件的输出文件夹路径
        shapefile_path: 用于裁剪的shapefile文件路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 设置GDAL配置选项，裁剪时使用所有可用线程
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')

    # 打开shapefile
    shapefile = ogr.Open(shapefile_path)
    if shapefile is None:
        raise RuntimeError(f"无法打开shapefile: {shapefile_path}")

    # 获取shapefile的几何边界
    layer = shapefile.GetLayer()
    # 如果需要，可以在这里添加空间过滤或其他几何处理

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"clipped_{filename}")

            print(f"正在处理: {filename}")

            # 设置裁剪选项
            warp_options = gdal.WarpOptions(
                cutlineDSName=shapefile_path,  # 裁剪用的shapefile
                cropToCutline=True,  # 将输出范围设置为裁剪几何的范围
                dstNodata=None,  # 根据需要设置Nodata值
                multithread=True,  # 启用多线程
            )

            # 执行裁剪操作
            try:
                gdal.Warp(
                    destNameOrDestDS=output_path,
                    srcDSOrSrcDSTab=input_path,
                    options=warp_options
                )
                print(f"已保存裁剪后的文件: {output_path}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

    # 关闭shapefile
    shapefile = None
    print("所有文件处理完成！")


# 使用示例
if __name__ == "__main__":
    # 设置路径 (根据实际情况修改这些路径)
    input_tif_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LAI_LULC/MODIS_LULC"
    output_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LULC_BCBoundingbox"
    shapefile_path = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_boundary/british_columnbia_no_crs_boundingBox.shp"

    # 调用函数执行裁剪
    clip_tifs_with_shapefile(input_tif_folder, output_folder, shapefile_path)