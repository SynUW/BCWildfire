import os
from osgeo import gdal, ogr, osr


def get_projection_epsg(dataset):
    """获取数据集的空间参考EPSG代码"""
    crs = dataset.GetSpatialRef()
    if crs is not None:
        return crs.GetAuthorityCode(None)
    return None


def is_wgs84(dataset):
    """检查数据集是否是WGS84坐标系"""
    epsg_code = get_projection_epsg(dataset)
    return epsg_code == '4326'


def clip_tifs_with_shapefile(input_folder, output_folder, shapefile_path):
    """
    使用shapefile裁剪文件夹内所有的tif文件，如果不是WGS84则先转换

    参数:
        input_folder: 包含待裁剪TIFF文件的文件夹路径
        output_folder: 裁剪后TIFF文件的输出文件夹路径
        shapefile_path: 用于裁剪的shapefile文件路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 设置GDAL配置选项
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')

    # 打开shapefile
    shapefile = ogr.Open(shapefile_path)
    if shapefile is None:
        raise RuntimeError(f"无法打开shapefile: {shapefile_path}")

    # 获取shapefile的空间参考
    layer = shapefile.GetLayer()
    shape_srs = layer.GetSpatialRef()

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"clipped_{filename}")

            print(f"\n正在处理: {filename}")

            # 打开输入TIFF文件
            input_ds = gdal.Open(input_path)
            if input_ds is None:
                print(f"无法打开文件: {input_path}")
                continue

            # 检查是否需要转换坐标系
            need_reproject = not is_wgs84(input_ds)

            # 准备裁剪选项
            warp_options = gdal.WarpOptions(
                cutlineDSName=shapefile_path,
                cropToCutline=True,
                dstNodata=None,
                multithread=True,
                # 如果需要重投影，则设置目标坐标系为WGS84
                dstSRS='EPSG:4326' if need_reproject else None
            )

            # 执行裁剪操作
            try:
                if need_reproject:
                    print("检测到非WGS84坐标系，将转换为WGS84后再裁剪")

                gdal.Warp(
                    destNameOrDestDS=output_path,
                    srcDSOrSrcDSTab=input_ds,
                    options=warp_options
                )
                print(f"已保存裁剪后的文件: {output_path}")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
            finally:
                # 关闭数据集
                input_ds = None

    # 关闭shapefile
    shapefile = None
    print("\n所有文件处理完成！")


# 使用示例
if __name__ == "__main__":
    # 设置路径 (根据实际情况修改这些路径)
    input_tif_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/DAILY_500_WGS84_BCBoundingbox"
    shapefile_path = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_boundary/british_columnbia_no_crs_boundingBox.shp"
    output_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/ERA5/DAILY_500_WGS84_BCBoundary"

    # 调用函数执行裁剪
    clip_tifs_with_shapefile(input_tif_folder, output_folder, shapefile_path)