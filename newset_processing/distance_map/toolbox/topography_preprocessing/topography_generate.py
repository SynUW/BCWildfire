"""
使用shapefile文件并加buffer裁剪tif文件
并用shapefile文件的文件名命名裁剪后的tif文件
"""
import os
import glob
import tempfile
from osgeo import gdal, ogr, osr
import numpy as np


def get_processing_region(shapefile_path):
    """获取处理区域，如果外接矩形小于1°×1°，则扩展到1°×1°"""
    try:
        # 打开Shapefile
        shape_ds = ogr.Open(shapefile_path)
        if shape_ds is None:
            raise ValueError(f"无法打开Shapefile: {shapefile_path}")
        shape_layer = shape_ds.GetLayer()

        # 获取整个图层的边界
        extent = shape_layer.GetExtent()
        min_x, max_x, min_y, max_y = extent

        # 计算当前宽度和高度
        width_deg = max_x - min_x
        height_deg = max_y - min_y

        # 如果小于1度，则扩展到1度
        if width_deg < 1.0 or height_deg < 1.0:
            # 计算中心点
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            # 新的半宽和半高
            half_size = 0.5  # 1度的一半

            # 计算新的边界
            new_min_x = center_x - half_size
            new_max_x = center_x + half_size
            new_min_y = center_y - half_size
            new_max_y = center_y + half_size

            # 创建新的矩形几何
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(new_min_x, new_min_y)
            ring.AddPoint(new_max_x, new_min_y)
            ring.AddPoint(new_max_x, new_max_y)
            ring.AddPoint(new_min_x, new_max_y)
            ring.AddPoint(new_min_x, new_min_y)

            polygon = ogr.Geometry(ogr.wkbPolygon)
            polygon.AddGeometry(ring)

            return polygon
        else:
            # 如果已经大于等于1度，返回原始外接矩形
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(min_x, min_y)
            ring.AddPoint(max_x, min_y)
            ring.AddPoint(max_x, max_y)
            ring.AddPoint(min_x, max_y)
            ring.AddPoint(min_x, min_y)

            polygon = ogr.Geometry(ogr.wkbPolygon)
            polygon.AddGeometry(ring)

            return polygon

    except Exception as e:
        print(f"获取处理区域时出错: {str(e)}")
        return None
    finally:
        if 'shape_ds' in locals():
            shape_ds = None


def create_temp_shapefile_with_region(region_geom, original_shp, output_shp):
    """创建包含处理区域的临时shapefile"""
    try:
        # 获取原始shapefile的驱动和空间参考
        driver = ogr.GetDriverByName('ESRI Shapefile')
        original_ds = ogr.Open(original_shp)
        if original_ds is None:
            raise ValueError(f"无法打开原始Shapefile: {original_shp}")

        srs = original_ds.GetLayer().GetSpatialRef()

        # 创建输出shapefile
        if os.path.exists(output_shp):
            driver.DeleteDataSource(output_shp)

        out_ds = driver.CreateDataSource(output_shp)
        out_layer = out_ds.CreateLayer('processing_region', srs, ogr.wkbPolygon)

        # 创建字段（复制原始shapefile的第一个字段）
        in_layer = original_ds.GetLayer()
        in_feature_def = in_layer.GetLayerDefn()
        if in_feature_def.GetFieldCount() > 0:
            field_def = in_feature_def.GetFieldDefn(0)
            out_layer.CreateField(field_def)

        # 创建要素
        out_feature = ogr.Feature(out_layer.GetLayerDefn())
        out_feature.SetGeometry(region_geom)

        # 设置字段值（如果有）
        if in_feature_def.GetFieldCount() > 0:
            first_feature = in_layer.GetFeature(0)
            out_feature.SetField(0, first_feature.GetField(0))

        out_layer.CreateFeature(out_feature)

        return True
    except Exception as e:
        print(f"创建临时shapefile时出错: {str(e)}")
        return False
    finally:
        if 'original_ds' in locals():
            original_ds = None
        if 'out_ds' in locals():
            out_ds = None


def clip_raster_with_shapefile(raster_path, shapefile_path, output_path):
    """使用Shapefile裁剪栅格"""
    temp_shp = None
    try:
        # 打开栅格文件
        raster_ds = gdal.Open(raster_path)
        if raster_ds is None:
            raise ValueError(f"无法打开栅格文件: {raster_path}")

        # 获取处理区域
        region_geom = get_processing_region(shapefile_path)
        if region_geom is None:
            raise ValueError("无法获取处理区域")

        # 创建临时shapefile
        temp_dir = tempfile.gettempdir()
        temp_shp = os.path.join(temp_dir, f"temp_region_{os.getpid()}.shp")

        if not create_temp_shapefile_with_region(region_geom, shapefile_path, temp_shp):
            raise ValueError("创建临时处理区域shapefile失败")

        # 执行裁剪
        options = gdal.WarpOptions(
            cutlineDSName=temp_shp,
            cropToCutline=True,
            dstNodata=np.nan
        )

        result = gdal.Warp(output_path, raster_ds, options=options)
        if result is None:
            raise ValueError("裁剪操作失败")

        return True
    except Exception as e:
        print(f"裁剪过程中出错: {str(e)}")
        return False
    finally:
        # 确保所有数据集被正确关闭
        if 'raster_ds' in locals():
            raster_ds = None

        # 清理临时文件
        if temp_shp and os.path.exists(temp_shp):
            try:
                driver = ogr.GetDriverByName('ESRI Shapefile')
                driver.DeleteDataSource(temp_shp)
                for ext in ['.shx', '.dbf', '.prj']:
                    temp_file = temp_shp.replace('.shp', ext)
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except Exception as e:
                print(f"删除临时文件时出错: {str(e)}")


def batch_clip_raster_with_shapefiles(raster_path, shapefile_folder, output_folder):
    """批量使用Shapefile裁剪栅格"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有Shapefile
    shapefiles = glob.glob(os.path.join(shapefile_folder, '*.shp'))

    total = len(shapefiles)
    success = 0

    for i, shp_path in enumerate(shapefiles, 1):
        # 从Shapefile路径获取名称
        shp_name = os.path.splitext(os.path.basename(shp_path))[0]

        # 设置输出路径
        output_path = os.path.join(output_folder, f"{shp_name}.tif")

        print(f"正在处理 ({i}/{total}): {shp_name}")

        try:
            # 执行裁剪
            if clip_raster_with_shapefile(raster_path, shp_path, output_path):
                print(f"成功裁剪并保存到: {output_path}")
                success += 1
            else:
                print(f"处理 {shp_name} 失败")
        except Exception as e:
            print(f"处理 {shp_name} 时出错: {str(e)}")

    print(f"\n处理完成! 成功: {success}/{total}")


if __name__ == '__main__':
    # 配置路径 (根据实际情况修改)
    input_raster = r"D:\wildfire_dataset\self_built\drivers\topographies.tif"  # 要裁剪的GeoTIFF文件
    shapefile_dir = r"D:\wildfire_dataset\self_built\GlobFire Fire Perimeters_BC_Filtered_FierEventsDate"  # 包含Shapefile的文件夹
    output_dir = r"D:\wildfire_dataset\self_built\drivers\Topography_clip"  # 输出文件夹

    # 执行批量裁剪
    batch_clip_raster_with_shapefiles(input_raster, shapefile_dir, output_dir)
