import os
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm
import pandas as pd


def check_folder_consistency(folder_path):
    """检查单个文件夹内所有GeoTIFF文件的一致性"""
    file_info = []
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]



    if not tif_files:
        return None

    # 检查第一个文件作为基准
    first_file = os.path.join(folder_path, tif_files[0])
    with rasterio.open(first_file) as src:
        base_info = {
            'bands': src.count,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds,
            'crs': src.crs,
            'transform': src.transform
        }

    # 检查文件夹内所有文件
    inconsistent_files = []
    for filename in tqdm(tif_files, desc=f"Checking {os.path.basename(folder_path)}"):
        filepath = os.path.join(folder_path, filename)
        try:
            with rasterio.open(filepath) as src:
                current_info = {
                    'bands': src.count,
                    'width': src.width,
                    'height': src.height,
                    'bounds': src.bounds,
                    'crs': src.crs,
                    'transform': src.transform
                }

                # 比较关键属性
                if (current_info['bands'] != base_info['bands'] or
                        current_info['width'] != base_info['width'] or
                        current_info['height'] != base_info['height'] or
                        current_info['bounds'] != base_info['bounds'] or
                        current_info['crs'] != base_info['crs'] or
                        not transforms_equal(current_info['transform'], base_info['transform'])):
                    inconsistent_files.append({
                        'file': filename,
                        'differences': compare_info(base_info, current_info)
                    })

        except Exception as e:
            inconsistent_files.append({
                'file': filename,
                'error': str(e)
            })

    # 汇总结果
    result = {
        'folder': folder_path,
        'base_info': base_info,
        'total_files': len(tif_files),
        'inconsistent_files': inconsistent_files,
        'is_consistent': len(inconsistent_files) == 0
    }
    return result


def transforms_equal(transform1, transform2):
    """比较两个仿射变换是否相同"""
    if isinstance(transform1, Affine) and isinstance(transform2, Affine):
        return all(abs(a - b) < 1e-10 for a, b in zip(transform1, transform2))
    return transform1 == transform2


def compare_info(base, current):
    """比较两个文件信息的差异"""
    differences = []
    for key in ['bands', 'width', 'height', 'bounds', 'crs']:
        if base[key] != current[key]:
            differences.append(f"{key}: {base[key]} ≠ {current[key]}")

    if not transforms_equal(base['transform'], current['transform']):
        differences.append(f"transform: {base['transform']} ≠ {current['transform']}")

    return "; ".join(differences) if differences else "No significant differences"


def compare_across_folders(folder_paths):
    """比较多个文件夹之间的空间属性"""
    folder_results = []
    for folder in folder_paths:
        result = check_folder_consistency(folder)
        if result:
            folder_results.append(result)

    if not folder_results:
        print("没有找到有效的GeoTIFF文件")
        return

    # 提取各文件夹的基准信息进行比较
    comparison = []
    base_info = folder_results[0]['base_info']

    for result in folder_results:
        folder_name = os.path.basename(result['folder'])
        comparison.append({
            'Folder': folder_name,
            'Bands': result['base_info']['bands'],
            'Width': result['base_info']['width'],
            'Height': result['base_info']['height'],
            'CRS': str(result['base_info']['crs']),
            'Bounds': result['base_info']['bounds'],
            'Internal_Consistency': "✔" if result['is_consistent'] else f"✖ ({len(result['inconsistent_files'])} files)"
        })

    # 创建比较表格
    df = pd.DataFrame(comparison)
    print("\n跨文件夹比较结果:")
    print(df.to_string(index=False))

    # 检查跨文件夹一致性
    all_consistent = True
    for key in ['Bands', 'Width', 'Height', 'CRS', 'Bounds']:
        if len(df[key].unique()) > 1:
            print(f"\n警告: 文件夹间 {key} 不一致!")
            print(df[['Folder', key]].to_string(index=False))
            all_consistent = False

    if all_consistent:
        print("\n所有文件夹的空间属性完全一致")
    else:
        print("\n注意: 不同文件夹间的空间属性存在差异")


if __name__ == '__main__':
    # 替换为你的文件夹路径列表
    folders_to_check = [
        r'Y:\mnt\raid\zhengsen\wildfire_dataset\self_built_materials\LAI_LULC\LAI_500',
        r'Y:\mnt\raid\zhengsen\wildfire_dataset\self_built_materials\LAI_LULC\LULC_500',
        r'Y:\mnt\raid\zhengsen\wildfire_dataset\self_built_materials\Topo_Distance_500',
        r'Y:\mnt\raid\zhengsen\wildfire_dataset\self_built_materials\Reflection_500_merge',
        r'Y:\mnt\raid\zhengsen\wildfire_dataset\self_built_materials\ERA5\DAILY_500',
        r'Y:\mnt\raid\zhengsen\wildfire_dataset\self_built_materials\FIRMS\Terra_Aqua_Daily',
    ]

    # 检查每个文件夹内部的一致性
    for folder in folders_to_check:
        result = check_folder_consistency(folder)
        if result:
            print(f"\n检查结果: {os.path.basename(folder)}")
            print(f"文件数量: {result['total_files']}")
            print(f"内部一致性: {'是' if result['is_consistent'] else '否'}")

            if not result['is_consistent']:
                print("\n不一致的文件:")
                for item in result['inconsistent_files']:
                    if 'error' in item:
                        print(f"{item['file']} - 错误: {item['error']}")
                    else:
                        print(f"{item['file']} - 差异: {item['differences']}")

    # 比较不同文件夹间的属性
    compare_across_folders(folders_to_check)