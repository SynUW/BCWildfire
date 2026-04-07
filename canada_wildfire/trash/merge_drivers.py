import os
import h5py
import numpy as np

def merge_sampling_flat_h5(year, h5_dir, driver_order, output_dir):
    # 1. 收集该年份所有驱动的h5文件
    h5_files = []
    for driver in driver_order:
        pattern = f'wildfire_samples_{year}_{driver}.h5'
        path = os.path.join(h5_dir, pattern)
        if os.path.exists(path):
            h5_files.append(path)
        else:
            print(f"警告: 未找到 {path}")

    if len(h5_files) < 2:
        print(f"{year} 年可用驱动文件不足2个，跳过")
        return

    # 2. 统计所有公共key
    key_sets = []
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as f:
            key_sets.append(set(f.keys()))
    common_keys = set.intersection(*key_sets)
    print(f"{year}年公共数据集数量: {len(common_keys)}")

    # 3. 合并并写入新h5
    output_path = os.path.join(output_dir, f'wildfire_samples_{year}_merged.h5')
    with h5py.File(output_path, 'w') as fout:
        fout.attrs['driver_order'] = ','.join(driver_order)
        for key in sorted(common_keys):
            datas = []
            for h5_path in h5_files:
                with h5py.File(h5_path, 'r') as f:
                    datas.append(f[key][:])
            # 按channel维拼接
            arrs = [np.array(d) for d in datas]
            # 判断 shape 是否为 (time, c, h, w) 或 (c, h, w)
            if arrs[0].ndim == 4:
                merged = np.concatenate(arrs, axis=1)
            elif arrs[0].ndim == 3:
                merged = np.concatenate(arrs, axis=0)
            else:
                raise ValueError(f"不支持的数据shape: {arrs[0].shape}")
            fout.create_dataset(key, data=merged, dtype=merged.dtype, compression='gzip', compression_opts=6)
    print(f"{year}年合并完成，输出: {output_path}")

if __name__ == '__main__':
    h5_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/samples_right'  # sampling_flat.py输出目录
    output_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/samples_merged_right'  # 合并输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 指定驱动顺序，firms fire detection 必须在首位
    driver_order = [
        'Firms_Detection_resampled',
        'ERA5_multi_bands',
        'LULC_BCBoundingbox_resampled',
        'Topo_Distance_WGS84_resize_resampled',
        'NDVI_EVI',
        'Reflection_500_merge_TerraAquaWGS84_clip',
        'MODIS_Terra_Aqua_B20_21_merged_resampled',
        'MOD21A1DN_multibands_filtered_resampled',
        'LAI_BCBoundingbox_resampled',
        # ... 其它驱动
    ]
    # 遍历所有年份
    years = sorted({f.split('_')[2] for f in os.listdir(h5_dir) if f.endswith('.h5')})
    for year in years:
        merge_sampling_flat_h5(year, h5_dir, driver_order, output_dir)