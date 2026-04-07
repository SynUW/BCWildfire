import rasterio
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm


def calculate_r2(tif1_path, tif2_path, threshold=2):
    """
    计算两张TIF图像之间的R²，过滤掉小于阈值的像元
    
    参数:
        tif1_path: 第一张TIF图像路径
        tif2_path: 第二张TIF图像路径
        threshold: 像元值阈值，默认为2
    
    返回:
        r2: 决定系数
        valid_pixels: 有效像元数量
    """
    start_time = time.time()
    
    print("正在读取TIF文件...")
    with rasterio.open(tif1_path) as src1, rasterio.open(tif2_path) as src2:
        # 检查图像大小是否一致
        if src1.shape != src2.shape:
            raise ValueError("两张图像大小不一致")
        
        # 读取数据
        data1 = src1.read(1)  # 读取第一个波段
        data2 = src2.read(1)
        
        print(f"图像大小: {data1.shape}")
        
        # 创建掩码，过滤掉小于阈值的像元
        mask = (data1 >= threshold) & (data2 >= threshold)
        valid_pixels = np.sum(mask)
        
        if valid_pixels == 0:
            raise ValueError("没有有效的像元（所有像元都小于阈值）")
        
        print(f"有效像元数量: {valid_pixels}")
        
        # 提取有效像元的值
        valid_data1 = data1[mask]
        valid_data2 = data2[mask]
        
        # 计算相关系数
        correlation = np.corrcoef(valid_data1, valid_data2)[0, 1]
        r2 = correlation ** 2
        
        print(f"R²值: {r2:.4f}")
        print(f"相关系数: {correlation:.4f}")
        
        # 计算一些基本统计信息
        print("\n统计信息:")
        print(f"图像1 - 均值: {np.mean(valid_data1):.2f}, 标准差: {np.std(valid_data1):.2f}")
        print(f"图像2 - 均值: {np.mean(valid_data2):.2f}, 标准差: {np.std(valid_data2):.2f}")
        
        # 计算RMSE
        rmse = np.sqrt(np.mean((valid_data1 - valid_data2) ** 2))
        print(f"RMSE: {rmse:.2f}")
        
        # 计算MAE
        mae = np.mean(np.abs(valid_data1 - valid_data2))
        print(f"MAE: {mae:.2f}")
    
    print(f"\n处理完成! 总耗时: {time.time() - start_time:.2f} 秒")
    
    return r2, valid_pixels


def main():
    # 输入文件路径
    tif1_path = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/sum_bands/sum_last_365_bands.tif"
    tif2_path = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/sum_bands/sum_first_bands.tif"
    
    # 计算R²
    r2, valid_pixels = calculate_r2(tif1_path, tif2_path, threshold=2)
    
    # 保存结果到文本文件
    output_file = "r2_results.txt"
    with open(output_file, "w") as f:
        f.write(f"R²值: {r2:.4f}\n")
        f.write(f"有效像元数量: {valid_pixels}\n")
        f.write(f"阈值: 2\n")


if __name__ == "__main__":
    main()