import os
import numpy as np
from PIL import Image
import random

# 指定输入文件夹和输出PNG路径
input_folder = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/pixel_data_selection"
output_png = "pixel_positions_selection.png"

# 根据您的坐标范围调整图像尺寸
# x坐标范围是0-2601 → height=2602
# y坐标范围是0-5564 → width=5565
width, height = 5565, 2602

# 初始化一个全零图像 (注意这里的行列顺序)
image = np.zeros((height, width), dtype=np.uint8)

# 获取所有npy文件并随机抽样10%
all_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
sample_size = int(len(all_files) * 1)  # 计算10%的样本量
sampled_files = random.sample(all_files, sample_size)

print(f"总文件数: {len(all_files)}")
print(f"抽样文件数: {sample_size}")

# 遍历抽样后的文件
for filename in sampled_files:
    # 从文件名中提取x和y坐标
    name_without_ext = os.path.splitext(filename)[0]  # 移除.npy后缀
    parts = name_without_ext.split('_')
    if len(parts) >= 2:
        try:
            x = int(parts[0])  # 第一个部分作为x坐标
            y = int(parts[1])  # 第二个部分作为y坐标
            # 确保坐标在图像范围内
            if 0 <= x < height and 0 <= y < width:  # 注意这里的检查
                image[x, y] = 255  # 使用x, y来标记位置
        except ValueError:
            print(f"跳过无效的文件名: {filename}")

# 将numpy数组转换为PIL图像并保存为PNG
img = Image.fromarray(image)
img.save(output_png)
print(f"已保存像素位置标记图: {output_png}")