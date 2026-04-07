import os
import shutil
from tqdm import tqdm

def copy_files(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    files = os.listdir(source_folder)
    for filename in tqdm(files):
        # 构建旧文件的完整路径
        old_filepath = os.path.join(source_folder, filename)

        # 检查是否是文件（而不是子文件夹）
        if os.path.isfile(old_filepath):
            # 构建新文件的完整路径
            new_filepath = os.path.join(target_folder, filename)

            # 复制文件
            shutil.copy2(old_filepath, new_filepath)
            # print(f"Copied: {old_filepath} -> {new_filepath}")

if __name__ == "__main__":
    # 设置源文件夹和目标文件夹路径
    source_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Firms_Terra_500_Daily"  # 源文件夹路径
    target_folder = r"/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/Terra_Aqua_Daily"

    # 调用函数
    copy_files(source_folder, target_folder)