import os
import glob
import logging
from tqdm import tqdm
import re

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('File_Name_Formatter')

def format_filename(filename):
    """将文件名从driver_yyyymmdd.tif格式转换为driver_yyyy_mm_dd.tif格式"""
    # 匹配driver_yyyymmdd.tif格式
    match = re.match(r'(.*?)_(\d{4})(\d{2})(\d{2})\.tif$', filename)
    if match:
        prefix = match.group(1)
        year = match.group(2)
        month = match.group(3)
        day = match.group(4)
        return f"{prefix}_{year}_{month}_{day}.tif"
    return filename

def process_directory(input_dir):
    """处理目录及其子目录中的所有文件"""
    try:
        # 获取所有子目录
        subdirs = [d for d in glob.glob(os.path.join(input_dir, '*')) if os.path.isdir(d)]
        if not subdirs:
            subdirs = [input_dir]  # 如果没有子目录，就处理当前目录
        
        total_files = 0
        renamed_files = 0
        
        # 处理每个子目录
        for subdir in tqdm(subdirs, desc="处理目录"):
            # 获取目录中的所有tif文件
            tif_files = glob.glob(os.path.join(subdir, '*.tif'))
            
            for file in tif_files:
                total_files += 1
                filename = os.path.basename(file)
                new_filename = format_filename(filename)
                
                # 如果文件名需要修改
                if new_filename != filename:
                    new_file = os.path.join(subdir, new_filename)
                    # 检查新文件名是否已存在
                    if os.path.exists(new_file):
                        logger.warning(f"文件已存在，跳过重命名: {new_file}")
                        continue
                    
                    # 重命名文件
                    os.rename(file, new_file)
                    renamed_files += 1
                    logger.info(f"重命名文件: {filename} -> {new_filename}")
        
        logger.info(f"\n处理完成:")
        logger.info(f"总文件数: {total_files}")
        logger.info(f"重命名文件数: {renamed_files}")
        
        return True
        
    except Exception as e:
        logger.error(f"处理目录时出错: {str(e)}")
        return False

def main():
    # 设置输入目录
    input_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data'
    
    # 处理目录
    process_directory(input_dir)

if __name__ == '__main__':
    main() 