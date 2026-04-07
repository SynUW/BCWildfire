"""
批量重命名文件，将文件名中的指定字符串替换为新的字符串。
"""
import os
import glob
from tqdm import tqdm
import logging
import multiprocessing
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('File_Renamer')

def rename_file(args):
    """
    重命名单个文件
    
    参数:
        args: (file_path, old_str, new_str) 元组
    """
    file_path, old_str, new_str = args
    try:
        # 获取目录和文件名
        dir_name = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        
        # 替换字符串
        new_name = file_name.replace(old_str, new_str)
        
        # 如果文件名没有变化，跳过
        if new_name == file_name:
            return True
            
        # 构建新的文件路径
        new_path = os.path.join(dir_name, new_name)
        
        # 重命名文件
        os.rename(file_path, new_path)
        return True
        
    except Exception as e:
        logger.error(f"重命名文件 {file_path} 时出错: {str(e)}")
        return False

def process_directory_wrapper(args):
    """
    包装函数，用于多进程处理目录
    """
    input_dir, old_str, new_str, file_pattern = args
    try:
        # 获取所有匹配的文件
        files = glob.glob(os.path.join(input_dir, file_pattern))
        
        if not files:
            logger.warning(f"在目录 {input_dir} 中没有找到匹配的文件")
            return False
            
        # 准备参数
        rename_args = [(file, old_str, new_str) for file in files]
        
        # 设置进程数为CPU内核数的85%
        num_processes = max(1, int(multiprocessing.cpu_count() * 0.85))
        
        # 使用进程池处理文件
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 使用tqdm显示进度
            list(tqdm(
                pool.imap(rename_file, rename_args),
                total=len(rename_args),
                desc=f"处理目录 {os.path.basename(input_dir)}"
            ))
            
        return True
        
    except Exception as e:
        logger.error(f"处理目录 {input_dir} 时出错: {str(e)}")
        return False

def main():
    # 设置输入目录列表
    input_dirs = [
        '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data/ERA5_soil_water_resampled',
    ]
    
    # 设置要替换的字符串
    old_str = 'ERA5_soil_water'  # 要替换的字符串
    new_str = 'ERA5SoilWater'  # 新的字符串
    
    # 设置文件匹配模式
    file_pattern = '*.tif'  # 例如：'*.tif' 表示所有tif文件
    
    # 准备参数
    args = [(input_dir, old_str, new_str, file_pattern) for input_dir in input_dirs]
    
    # 处理所有目录
    for arg in args:
        process_directory_wrapper(arg)
    
    logger.info("所有文件重命名完成")

if __name__ == '__main__':
    main() 