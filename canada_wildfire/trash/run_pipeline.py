#!/usr/bin/env python3
"""
野火数据处理自动化流水线
自动执行像素采样和数据合并的完整流程
"""

import os
import sys
import subprocess
import logging
import time
from datetime import datetime
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('Pipeline')

class WildfirePipeline:
    def __init__(self, working_dir=None):
        """
        初始化流水线
        
        Args:
            working_dir: 工作目录，默认为当前脚本所在目录
        """
        if working_dir is None:
            self.working_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.working_dir = working_dir
        
        self.pixel_sampling_script = os.path.join(self.working_dir, 'pixel_sampling.py')
        self.merge_script = os.path.join(self.working_dir, 'merge_pixel_samples.py')
        
        logger.info(f"工作目录: {self.working_dir}")
        logger.info(f"像素采样脚本: {self.pixel_sampling_script}")
        logger.info(f"数据合并脚本: {self.merge_script}")
    
    def check_scripts_exist(self):
        """检查必需的脚本是否存在"""
        missing_scripts = []
        
        if not os.path.exists(self.pixel_sampling_script):
            missing_scripts.append(self.pixel_sampling_script)
        
        if not os.path.exists(self.merge_script):
            missing_scripts.append(self.merge_script)
        
        if missing_scripts:
            for script in missing_scripts:
                logger.error(f"脚本不存在: {script}")
            return False
        
        logger.info("所有必需脚本都存在")
        return True
    
    def run_command(self, command, description, timeout=None):
        """
        执行命令并监控结果
        
        Args:
            command: 要执行的命令（列表形式）
            description: 命令描述
            timeout: 超时时间（秒），None表示无超时
        
        Returns:
            bool: 执行是否成功
        """
        logger.info(f"=" * 60)
        logger.info(f"开始执行: {description}")
        logger.info(f"命令: {' '.join(command)}")
        logger.info(f"=" * 60)
        
        start_time = time.time()
        
        try:
            # 执行命令
            process = subprocess.Popen(
                command,
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                
                # 检查超时
                if timeout and (time.time() - start_time) > timeout:
                    logger.error(f"命令执行超时 ({timeout}秒)")
                    process.terminate()
                    return False
            
            # 获取返回码
            return_code = process.poll()
            
            elapsed_time = time.time() - start_time
            
            if return_code == 0:
                logger.info(f"✅ {description} 执行成功")
                logger.info(f"执行时间: {elapsed_time:.2f} 秒")
                return True
            else:
                logger.error(f"❌ {description} 执行失败")
                logger.error(f"返回码: {return_code}")
                logger.error(f"执行时间: {elapsed_time:.2f} 秒")
                return False
                
        except Exception as e:
            logger.error(f"❌ 执行命令时出错: {str(e)}")
            return False
    
    def run_pixel_sampling(self):
        """执行像素采样"""
        command = [sys.executable, self.pixel_sampling_script]
        return self.run_command(
            command, 
            "像素采样 (pixel_sampling.py)",
            timeout=None  # 不设置超时，因为这个过程可能很长
        )
    
    def run_merge_windows(self, pixel_samples_dir=None, output_dir=None):
        """执行时间窗口数据合并"""
        command = [sys.executable, self.merge_script, '--mode', 'windows']
        
        if pixel_samples_dir:
            command.extend(['--pixel_samples_dir', pixel_samples_dir])
        
        if output_dir:
            command.extend(['--output_dir', output_dir])
        
        return self.run_command(
            command,
            "时间窗口数据合并 (merge_pixel_samples.py --mode windows)",
            timeout=None
        )
    
    def run_full_pipeline(self, pixel_samples_dir=None, output_dir=None, skip_sampling=False):
        """
        执行完整的流水线
        
        Args:
            pixel_samples_dir: 像素样本目录
            output_dir: 输出目录
            skip_sampling: 是否跳过采样步骤
        
        Returns:
            bool: 流水线是否成功完成
        """
        logger.info("🚀 开始执行野火数据处理流水线")
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        pipeline_start_time = time.time()
        
        # 检查脚本是否存在
        if not self.check_scripts_exist():
            logger.error("❌ 流水线中止：缺少必需的脚本")
            return False
        
        success = True
        
        # 步骤1: 像素采样
        if not skip_sampling:
            logger.info("📊 步骤 1/2: 执行像素采样...")
            if not self.run_pixel_sampling():
                logger.error("❌ 像素采样失败，流水线中止")
                return False
            logger.info("✅ 像素采样完成")
        else:
            logger.info("⏭️ 跳过像素采样步骤")
        
        # 步骤2: 数据合并
        logger.info("🔄 步骤 2/2: 执行时间窗口数据合并...")
        if not self.run_merge_windows(pixel_samples_dir, output_dir):
            logger.error("❌ 数据合并失败")
            success = False
        else:
            logger.info("✅ 数据合并完成")
        
        # 流水线完成
        pipeline_elapsed_time = time.time() - pipeline_start_time
        
        logger.info("=" * 60)
        if success:
            logger.info("🎉 野火数据处理流水线执行成功！")
        else:
            logger.info("❌ 野火数据处理流水线执行失败")
        
        logger.info(f"总执行时间: {pipeline_elapsed_time:.2f} 秒 ({pipeline_elapsed_time/60:.1f} 分钟)")
        logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        return success
    
    def check_output_files(self, pixel_samples_dir=None):
        """检查输出文件是否存在"""
        if pixel_samples_dir is None:
            pixel_samples_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_samples'
        
        if not os.path.exists(pixel_samples_dir):
            logger.warning(f"像素样本目录不存在: {pixel_samples_dir}")
            return False
        
        # 查找时间窗口格式的文件
        import glob
        window_files = glob.glob(os.path.join(pixel_samples_dir, '*_????_????_*_full.h5'))
        
        if window_files:
            logger.info(f"找到 {len(window_files)} 个时间窗口文件:")
            for file_path in sorted(window_files)[:10]:  # 只显示前10个
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                logger.info(f"  {filename} ({file_size:.1f} MB)")
            
            if len(window_files) > 10:
                logger.info(f"  ... 还有 {len(window_files) - 10} 个文件")
            
            return True
        else:
            logger.warning("没有找到时间窗口格式的文件")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='野火数据处理自动化流水线')
    parser.add_argument('--skip-sampling', action='store_true',
                       help='跳过像素采样步骤，只执行数据合并')
    parser.add_argument('--pixel-samples-dir', 
                       default=None,
                       help='像素样本数据目录')
    parser.add_argument('--output-dir',
                       default=None, 
                       help='合并后数据输出目录')
    parser.add_argument('--working-dir',
                       default=None,
                       help='工作目录（脚本所在目录）')
    parser.add_argument('--check-output', action='store_true',
                       help='只检查输出文件，不执行流水线')
    
    args = parser.parse_args()
    
    # 创建流水线实例
    pipeline = WildfirePipeline(working_dir=args.working_dir)
    
    if args.check_output:
        # 只检查输出文件
        logger.info("检查输出文件...")
        pipeline.check_output_files(args.pixel_samples_dir)
        return
    
    # 执行流水线
    success = pipeline.run_full_pipeline(
        pixel_samples_dir=args.pixel_samples_dir,
        output_dir=args.output_dir,
        skip_sampling=args.skip_sampling
    )
    
    if success:
        logger.info("流水线执行成功，检查输出文件...")
        pipeline.check_output_files(args.pixel_samples_dir)
        sys.exit(0)
    else:
        logger.error("流水线执行失败")
        sys.exit(1)


if __name__ == "__main__":
    main() 