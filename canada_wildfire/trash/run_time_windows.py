#!/usr/bin/env python3
"""
时间窗口数据处理脚本
自动执行 pixel_sampling.py (时间窗口模式) + merge_pixel_samples.py --mode windows
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name, args=None, description=""):
    """执行脚本并返回结果"""
    print(f"\n{'='*60}")
    print(f"🚀 开始执行: {description}")
    print(f"脚本: {script_name}")
    if args:
        print(f"参数: {' '.join(args)}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    start_time = time.time()
    
    try:
        # 执行命令，实时显示输出
        result = subprocess.run(
            cmd, 
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ {description} 执行成功！")
        print(f"⏱️ 执行时间: {elapsed_time:.2f} 秒 ({elapsed_time/60:.1f} 分钟)")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ {description} 执行失败！")
        print(f"⏱️ 执行时间: {elapsed_time:.2f} 秒")
        print(f"错误码: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ 执行出错: {str(e)}")
        return False

def main():
    """主函数"""
    print("🔥 野火时间窗口数据处理流水线")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_start = time.time()
    
    # 步骤1: 执行像素采样（时间窗口模式）
    success1 = run_script(
        'pixel_sampling.py', 
        description="时间窗口像素采样"
    )
    
    if not success1:
        print("\n❌ 像素采样失败，流水线终止")
        sys.exit(1)
    
    # 步骤2: 执行数据合并（时间窗口模式）
    success2 = run_script(
        'merge_pixel_samples.py',
        args=['--mode', 'windows'],
        description="时间窗口数据合并"
    )
    
    # 总结
    pipeline_elapsed = time.time() - pipeline_start
    
    print(f"\n{'='*60}")
    if success1 and success2:
        print("🎉 时间窗口数据处理流水线执行成功！")
        print("📁 生成的数据集:")
        print("   - 原始数据: pixel_samples/ 目录")
        print("   - 合并数据: pixel_samples_merged/ 目录")
    else:
        print("❌ 流水线执行失败")
    
    print(f"⏱️ 总执行时间: {pipeline_elapsed:.2f} 秒 ({pipeline_elapsed/60:.1f} 分钟)")
    print(f"📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # 返回适当的退出码
    if success1 and success2:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 