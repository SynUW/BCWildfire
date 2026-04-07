#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试逐日驱动因素分析功能

验证新的DailyDriverAnalyzer类能够正确分析野火点燃像素前10天的逐日驱动因素分布。
"""

import os
import sys
import logging
from daily_driver_analysis import DailyDriverAnalyzer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_daily_driver_analysis():
    """测试逐日驱动因素分析功能"""
    
    # 配置参数
    data_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked"
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials"
    firms_dir = os.path.join(data_dir, "Firms_Detection_resampled")
    
    logger.info("开始测试逐日驱动因素分析功能")
    
    try:
        # 创建分析器实例
        analyzer = DailyDriverAnalyzer(
            data_dir=data_dir,
            output_dir=output_dir,
            firms_dir=firms_dir,
            days_before=3  # 测试用较少天数
        )
        
        logger.info("测试配置:")
        logger.info(f"  数据目录: {data_dir}")
        logger.info(f"  输出目录: {output_dir}")
        logger.info(f"  FIRMS目录: {firms_dir}")
        logger.info(f"  分析天数: {analyzer.days_before}天")
        logger.info(f"  找到的动态驱动因素: {list(analyzer.dynamic_drivers.keys())}")
        
        # 测试燃烧事件识别
        logger.info("测试燃烧事件识别...")
        num_events = analyzer.identify_burn_events()
        logger.info(f"识别到 {num_events} 个燃烧事件")
        
        if num_events == 0:
            logger.warning("未找到任何燃烧事件，可能FIRMS目录路径有问题")
            return False
        
        # 显示前几个燃烧事件的信息
        logger.info("前5个燃烧事件:")
        for i, event in enumerate(analyzer.burn_events[:5]):
            logger.info(f"  事件{i+1}: {event['date'].strftime('%Y-%m-%d')}, "
                       f"燃烧像元数: {event['total_burned']}")
        
        # 运行逐日分析（限制为前几个事件进行快速测试）
        logger.info("运行逐日分析（测试模式：仅前3个事件）...")
        original_events = analyzer.burn_events
        analyzer.burn_events = analyzer.burn_events[:3]  # 只测试前3个事件
        
        results = analyzer.analyze_daily_distributions()
        
        if results:
            logger.info(f"生成了 {len(results)} 条分析记录")
            
            # 显示几条示例结果
            logger.info("示例结果:")
            for i, result in enumerate(results[:3]):
                logger.info(f"  {i+1}. {result['feature_name']} "
                           f"燃烧前{result['days_before']}天: "
                           f"燃烧均值={result['burned_mean']:.4f}, "
                           f"对照均值={result['control_mean']:.4f}, "
                           f"差异={result['mean_difference']:.4f}")
            
            # 保存结果
            logger.info("保存测试结果...")
            analyzer._save_results(results)
            
            # 检查输出文件
            expected_files = [
                f'daily_driver_analysis_{analyzer.days_before}days.csv',
                f'daily_driver_summary_{analyzer.days_before}days.txt',
                f'daily_driver_pivot_{analyzer.days_before}days.csv'
            ]
            
            logger.info("检查输出文件:")
            for filename in expected_files:
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    logger.info(f"  ✓ {filename} ({file_size} bytes)")
                else:
                    logger.warning(f"  ✗ {filename} (文件不存在)")
            
        else:
            logger.warning("未生成任何分析结果")
            return False
        
        logger.info("逐日驱动因素分析测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_daily_driver_analysis()
    if success:
        logger.info("所有测试通过!")
        sys.exit(0)
    else:
        logger.error("测试失败!")
        sys.exit(1) 