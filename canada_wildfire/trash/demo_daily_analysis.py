#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐日驱动因素分析演示脚本

这是一个简化版的演示，用于快速验证功能和输出格式。
"""

import os
import sys
import logging
from daily_driver_analysis import DailyDriverAnalyzer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_daily_analysis():
    """演示逐日驱动因素分析功能"""
    
    logger.info("="*60)
    logger.info("野火点燃像素前10天逐日驱动因素分布分析演示")
    logger.info("="*60)
    
    # 配置参数
    data_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_downsampled_10x_masked"
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials"
    firms_dir = os.path.join(data_dir, "Firms_Detection_resampled")
    
    logger.info("分析配置:")
    logger.info(f"  数据目录: {data_dir}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  FIRMS目录: {firms_dir}")
    
    try:
        # 创建分析器 - 只分析前3天作为演示
        analyzer = DailyDriverAnalyzer(
            data_dir=data_dir,
            output_dir=output_dir,
            firms_dir=firms_dir,
            days_before=3  # 演示用，只分析前3天
        )
        
        logger.info(f"\n找到的动态驱动因素：")
        for i, driver in enumerate(analyzer.dynamic_drivers.keys(), 1):
            logger.info(f"  {i}. {driver}")
        
        # 识别燃烧事件
        logger.info(f"\n步骤1: 识别燃烧事件")
        num_events = analyzer.identify_burn_events()
        logger.info(f"识别到 {num_events} 个燃烧事件")
        
        if num_events == 0:
            logger.error("未找到燃烧事件，无法继续分析")
            return False
        
        # 显示燃烧事件概况
        logger.info("\n燃烧事件概况:")
        total_burned_pixels = sum(event['total_burned'] for event in analyzer.burn_events)
        logger.info(f"  总燃烧像元数: {total_burned_pixels}")
        logger.info(f"  平均每事件燃烧像元: {total_burned_pixels/num_events:.1f}")
        logger.info(f"  时间范围: {analyzer.burn_events[0]['date'].strftime('%Y-%m-%d')} 至 "
                   f"{analyzer.burn_events[-1]['date'].strftime('%Y-%m-%d')}")
        
        # 只处理前5个事件进行演示
        logger.info(f"\n步骤2: 逐日驱动因素分析（演示：仅前5个事件）")
        original_events = analyzer.burn_events
        analyzer.burn_events = analyzer.burn_events[:5]
        
        logger.info("正在分析以下燃烧事件:")
        for i, event in enumerate(analyzer.burn_events):
            logger.info(f"  {i+1}. {event['date'].strftime('%Y-%m-%d')}: {event['total_burned']} 个燃烧像元")
        
        # 进行分析
        results = analyzer.analyze_daily_distributions()
        
        if not results:
            logger.warning("未生成任何分析结果")
            return False
        
        logger.info(f"\n生成了 {len(results)} 条分析记录")
        
        # 显示一些统计信息
        logger.info("\n分析结果概览:")
        days_coverage = set(r['days_before'] for r in results)
        drivers_coverage = set(r['driver_name'] for r in results)
        logger.info(f"  覆盖天数: {sorted(days_coverage)}")
        logger.info(f"  涉及驱动因素: {len(drivers_coverage)} 个")
        
        # 显示部分结果示例
        logger.info("\n结果示例（前5条记录）:")
        logger.info("日期\t\t燃烧前\t驱动因素\t\t\t燃烧均值\t对照均值\t差异")
        logger.info("-" * 80)
        
        for i, result in enumerate(results[:5]):
            logger.info(f"{result['target_date']}\t{result['days_before']}天\t"
                       f"{result['feature_name'][:25]:<25}\t"
                       f"{result['burned_mean']:8.4f}\t"
                       f"{result['control_mean']:8.4f}\t"
                       f"{result['mean_difference']:8.4f}")
        
        # 保存结果
        logger.info(f"\n步骤3: 保存分析结果")
        detail_file = analyzer._save_results(results)
        
        # 检查输出文件
        output_files = [
            f'daily_driver_analysis_{analyzer.days_before}days.csv',
            f'daily_driver_summary_{analyzer.days_before}days.txt',
            f'daily_driver_pivot_{analyzer.days_before}days.csv'
        ]
        
        logger.info("\n生成的输出文件:")
        for filename in output_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(f"  ✓ {filename} ({file_size:,} bytes)")
            else:
                logger.warning(f"  ✗ {filename} (未生成)")
        
        # 显示一些关键发现
        logger.info(f"\n步骤4: 关键发现总结")
        
        # 按天数分组分析
        from collections import defaultdict
        day_stats = defaultdict(list)
        for result in results:
            if not pd.isna(result['mean_difference']):
                day_stats[result['days_before']].append(result['mean_difference'])
        
        logger.info("各天平均差异:")
        for day in sorted(day_stats.keys()):
            avg_diff = sum(day_stats[day]) / len(day_stats[day])
            logger.info(f"  燃烧前第{day}天: {avg_diff:8.4f} (基于{len(day_stats[day])}条记录)")
        
        # 找出差异最大的特征
        logger.info("\n差异最大的前5个特征-天数组合:")
        import pandas as pd
        df = pd.DataFrame(results)
        df_sorted = df.dropna(subset=['mean_difference']).copy()
        df_sorted['abs_diff'] = abs(df_sorted['mean_difference'])
        df_sorted = df_sorted.sort_values('abs_diff', ascending=False)
        
        for i, (_, row) in enumerate(df_sorted.head(5).iterrows()):
            logger.info(f"  {i+1}. {row['feature_name']} 燃烧前{row['days_before']}天: "
                       f"差异 = {row['mean_difference']:8.4f}")
        
        logger.info("\n" + "="*60)
        logger.info("演示完成！")
        logger.info("="*60)
        
        logger.info("\n功能说明:")
        logger.info("✓ 成功识别了燃烧事件和有效像元")
        logger.info("✓ 为每个燃烧事件选择了相同数量的对照未燃烧像元")
        logger.info("✓ 统计了燃烧前每一天每个驱动因素的分布（均值、中值、标准差等）")
        logger.info("✓ 计算了燃烧组vs对照组的差异")
        logger.info("✓ 生成了详细的CSV文件、汇总报告和透视表")
        
        logger.info("\n要进行完整的10天分析，请运行:")
        logger.info("python daily_driver_analysis.py")
        
        return True
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import pandas as pd  # 确保pandas导入
    
    success = demo_daily_analysis()
    if success:
        logger.info("演示成功完成!")
        sys.exit(0)
    else:
        logger.error("演示失败!")
        sys.exit(1) 