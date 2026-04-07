#!/usr/bin/env python3
"""
时间窗口分布统计转换器
将pixel_driver_distribution.py生成的30天和365天时间窗口统计数据
转换成与driver_distribution_analysis.csv相同的整体分布格式
"""

import os
import pandas as pd
import logging
from datetime import datetime
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('TemporalDistributionConverter')

class TemporalDistributionConverter:
    def __init__(self, input_dir, output_dir):
        """
        初始化时间窗口分布转换器
        
        Args:
            input_dir: 包含pixel_driver_distribution结果的目录
            output_dir: 输出目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"输入目录: {input_dir}")
        logger.info(f"输出目录: {output_dir}")
    
    def convert_temporal_distributions(self, time_windows=[30, 365]):
        """
        转换时间窗口分布数据
        
        Args:
            time_windows: 时间窗口列表（天数）
        """
        logger.info(f"开始转换时间窗口分布数据: {time_windows}天")
        
        all_converted_data = []
        
        for window_days in time_windows:
            logger.info(f"处理 {window_days} 天时间窗口...")
            
            # 读取对应的时间窗口分布统计文件
            temporal_file = os.path.join(
                self.input_dir, 
                f'temporal_distribution_analysis_{window_days}days.csv'
            )
            
            if not os.path.exists(temporal_file):
                logger.warning(f"未找到文件: {temporal_file}")
                continue
            
            try:
                df = pd.read_csv(temporal_file)
                logger.info(f"成功读取 {temporal_file}，包含 {len(df)} 条记录")
                
                # 为每条记录添加时间窗口信息
                df_with_window = df.copy()
                df_with_window['time_window_days'] = window_days
                df_with_window['analysis_type'] = f'temporal_{window_days}days'
                
                # 修改特征名称，去掉temporal前缀并添加时间窗口后缀
                df_with_window['original_feature_name'] = df_with_window['feature_name'].copy()
                df_with_window['feature_name'] = df_with_window['feature_name'].apply(
                    lambda x: self._clean_feature_name(x, window_days)
                )
                
                all_converted_data.append(df_with_window)
                logger.info(f"  转换了 {len(df_with_window)} 个特征")
                
            except Exception as e:
                logger.error(f"读取文件 {temporal_file} 时出错: {str(e)}")
                continue
        
        if all_converted_data:
            # 合并所有时间窗口的数据
            combined_df = pd.concat(all_converted_data, ignore_index=True)
            
            # 保存合并后的数据
            output_file = os.path.join(self.output_dir, 'temporal_distribution_combined.csv')
            combined_df.to_csv(output_file, index=False)
            logger.info(f"合并的时间窗口分布数据保存到: {output_file}")
            
            # 分别保存每个时间窗口的清理版本
            for window_days in time_windows:
                window_data = combined_df[combined_df['time_window_days'] == window_days].copy()
                if len(window_data) > 0:
                    # 去掉额外的列，保持与driver_distribution_analysis.csv相同的格式
                    clean_data = self._clean_temporal_data(window_data)
                    
                    clean_output_file = os.path.join(
                        self.output_dir, 
                        f'temporal_distribution_clean_{window_days}days.csv'
                    )
                    clean_data.to_csv(clean_output_file, index=False)
                    logger.info(f"{window_days}天时间窗口清理数据保存到: {clean_output_file}")
            
            # 创建汇总报告
            self._create_summary_report(combined_df)
            
            # 创建对比分析
            self._create_comparison_analysis(combined_df, time_windows)
            
        else:
            logger.warning("没有成功转换任何数据")
    
    def _clean_feature_name(self, feature_name, window_days):
        """
        清理特征名称
        
        Args:
            feature_name: 原始特征名称
            window_days: 时间窗口天数
            
        Returns:
            清理后的特征名称
        """
        # 去掉temporal_XXdays前缀
        if f'_temporal_{window_days}days_' in feature_name:
            cleaned_name = feature_name.replace(f'_temporal_{window_days}days_', '_')
        else:
            cleaned_name = feature_name
        
        # 添加时间窗口后缀
        cleaned_name = f"{cleaned_name}_temporal_{window_days}d"
        
        return cleaned_name
    
    def _clean_temporal_data(self, df):
        """
        清理时间窗口数据，保持与driver_distribution_analysis.csv相同的格式
        
        Args:
            df: 包含时间窗口数据的DataFrame
            
        Returns:
            清理后的DataFrame
        """
        # 保留与driver_distribution_analysis.csv相同的列
        essential_columns = [
            'feature_name',
            'burned_count', 'burned_mean', 'burned_std', 'burned_median',
            'burned_min', 'burned_max', 'burned_q25', 'burned_q75',
            'unburned_count', 'unburned_mean', 'unburned_std', 'unburned_median',
            'unburned_min', 'unburned_max', 'unburned_q25', 'unburned_q75',
            'mean_difference', 'median_difference', 'std_ratio',
            'effect_size_cohens_d', 'effect_magnitude',
            't_statistic', 'p_value', 'significant',
            'mannwhitney_u_statistic', 'mannwhitney_p_value', 'mannwhitney_significant',
            'ks_statistic', 'ks_p_value', 'distributions_different'
        ]
        
        # 只保留存在的列
        available_columns = [col for col in essential_columns if col in df.columns]
        clean_df = df[available_columns].copy()
        
        return clean_df
    
    def _create_summary_report(self, df):
        """
        创建时间窗口分布转换汇总报告
        
        Args:
            df: 合并后的数据框
        """
        summary_file = os.path.join(self.output_dir, 'temporal_conversion_summary.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("时间窗口分布统计转换汇总报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"转换时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入目录: {self.input_dir}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            
            # 统计每个时间窗口的特征数量
            window_stats = df.groupby('time_window_days').agg({
                'feature_name': 'count',
                'significant': 'sum',
                'effect_size_cohens_d': ['mean', 'std']
            }).round(4)
            
            f.write("时间窗口统计:\n")
            f.write("-" * 40 + "\n")
            for window_days in sorted(df['time_window_days'].unique()):
                window_data = df[df['time_window_days'] == window_days]
                significant_count = len(window_data[window_data['significant'] == True])
                f.write(f"{window_days}天时间窗口:\n")
                f.write(f"  - 总特征数: {len(window_data)}\n")
                f.write(f"  - 显著特征数: {significant_count}\n")
                f.write(f"  - 显著率: {significant_count/len(window_data)*100:.1f}%\n")
                f.write(f"  - 平均效应大小: {window_data['effect_size_cohens_d'].mean():.3f}\n\n")
            
            # 整体统计
            f.write("整体统计:\n")
            f.write("-" * 40 + "\n")
            f.write(f"总记录数: {len(df)}\n")
            f.write(f"总显著特征数: {len(df[df['significant'] == True])}\n")
            f.write(f"总体显著率: {len(df[df['significant'] == True])/len(df)*100:.1f}%\n\n")
            
            # 按时间窗口显示最重要的特征
            for window_days in sorted(df['time_window_days'].unique()):
                window_data = df[df['time_window_days'] == window_days]
                
                f.write(f"{window_days}天时间窗口最重要的前10个特征:\n")
                f.write("-" * 50 + "\n")
                
                top_features = window_data.nlargest(10, 'effect_size_cohens_d', 'first')[
                    ['feature_name', 'effect_size_cohens_d', 'effect_magnitude', 'significant']
                ]
                
                for i, (_, row) in enumerate(top_features.iterrows(), 1):
                    f.write(f"{i:2d}. {row['feature_name']:<40} "
                           f"d={row['effect_size_cohens_d']:7.3f} "
                           f"({row['effect_magnitude']:<8}) "
                           f"显著={'✓' if row['significant'] else '✗'}\n")
                f.write("\n")
        
        logger.info(f"转换汇总报告保存到: {summary_file}")
    
    def _create_comparison_analysis(self, df, time_windows):
        """
        创建时间窗口对比分析
        
        Args:
            df: 合并后的数据框
            time_windows: 时间窗口列表
        """
        comparison_file = os.path.join(self.output_dir, 'temporal_window_comparison.txt')
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("时间窗口对比分析\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("说明: 比较不同时间窗口下驱动因素的重要性变化\n\n")
            
            # 获取所有唯一的驱动因素基础名称
            all_features = set()
            for _, row in df.iterrows():
                # 提取驱动因素基础名称（去掉时间窗口后缀）
                feature_base = row['feature_name'].replace(f"_temporal_{row['time_window_days']}d", "")
                all_features.add(feature_base)
            
            all_features = sorted(list(all_features))
            
            f.write("驱动因素在不同时间窗口下的效应大小对比:\n")
            f.write("-" * 60 + "\n")
            
            # 为每个驱动因素比较不同时间窗口的效应
            comparison_data = []
            
            for feature_base in all_features:
                feature_comparison = {'feature_base': feature_base}
                
                for window_days in time_windows:
                    feature_full_name = f"{feature_base}_temporal_{window_days}d"
                    feature_data = df[df['feature_name'] == feature_full_name]
                    
                    if len(feature_data) > 0:
                        effect_size = feature_data.iloc[0]['effect_size_cohens_d']
                        is_significant = feature_data.iloc[0]['significant']
                        feature_comparison[f'effect_{window_days}d'] = effect_size
                        feature_comparison[f'significant_{window_days}d'] = is_significant
                    else:
                        feature_comparison[f'effect_{window_days}d'] = None
                        feature_comparison[f'significant_{window_days}d'] = None
                
                comparison_data.append(feature_comparison)
            
            # 按30天效应大小排序显示前20个特征
            valid_comparisons = [
                comp for comp in comparison_data 
                if comp.get('effect_30d') is not None
            ]
            valid_comparisons.sort(key=lambda x: abs(x['effect_30d']), reverse=True)
            
            for i, comp in enumerate(valid_comparisons[:20], 1):
                f.write(f"{i:2d}. {comp['feature_base']:<40}\n")
                for window_days in time_windows:
                    effect_key = f'effect_{window_days}d'
                    sig_key = f'significant_{window_days}d'
                    
                    if comp[effect_key] is not None:
                        effect_str = f"{comp[effect_key]:7.3f}"
                        sig_str = "✓" if comp[sig_key] else "✗"
                        f.write(f"    {window_days:3d}天: 效应={effect_str} 显著={sig_str}\n")
                    else:
                        f.write(f"    {window_days:3d}天: 无数据\n")
                f.write("\n")
            
            # 分析时间窗口效应的趋势
            f.write("时间窗口效应趋势分析:\n")
            f.write("-" * 40 + "\n")
            
            increasing_effects = 0
            decreasing_effects = 0
            stable_effects = 0
            
            for comp in valid_comparisons:
                if len(time_windows) >= 2:
                    effect_30 = comp.get('effect_30d')
                    effect_365 = comp.get('effect_365d')
                    
                    if effect_30 is not None and effect_365 is not None:
                        diff = abs(effect_365) - abs(effect_30)
                        if diff > 0.1:
                            increasing_effects += 1
                        elif diff < -0.1:
                            decreasing_effects += 1
                        else:
                            stable_effects += 1
            
            total_compared = increasing_effects + decreasing_effects + stable_effects
            if total_compared > 0:
                f.write(f"长期效应增强的特征: {increasing_effects} ({increasing_effects/total_compared*100:.1f}%)\n")
                f.write(f"长期效应减弱的特征: {decreasing_effects} ({decreasing_effects/total_compared*100:.1f}%)\n")
                f.write(f"效应相对稳定的特征: {stable_effects} ({stable_effects/total_compared*100:.1f}%)\n")
        
        logger.info(f"时间窗口对比分析保存到: {comparison_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='转换时间窗口分布统计数据')
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_driver_distribution_results',
        help='包含pixel_driver_distribution结果的目录'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/pixel_driver_distribution_results',
        help='输出目录'
    )
    parser.add_argument(
        '--time_windows', 
        nargs='+', 
        type=int, 
        default=[30, 365],
        help='时间窗口列表（天数）'
    )
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = TemporalDistributionConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # 执行转换
    converter.convert_temporal_distributions(time_windows=args.time_windows)
    
    logger.info("时间窗口分布统计转换完成！")

if __name__ == "__main__":
    main() 