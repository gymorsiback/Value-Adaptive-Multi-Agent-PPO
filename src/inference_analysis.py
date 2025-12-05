#!/usr/bin/env python3
"""
Enhanced Result Analysis Tool
分析推理结果和训练数据，给出清晰易懂的评价
"""

import json
import numpy as np
from pathlib import Path
import sys

def load_training_data(filename):
    """Load training data"""
    script_dir = Path(__file__).parent  # vamappo/src
    results_dir = script_dir.parent / "results"  # vamappo/results
    data_path = results_dir / filename
    
    print(f"Looking for training data at: {data_path.resolve()}")
    
    if not data_path.exists():
        print("❌ 训练数据文件不存在")
        print(f"Expected path: {data_path}")
        return None
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return data

def analyze_training_data(data):
    """Analyze PPO Value Adaptive training data"""
    print("🔍 PPO Value Adaptive 训练数据分析")
    print("=" * 60)
    
    # 获取基本信息
    metadata = data.get('metadata', {})
    total_episodes = metadata.get('total_episodes', 0)
    training_duration = metadata.get('training_duration', 0)
    algorithm = metadata.get('algorithm', 'Unknown')
    
    print(f"📊 训练基本信息:")
    print(f"   • 算法: {algorithm}")
    print(f"   • 训练回合数: {total_episodes}")
    print(f"   • 训练时长: {training_duration/3600:.2f} 小时")
    print(f"   • 平均每回合时间: {training_duration/total_episodes:.2f} 秒")
    print()
    
    # 分析置信区间数据
    ci_data = data.get('confidence_intervals', {})
    if ci_data:
        print("📈 关键指标置信区间分析 (95% CI):")
        print("-" * 50)
        
        # 总奖励分析
        if 'total_rewards' in ci_data:
            rewards_ci = ci_data['total_rewards']
            final_rewards_ci = ci_data.get('final_performance', {})
            
            print(f"🎯 奖励性能:")
            print(f"   • 整体平均奖励: {rewards_ci['mean']:.3f} ± {rewards_ci['ci_width']/2:.3f}")
            print(f"   • 置信区间: [{rewards_ci['ci_lower']:.3f}, {rewards_ci['ci_upper']:.3f}]")
            print(f"   • 相对精度: ±{rewards_ci['relative_ci_width_percent']:.2f}%")
            
            if final_rewards_ci:
                print(f"   • 最终性能: {final_rewards_ci['mean']:.3f} ± {final_rewards_ci['ci_width']/2:.3f}")
                print(f"   • 最终置信区间: [{final_rewards_ci['ci_lower']:.3f}, {final_rewards_ci['ci_upper']:.3f}]")
                print(f"   • 最终精度: ±{final_rewards_ci['relative_ci_width_percent']:.2f}%")
                
                # 性能改进
                improvement = ((final_rewards_ci['mean'] - rewards_ci['mean']) / rewards_ci['mean']) * 100
                print(f"   • 性能提升: {improvement:.1f}%")
        
        print()
        
        # 完成率分析
        if 'completion_rates' in ci_data:
            completion_ci = ci_data['completion_rates']
            final_completion_ci = ci_data.get('final_completion_rate', {})
            
            print(f"✅ 任务完成率:")
            print(f"   • 整体完成率: {completion_ci['mean']:.1%} ± {completion_ci['ci_width']/2:.1%}")
            print(f"   • 置信区间: [{completion_ci['ci_lower']:.1%}, {completion_ci['ci_upper']:.1%}]")
            print(f"   • 相对精度: ±{completion_ci['relative_ci_width_percent']:.2f}%")
            
            if final_completion_ci:
                print(f"   • 最终完成率: {final_completion_ci['mean']:.1%} ± {final_completion_ci['ci_width']/2:.1%}")
                print(f"   • 最终置信区间: [{final_completion_ci['ci_lower']:.1%}, {final_completion_ci['ci_upper']:.1%}]")
                
                # 完成率改进
                completion_improvement = ((final_completion_ci['mean'] - completion_ci['mean']) / completion_ci['mean']) * 100
                print(f"   • 完成率提升: {completion_improvement:.1f}%")
        
        print()
        
        # 自适应学习率分析
        if 'adaptive_lr_factors' in ci_data:
            lr_ci = ci_data['adaptive_lr_factors']
            final_lr_ci = ci_data.get('final_adaptive_lr', {})
            
            print(f"🔧 自适应学习率:")
            print(f"   • 整体LR因子: {lr_ci['mean']:.3f} ± {lr_ci['ci_width']/2:.3f}")
            print(f"   • 置信区间: [{lr_ci['ci_lower']:.3f}, {lr_ci['ci_upper']:.3f}]")
            print(f"   • 相对精度: ±{lr_ci['relative_ci_width_percent']:.2f}%")
            
            if final_lr_ci:
                print(f"   • 最终LR因子: {final_lr_ci['mean']:.3f} ± {final_lr_ci['ci_width']/2:.3f}")
                print(f"   • 收敛精度: ±{final_lr_ci['relative_ci_width_percent']:.2f}%")
        
        print()
        
        # 价值不确定性分析
        if 'value_uncertainties' in ci_data:
            uncertainty_ci = ci_data['value_uncertainties']
            final_uncertainty_ci = ci_data.get('final_uncertainty', {})
            
            print(f"🎲 价值不确定性:")
            print(f"   • 整体不确定性: {uncertainty_ci['mean']:.4f} ± {uncertainty_ci['ci_width']/2:.4f}")
            print(f"   • 置信区间: [{uncertainty_ci['ci_lower']:.4f}, {uncertainty_ci['ci_upper']:.4f}]")
            
            if final_uncertainty_ci:
                print(f"   • 最终不确定性: {final_uncertainty_ci['mean']:.4f} ± {final_uncertainty_ci['ci_width']/2:.4f}")
                
                # 不确定性下降
                uncertainty_reduction = ((uncertainty_ci['mean'] - final_uncertainty_ci['mean']) / uncertainty_ci['mean']) * 100
                print(f"   • 不确定性降低: {uncertainty_reduction:.1f}%")
        
        print()
    
    # 稳定性分析
    stability = data.get('stability_metrics', {})
    if stability:
        print("📊 训练稳定性分析:")
        print("-" * 30)
        
        reward_cv = stability.get('reward_coefficient_of_variation_percent', 0)
        trend_corr = stability.get('reward_trend_correlation', 0)
        lr_cv = stability.get('adaptive_lr_coefficient_of_variation_percent', 0)
        
        print(f"   • 奖励变异系数: {reward_cv:.2f}%")
        print(f"   • 趋势相关性: {trend_corr:.3f}")
        print(f"   • LR因子变异系数: {lr_cv:.2f}%")
        
        # 稳定性评估
        if reward_cv < 10:
            stability_grade = "优秀 🎯"
        elif reward_cv < 20:
            stability_grade = "良好 📈"
        elif reward_cv < 30:
            stability_grade = "一般 📊"
        else:
            stability_grade = "较差 📉"
        
        print(f"   • 奖励稳定性: {stability_grade}")
        
        if trend_corr > 0.8:
            trend_grade = "强烈上升 🚀"
        elif trend_corr > 0.5:
            trend_grade = "明显上升 📈"
        elif trend_corr > 0.2:
            trend_grade = "轻微上升 📊"
        else:
            trend_grade = "无明显趋势 📉"
        
        print(f"   • 学习趋势: {trend_grade}")
        print()
    
    # 综合评价
    print("🏆 综合训练评价:")
    print("-" * 30)
    
    # 计算综合得分
    scores = {}
    
    if ci_data.get('final_performance'):
        final_reward = ci_data['final_performance']['mean']
        scores['performance'] = min(100, (final_reward / 10) * 100)  # 假设满分是10
    
    if ci_data.get('final_completion_rate'):
        final_completion = ci_data['final_completion_rate']['mean']
        scores['completion'] = final_completion * 100
    
    if ci_data.get('final_adaptive_lr'):
        lr_precision = ci_data['final_adaptive_lr']['relative_ci_width_percent']
        scores['lr_stability'] = max(0, 100 - lr_precision * 10)
    
    if stability.get('reward_coefficient_of_variation_percent'):
        reward_cv = stability['reward_coefficient_of_variation_percent']
        scores['reward_stability'] = max(0, 100 - reward_cv * 3)
    
    if stability.get('reward_trend_correlation'):
        trend_corr = stability['reward_trend_correlation']
        scores['learning_trend'] = trend_corr * 100
    
    # 打印各项得分
    for metric, score in scores.items():
        print(f"   • {metric.replace('_', ' ').title()}: {score:.1f}/100")
    
    if scores:
        overall_score = np.mean(list(scores.values()))
        print(f"   • 综合得分: {overall_score:.1f}/100")
        
        if overall_score >= 85:
            grade = "优秀 🏆"
            desc = "训练效果优秀，模型已达到高性能水平"
        elif overall_score >= 75:
            grade = "良好 🥈"
            desc = "训练效果良好，模型性能稳定"
        elif overall_score >= 65:
            grade = "及格 🥉"
            desc = "训练基本达标，还有改进空间"
        elif overall_score >= 50:
            grade = "需改进 ⚠️"
            desc = "训练效果一般，需要优化"
        else:
            grade = "不理想 ❌"
            desc = "训练效果不理想，需要重新设计"
        
        print(f"   • 最终评级: {grade}")
        print(f"   • 评价: {desc}")
    
    print()
    
    # 训练建议
    print("💡 训练优化建议:")
    print("-" * 25)
    
    suggestions = []
    
    if ci_data.get('final_completion_rate', {}).get('mean', 0) < 0.5:
        suggestions.append("🎯 提高任务完成率:")
        suggestions.append("   - 调整奖励函数权重")
        suggestions.append("   - 增加完成奖励")
    
    if stability.get('reward_coefficient_of_variation_percent', 0) > 15:
        suggestions.append("📊 改善训练稳定性:")
        suggestions.append("   - 降低学习率")
        suggestions.append("   - 增加训练批次大小")
    
    if ci_data.get('final_adaptive_lr', {}).get('relative_ci_width_percent', 0) > 2:
        suggestions.append("🔧 优化自适应学习率:")
        suggestions.append("   - 增加LR平滑机制")
        suggestions.append("   - 调整适应速度")
    
    if stability.get('reward_trend_correlation', 0) < 0.5:
        suggestions.append("📈 加强学习趋势:")
        suggestions.append("   - 检查梯度流")
        suggestions.append("   - 调整网络结构")
    
    if not suggestions:
        suggestions = ["🎉 训练表现优秀，继续保持当前策略"]
    
    for suggestion in suggestions:
        print(f"   {suggestion}")

def load_inference_results():
    """Load inference results (original function)"""
    # 修复路径：确保指向正确的 vamappo/results 目录
    script_dir = Path(__file__).parent  # vamappo/src
    results_dir = script_dir.parent / "results"  # vamappo/results
    report_path = results_dir / "inference_report_1000000.json"
    
    print(f"Looking for file at: {report_path.resolve()}")
    
    if not report_path.exists():
        print("❌ 推理报告文件不存在")
        print(f"Expected path: {report_path}")
        return None
    
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    return data

def analyze_performance(data):
    """Analyze and interpret performance"""
    print("🔍 分布式PPO模型推理结果分析")
    print("=" * 50)
    
    # 兼容新旧格式的性能指标获取
    if 'enhanced_performance_statistics' in data:
        perf_stats = data['enhanced_performance_statistics']
        analysis_type = "增强版"
    elif 'performance_statistics' in data:
        perf_stats = data['performance_statistics']
        analysis_type = "标准版"
    else:
        print("❌ 无法找到性能统计数据")
        return
    
    # 获取性能指标
    reward_stats = perf_stats['reward_metrics']
    completion_stats = perf_stats['completion_metrics']
    efficiency_stats = perf_stats['efficiency_metrics']
    model_usage = perf_stats['model_usage']
    server_usage = perf_stats['server_usage']
    
    print(f"📊 测试规模: {data['inference_metadata']['total_episodes']} 个测试episode ({analysis_type})")
    print()
    
    # 1. 奖励分析
    print("🎯 奖励表现分析:")
    avg_reward = reward_stats['mean']
    std_reward = reward_stats['std']
    print(f"   • 平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"   • 奖励范围: [{reward_stats['min']:.2f}, {reward_stats['max']:.2f}]")
    
    # 评判奖励表现
    if avg_reward >= 40:
        reward_grade = "优秀 ⭐⭐⭐⭐⭐"
        reward_desc = "模型决策质量很高"
    elif avg_reward >= 30:
        reward_grade = "良好 ⭐⭐⭐⭐"
        reward_desc = "模型决策质量较好"
    elif avg_reward >= 20:
        reward_grade = "一般 ⭐⭐⭐"
        reward_desc = "模型决策质量中等"
    elif avg_reward >= 10:
        reward_grade = "较差 ⭐⭐"
        reward_desc = "模型决策需要改进"
    else:
        reward_grade = "很差 ⭐"
        reward_desc = "模型决策有严重问题"
    
    print(f"   • 评级: {reward_grade}")
    print(f"   • 评价: {reward_desc}")
    print()
    
    # 2. 增强的任务完成分析
    print("✅ 任务完成分析:")
    completion_rate = completion_stats['mean_completion_rate']
    success_rate = completion_stats['success_rate']
    print(f"   • 平均完成率: {completion_rate:.1%}")
    print(f"   • 任务成功率: {success_rate:.1%}")
    
    # 如果有增强指标，显示额外信息
    if 'total_successful_episodes' in completion_stats:
        successful_episodes = completion_stats['total_successful_episodes']
        print(f"   • 成功完成的任务数: {successful_episodes}")
        
    if 'completion_consistency' in completion_stats:
        consistency = completion_stats['completion_consistency']
        print(f"   • 完成率一致性: {consistency:.3f}")
    
    # 任务完成质量评估
    if success_rate >= 0.8:
        task_grade = "优秀 🎯"
        task_desc = "任务完成质量很高"
    elif success_rate >= 0.6:
        task_grade = "良好 ✅"
        task_desc = "任务完成表现良好"
    elif success_rate >= 0.4:
        task_grade = "一般 📋"
        task_desc = "任务完成表现中等"
    elif success_rate >= 0.2:
        task_grade = "较差 ⚠️"
        task_desc = "任务完成需要改进"
    else:
        task_grade = "很差 ❌"
        task_desc = "任务完成有严重问题"
    
    print(f"   • 任务完成评级: {task_grade}")
    print(f"   • 评价: {task_desc}")
    print()
    
    # 3. 增强的效率分析
    print("⚡ 系统效率分析:")
    avg_response_time = efficiency_stats['mean_response_time']
    load_balance = efficiency_stats['mean_load_balance']
    print(f"   • 平均响应时间: {avg_response_time:.3f}秒")
    print(f"   • 负载均衡得分: {load_balance:.3f}")
    
    # 如果有增强指标，显示额外信息
    if 'mean_action_efficiency' in efficiency_stats:
        action_efficiency = efficiency_stats['mean_action_efficiency']
        print(f"   • 动作执行效率: {action_efficiency:.3f}")
        
    if 'mean_resource_utilization' in efficiency_stats:
        resource_util = efficiency_stats['mean_resource_utilization']
        print(f"   • 资源利用率: {resource_util:.3f}")
    
    if avg_response_time < 0.05:
        time_grade = "非常快 🚀"
    elif avg_response_time < 0.1:
        time_grade = "较快 ⚡"
    elif avg_response_time < 0.5:
        time_grade = "正常 ⏱️"
    else:
        time_grade = "较慢 🐌"
    
    print(f"   • 响应速度: {time_grade}")
    print()
    
    # 4. 增强的资源使用分析
    print("🖥️ 资源使用分析:")
    most_used_model = model_usage.get('most_used_model', 'N/A')
    model_percentages = model_usage.get('usage_percentages', {})
    most_used_server = server_usage['most_used_server']
    balance_coeff = server_usage['utilization_balance_coefficient']
    
    print(f"   • 最常用模型类型: {most_used_model}")
    print("   • 模型类型使用分布:")
    for model_type, percentage in model_percentages.items():
        print(f"      - 类型{model_type}: {percentage:.1f}%")
    
    print(f"   • 最常用服务器: #{most_used_server}")
    print(f"   • 服务器负载均衡系数: {balance_coeff:.3f}")
    
    # 增强指标
    if 'active_servers' in server_usage:
        active_servers = server_usage['active_servers']
        server_diversity = server_usage.get('server_diversity', 0)
        print(f"   • 活跃服务器数量: {active_servers}/25")
        print(f"   • 服务器多样性: {server_diversity:.1%}")
    
    if 'model_diversity' in model_usage:
        model_diversity = model_usage['model_diversity']
        print(f"   • 模型类型多样性: {model_diversity} 种")
    
    if balance_coeff < 0.5:
        balance_grade = "优秀 ⚖️"
    elif balance_coeff < 1.0:
        balance_grade = "良好 📊"
    elif balance_coeff < 2.0:
        balance_grade = "一般 📈"
    else:
        balance_grade = "不均衡 ⚠️"
    
    print(f"   • 负载均衡评价: {balance_grade}")
    print()
    
    # 5. 服务器使用详情
    total_usage = server_usage['total_usage_per_server']
    active_servers = sum(1 for usage in total_usage if usage > 0)
    total_servers = len(total_usage)
    
    print("🌐 服务器使用详情:")
    print(f"   • 活跃服务器: {active_servers}/{total_servers}")
    print(f"   • 服务器利用率: {active_servers/total_servers:.1%}")
    
    # 显示最活跃的几个服务器
    server_usage_pairs = [(i, usage) for i, usage in enumerate(total_usage) if usage > 0]
    server_usage_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("   • 使用最多的5个服务器:")
    for i, (server_id, usage) in enumerate(server_usage_pairs[:5]):
        print(f"      {i+1}. 服务器#{server_id}: {usage}次")
    print()
    
    # 6. 增强的综合评价
    print("🎖️ 综合性能评价:")
    print("-" * 30)
    
    # 计算增强的综合得分
    reward_score = min(100, max(0, avg_reward * 2))  # 奖励转换为0-100分
    efficiency_score = 100 if avg_response_time < 0.02 else max(0, 100 - avg_response_time * 1000)
    balance_score = max(0, 100 - balance_coeff * 30)
    utilization_score = (active_servers / total_servers) * 100
    success_score = success_rate * 100  # 成功率得分
    
    # 如果有动作效率指标，加入计算
    if 'mean_action_efficiency' in efficiency_stats:
        action_efficiency = efficiency_stats['mean_action_efficiency']
        action_score = action_efficiency * 100
        overall_score = (reward_score * 0.3 + efficiency_score * 0.2 + 
                        balance_score * 0.15 + utilization_score * 0.1 +
                        success_score * 0.2 + action_score * 0.05)
        print(f"   • 动作效率得分: {action_score:.1f}/100")
    else:
        overall_score = (reward_score * 0.4 + efficiency_score * 0.3 + 
                        balance_score * 0.2 + utilization_score * 0.1)
    
    print(f"   • 决策质量得分: {reward_score:.1f}/100")
    print(f"   • 响应效率得分: {efficiency_score:.1f}/100")
    print(f"   • 负载均衡得分: {balance_score:.1f}/100")
    print(f"   • 资源利用得分: {utilization_score:.1f}/100")
    print(f"   • 任务成功得分: {success_score:.1f}/100")
    print(f"   • 综合得分: {overall_score:.1f}/100")
    
    if overall_score >= 85:
        final_grade = "优秀 🏆"
        recommendation = "模型表现优秀，已达到生产级别标准"
    elif overall_score >= 75:
        final_grade = "良好 🥈"
        recommendation = "模型表现良好，可考虑投入使用"
    elif overall_score >= 65:
        final_grade = "及格 🥉"
        recommendation = "模型基本达标，建议进一步优化"
    elif overall_score >= 50:
        final_grade = "需改进 ⚠️"
        recommendation = "模型需要重要改进才能投入使用"
    else:
        final_grade = "不合格 ❌"
        recommendation = "模型需要重新设计或重新训练"
    
    print(f"   • 最终评级: {final_grade}")
    print(f"   • 建议: {recommendation}")
    print()
    
    # 7. 智能改进建议
    print("💡 智能改进建议:")
    print("-" * 20)
    
    suggestions = []
    
    if success_rate < 0.5:
        suggestions.append("🎯 任务完成率改进:")
        suggestions.append("      - 优化动作选择策略")
        suggestions.append("      - 检查奖励函数设计")
        suggestions.append("      - 增加任务完成奖励")
    
    if balance_coeff > 1.5:
        suggestions.append("⚖️ 负载均衡改进:")
        suggestions.append("      - 调整负载均衡权重")
        suggestions.append("      - 增加服务器多样性奖励")
        suggestions.append("      - 优化地理位置因子")
    
    if avg_reward < 25:
        suggestions.append("🎯 决策质量提升:")
        suggestions.append("      - 继续训练更多步数")
        suggestions.append("      - 调整学习率参数")
        suggestions.append("      - 优化神经网络结构")
    
    if active_servers < total_servers * 0.6:
        suggestions.append("🌐 服务器利用率改进:")
        suggestions.append("      - 检查服务器模型配置")
        suggestions.append("      - 调整地理位置权重")
        suggestions.append("      - 增加探索策略")
    
    if 'mean_action_efficiency' in efficiency_stats and efficiency_stats['mean_action_efficiency'] < 0.6:
        suggestions.append("⚡ 动作效率提升:")
        suggestions.append("      - 优化动作选择算法")
        suggestions.append("      - 减少无效动作")
        suggestions.append("      - 改进状态表示")
    
    if not suggestions:
        suggestions = ["🎉 模型表现良好，继续保持当前训练策略"]
    
    for suggestion in suggestions:
        print(f"   {suggestion}")
    
    # 8. 性能趋势分析（如果是增强版）
    if analysis_type == "增强版":
        print()
        print("📈 性能趋势分析:")
        print("-" * 20)
        if success_rate > 0.6:
            print("   ✅ 任务成功率表现良好，模型学习效果显著")
        if avg_response_time < 0.05:
            print("   🚀 响应速度优秀，满足实时性要求")
        if 'mean_action_efficiency' in efficiency_stats and efficiency_stats['mean_action_efficiency'] > 0.7:
            print("   ⚡ 动作执行效率高，资源使用合理")
        if balance_coeff < 1.0:
            print("   ⚖️ 负载均衡良好，系统稳定性强")

def main():
    """Main function with command line argument support"""
    if len(sys.argv) > 1:
        # 分析训练数据
        filename = sys.argv[1]
        print(f"🔍 分析训练数据文件: {filename}")
        data = load_training_data(filename)
        if data:
            analyze_training_data(data)
        else:
            print("❌ 无法加载训练数据文件")
    else:
        # 默认分析推理结果
        print("🔍 分析推理结果...")
        data = load_inference_results()
        if data:
            analyze_performance(data)
        else:
            print("❌ 无法加载推理结果文件")
            print("\n💡 使用提示:")
            print("   python inference_analysis.py                    # 分析推理结果")
            print("   python inference_analysis.py <training_file>    # 分析训练数据")

if __name__ == "__main__":
    main() 