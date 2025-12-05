#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from pathlib import Path

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

warnings.filterwarnings('ignore')

def load_data():
    algorithms = ['Algorithm_1', 'Algorithm_2', 'main', 'Greedy', 'Random']
    data = {}
    
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    print(f"Script directory: {script_dir}")
    print(f"Current working directory: {os.getcwd()}")
    
    for algorithm in algorithms:
        data[algorithm] = {}
        print(f"\nProcessing {algorithm}...")
        
        # 1. Average Reward
        reward_file = script_dir / f"{algorithm}_reward_distribution.csv"
        print(f"  Looking for reward file: {reward_file}")
        try:
            if reward_file.exists():
                df = pd.read_csv(reward_file)
                print(f"  Reward file loaded, shape: {df.shape}")
                if 'reward' in df.columns:
                    data[algorithm]['avg_reward'] = df['reward'].mean()
                    print(f"  Average reward: {data[algorithm]['avg_reward']:.3f}")
                else:
                    print(f"  No 'reward' column found, columns: {df.columns.tolist()}")
                    data[algorithm]['avg_reward'] = 0
            else:
                print(f"  Reward file not found!")
                data[algorithm]['avg_reward'] = 0
        except Exception as e:
            print(f"  Error loading reward file: {e}")
            data[algorithm]['avg_reward'] = 0
        
        # 2. Success Rate
        success_file = script_dir / f"{algorithm}_task_success_rate.csv"
        print(f"  Looking for success file: {success_file}")
        try:
            if success_file.exists():
                df = pd.read_csv(success_file)
                print(f"  Success file loaded, shape: {df.shape}")
                success_row = df[df['status'] == 'Successful']
                if not success_row.empty:
                    data[algorithm]['success_rate'] = float(success_row['percentage'].values[0])
                    print(f"  Success rate: {data[algorithm]['success_rate']:.1f}%")
                else:
                    print(f"  No 'Successful' status found")
                    data[algorithm]['success_rate'] = 0
            else:
                print(f"  Success file not found!")
                data[algorithm]['success_rate'] = 0
        except Exception as e:
            print(f"  Error loading success file: {e}")
            data[algorithm]['success_rate'] = 0
        
        # 3. Response Time
        network_file = script_dir / f"{algorithm}_network_performance_metrics.csv"
        print(f"  Looking for network file: {network_file}")
        try:
            if network_file.exists():
                df = pd.read_csv(network_file)
                print(f"  Network file loaded, shape: {df.shape}")
                transmission_row = df[df['metric'] == 'Avg Transmission Time']
                if not transmission_row.empty:
                    data[algorithm]['response_time'] = float(transmission_row['value'].values[0]) / 1000.0
                    print(f"  Response time (transmission): {data[algorithm]['response_time']:.3f}s")
                else:
                    latency_row = df[df['metric'] == 'Avg Latency']
                    if not latency_row.empty:
                        data[algorithm]['response_time'] = float(latency_row['value'].values[0]) / 1000.0
                        print(f"  Response time (latency): {data[algorithm]['response_time']:.3f}s")
                    else:
                        print(f"  No transmission time or latency metric found")
                        data[algorithm]['response_time'] = 0
            else:
                print(f"  Network file not found!")
                data[algorithm]['response_time'] = 0
        except Exception as e:
            print(f"  Error loading network file: {e}")
            data[algorithm]['response_time'] = 0
        
        # 4. Communication Overhead
        comm_file = script_dir / f"{algorithm}_communication_overhead_distribution.csv"
        print(f"  Looking for comm overhead file: {comm_file}")
        try:
            if comm_file.exists():
                df = pd.read_csv(comm_file)
                print(f"  Comm overhead file loaded, shape: {df.shape}")
                if 'communication_overhead' in df.columns:
                    data[algorithm]['comm_overhead'] = df['communication_overhead'].mean()
                    print(f"  Comm overhead: {data[algorithm]['comm_overhead']:.3f}")
                else:
                    print(f"  No 'communication_overhead' column found, columns: {df.columns.tolist()}")
                    data[algorithm]['comm_overhead'] = 0
            else:
                print(f"  Comm overhead file not found!")
                data[algorithm]['comm_overhead'] = 0
        except Exception as e:
            print(f"  Error loading comm overhead file: {e}")
            data[algorithm]['comm_overhead'] = 0
        
        # 5. Server Diversity
        server_file = script_dir / f"{algorithm}_active_server_usage_distribution.csv"
        print(f"  Looking for server usage file: {server_file}")
        try:
            if server_file.exists():
                df = pd.read_csv(server_file)
                print(f"  Server usage file loaded, shape: {df.shape}")
                if 'usage_count' in df.columns:
                    usage_counts = df['usage_count'].values
                    usage_counts = usage_counts[usage_counts > 0]
                    if len(usage_counts) > 1:
                        data[algorithm]['diversity'] = np.std(usage_counts) / np.mean(usage_counts)
                    else:
                        data[algorithm]['diversity'] = 0
                    print(f"  Server diversity: {data[algorithm]['diversity']:.3f}")
                else:
                    print(f"  No 'usage_count' column found, columns: {df.columns.tolist()}")
                    data[algorithm]['diversity'] = 0
            else:
                print(f"  Server usage file not found!")
                data[algorithm]['diversity'] = 0
        except Exception as e:
            print(f"  Error loading server usage file: {e}")
            data[algorithm]['diversity'] = 0
        
        # 6. Communication Efficiency
        efficiency_file = script_dir / f"{algorithm}_communication_efficiency.csv"
        print(f"  Looking for efficiency file: {efficiency_file}")
        try:
            if efficiency_file.exists():
                df = pd.read_csv(efficiency_file)
                print(f"  Efficiency file loaded, shape: {df.shape}")
                efficiency_row = df[df['metric'] == 'Communication Efficiency']
                if not efficiency_row.empty:
                    data[algorithm]['efficiency'] = float(efficiency_row['value'].values[0])
                    print(f"  Communication efficiency: {data[algorithm]['efficiency']:.3f}")
                else:
                    print(f"  No 'Communication Efficiency' metric found")
                    data[algorithm]['efficiency'] = 0
            else:
                print(f"  Efficiency file not found!")
                data[algorithm]['efficiency'] = 0
        except Exception as e:
            print(f"  Error loading efficiency file: {e}")
            data[algorithm]['efficiency'] = 0
    
    return data

def create_chart(metric_key, metric_name, unit, data, script_dir):
    algorithms = ['Algorithm_1', 'Algorithm_2', 'main', 'Greedy', 'Random']
    # 算法显示名称映射
    algorithm_display_names = ['MAPPO', 'METAPPO', 'Ours', 'Greedy', 'Random']
    values = [data[alg][metric_key] for alg in algorithms]
    
    # 防色盲颜色方案和对应的稀疏下划线样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 蓝、橙、绿、红、紫
    hatches = ['/', '.', 'x', '|', '+']  # 更稀疏的下划线样式
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(algorithm_display_names, values, color=colors, alpha=0.8, edgecolor='black', 
                   hatch=hatches)
    
    # 添加数值标签
    max_val = max(values) if values else 1
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if unit == '%':
            label = f'{value:.1f}%'
        elif metric_key in ['response_time', 'comm_overhead']:
            label = f'{value:.3f}'
        else:
            label = f'{value:.3f}'
        
        # 特殊处理success_rate图表，将标签放在条形图内部
        if metric_key == 'success_rate':
            # 将标签放在条形图的中间位置
            plt.text(bar.get_x() + bar.get_width()/2., height/2,
                    label, ha='center', va='center', fontsize=24, color='white')
        else:
            # 其他图表将标签放在条形图上方
            plt.text(bar.get_x() + bar.get_width()/2., height + max_val*0.02,
                    label, ha='center', va='bottom', fontsize=24)
    
    # 设置标题和标签（不加粗，字体大小30）
    plt.title(f'{metric_name} Comparison', fontsize=30)
    plt.xlabel('Algorithm', fontsize=28)
    plt.ylabel(f'{metric_name} ({unit})' if unit else metric_name, fontsize=28)
    
    # 设置刻度字体大小
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    
    # 设置网格
    plt.grid(True, alpha=0.3, axis='y')
    
    # Y轴长度增加四分之一
    if metric_key == 'success_rate':
        plt.ylim(0, 100)
    else:
        plt.ylim(0, max_val * 1.25)  # 原来是1.0，现在增加到1.25
    
    plt.tight_layout()
    
    # 保存到脚本目录
    output_path = script_dir / f"{metric_key}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def print_summary(data):
    algorithms = ['Algorithm_1', 'Algorithm_2', 'main', 'Greedy', 'Random']
    algorithm_display_names = ['MAPPO', 'METAPPO', 'Ours', 'Greedy', 'Random']
    metrics = [
        ('success_rate', 'Success Rate (%)'),
        ('response_time', 'Response Time (s)'),
        ('comm_overhead', 'Communication Overhead (s)'),
        ('diversity', 'Server Diversity'),
        ('efficiency', 'Communication Efficiency')
    ]
    
    print("\n" + "="*80)
    print("Algorithm Performance Summary")
    print("="*80)
    
    header = f"{'Metric':<25}"
    for name in algorithm_display_names:
        header += f"{name:<15}"
    header += f"{'Best':<15}"
    print(header)
    print("-" * len(header))
    
    for metric_key, metric_display in metrics:
        row = f"{metric_display:<25}"
        values = []
        
        for i, algorithm in enumerate(algorithms):
            value = data[algorithm][metric_key]
            values.append(value)
            
            if metric_key == 'success_rate':
                row += f"{value:.1f}%{'':<10}"
            elif metric_key in ['response_time', 'comm_overhead']:
                row += f"{value:.3f}{'':<11}"
            else:
                row += f"{value:.3f}{'':<11}"
        
        # 确定最佳算法
        higher_is_better = ['success_rate', 'diversity', 'efficiency']
        lower_is_better = ['response_time', 'comm_overhead']
        
        if metric_key in higher_is_better and any(v > 0 for v in values):
            best_idx = values.index(max(values))
        elif metric_key in lower_is_better and any(v > 0 for v in values):
            best_idx = values.index(min(values))
        else:
            best_idx = None
        
        if best_idx is not None:
            best_algorithm = algorithm_display_names[best_idx]
            row += f"{best_algorithm:<15}"
        else:
            row += f"{'N/A':<15}"
        
        print(row)

def main():
    print("Algorithm Performance Comparison")
    print("=" * 40)
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    
    # 加载数据
    data = load_data()
    
    # 创建5个单独的图表（移除avg_reward）
    metrics = [
        ('success_rate', 'Success Rate', '%'),
        ('response_time', 'Response Time', 's'),
        ('comm_overhead', 'Communication Overhead', 's'),
        ('diversity', 'Server Diversity', ''),
        ('efficiency', 'Communication Efficiency', '')
    ]
    
    print("\nCreating individual metric charts...")
    for metric_key, metric_name, unit in metrics:
        create_chart(metric_key, metric_name, unit, data, script_dir)
    
    # 打印汇总
    print_summary(data)
    
    print("\nAll 5 charts generated successfully!")

if __name__ == "__main__":
    main() 