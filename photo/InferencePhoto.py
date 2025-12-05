#!/usr/bin/env python3
"""
PPO Value Adaptive Multi-Server Inference Visualization System

This module provides comprehensive visualization capabilities for analyzing
inference performance across multiple servers in the PPO Value Adaptive system.

Features:
- Multi-server inference analysis
- Individual performance charts for each server
- Comparison charts across servers
- Comprehensive performance metrics visualization

Author: PPO Value Adaptive Team
Version: 2.4
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Increase font sizes for better readability
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.5,        # 设置坐标轴边框线宽
    'axes.edgecolor': 'black'     # 设置坐标轴边框颜色为黑色
})

warnings.filterwarnings('ignore')

class MultiServerInferenceVisualizer:
    """Visualizer for multi-server inference analysis"""
    
    def __init__(self):
        """Initialize the multi-server visualizer"""
        self.steps = 1000000  # Training steps
        self.servers = ['Server1', 'Server2', 'Server3']
        self.server_display_names = {
            'Server1': 'S1',
            'Server2': 'S2', 
            'Server3': 'S3'
        }
        
        # Color scheme for different servers (color-blind friendly)
        self.colorblind_colors = {
            'Server1': '#1f77b4',  # Blue
            'Server2': '#ff7f0e',  # Orange  
            'Server3': '#2ca02c'   # Green
        }
        
        # Pattern scheme for different servers
        self.patterns = {
            'Server1': '///',      # Diagonal lines
            'Server2': '...',      # Dots
            'Server3': 'xxx'       # Crosses
        }
        
        # Set up paths
        self.base_dir = Path(__file__).parent.parent
        self.results_dir = self.base_dir / "results"
        self.photo_results_dir = Path(__file__).parent / "results"
        
        # Create results directory if it doesn't exist
        self.photo_results_dir.mkdir(exist_ok=True)
        
        # JSON file mappings
        self.json_files = {
            'Server1': f'ppo_value_adaptive_inference_report_{self.steps}_Server1.json',
            'Server2': f'ppo_value_adaptive_inference_report_{self.steps}_Server2.json',
            'Server3': f'ppo_value_adaptive_inference_report_{self.steps}_Server3.json'
        }
        
        # Initialize data storage
        self.inference_data = {}
        
    def load_json_data(self):
        """Load all JSON inference report files"""
        print("Loading JSON inference report files...")
        print(f"Looking in directory: {self.results_dir}")
        
        for server in self.servers:
            filename = self.json_files[server]
            filepath = self.results_dir / filename
            
            print(f"Checking for file: {filepath}")
            
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.inference_data[server] = data
                    print(f"✅ Loaded: {filename}")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {str(e)}")
                    self.inference_data[server] = None
            else:
                print(f"⚠️ Warning: File not found: {filename}")
                self.inference_data[server] = None
                
    def create_all_visualizations(self):
        """Create all visualization charts (individual + comparison charts)"""
        print(f"\nGenerating performance analysis visualizations...")
        print("=" * 60)
        
        # Individual chart configurations (4 charts per server = 12 total, plus 9 additional charts)
        individual_chart_configs = [
            ("Active Server Usage Distribution", "active_server_usage_distribution"),
            ("Communication Efficiency", "communication_efficiency"),
            ("Communication Overhead Distribution", "communication_overhead_distribution"),
            ("Reward Distribution", "reward_distribution")
        ]
        
        total_charts = 0
        successful_charts = 0
        
        # Generate individual charts for each server
        for server in self.servers:
            if self.inference_data.get(server) is None:
                print(f"⚠️ Skipping {server} - no data loaded")
                continue
                
            print(f"\nGenerating individual charts for {server}...")
            
            for i, (chart_title, file_suffix) in enumerate(individual_chart_configs, 1):
                total_charts += 1
                try:
                    plt.figure(figsize=(12, 9))
                    
                    # Call the appropriate chart creation method
                    if file_suffix == "active_server_usage_distribution":
                        self._create_active_server_usage_distribution(server)
                    elif file_suffix == "communication_efficiency":
                        self._create_communication_efficiency(server)
                    elif file_suffix == "communication_overhead_distribution":
                        self._create_communication_overhead_distribution(server)
                    elif file_suffix == "reward_distribution":
                        self._create_reward_distribution(server)
                    
                    # Set title with S1/S2/S3 format
                    display_name = self.server_display_names[server]
                    plt.title(f"{chart_title} - {display_name}", fontsize=28, fontweight='normal')
                    plt.tick_params(axis='both', which='major', labelsize=25)
                    
                    # Save chart
                    filename = f"inference_{file_suffix}_{server.lower()}_{self.steps}.png"
                    filepath = self.photo_results_dir / filename
                    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                    print(f"  ✅ Chart {i}/4 saved: {filename}")
                    
                    plt.close()  # Close current chart to free memory
                    successful_charts += 1
                        
                except Exception as e:
                    print(f"  ❌ Error creating chart {i} ({chart_title}): {str(e)}")
                    plt.close()
        
        # Generate new independent charts (replacing comparison charts)
        print(f"\nGenerating additional independent charts...")
        
        # Generate model type usage charts for each server
        try:
            self._create_model_type_usage_charts()
            total_charts += 3
            successful_charts += 3
        except Exception as e:
            print(f"  ❌ Error creating model type usage charts: {str(e)}")
        
        # Generate network performance metrics charts for each server
        try:
            self._create_network_performance_metrics_charts()
            total_charts += 3
            successful_charts += 3
        except Exception as e:
            print(f"  ❌ Error creating network performance metrics charts: {str(e)}")
        
        # Generate task success rate charts for each server
        try:
            self._create_task_success_rate_charts()
            total_charts += 3
            successful_charts += 3
        except Exception as e:
            print(f"  ❌ Error creating task success rate charts: {str(e)}")
        
        print("\n" + "=" * 60)
        print(f"Chart generation completed: {successful_charts}/{total_charts} successful")
        print(f"📁 Charts saved to: {self.photo_results_dir}")
        
    def _create_active_server_usage_distribution(self, server):
        """Create active server usage distribution bar chart"""
        data = self.inference_data[server]
        
        # Extract server usage data from performance_statistics
        if 'performance_statistics' not in data or 'server_usage' not in data['performance_statistics']:
            return
            
        server_stats = data['performance_statistics']['server_usage']
        total_usage = server_stats.get('total_usage_per_server', [])
        
        if not total_usage:
            return
            
        # Filter out servers with zero usage
        server_indices = []
        usage_counts = []
        for i, usage in enumerate(total_usage):
            if usage > 0:
                server_indices.append(i)
                usage_counts.append(usage)
        
        if not usage_counts:
            return
            
        colors = plt.cm.Set3(np.linspace(0, 1, len(server_indices)))
        bars = plt.bar([f'Server {i}' for i in server_indices], usage_counts, color=colors, alpha=0.7)
        plt.xlabel('Server Index', fontsize=25)
        plt.ylabel('Usage Count', fontsize=25)
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        for bar, count in zip(bars, usage_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(usage_counts) * 0.01,
                     f'{int(count)}', ha='center', va='bottom', fontsize=25)
        
        # 增加y轴比例，防止数值超过图框
        if usage_counts:
            max_count = max(usage_counts)
            plt.ylim(0, max_count * 1.25)  # 增加25%的空间
    
    def _create_communication_efficiency(self, server):
        """Create communication efficiency scatter plot"""
        data = self.inference_data[server]
        
        if 'detailed_results' not in data:
            return
            
        comm_overheads = []
        rewards = []
        completion_rates = []
        
        for episode in data['detailed_results']:
            if all(key in episode for key in ['communication_overhead', 'total_reward', 'completion_rate']):
                comm_overheads.append(episode['communication_overhead'])
                rewards.append(episode['total_reward'])
                completion_rates.append(episode['completion_rate'])
        
        if not comm_overheads:
            return
            
        scatter = plt.scatter(comm_overheads, rewards, c=completion_rates, cmap='RdYlGn', 
                   alpha=0.6, s=30, edgecolors='black', linewidth=0.5, vmin=0, vmax=1)
        
        # Add trend line if enough data
        if len(comm_overheads) > 1 and len(set(comm_overheads)) > 1:
            z = np.polyfit(comm_overheads, rewards, 1)
            p = np.poly1d(z)
            plt.plot(sorted(comm_overheads), p(sorted(comm_overheads)), "r--", alpha=0.8, linewidth=2)
        
        plt.xlabel('Communication Overhead (s)', fontsize=25)
        plt.ylabel('Reward', fontsize=25)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Completion Rate', fontsize=25)
        cbar.ax.tick_params(labelsize=25)
        plt.grid(True, alpha=0.3)
    
    def _create_communication_overhead_distribution(self, server):
        """Create communication overhead distribution histogram"""
        data = self.inference_data[server]
        
        if 'detailed_results' not in data:
            return
            
        comm_overheads = []
        for episode in data['detailed_results']:
            if 'communication_overhead' in episode:
                comm_overheads.append(episode['communication_overhead'])
        
        if not comm_overheads:
            return
            
        plt.hist(comm_overheads, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        plt.axvline(float(np.mean(comm_overheads)), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(comm_overheads):.3f}s')
        plt.xlabel('Communication Overhead (s)', fontsize=25)
        plt.ylabel('Frequency', fontsize=25)
        plt.legend(fontsize=25)
        plt.grid(True, alpha=0.3)
    
    def _create_reward_distribution(self, server):
        """Create reward distribution histogram"""
        data = self.inference_data[server]
        
        if 'detailed_results' not in data:
            return
            
        rewards = []
        for episode in data['detailed_results']:
            if 'total_reward' in episode:
                rewards.append(episode['total_reward'])
        
        if not rewards:
            return
            
        plt.hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(float(np.mean(rewards)), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(rewards):.2f}')
        plt.xlabel('Episode Reward', fontsize=25)
        plt.ylabel('Frequency', fontsize=25)
        plt.legend(fontsize=25, loc='upper left')
        plt.grid(True, alpha=0.3)

    def _create_model_type_usage_charts(self):
        """Create 3 independent model type usage charts for each server"""
        for server in self.servers:
            if self.inference_data.get(server) is None:
                continue
                
            data = self.inference_data[server]
            if 'performance_statistics' not in data or 'model_usage' not in data['performance_statistics']:
                continue
                
            model_usage = data['performance_statistics']['model_usage']
            usage_percentages = model_usage.get('usage_percentages', {})
            
            if not usage_percentages:
                continue
            
            plt.figure(figsize=(10, 8))
            
            models = sorted(list(usage_percentages.keys()))
            percentages = [usage_percentages[model] for model in models]
            
            # Use a consistent color scheme with patterns for color-blind accessibility
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            patterns = ['///', '...', 'xxx', '|||', '+++', 'ooo', '***']  # 防色盲图案
            
            bars = plt.bar([f'Model {m}' for m in models], percentages, 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                          hatch=[patterns[i % len(patterns)] for i in range(len(models))])
            
            # Add value labels on bars
            for bar, percentage in zip(bars, percentages):
                if percentage > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(percentages) * 0.01,
                            f'{percentage:.1f}%', ha='center', va='bottom', fontsize=20)
            
            plt.title(f'Model Type Usage - {self.server_display_names[server]}', fontsize=28, fontweight='normal')
            plt.xlabel('Model Type', fontsize=25)
            plt.ylabel('Usage Percentage (%)', fontsize=25)
            plt.tick_params(axis='both', which='major', labelsize=22)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Adjust y-axis limit
            if percentages:
                plt.ylim(0, max(percentages) * 1.2)
            
            # Save chart
            filename = f"inference_model_type_usage_{server.lower()}_{self.steps}.png"
            filepath = self.photo_results_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  ✅ Chart saved: {filename}")
            plt.close()

    def _create_network_performance_metrics_charts(self):
        """Create 3 independent network performance metrics charts for each server"""
        for server in self.servers:
            if self.inference_data.get(server) is None:
                continue
                
            data = self.inference_data[server]
            if 'performance_statistics' not in data or 'communication_analysis' not in data['performance_statistics']:
                continue
                
            comm_analysis = data['performance_statistics']['communication_analysis']
            
            # Extract key metrics
            metrics = {
                'Transmission': comm_analysis.get('avg_transmission_time', 0) * 1000,  # Convert to ms
                'Latency': comm_analysis.get('avg_latency', 0),  # Already in ms
                'Bandwidth': comm_analysis.get('avg_bandwidth_usage', 0) / 100  # Scale for visualization
            }
            
            if not any(v > 0 for v in metrics.values()):
                continue
            
            plt.figure(figsize=(10, 8))
            
            metric_names = ['Avg\nTransmission', 'Avg\nLatency', 'Avg\nBandwidth']
            values = list(metrics.values())
            
            # Use a consistent color scheme with patterns for color-blind accessibility
            colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
            patterns = ['///', '...', 'xxx']  # 防色盲图案
            bars = plt.bar(metric_names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                          hatch=patterns)
            
            # Add value labels on bars
            units = ['ms', 'ms', 'x100 Mbps']
            for bar, value, unit in zip(bars, values, units):
                if value > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.02,
                            f'{value:.1f}\n{unit}', ha='center', va='bottom', fontsize=18)
            
            plt.title(f'Network Performance Metrics - {self.server_display_names[server]}', fontsize=28, fontweight='normal')
            plt.xlabel('Network Metrics', fontsize=25)
            plt.ylabel('Value', fontsize=25)
            plt.tick_params(axis='both', which='major', labelsize=22)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Adjust y-axis limit
            if values:
                plt.ylim(0, max(values) * 1.3)
            
            # Save chart
            filename = f"inference_network_performance_metrics_{server.lower()}_{self.steps}.png"
            filepath = self.photo_results_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  ✅ Chart saved: {filename}")
            plt.close()

    def _create_task_success_rate_charts(self):
        """Create 3 independent task success rate charts for each server"""
        for server in self.servers:
            if self.inference_data.get(server) is None:
                continue
                
            data = self.inference_data[server]
            if 'performance_statistics' not in data or 'completion_metrics' not in data['performance_statistics']:
                continue
            
            completion_stats = data['performance_statistics']['completion_metrics']
            
            # Create task success rate data from available metrics
            success_rates = {
                'Overall Success': completion_stats.get('success_rate', 0) * 100,
                'Mean Completion': completion_stats.get('mean_completion_rate', 0) * 100,
                'Consistency': completion_stats.get('completion_consistency', 0) * 100
            }
            
            if not any(rate > 0 for rate in success_rates.values()):
                continue
            
            plt.figure(figsize=(10, 8))
            
            task_types = ['Overall\nSuccess', 'Mean\nCompletion', 'Consistency']
            rates = list(success_rates.values())
            
            # Use a consistent color scheme with patterns for color-blind accessibility
            colors = ['#9b59b6', '#f39c12', '#1abc9c']  # Purple, Orange, Teal
            patterns = ['///', '...', 'xxx']  # 防色盲图案
            bars = plt.bar(task_types, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                          hatch=patterns)
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                if rate > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rates) * 0.01,
                            f'{rate:.1f}%', ha='center', va='bottom', fontsize=20)
            
            plt.title(f'Task Success Rate - {self.server_display_names[server]}', fontsize=28, fontweight='normal')
            plt.xlabel('Task Type', fontsize=25)
            plt.ylabel('Success Rate (%)', fontsize=25)
            plt.tick_params(axis='both', which='major', labelsize=22)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Adjust y-axis limit
            if rates:
                plt.ylim(0, max(rates) * 1.2)
            
            # Save chart
            filename = f"inference_task_success_rate_{server.lower()}_{self.steps}.png"
            filepath = self.photo_results_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  ✅ Chart saved: {filename}")
            plt.close()


def main():
    """Main function to run the multi-server visualization"""
    print("PPO Value Adaptive Multi-Server Inference Visualization System")
    print("=" * 70)
    print("Configuration:")
    print("  • Servers: S1, S2, S3")
    print("  • Training Steps: 1000000")
    print("  • Individual Charts: 21 (7 charts × 3 servers)")
    print("=" * 70)
    
    # Initialize visualizer
    visualizer = MultiServerInferenceVisualizer()
    
    # Load JSON data
    visualizer.load_json_data()
    
    # Create visualizations
    visualizer.create_all_visualizations()
    
    print("\n✅ Multi-server visualization complete!")
    print("📊 Generated chart types:")
    print("  Individual Charts (per server):")
    print("    • Active Server Usage Distribution")
    print("    • Communication Efficiency") 
    print("    • Communication Overhead Distribution")
    print("    • Reward Distribution")
    print("    • Model Type Usage")
    print("    • Network Performance Metrics")
    print("    • Task Success Rate")


if __name__ == "__main__":
    main()
