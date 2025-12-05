#!/usr/bin/env python3
"""
PPO Value Adaptive Model Cross-Server Generalization Visualization System

Reads CSV files exported by GeneralizabilityPhoto.py and creates comparative visualization charts
Generates 5 performance comparison charts showing Server1/Server2/Server3 performance on Server4
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib font parameters
plt.rcParams.update({
    'font.size': 25,
    'axes.titlesize': 28,
    'axes.labelsize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 25,
    'figure.titlesize': 30
})


class GeneralizationVisualizer:
    """Create visualizations from generalization analysis CSV data"""
    
    def __init__(self, data_dir: str = None, target_server: str = "Server4"):
        """
        Initialize visualizer
        
        Args:
            data_dir: Directory containing CSV files
            target_server: Target testing server
        """
        self.target_server = target_server
        if data_dir is None:
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)
        
        self.training_servers = ["Server1", "Server2", "Server3"]
        
        # Define colors and patterns for different training servers
        self.server_colors = {
            'Server1': '#3498db',  # Blue
            'Server2': '#e74c3c',  # Red
            'Server3': '#2ecc71'   # Green
        }
        
        # Define hatch patterns for color-blind accessibility
        self.server_hatches = {
            'Server1': '///',   # Forward diagonal
            'Server2': '\\\\\\',  # Backward diagonal  
            'Server3': '|||'    # Vertical lines
        }
        
        # Short labels
        self.server_labels = {
            'Server1': 'S1',
            'Server2': 'S2',
            'Server3': 'S3'
        }
    
    def load_csv_data(self, file_suffix: str):
        """Load CSV data for a specific chart"""
        csv_filename = f"inference_generalization_{self.target_server.lower()}_{file_suffix}.csv"
        csv_path = self.data_dir / csv_filename
        if csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            print(f"Warning: CSV file not found: {csv_filename}")
            return None
    

    
    def create_active_server_usage_chart(self):
        """Create active server usage distribution comparison chart"""
        fig = plt.figure(figsize=(18, 8))
        
        # Find max usage for consistent y-axis
        max_usage = 0
        for train_server in self.training_servers:
            df = pd.read_csv(self.data_dir / f"inference_generalization_{self.target_server.lower()}_active_server_usage_distribution_{train_server}.csv")
            if len(df) > 0:
                max_usage = max(max_usage, df['usage_count'].max())
        
        # Create subplots with proper configuration
        for idx, train_server in enumerate(self.training_servers):
            ax = plt.subplot(1, 3, idx + 1)
            
            df = pd.read_csv(self.data_dir / f"inference_generalization_{self.target_server.lower()}_active_server_usage_distribution_{train_server}.csv")
            
            # Filter active servers (usage > 0)
            active_df = df[df['usage_count'] > 0]
            
            if len(active_df) > 0:
                # Create bars with wider width to reduce spacing
                bars = ax.bar(range(len(active_df)), active_df['usage_count'], 
                              width=0.8, alpha=0.7, color=self.server_colors[train_server], 
                              edgecolor='black')
                # Add hatch pattern
                for bar in bars:
                    bar.set_hatch(self.server_hatches[train_server])
                
                # Set x-ticks with rotation for better spacing
                ax.set_xticks(range(len(active_df)))
                ax.set_xticklabels([f'S{idx}' for idx in active_df['server_index']], 
                                  rotation=45, ha='right', fontsize=16)
            
            # Ensure consistent axis properties for all subplots
            ax.set_ylim(0, max_usage * 1.1)
            ax.set_title(self.server_labels[train_server], fontsize=22, pad=10)
            ax.set_xlabel('Active Server Index', fontsize=20)
            if idx == 0:
                ax.set_ylabel('Usage Count', fontsize=20)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Ensure consistent tick parameters
            ax.tick_params(axis='x', which='major', labelsize=16)
            ax.tick_params(axis='y', which='major', labelsize=18)
            
            # Set tight x-axis to reduce empty space
            if len(active_df) > 0:
                ax.set_xlim(-0.5, len(active_df) - 0.5)
            else:
                ax.set_xlim(-0.5, 0.5)
        
        # Move title lower to be closer to subplots
        fig.suptitle('Active Server Usage Distribution Comparison', fontsize=28, y=0.90)
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        
        filename = f"inference_generalization_{self.target_server.lower()}_active_server_usage_distribution.png"
        plt.savefig(self.data_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {filename}")
        plt.close()
    
    def create_model_type_usage_chart(self):
        """Create model type usage comparison chart"""
        df = self.load_csv_data("model_type_usage")
        if df is None:
            return
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Prepare data
        model_types = df['model_type'].tolist()
        x = np.arange(len(model_types))
        width = 0.25
        
        for idx, train_server in enumerate(self.training_servers):
            usage_values = df[f'{train_server}_usage'].tolist()
            
            bars = ax.bar(x + idx * width, usage_values, width, 
                         label=self.server_labels[train_server], color=self.server_colors[train_server], 
                         alpha=0.7, edgecolor='black')
            
            # Add hatch pattern
            for bar in bars:
                bar.set_hatch(self.server_hatches[train_server])
            
            # Add value labels with larger font
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=20)
        
        ax.set_xlabel('Model Type', fontsize=25)
        ax.set_ylabel('Usage Percentage (%)', fontsize=25)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'Type {t}' for t in model_types], fontsize=20)
        ax.legend(fontsize=22, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.title('Model Type Usage Comparison', fontsize=28)
        plt.tight_layout()
        
        filename = f"inference_generalization_{self.target_server.lower()}_model_type_usage.png"
        plt.savefig(self.data_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {filename}")
        plt.close()
    
    def create_reward_vs_response_time_chart(self):
        """Create reward vs response time comparison chart"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, train_server in enumerate(self.training_servers):
            df = pd.read_csv(self.data_dir / f"inference_generalization_{self.target_server.lower()}_reward_vs_response_time_{train_server}.csv")
            
            if len(df) > 0:
                rewards = df['reward'].tolist()
                response_times = df['response_time'].tolist()
                
                # Create 2D histogram
                hist, xedges, yedges = np.histogram2d(response_times, rewards, bins=[8, 8])
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                
                # Use blue colormap
                im = axes[idx].imshow(hist.T, origin='lower', extent=extent, 
                                     aspect='auto', cmap='Blues', interpolation='nearest')
                
                # Add grid for square effect with fewer x-ticks
                axes[idx].set_xticks(xedges[::2])  # Show every other tick
                axes[idx].set_yticks(yedges)
                axes[idx].grid(True, color='white', linewidth=0.5)
                
                axes[idx].set_xlabel('Response Time (s)' if idx == 1 else '', fontsize=20)
                axes[idx].set_ylabel('Reward' if idx == 0 else '', fontsize=20)
                axes[idx].set_title(self.server_labels[train_server], fontsize=22)
                axes[idx].tick_params(axis='x', which='major', labelsize=16, rotation=45)
                axes[idx].tick_params(axis='y', which='major', labelsize=18)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[idx])
                cbar.set_label('Count', fontsize=16)
                cbar.ax.tick_params(labelsize=14)
        
        plt.suptitle('Reward vs Response Time Comparison', fontsize=28, y=0.9)
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        
        filename = f"inference_generalization_{self.target_server.lower()}_reward_vs_response_time.png"
        plt.savefig(self.data_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {filename}")
        plt.close()
    
    def create_key_performance_metrics_chart(self):
        """Create key performance metrics comparison chart"""
        df = self.load_csv_data("key_performance_metrics")
        if df is None:
            return
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 修改：去掉Completion Rate，只保留3个指标
        metrics_names = ['Avg\nReward', 'Success\nRate', 'Response\nTime']
        x = np.arange(len(metrics_names))
        width = 0.25
        
        # Find max values for each metric for normalization
        max_values = [
            max([df.loc[0, server] for server in self.training_servers]),
            1.0,  # Success rate max is 1
            max([df.loc[3, server] for server in self.training_servers])  # 修改：Response Time在第4行(索引3)
        ]
        
        for idx, train_server in enumerate(self.training_servers):
            # 修改：取第0、1、3行，跳过第2行的Completion Rate
            values = [df.loc[0, train_server], df.loc[1, train_server], df.loc[3, train_server]]
            
            bars = ax.bar(x + idx * width, values, width, 
                         label=self.server_labels[train_server], color=self.server_colors[train_server], 
                         alpha=0.7, edgecolor='black')
            
            # Add hatch pattern
            for bar in bars:
                bar.set_hatch(self.server_hatches[train_server])
            
            # Add value labels with larger font
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(max_values)*0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=20)
        
        ax.set_xlabel('Performance Metrics', fontsize=25)
        ax.set_ylabel('Value', fontsize=25)
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_names, fontsize=22)
        ax.legend(fontsize=22, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_ylim(0, max(max_values) * 1.2)
        plt.title('Key Performance Metrics Comparison', fontsize=28)
        plt.tight_layout()
        
        filename = f"inference_generalization_{self.target_server.lower()}_key_performance_metrics.png"
        plt.savefig(self.data_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {filename}")
        plt.close()
    

    
    def create_communication_overhead_episodes_chart(self):
        """Create communication overhead by episodes chart"""
        df = self.load_csv_data("communication_overhead_episodes")
        if df is None:
            return
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        n_bins = 8
        positions = []
        all_data = []
        colors = []
        hatches = []
        
        for train_server in self.training_servers:
            server_df = df[df['server'] == train_server]
            overheads = server_df['communication_overhead'].tolist()
            bin_size = len(overheads) // n_bins
            
            for bin_idx in range(n_bins):
                start_idx = bin_idx * bin_size
                end_idx = (bin_idx + 1) * bin_size if bin_idx < n_bins - 1 else len(overheads)
                bin_data = overheads[start_idx:end_idx]
                if bin_data:
                    all_data.append(bin_data)
                    pos = bin_idx * 4 + self.training_servers.index(train_server) * 1  # Increased spacing
                    positions.append(pos)
                    colors.append(self.server_colors[train_server])
                    hatches.append(self.server_hatches[train_server])
        
        bp = ax.boxplot(all_data, positions=positions, widths=0.8, patch_artist=True,
                       showfliers=False)
        
        for patch, color, hatch in zip(bp['boxes'], colors, hatches):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_hatch(hatch)
        
        # Add legend
        handles = [mpatches.Rectangle((0,0),1,1, facecolor=self.server_colors[server], 
                                    hatch=self.server_hatches[server], alpha=0.7, edgecolor='black') 
                  for server in self.training_servers]
        ax.legend(handles, [self.server_labels[s] for s in self.training_servers], 
                 fontsize=22, loc='upper right')
        
        bin_labels = [f'{i*25+1}-{(i+1)*25}' for i in range(n_bins)]
        ax.set_xticks([i * 4 + 1 for i in range(n_bins)])  # Adjusted positions
        ax.set_xticklabels(bin_labels, fontsize=18, rotation=0)  # No rotation needed with more space
        ax.set_xlabel('Episode Range', fontsize=25)
        ax.set_ylabel('Communication Overhead (s)', fontsize=25)
        ax.set_title('Communication Overhead by Episodes', fontsize=28)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.margins(x=0.02)  # Add margins
        plt.tight_layout()
        
        filename = f"inference_generalization_{self.target_server.lower()}_communication_overhead_episodes.png"
        plt.savefig(self.data_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {filename}")
        plt.close()
    

    
    def create_all_charts(self):
        """Create all comparative visualization charts"""
        print("="*60)
        print("GENERATING COMPARATIVE VISUALIZATIONS")
        print("="*60)
        
        print("\n1. Creating Active Server Usage Distribution Chart...")
        self.create_active_server_usage_chart()
        
        print("\n2. Creating Model Type Usage Chart...")
        self.create_model_type_usage_chart()
        
        print("\n3. Creating Reward vs Response Time Chart...")
        self.create_reward_vs_response_time_chart()
        
        print("\n4. Creating Key Performance Metrics Chart...")
        self.create_key_performance_metrics_chart()
        
        print("\n5. Creating Communication Overhead by Episodes Chart...")
        self.create_communication_overhead_episodes_chart()
        
        print(f"\n✅ All comparative charts saved to: {self.data_dir}")
        print(f"📁 Chart naming pattern: inference_generalization_{self.target_server.lower()}_[chart_name].png")


def main():
    """Main function to generate all visualization charts"""
    
    # Initialize visualizer
    visualizer = GeneralizationVisualizer(target_server="Server4")
    
    # Create all charts
    visualizer.create_all_charts()


if __name__ == "__main__":
    main()
