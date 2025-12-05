#!/usr/bin/env python3
"""
PPO Value Adaptive Training Data Visualization Tool - Multi-Server Comparison
Professional chart extraction and generation system for training data analysis.
Generates 11 comparison charts from multiple training datasets.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd


class MultiServerTrainingVisualizer:
    """
    Multi-server training data visualizer for comparative analysis.
    Reads multiple training datasets and generates comparison charts.
    """
    
    def __init__(self, output_directory: Optional[str] = None):
        """
        Initialize multi-server training visualizer.
        
        Args:
            output_directory: Output directory path, uses default results folder if None
        """
        self.logger = self._setup_logging()
        
        # Path configuration
        current_dir = Path(__file__).parent
        self.results_dir = current_dir.parent / "results"
        self.output_dir = Path(output_directory) if output_directory else (current_dir / "results")
        
        # Training data files configuration
        self.training_files = {
            'Server1': {
                'json': 'ppo_value_adaptive_training_data_1000000.json',
                'critic': 'ppo_value_adaptive_training_critic_model_1000000.pth',
                'actor': 'ppo_value_adaptive_training_actor_model_1000000.pth',
                'steps': '1000000'
            },
            'Server2': {
                'json': 'ppo_value_adaptive_training_data_1000000_Server2.json',
                'critic': 'ppo_value_adaptive_training_critic_model_1000000_Server2.pth',
                'actor': 'ppo_value_adaptive_training_actor_model_1000000_Server2.pth',
                'steps': '1000000'
            },
            'Server3': {
                'json': 'ppo_value_adaptive_training_data_1000000_Server3.json',
                'critic': 'ppo_value_adaptive_training_critic_model_1000000_Server3.pth',
                'actor': 'ppo_value_adaptive_training_actor_model_1000000_Server3.pth',
                'steps': '1000000'
            }
        }
        
        # Colors for different servers
        self.server_colors = {
            'Server1': '#2E86C1',  # Blue
            'Server2': '#E74C3C',  # Red
            'Server3': '#27AE60'   # Green
        }
        
        # Display name mapping
        self.server_display_names = {
            'Server1': 'S1',
            'Server2': 'S2', 
            'Server3': 'S3'
        }
        
        # Training data cache
        self.training_data = {}
        
        self.logger.info("Multi-Server Training Visualizer initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def load_all_training_data(self) -> bool:
        """
        Load training data from all server configurations.
        
        Returns:
            True if at least one dataset loaded successfully, False otherwise
        """
        success_count = 0
        
        for server_name, files in self.training_files.items():
            try:
                data_path = self.results_dir / files['json']
                
                if not data_path.exists():
                    self.logger.warning(f"Training data file not found: {data_path}")
                    continue
                
                self.logger.info(f"Loading training data for {server_name}: {files['json']}")
                
                with open(data_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                # Validate data
                if not isinstance(data, dict):
                    self.logger.error(f"Invalid data format for {server_name}")
                    continue
                
                episodes = len(data.get('total_rewards', []))
                if episodes == 0:
                    self.logger.error(f"No training episodes found for {server_name}")
                    continue
                
                self.training_data[server_name] = data
                self.logger.info(f"Loaded {episodes} episodes for {server_name}")
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to load data for {server_name}: {str(e)}")
        
        self.logger.info(f"Successfully loaded data for {success_count}/{len(self.training_files)} servers")
        return success_count > 0
    
    def generate_adaptive_learning_rate_charts(self) -> List[bool]:
        """Generate 3 independent adaptive learning rate charts."""
        results = []
        
        for server_name, data in self.training_data.items():
            try:
                if 'adaptive_lr_factors' not in data:
                    self.logger.warning(f"No learning rate data for {server_name}")
                    results.append(False)
                    continue
                
                lr_factors = data['adaptive_lr_factors']
                iterations = list(range(len(lr_factors)))
                
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                
                # Plot learning rate factor
                ax.plot(iterations, lr_factors, color='#2E86C1',  # 统一使用蓝色
                       linewidth=2.5, alpha=0.8, label=f'{self.server_display_names[server_name]} LR Factor')
                
                # Add trend line
                if len(lr_factors) > 10:
                    z = np.polyfit(iterations, lr_factors, 3)
                    p = np.poly1d(z)
                    ax.plot(iterations, p(iterations), '--', color='gray', 
                           alpha=0.6, linewidth=1.5, label='Trend')
                
                ax.set_title(f'Adaptive Learning Rate Factor - {self.server_display_names[server_name]}', 
                           fontsize=28, fontweight='normal', pad=20)
                ax.set_xlabel('Training Iteration', fontsize=25)
                ax.set_ylabel('Learning Rate Factor', fontsize=25)
                ax.tick_params(axis='both', which='major', labelsize=25)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=20)
                
                # Save chart
                filename = f"train_adaptive_learning_rate_{self.training_files[server_name]['steps']}_{server_name}.png"
                output_path = self.output_dir / filename
                fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
                self.logger.info(f"Generated adaptive learning rate chart for {server_name}")
                results.append(True)
                
            except Exception as e:
                self.logger.error(f"Failed to generate adaptive learning rate chart for {server_name}: {str(e)}")
                results.append(False)
        
        return results
    
    def generate_reward_distribution_chart(self) -> bool:
        """Generate merged reward distribution bar chart."""
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            # Collect all reward data
            all_rewards = {}
            for server_name, data in self.training_data.items():
                if 'total_rewards' in data:
                    all_rewards[server_name] = data['total_rewards']
            
            if not all_rewards:
                self.logger.warning("No reward data available for distribution chart")
                return False
            
            # Create bins and calculate histograms
            all_values = np.concatenate([rewards for rewards in all_rewards.values()])
            bins = np.linspace(all_values.min(), all_values.max(), 30)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            bar_width = (bins[1] - bins[0]) / (len(all_rewards) + 0.5)
            
            for i, (server_name, rewards) in enumerate(all_rewards.items()):
                counts, _ = np.histogram(rewards, bins=bins)
                x_positions = bin_centers + (i - len(all_rewards)/2 + 0.5) * bar_width
                
                ax.bar(x_positions, counts, bar_width, 
                      color=self.server_colors[server_name], alpha=0.7, 
                      label=f'{self.server_display_names[server_name]} (μ={np.mean(rewards):.2f})')
            
            ax.set_title('Reward Distribution Comparison', fontsize=28, fontweight='normal', pad=20)
            ax.set_xlabel('Episode Reward', fontsize=25)
            ax.set_ylabel('Frequency', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=25)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=20)
            
            # Save chart
            filename = "train_reward_distribution_comparison.png"
            output_path = self.output_dir / filename
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            self.logger.info("Generated merged reward distribution chart")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate reward distribution chart: {str(e)}")
            return False
    
    def generate_system_performance_charts(self) -> List[bool]:
        """Generate 3 dual-axis system performance charts (completion rate + load balance)."""
        results = []
        
        # Generate 3 dual-axis charts (completion rate + load balance)
        for server_name, data in self.training_data.items():
            try:
                if 'completion_rates' not in data or 'load_balance_scores' not in data:
                    self.logger.warning(f"Missing system performance data for {server_name}")
                    results.append(False)
                    continue
                
                fig, ax1 = plt.subplots(figsize=(14, 8))
                fig.patch.set_facecolor('white')
                ax1.set_facecolor('white')
                
                completion_rates = data['completion_rates']
                load_scores = data['load_balance_scores']
                iterations = list(range(len(completion_rates)))
                
                # Plot completion rate on primary y-axis (left) - 统一使用蓝色
                color1 = '#2E86C1'  # 蓝色 - Completion Rate
                ax1.set_xlabel('Training Iteration', fontsize=25)
                ax1.set_ylabel('Completion Rate', color=color1, fontsize=25)
                
                # Calculate confidence intervals for completion rate
                if 'confidence_intervals' in data and 'completion_rates' in data['confidence_intervals']:
                    ci_data = data['confidence_intervals']['completion_rates']
                    relative_margin = 2.0 * (ci_data['ci_upper'] - ci_data['ci_lower']) / (2 * ci_data['mean'])
                    ci_upper = [val + val * relative_margin for val in completion_rates]
                    ci_lower = [max(0, val - val * relative_margin) for val in completion_rates]
                    ax1.fill_between(iterations, ci_lower, ci_upper, color=color1, alpha=0.15)
                
                line1 = ax1.plot(iterations, completion_rates, color=color1, linewidth=2.5, 
                               alpha=0.8, label='Completion Rate')
                ax1.tick_params(axis='y', labelcolor=color1, labelsize=20)
                ax1.tick_params(axis='x', labelsize=20)
                ax1.grid(True, alpha=0.3)
                
                # Set appropriate y-axis range for completion rate
                if completion_rates:
                    y_min = max(0, min(completion_rates) - 0.05)
                    y_max = min(1.0, max(completion_rates) + 0.05)
                    ax1.set_ylim(y_min, y_max)
                
                # Create secondary y-axis for load balance score - 统一使用橙色
                ax2 = ax1.twinx()
                color2 = '#FF6B35'  # 橙色 - Load Balance Score
                ax2.set_ylabel('Load Balance Score', color=color2, fontsize=25)
                
                # Calculate confidence intervals for load balance
                if 'confidence_intervals' in data and 'load_balance_scores' in data['confidence_intervals']:
                    ci_data = data['confidence_intervals']['load_balance_scores']
                    relative_margin = 2.0 * (ci_data['ci_upper'] - ci_data['ci_lower']) / (2 * ci_data['mean'])
                    ci_upper = [val + val * relative_margin for val in load_scores]
                    ci_lower = [max(0, val - val * relative_margin) for val in load_scores]
                    ax2.fill_between(iterations, ci_lower, ci_upper, color=color2, alpha=0.15)
                
                line2 = ax2.plot(iterations, load_scores, color=color2, linewidth=2.5, 
                               alpha=0.8, linestyle='--', label='Load Balance Score')
                ax2.tick_params(axis='y', labelcolor=color2, labelsize=20)
                
                # Set title and legend
                ax1.set_title(f'System Performance - {self.server_display_names[server_name]}', 
                            fontsize=28, fontweight='normal', pad=20)
                
                # Create combined legend - 根据服务器设置不同的图例位置
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                if server_name == 'Server1':
                    ax1.legend(lines, labels, loc='lower right', fontsize=18)   # Server1放右下角
                else:
                    ax1.legend(lines, labels, loc='upper left', fontsize=18)    # Server2和Server3放左上角
                
                filename = f"train_system_performance_combined_{self.training_files[server_name]['steps']}_{server_name}.png"
                output_path = self.output_dir / filename
                fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
                self.logger.info(f"Generated combined system performance chart for {server_name}")
                results.append(True)
                
            except Exception as e:
                self.logger.error(f"Failed to generate system performance chart for {server_name}: {str(e)}")
                results.append(False)
        
        return results
    
    def generate_training_losses_charts(self) -> List[bool]:
        """Generate 3 dual-axis training losses charts (actor + critic)."""
        results = []
        
        # Generate 3 dual-axis charts (actor loss + critic loss)
        for server_name, data in self.training_data.items():
            try:
                if 'actor_losses' not in data or 'critic_losses' not in data:
                    self.logger.warning(f"Missing training losses data for {server_name}")
                    results.append(False)
                    continue
                
                fig, ax1 = plt.subplots(figsize=(14, 8))
                fig.patch.set_facecolor('white')
                ax1.set_facecolor('white')
                
                actor_losses = data['actor_losses']
                critic_losses = data['critic_losses']
                iterations = list(range(len(actor_losses)))
                
                # Plot actor loss on primary y-axis (left) - 统一使用绿色
                color1 = '#27AE60'  # 绿色 - Actor Loss
                ax1.set_xlabel('Training Iteration', fontsize=25)
                ax1.set_ylabel('Actor Loss', color=color1, fontsize=25)
                
                # Calculate confidence intervals for actor loss
                if len(actor_losses) > 10:
                    window_size = min(20, len(actor_losses) // 5)
                    rolling_std = []
                    for i in range(len(actor_losses)):
                        start_idx = max(0, i - window_size // 2)
                        end_idx = min(len(actor_losses), i + window_size // 2 + 1)
                        window_data = actor_losses[start_idx:end_idx]
                        rolling_std.append(np.std(window_data))
                    
                    ci_upper = [val + 1.0 * std for val, std in zip(actor_losses, rolling_std)]
                    ci_lower = [val - 1.0 * std for val, std in zip(actor_losses, rolling_std)]
                    ax1.fill_between(iterations, ci_lower, ci_upper, color=color1, alpha=0.12)
                
                line1 = ax1.plot(iterations, actor_losses, color=color1, linewidth=2.5, 
                               alpha=0.8, label='Actor Loss')
                ax1.tick_params(axis='y', labelcolor=color1, labelsize=20)
                ax1.tick_params(axis='x', labelsize=20)
                ax1.grid(True, alpha=0.3)
                
                # Create secondary y-axis for critic loss - 统一使用紫色
                ax2 = ax1.twinx()
                color2 = '#8E44AD'  # 紫色 - Critic Loss
                ax2.set_ylabel('Critic Loss', color=color2, fontsize=25)
                
                # Calculate confidence intervals for critic loss
                data_std = np.std(critic_losses)
                data_mean = np.mean(critic_losses)
                relative_margin = 3.0 * 1.96 * data_std / (data_mean * np.sqrt(len(critic_losses)))
                
                ci_upper = [val + abs(val) * relative_margin for val in critic_losses]
                ci_lower = [max(val * 0.1, val - abs(val) * relative_margin) for val in critic_losses]
                ax2.fill_between(iterations, ci_lower, ci_upper, color=color2, alpha=0.15)
                
                line2 = ax2.plot(iterations, critic_losses, color=color2, linewidth=2.5, 
                               alpha=0.8, linestyle='--', label='Critic Loss')
                ax2.tick_params(axis='y', labelcolor=color2, labelsize=20)
                
                # Set title and legend
                ax1.set_title(f'Training Losses - {self.server_display_names[server_name]}', 
                            fontsize=28, fontweight='normal', pad=20)
                
                # Create combined legend - 根据服务器设置不同的图例位置
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                if server_name == 'Server2':
                    ax1.legend(lines, labels, loc='center right', fontsize=18)  # Server2放右侧中间
                elif server_name == 'Server3':
                    ax1.legend(lines, labels, loc='lower right', fontsize=18)   # Server3放右下角
                else:
                    ax1.legend(lines, labels, loc='upper right', fontsize=18)   # Server1放右上角
                
                filename = f"train_training_losses_combined_{self.training_files[server_name]['steps']}_{server_name}.png"
                output_path = self.output_dir / filename
                fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
                self.logger.info(f"Generated combined training losses chart for {server_name}")
                results.append(True)
                
            except Exception as e:
                self.logger.error(f"Failed to generate training losses chart for {server_name}: {str(e)}")
                results.append(False)
        
        return results
    
    def generate_training_rewards_progress_charts(self) -> List[bool]:
        """Generate 3 independent training rewards progress charts."""
        results = []
        
        for server_name, data in self.training_data.items():
            try:
                if 'total_rewards' not in data:
                    self.logger.warning(f"No total rewards data for {server_name}")
                    results.append(False)
                    continue
                
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                
                rewards = data['total_rewards']
                iterations = list(range(len(rewards)))
                
                # Plot raw data (main curves, clear and visible)
                ax.plot(iterations, rewards, color='#2E86C1',  # 统一使用蓝色
                       alpha=0.8, linewidth=2.5, label=f'{self.server_display_names[server_name]}')
                
                # Add confidence interval around raw data
                if 'confidence_intervals' in data and 'total_rewards' in data['confidence_intervals']:
                    ci_data = data['confidence_intervals']['total_rewards']
                    ci_upper_val = ci_data['ci_upper']
                    ci_lower_val = ci_data['ci_lower']
                    
                    # Create confidence band around raw data with enhanced visibility
                    reward_mean = ci_data['mean']
                    ci_width_ratio = 3.0 * (ci_upper_val - ci_lower_val) / reward_mean  # Amplify by 3x
                    
                    ci_upper_raw = [val * (1 + ci_width_ratio/2) for val in rewards]
                    ci_lower_raw = [val * (1 - ci_width_ratio/2) for val in rewards]
                    
                    # Plot confidence interval around raw data
                    ax.fill_between(iterations, ci_lower_raw, ci_upper_raw, 
                                   color='#2E86C1', alpha=0.2)  # 统一使用蓝色
                
                ax.set_title(f'Training Rewards Progress - {self.server_display_names[server_name]}', 
                           fontsize=28, fontweight='normal', pad=20)
                ax.set_xlabel('Training Episode', fontsize=25)
                ax.set_ylabel('Episode Reward', fontsize=25)
                ax.tick_params(axis='both', which='major', labelsize=25)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=20)
                
                filename = f"train_training_rewards_progress_{self.training_files[server_name]['steps']}_{server_name}.png"
                output_path = self.output_dir / filename
                fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
                self.logger.info(f"Generated training rewards progress chart for {server_name}")
                results.append(True)
                        
            except Exception as e:
                self.logger.error(f"Failed to generate training rewards progress chart for {server_name}: {str(e)}")
                results.append(False)
        
        return results
    
    def generate_all_charts(self) -> Dict[str, Any]:
        """
        Generate all 12 individual charts (3 servers × 4 chart types).
        
        Returns:
            Dictionary with generation results
        """
        self.logger.info("Starting multi-server chart generation")
        self.logger.info("=" * 60)
        
        if not self.load_all_training_data():
            self.logger.error("Failed to load training data")
            return {'success': False, 'message': 'No training data loaded'}
        
        results = {
            'adaptive_learning_rate': [],
            'reward_distribution': False,
            'system_performance': [],
            'training_losses': [],
            'training_rewards_progress': []
        }
        
        # Generate all chart types
        try:
            # 3 independent adaptive learning rate charts
            results['adaptive_learning_rate'] = self.generate_adaptive_learning_rate_charts()
            
            # 1 merged reward distribution chart - REMOVED per user request
            # results['reward_distribution'] = self.generate_reward_distribution_chart()
            results['reward_distribution'] = False
            
            # 3 system performance charts
            results['system_performance'] = self.generate_system_performance_charts()
            
            # 3 training losses charts
            results['training_losses'] = self.generate_training_losses_charts()
            
            # 3 training rewards progress charts
            results['training_rewards_progress'] = self.generate_training_rewards_progress_charts()
            
        except Exception as e:
            self.logger.error(f"Error during chart generation: {str(e)}")
            results['error'] = str(e)
        
        # Calculate success statistics
        total_charts = (
            len(results['adaptive_learning_rate']) +
            (1 if results['reward_distribution'] else 0) +
            len(results['system_performance']) +
            len(results['training_losses']) +
            len(results['training_rewards_progress'])
        )
        
        successful_charts = (
            sum(results['adaptive_learning_rate']) +
            (1 if results['reward_distribution'] else 0) +
            sum(results['system_performance']) +
            sum(results['training_losses']) +
            sum(results['training_rewards_progress'])
        )
        
        self.logger.info("=" * 60)
        self.logger.info(f"Chart generation completed: {successful_charts}/{total_charts} successful")
        self.logger.info(f"Charts saved to: {self.output_dir}")
        
        if successful_charts > 0:
            self.logger.info("Generated chart types:")
            self.logger.info("  • 3 Adaptive Learning Rate charts (individual)")
            self.logger.info("  • 3 System Performance charts (dual-axis: completion rate + load balance)")
            self.logger.info("  • 3 Training Losses charts (dual-axis: actor loss + critic loss)")
            self.logger.info("  • 3 Training Rewards Progress charts (individual)")
        
        results['total'] = total_charts
        results['successful'] = successful_charts
        results['success'] = successful_charts > 0
        
        return results


def setup_matplotlib_configuration():
    """Setup matplotlib configuration for professional chart output."""
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['savefig.format'] = 'png'
    plt.rcParams['savefig.dpi'] = 300
    
    # Disable shadows and effects for clean appearance
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.0


def main():
    """Main application entry point."""
    import argparse
    
    # Setup matplotlib configuration
    setup_matplotlib_configuration()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("PPO Multi-Server Training Data Visualization Tool")
    logger.info("=" * 60)
    
    parser = argparse.ArgumentParser(
        description="Generate multi-server training comparison charts",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Output directory for generated charts')
    
    args = parser.parse_args()
    
    try:
        # Initialize visualizer
        visualizer = MultiServerTrainingVisualizer(output_directory=args.output_dir)
        
        # Generate all charts
        results = visualizer.generate_all_charts()
        
        if results['success']:
            logger.info("=" * 60)
            logger.info("✅ Chart generation completed successfully")
            logger.info(f"📁 Charts saved to: {visualizer.output_dir}")
            logger.info(f"📊 Generated {results['successful']}/{results['total']} charts")
        else:
            logger.error("❌ Chart generation failed")
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()