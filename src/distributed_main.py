"""
Distributed Multi-Modal Model Orchestration System - Main Training File
Supports both single-agent PPO Value Adaptive and VAMAPPO multi-agent training
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distributed_env import DistributedModelOrchestrationEnv
from distributed_ppo import DistributedPPOValueAdaptive, VAMAPPO
from gpu_config import configure_gpu_for_training, monitor_gpu_memory, cleanup_gpu

class ValueAdaptiveTrainingAnalyzer:
    """Enhanced training analysis and visualization for PPO Value Adaptive"""
    
    def __init__(self):
        self.training_data = {
            'total_rewards': [],
            'completion_rates': [],
            'load_balance_scores': [],
            'actor_losses': [],
            'critic_losses': [],
            'episode_lengths': [],
            'timestamps': [],
            # Value adaptive specific metrics
            'value_uncertainties': [],
            'adaptive_lr_factors': [],
            'exploration_factors': [],
            'value_adaptation_rates': [],
            'td_errors': []
        }
    
    def update_metrics(self, metrics: Dict):
        """Update training metrics with value adaptive enhancements"""
        for key, value in metrics.items():
            if key in self.training_data:
                self.training_data[key].append(value)
                
        # Add timestamp for tracking
        self.training_data['timestamps'].append(time.time())
    
    def save_data(self, filepath: str):
        """Save training data to JSON with enhanced metadata and confidence intervals"""
        import scipy.stats as stats
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in self.training_data.items():
            if isinstance(value, list):
                json_data[key] = value
            else:
                json_data[key] = value.tolist() if hasattr(value, 'tolist') else value
        
        # Calculate confidence intervals for key metrics
        confidence_intervals = {}
        confidence_level = 0.95  # 95% confidence interval
        alpha = 1 - confidence_level
        
        def calculate_ci(data, label):
            """Calculate confidence interval for a dataset"""
            if len(data) < 2:
                return None
            
            # Ensure data is a numpy array with proper dtype
            data_array = np.array(data, dtype=np.float64)
            
            mean = np.mean(data_array)
            std = np.std(data_array, ddof=1)  # Sample standard deviation
            n = len(data_array)
            
            # Calculate t-statistic for small samples, z-statistic for large samples
            if n < 30:
                t_value = stats.t.ppf(1 - alpha/2, df=n-1)
                margin_error = t_value * (std / np.sqrt(n))
            else:
                z_value = stats.norm.ppf(1 - alpha/2)
                margin_error = z_value * (std / np.sqrt(n))
            
            ci_lower = mean - margin_error
            ci_upper = mean + margin_error
            ci_width = 2 * margin_error
            relative_width = (ci_width / abs(mean)) * 100 if mean != 0 else 0
            
            return {
                'mean': float(mean),
                'std': float(std),
                'n_samples': int(n),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'ci_width': float(ci_width),
                'relative_ci_width_percent': float(relative_width),
                'confidence_level': confidence_level
            }
        
        # Calculate CIs for key metrics
        if json_data.get('total_rewards'):
            # Overall performance CI
            confidence_intervals['total_rewards'] = calculate_ci(json_data['total_rewards'], 'Total Rewards')
            
            # Final performance CI (last 25% of episodes)
            final_quarter = json_data['total_rewards'][-max(1, len(json_data['total_rewards'])//4):]
            confidence_intervals['final_performance'] = calculate_ci(final_quarter, 'Final Performance')
        
        if json_data.get('completion_rates'):
            confidence_intervals['completion_rates'] = calculate_ci(json_data['completion_rates'], 'Completion Rates')
            
            # Final completion rate CI
            final_quarter = json_data['completion_rates'][-max(1, len(json_data['completion_rates'])//4):]
            confidence_intervals['final_completion_rate'] = calculate_ci(final_quarter, 'Final Completion Rate')
        
        if json_data.get('load_balance_scores'):
            confidence_intervals['load_balance_scores'] = calculate_ci(json_data['load_balance_scores'], 'Load Balance Scores')
        
        if json_data.get('value_uncertainties'):
            confidence_intervals['value_uncertainties'] = calculate_ci(json_data['value_uncertainties'], 'Value Uncertainties')
            
            # Final uncertainty CI
            final_quarter = json_data['value_uncertainties'][-max(1, len(json_data['value_uncertainties'])//4):]
            confidence_intervals['final_uncertainty'] = calculate_ci(final_quarter, 'Final Uncertainty')
        
        if json_data.get('adaptive_lr_factors'):
            confidence_intervals['adaptive_lr_factors'] = calculate_ci(json_data['adaptive_lr_factors'], 'Adaptive LR Factors')
            
            # Final adaptive LR CI
            final_quarter = json_data['adaptive_lr_factors'][-max(1, len(json_data['adaptive_lr_factors'])//4):]
            confidence_intervals['final_adaptive_lr'] = calculate_ci(final_quarter, 'Final Adaptive LR')
        
        if json_data.get('episode_lengths'):
            confidence_intervals['episode_lengths'] = calculate_ci(json_data['episode_lengths'], 'Episode Lengths')
        
        # Add enhanced metadata with confidence intervals
        json_data['metadata'] = {
            'algorithm': 'PPO_Value_Adaptive',
            'total_episodes': len(json_data.get('total_rewards', [])),
            'training_duration': time.time() - json_data['timestamps'][0] if json_data['timestamps'] else 0,
            'final_exploration_factor': json_data['exploration_factors'][-1] if json_data['exploration_factors'] else 1.0,
            'final_adaptive_lr': json_data['adaptive_lr_factors'][-1] if json_data['adaptive_lr_factors'] else 1.0,
            'confidence_level': confidence_level
        }
        
        # Add confidence intervals to JSON
        json_data['confidence_intervals'] = confidence_intervals
        
        # Calculate and add stability metrics
        stability_metrics = {}
        if json_data.get('total_rewards') and len(json_data['total_rewards']) > 10:
            rewards_array = np.array(json_data['total_rewards'], dtype=np.float64)
            # Coefficient of variation (CV) - lower is more stable
            cv = (np.std(rewards_array) / np.mean(rewards_array)) * 100
            stability_metrics['reward_coefficient_of_variation_percent'] = float(cv)
            
            # Trend stability (correlation with episode number) - 使用numpy直接计算
            episodes = np.arange(len(rewards_array), dtype=np.float64)
            correlation_matrix = np.corrcoef(episodes, rewards_array)
            correlation = correlation_matrix[0, 1]
            # 处理可能的NaN值
            if np.isnan(correlation):
                correlation = 0.0
            stability_metrics['reward_trend_correlation'] = float(correlation)
        
        if json_data.get('adaptive_lr_factors') and len(json_data['adaptive_lr_factors']) > 10:
            lr_factors = np.array(json_data['adaptive_lr_factors'], dtype=np.float64)
            cv = (np.std(lr_factors) / np.mean(lr_factors)) * 100
            stability_metrics['adaptive_lr_coefficient_of_variation_percent'] = float(cv)
        
        json_data['stability_metrics'] = stability_metrics
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Training data saved to: {filepath}")
        
        # Print confidence interval summary
        if confidence_intervals:
            print("\n📊 Confidence Interval Summary (95% CI):")
            for metric, ci_data in confidence_intervals.items():
                if ci_data:
                    print(f"  {metric}: {ci_data['mean']:.4f} ± {ci_data['ci_width']/2:.4f} "
                          f"[{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}] "
                          f"(±{ci_data['relative_ci_width_percent']:.1f}%)")
        
        if stability_metrics:
            print(f"\n📈 Stability Metrics:")
            for metric, value in stability_metrics.items():
                print(f"  {metric}: {value:.2f}")
        
        print(f"✅ Enhanced training analysis saved with confidence intervals!")
    
    def generate_plots(self, save_dir: str, filename_prefix: str = "ppo_value_adaptive_training"):
        """Generate comprehensive training plots for value adaptive metrics"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('PPO Value Adaptive Training Progress - Comprehensive Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Total Rewards with trend line
        if self.training_data['total_rewards']:
            rewards = self.training_data['total_rewards']
            episodes = range(len(rewards))
            axes[0, 0].plot(episodes, rewards, 'b-', linewidth=2, alpha=0.8, label='Rewards')
            
            # Add trend line
            if len(rewards) > 10:
                z = np.polyfit(episodes, rewards, 1)
                p = np.poly1d(z)
                axes[0, 0].plot(episodes, p(episodes), 'r--', alpha=0.8, label='Trend')
            
            axes[0, 0].set_title('Total Rewards per Episode', fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Training Losses
        if self.training_data['actor_losses'] and self.training_data['critic_losses']:
            axes[0, 1].plot(self.training_data['actor_losses'], 'r-', linewidth=2, alpha=0.8, label='Actor Loss')
            axes[0, 1].plot(self.training_data['critic_losses'], 'b-', linewidth=2, alpha=0.8, label='Critic Loss')
            axes[0, 1].set_title('Training Losses', fontweight='bold')
            axes[0, 1].set_xlabel('Update Iteration')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Value Uncertainties
        if self.training_data['value_uncertainties']:
            uncertainties = self.training_data['value_uncertainties']
            axes[0, 2].plot(uncertainties, 'g-', linewidth=2, alpha=0.8)
            axes[0, 2].fill_between(range(len(uncertainties)), uncertainties, alpha=0.3, color='green')
            axes[0, 2].set_title('Value Function Uncertainties', fontweight='bold')
            axes[0, 2].set_xlabel('Update Iteration')
            axes[0, 2].set_ylabel('Uncertainty')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Adaptive Learning Rate Factors
        if self.training_data['adaptive_lr_factors']:
            lr_factors = self.training_data['adaptive_lr_factors']
            axes[1, 0].plot(lr_factors, 'orange', linewidth=2, alpha=0.8)
            axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline')
            axes[1, 0].set_title('Adaptive Learning Rate Factors', fontweight='bold')
            axes[1, 0].set_xlabel('Update Iteration')
            axes[1, 0].set_ylabel('LR Adaptation Factor')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Exploration Factors with decay curve
        if self.training_data['exploration_factors']:
            exploration = self.training_data['exploration_factors']
            axes[1, 1].plot(exploration, 'purple', linewidth=2, alpha=0.8)
            axes[1, 1].set_title('Exploration Factor Decay', fontweight='bold')
            axes[1, 1].set_xlabel('Update Iteration')
            axes[1, 1].set_ylabel('Exploration Factor')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Task Completion Rates
        if self.training_data['completion_rates']:
            completion = self.training_data['completion_rates']
            axes[1, 2].plot(completion, 'brown', linewidth=2, alpha=0.8)
            axes[1, 2].fill_between(range(len(completion)), completion, alpha=0.3, color='brown')
            axes[1, 2].set_title('Task Completion Rates', fontweight='bold')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Completion Rate')
            axes[1, 2].set_ylim(0, 1.1)
            axes[1, 2].grid(True, alpha=0.3)
        
        # Plot 7: Load Balance Scores
        if self.training_data['load_balance_scores']:
            load_balance = self.training_data['load_balance_scores']
            axes[2, 0].plot(load_balance, 'teal', linewidth=2, alpha=0.8)
            axes[2, 0].set_title('Load Balance Scores', fontweight='bold')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Load Balance Score')
            axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 8: Episode Lengths
        if self.training_data['episode_lengths']:
            ep_lengths = self.training_data['episode_lengths']
            axes[2, 1].plot(ep_lengths, 'navy', linewidth=2, alpha=0.8)
            axes[2, 1].set_title('Episode Lengths', fontweight='bold')
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('Steps per Episode')
            axes[2, 1].grid(True, alpha=0.3)
        
        # Plot 9: Value Adaptation Rates
        if self.training_data['value_adaptation_rates']:
            adaptation_rates = self.training_data['value_adaptation_rates']
            axes[2, 2].plot(adaptation_rates, 'darkred', linewidth=2, alpha=0.8)
            axes[2, 2].axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Baseline')
            axes[2, 2].set_title('Value Adaptation Rates', fontweight='bold')
            axes[2, 2].set_xlabel('Iteration')
            axes[2, 2].set_ylabel('Adaptation Rate')
            axes[2, 2].legend()
            axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = f"{save_dir}/{filename_prefix}_comprehensive.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {plot_filename}")
        
        # Generate additional summary statistics plot
        self._generate_summary_statistics(save_dir, filename_prefix)
    
    def _generate_summary_statistics(self, save_dir: str, filename_prefix: str):
        """Generate summary statistics visualization"""
        if not any(self.training_data.values()):
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PPO Value Adaptive Training Summary Statistics', 
                    fontsize=14, fontweight='bold')
        
        # Performance Distribution
        if self.training_data['total_rewards']:
            rewards = self.training_data['total_rewards'][-50:]  # Last 50 episodes
            axes[0, 0].hist(rewards, bins=15, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].axvline(np.mean(rewards), color='red', linestyle='--', 
                            label=f'Mean: {np.mean(rewards):.2f}')
            axes[0, 0].set_title('Reward Distribution (Last 50 Episodes)')
            axes[0, 0].set_xlabel('Total Reward')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Uncertainty Distribution
        if self.training_data['value_uncertainties']:
            uncertainties = self.training_data['value_uncertainties'][-100:]  # Last 100 updates
            axes[0, 1].hist(uncertainties, bins=15, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].axvline(np.mean(uncertainties), color='red', linestyle='--',
                            label=f'Mean: {np.mean(uncertainties):.3f}')
            axes[0, 1].set_title('Value Uncertainty Distribution')
            axes[0, 1].set_xlabel('Uncertainty')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Progress Comparison
        if (self.training_data['total_rewards'] and 
            len(self.training_data['total_rewards']) > 20):
            
            rewards = self.training_data['total_rewards']
            early_rewards = rewards[:len(rewards)//4]
            late_rewards = rewards[-len(rewards)//4:]
            
            x_pos = [1, 2]
            means = [np.mean(early_rewards), np.mean(late_rewards)]
            stds = [np.std(early_rewards), np.std(late_rewards)]
            
            axes[1, 0].bar(x_pos, means, yerr=stds, alpha=0.7, 
                          color=['lightcoral', 'lightblue'], capsize=5)
            axes[1, 0].set_title('Learning Progress Comparison')
            axes[1, 0].set_xlabel('Training Phase')
            axes[1, 0].set_ylabel('Average Reward')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(['Early Training', 'Late Training'])
            axes[1, 0].grid(True, alpha=0.3)
        
        # Adaptive Metrics Correlation
        if (self.training_data['adaptive_lr_factors'] and 
            self.training_data['value_uncertainties']):
            
            lr_factors = self.training_data['adaptive_lr_factors']
            uncertainties = self.training_data['value_uncertainties']
            
            # Ensure same length
            min_len = min(len(lr_factors), len(uncertainties))
            lr_factors = lr_factors[:min_len]
            uncertainties = uncertainties[:min_len]
            
            axes[1, 1].scatter(uncertainties, lr_factors, alpha=0.6, 
                             color='purple', s=20)
            axes[1, 1].set_title('Uncertainty vs Adaptive LR Correlation')
            axes[1, 1].set_xlabel('Value Uncertainty')
            axes[1, 1].set_ylabel('Adaptive LR Factor')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add correlation coefficient if possible
            if len(lr_factors) > 1:
                corr = np.corrcoef(uncertainties, lr_factors)[0, 1]
                axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                              transform=axes[1, 1].transAxes, 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        summary_filename = f"{save_dir}/{filename_prefix}_summary_stats.png"
        plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary statistics saved to {summary_filename}")

def detect_server_from_path(data_dir_path):
    """
    从数据路径中检测当前使用的服务器
    
    Args:
        data_dir_path: 数据目录路径
        
    Returns:
        server_name: Server1, Server2, Server3, 或 Server4
    """
    data_dir_str = str(data_dir_path)
    if "Server1" in data_dir_str:
        return "Server1"
    elif "Server2" in data_dir_str:
        return "Server2"
    elif "Server3" in data_dir_str:
        return "Server3"
    elif "Server4" in data_dir_str:
        return "Server4"
    else:
        # 默认返回Server1，如果无法检测
        return "Server1"

def generate_unique_filename(base_dir, filename_template, total_timesteps, server_name):
    """
    生成唯一的文件名，避免覆盖现有文件
    
    Args:
        base_dir: 保存目录
        filename_template: 文件名模板（包含{}占位符）
        total_timesteps: 训练步数
        server_name: 服务器名称
        
    Returns:
        unique_filename: 唯一的文件名
    """
    import os
    
    # 生成基础文件名
    base_filename = filename_template.format(total_timesteps, server_name)
    base_path = os.path.join(base_dir, base_filename)
    
    # 如果文件不存在，直接返回基础文件名
    if not os.path.exists(base_path):
        return base_filename
    
    # 如果文件存在，添加序号
    counter = 1
    while True:
        # 分离文件名和扩展名
        name_part, ext_part = os.path.splitext(base_filename)
        numbered_filename = f"{name_part}_{counter}{ext_part}"
        numbered_path = os.path.join(base_dir, numbered_filename)
        
        if not os.path.exists(numbered_path):
            return numbered_filename
        
        counter += 1
        
        # 防止无限循环，最多尝试1000次
        if counter > 1000:
            import time
            timestamp = int(time.time())
            timestamped_filename = f"{name_part}_{timestamp}{ext_part}"
            return timestamped_filename

def save_models_with_timesteps(model, total_timesteps, server_name):
    """
    Save models with timestep-specific naming convention including server name
    
    Args:
        model: Trained PPO model
        total_timesteps: Number of training timesteps for naming
        server_name: Server name (Server1, Server2, Server3, or Server4)
    """
    # 生成唯一的文件名
    actor_filename = generate_unique_filename(
        model.save_dir, 
        'ppo_value_adaptive_training_actor_model_{}_{}.pth',
        total_timesteps,
        server_name
    )
    critic_filename = generate_unique_filename(
        model.save_dir,
        'ppo_value_adaptive_training_critic_model_{}_{}.pth', 
        total_timesteps,
        server_name
    )
    
    # 创建完整路径
    actor_path = os.path.join(model.save_dir, actor_filename)
    critic_path = os.path.join(model.save_dir, critic_filename)
    
    # Save models with training metadata
    torch.save({
        'model_state_dict': model.actor.state_dict(),
        'optimizer_state_dict': model.actor_optim.state_dict(),
        'scheduler_state_dict': model.actor_scheduler.state_dict(),
        'exploration_factor': model.current_exploration_factor,
        'iteration': model.logger['i_so_far'],
        'training_timesteps': total_timesteps,
        'server_dataset': server_name,
        'algorithm': 'PPO_Value_Adaptive'
    }, actor_path)
    
    torch.save({
        'model_state_dict': model.critic.state_dict(),
        'optimizer_state_dict': model.critic_optim.state_dict(),
        'scheduler_state_dict': model.critic_scheduler.state_dict(),
        'adaptive_lr_factor': model.adaptive_lr_factor,
        'value_history': model.critic.value_history,
        'uncertainty_history': model.critic.uncertainty_history,
        'iteration': model.logger['i_so_far'],
        'training_timesteps': total_timesteps,
        'server_dataset': server_name,
        'algorithm': 'PPO_Value_Adaptive'
    }, critic_path)
    
    print(f"✅ PPO Value Adaptive models saved with server-specific timestep naming:")
    print(f"  Actor: {actor_filename}")
    print(f"  Critic: {critic_filename}")
    
    return actor_path, critic_path

def train_value_adaptive(args):
    """
    Train distributed PPO Value Adaptive agent with enhanced optimization
    """
    print("Starting Distributed Multi-Modal Model Orchestration PPO Value Adaptive Training...")
    
    # GPU configuration
    device, gpu_info = configure_gpu_for_training()
    
    # Data paths setup - 更换为Server3数据集
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "database" / "Server3"
    
    topology_path = data_dir / "server3_topology.csv"
    servers_path = data_dir / "server3_information.csv"
    models_path = data_dir / "server3_RandomDeployment.csv"
    train_tasks_path = data_dir / "train.CSV"
    test_tasks_path = data_dir / "test.CSV"
    
    # Validate data files
    for path in [topology_path, servers_path, models_path, train_tasks_path, test_tasks_path]:
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
    
    print("Data file validation passed")
    
    # Create environment
    print("Creating distributed environment...")
    env = DistributedModelOrchestrationEnv(
        topology_path=str(topology_path),
        servers_path=str(servers_path),
        models_path=str(models_path),
        tasks_path=str(train_tasks_path),
        test_tasks_path=str(test_tasks_path)
    )
    
    # Enhanced PPO Value Adaptive hyperparameters - Optimized for better convergence
    ppo_value_adaptive_hyperparameters = {
        # Standard PPO parameters - Fine-tuned for convergence
        'timesteps_per_batch': 1024,       # Keep stable batch size
        'max_timesteps_per_episode': 25,   # Slightly longer for better completion
        'n_updates_per_iteration': 6,      # More updates for convergence
        'lr': 0.0002,                      # Slightly reduce for stability
        'gamma': 0.99,                     # Higher discount for long sequences
        'clip': 0.12,                      # Reduce clipping for stability
        'ent_coef': 0.03,                  # Reduce entropy as performance improves
        'value_clip': 0.15,                # Reduce value clipping
        'value_reg_coef': 0.3,             # Further reduce regularization
        'max_grad_norm': 0.8,              # Slightly smaller gradients
        'save_freq': 10,                   # Keep same frequency
        
        # Value Adaptive specific parameters - Optimized
        'adaptive_clip_range': [0.1, 0.3],            # Wider adaptive clipping range
        'uncertainty_regularization': 0.05,            # Reduced uncertainty penalty
        'exploration_schedule': 'exponential',          # Keep exponential decay
        'value_adaptation_frequency': 5,                # More frequent adaptation
        'adaptive_lr_bounds': [0.0001, 0.001],         # Higher learning rate bounds
        'uncertainty_threshold': 0.3,                  # Lower threshold for more adaptation
        'exploration_decay_rate': 0.998,               # Slower exploration decay
        'value_history_size': 50,                      # Smaller history for faster adaptation
    }
    
    # Create PPO Value Adaptive model
    print("Creating PPO Value Adaptive model...")
    model = DistributedPPOValueAdaptive(env, device=device, **ppo_value_adaptive_hyperparameters)
    
    # Training configuration - only use steps parameter
    if args.steps:
        total_timesteps = args.steps
    else:
        raise ValueError("Please specify training steps using --steps parameter")
    
    print(f"PPO Value Adaptive Training configuration:")
    print(f"  - Algorithm: PPO Value Adaptive")
    print(f"  - Total timesteps: {total_timesteps:,}")
    print(f"  - Batch size: {ppo_value_adaptive_hyperparameters['timesteps_per_batch']}")
    print(f"  - Base learning rate: {ppo_value_adaptive_hyperparameters['lr']}")
    print(f"  - Episode length: {ppo_value_adaptive_hyperparameters['max_timesteps_per_episode']}")
    print(f"  - Updates per iteration: {ppo_value_adaptive_hyperparameters['n_updates_per_iteration']}")
    print(f"  - Adaptive clipping range: {ppo_value_adaptive_hyperparameters['adaptive_clip_range']}")
    print(f"  - Exploration decay rate: {ppo_value_adaptive_hyperparameters['exploration_decay_rate']}")
    
    # Initialize enhanced training analyzer
    analyzer = ValueAdaptiveTrainingAnalyzer()
    
    # Start GPU memory monitoring
    if device.type == 'cuda':
        monitor_gpu_memory()
    
    start_time = time.time()
    print(f"Training start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Train model with value adaptive enhancements
    training_completed = False
    try:
        model.learn(total_timesteps, analyzer=analyzer)
        training_completed = True
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("❌ TRAINING INTERRUPTED BY USER")
        print("="*80)
        print("Training was stopped before completion. Models and data will be saved with 'interrupted' suffix.")
        total_timesteps = f"{total_timesteps}_interrupted"
    except Exception as e:
        print(f"\n" + "="*80)
        print(f"❌ TRAINING ERROR: {e}")
        print("="*80)
        raise
    
    training_time = time.time() - start_time
    end_time = time.strftime('%Y-%m-%d %H:%M:%S')
    
    print("=" * 80)
    if training_completed:
        print(f"✅ PPO Value Adaptive Training completed successfully!")
    else:
        print(f"⚠️ PPO Value Adaptive Training ended (interrupted)")
    print(f"Training end time: {end_time}")
    print(f"Total training time: {training_time//3600:.0f}h {(training_time%3600)//60:.0f}m {training_time%60:.0f}s")
    
    # Save models with timestep-specific naming (only new naming convention)
    server_name = detect_server_from_path(data_dir)
    actor_path, critic_path = save_models_with_timesteps(model, total_timesteps, server_name)
    
    # Generate training plots
    model.plot_training_curves()
    
    # Save training data and generate enhanced plots with server-specific naming
    results_dir = Path(model.save_dir)
    
    # 生成唯一的JSON文件名
    json_filename = generate_unique_filename(
        str(results_dir),
        'ppo_value_adaptive_training_data_{}_{}.json',
        total_timesteps,
        server_name
    )
    json_path = str(results_dir / json_filename)
    
    # 保存数据并生成图表
    analyzer.save_data(json_path)
    
    # 图表文件名前缀也包含服务器信息
    plot_prefix = f'ppo_value_adaptive_training_{total_timesteps}_{server_name}'
    analyzer.generate_plots(str(results_dir), plot_prefix)
    
    # Performance summary
    if analyzer.training_data['total_rewards']:
        final_rewards = analyzer.training_data['total_rewards'][-10:]
        avg_final_reward = np.mean(final_rewards)
        max_reward = max(analyzer.training_data['total_rewards'])
        
        print(f"\nTraining Performance Summary:")
        print(f"  - Average final reward (last 10 episodes): {avg_final_reward:.4f}")
        print(f"  - Maximum reward achieved: {max_reward:.4f}")
        
        if analyzer.training_data['completion_rates']:
            final_completion = np.mean(analyzer.training_data['completion_rates'][-10:])
            print(f"  - Final completion rate: {final_completion:.2%}")
        
        if analyzer.training_data['value_uncertainties']:
            final_uncertainty = np.mean(analyzer.training_data['value_uncertainties'][-10:])
            print(f"  - Final value uncertainty: {final_uncertainty:.4f}")
        
        print(f"  - Final exploration factor: {model.current_exploration_factor:.4f}")
        print(f"  - Final adaptive LR factor: {model.adaptive_lr_factor:.4f}")
    
    # Print final summary of saved files
    print(f"\n📁 Training artifacts saved to: {model.save_dir}")
    print(f"📋 Generated files (Server: {server_name}):")
    print(f"  - {os.path.basename(actor_path)}")
    print(f"  - {os.path.basename(critic_path)}")
    print(f"  - {os.path.basename(json_path)}")
    print(f"  - {plot_prefix}_comprehensive.png")
    print(f"  - {plot_prefix}_summary_stats.png")
    print(f"  - ppo_value_adaptive_training_curves.png")
    
    # Cleanup
    if device.type == 'cuda':
        cleanup_gpu()
    
    return model, analyzer

def evaluate_value_adaptive_model(actor_path: str, critic_path: str, n_episodes: int = 50):
    """
    Evaluate trained PPO Value Adaptive model
    
    Args:
        actor_path: Path to saved actor model
        critic_path: Path to saved critic model
        n_episodes: Number of evaluation episodes
        
    Returns:
        evaluation_results: Dictionary with evaluation metrics
    """
    print(f"Evaluating PPO Value Adaptive model...")
    print(f"Actor model: {actor_path}")
    print(f"Critic model: {critic_path}")
    
    # Setup evaluation environment - 使用Server3数据集
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "database" / "Server3"
    
    topology_path = data_dir / "server3_topology.csv"
    servers_path = data_dir / "server3_information.csv"
    models_path = data_dir / "server3_RandomDeployment.csv"
    test_tasks_path = data_dir / "test.CSV"
    
    # Create evaluation environment
    env = DistributedModelOrchestrationEnv(
        topology_path=str(topology_path),
        servers_path=str(servers_path),
        models_path=str(models_path),
        tasks_path=str(test_tasks_path),
        test_tasks_path=str(test_tasks_path)
    )
    
    # Create model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistributedPPOValueAdaptive(env, device=device)
    
    # Load model weights
    if os.path.exists(actor_path) and os.path.exists(critic_path):
        model.load_models('_final')
    else:
        print(f"Model files not found: {actor_path}, {critic_path}")
        return None
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    completion_rates = []
    load_balance_scores = []
    value_uncertainties = []
    
    print(f"Running {n_episodes} evaluation episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset(use_test_set=True)
        episode_reward = 0
        episode_length = 0
        done = False
        
        episode_uncertainties = []
        
        while not done and episode_length < model.max_timesteps_per_episode:
            # Get action from model
            action = model.get_action(obs)
            
            # Get value uncertainty for analysis
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                _, uncertainty, _ = model.critic(obs_tensor)
                # Handle both single values and batch tensors
                if uncertainty.numel() == 1:
                    episode_uncertainties.append(uncertainty.item())
                else:
                    episode_uncertainties.append(uncertainty.mean().item())
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if done or truncated:
                break
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Extract environment metrics
        env_metrics = env.get_evaluation_metrics()
        completion_rates.append(env_metrics.get('task_completion_rate', 0))
        load_balance_scores.append(env_metrics.get('load_balance_score', 0))
        value_uncertainties.append(np.mean(episode_uncertainties))
        
        if (episode + 1) % 10 == 0:
            print(f"  Completed {episode + 1}/{n_episodes} episodes")
    
    # Calculate evaluation results
    evaluation_results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_completion_rate': np.mean(completion_rates),
        'mean_load_balance': np.mean(load_balance_scores),
        'mean_value_uncertainty': np.mean(value_uncertainties),
        'episode_rewards': episode_rewards,
        'completion_rates': completion_rates,
        'load_balance_scores': load_balance_scores
    }
    
    # Print evaluation summary
    print(f"\nPPO Value Adaptive Evaluation Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {evaluation_results['mean_reward']:.4f} ± {evaluation_results['std_reward']:.4f}")
    print(f"  Reward Range: [{evaluation_results['min_reward']:.4f}, {evaluation_results['max_reward']:.4f}]")
    print(f"  Mean Episode Length: {evaluation_results['mean_episode_length']:.2f}")
    print(f"  Mean Completion Rate: {evaluation_results['mean_completion_rate']:.2%}")
    print(f"  Mean Load Balance Score: {evaluation_results['mean_load_balance']:.4f}")
    print(f"  Mean Value Uncertainty: {evaluation_results['mean_value_uncertainty']:.4f}")
    
    return evaluation_results

def train_vamappo(args):
    """
    Train VAMAPPO Multi-Agent System
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (model, analyzer) for trained VAMAPPO system and analysis
    """
    print(f"🚀 Starting VAMAPPO Multi-Agent Training")
    print(f"📊 Configuration:")
    print(f"  - Total timesteps: {args.steps}")
    print(f"  - Number of agents: {args.n_agents}")
    print(f"  - Server allocation strategy: {args.server_allocation}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Batch size: {args.batch_size}")
    
    # Detect server configuration
    base_dir = Path(__file__).parent.parent.parent
    server_name = detect_server_from_path(str(base_dir))
    data_dir = base_dir / "database" / server_name
    
    print(f"📁 Using data from: {server_name}")
    
    # Configure GPU/CPU
    device, gpu_info = configure_gpu_for_training()
    
    # Create multi-agent environment
    env = DistributedModelOrchestrationEnv(
        topology_path=str(data_dir / f"{server_name.lower()}_topology.csv"),
        servers_path=str(data_dir / f"{server_name.lower()}_information.csv"),
        models_path=str(data_dir / f"{server_name.lower()}_RandomDeployment.csv"),
        tasks_path=str(data_dir / "train.csv"),
        test_tasks_path=str(data_dir / "test.CSV"),
        use_full_state_space=False,  # Use compact state space for multi-agent
        multi_agent_mode=True,
        n_agents=args.n_agents,
        server_allocation_strategy=args.server_allocation
    )
    
    print(f"🌐 Multi-agent environment created:")
    print(f"  - Total servers: {env.n_servers}")
    print(f"  - Agents: {env.n_agents}")
    print(f"  - Server allocation:")
    for agent_id, servers in env.server_allocation.items():
        print(f"    {agent_id}: {len(servers)} servers")
    
    # Create VAMAPPO model
    # Adjust batch size for shorter training runs
    adjusted_batch_size = min(args.batch_size, max(200, args.steps // 10))
    
    hyperparameters = {
        'device': device,
        'lr': args.lr,
        'timesteps_per_batch': adjusted_batch_size,
        'max_timesteps_per_episode': args.max_ep_len,
        'save_freq': args.save_freq,
        'n_updates_per_iteration': 4,
        'gamma': 0.99,
        'clip': 0.2,
        'ent_coef': 0.01,
        'value_reg_coef': 1.0,
        'max_grad_norm': 0.5
    }
    
    print(f"📊 Adjusted batch size from {args.batch_size} to {adjusted_batch_size} for {args.steps} total steps")
    
    model = VAMAPPO(env, **hyperparameters)
    
    # Create enhanced analyzer for multi-agent
    analyzer = VAMAPPOTrainingAnalyzer()
    
    # Monitor GPU memory if available
    if device.type == 'cuda':
        print("VAMAPPO Training Start - GPU Memory Status:")
        monitor_gpu_memory()
    
    # Start training
    start_time = time.time()
    model.learn(total_timesteps=args.steps, analyzer=analyzer)
    training_time = time.time() - start_time
    
    print(f"⏰ Training completed in {training_time:.2f} seconds")
    
    # Save models
    print("💾 Saving VAMAPPO models...")
    model.save_models(suffix='_final')
    
    # Generate analysis and visualizations
    print("📈 Generating analysis and visualizations...")
    plot_prefix = generate_unique_filename(
        model.save_dir, 
        "vamappo_training", 
        args.steps, 
        server_name
    )
    
    # Save training data
    json_path = os.path.join(model.save_dir, f"{plot_prefix}_data.json")
    analyzer.save_data(json_path)
    
    # Generate plots
    analyzer.generate_plots(model.save_dir, plot_prefix)
    
    print(f"\n📁 VAMAPPO Training artifacts saved to: {model.save_dir}")
    print(f"📋 Generated files (Server: {server_name}):")
    print(f"  - vamappo_Agent_*_model_final.pth (models for each agent)")
    print(f"  - {os.path.basename(json_path)}")
    print(f"  - {plot_prefix}_comprehensive.png")
    print(f"  - {plot_prefix}_summary_stats.png")
    
    # Cleanup
    if device.type == 'cuda':
        cleanup_gpu()
    
    return model, analyzer

def evaluate_vamappo_model(results_dir: str, model_suffix: str, n_episodes: int, n_agents: int):
    """
    Evaluate trained VAMAPPO model
    
    Args:
        results_dir: Directory containing saved models
        model_suffix: Model file suffix
        n_episodes: Number of evaluation episodes
        n_agents: Number of agents
        
    Returns:
        evaluation_results: Dictionary with evaluation metrics
    """
    print(f"🔍 Evaluating VAMAPPO model...")
    print(f"📁 Models directory: {results_dir}")
    print(f"🤖 Number of agents: {n_agents}")
    
    # Setup evaluation environment
    base_dir = Path(__file__).parent.parent.parent
    server_name = detect_server_from_path(str(base_dir))
    data_dir = base_dir / "database" / server_name
    
    # Create multi-agent environment
    env = DistributedModelOrchestrationEnv(
        topology_path=str(data_dir / f"{server_name.lower()}_topology.csv"),
        servers_path=str(data_dir / f"{server_name.lower()}_information.csv"),
        models_path=str(data_dir / f"{server_name.lower()}_RandomDeployment.csv"),
        tasks_path=str(data_dir / "test.CSV"),
        test_tasks_path=str(data_dir / "test.CSV"),
        use_full_state_space=False,
        multi_agent_mode=True,
        n_agents=n_agents,
        server_allocation_strategy='geographic'
    )
    
    # Create model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAMAPPO(env, device=device)
    
    # Load model weights
    model.load_models(suffix=model_suffix)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    completion_rates = []
    load_balance_scores = []
    agent_rewards = {f'Agent_{i}': [] for i in range(n_agents)}
    coordination_efficiency = []
    
    print(f"🧪 Running {n_episodes} evaluation episodes...")
    
    for episode in range(n_episodes):
        observations, infos = env.reset(use_test_set=True)
        episode_reward = 0
        episode_length = 0
        done = False
        
        agent_episode_rewards = {f'Agent_{i}': 0 for i in range(n_agents)}
        
        while not done and episode_length < model.max_timesteps_per_episode:
            # Get joint actions
            joint_output = model.multi_agent_system.get_joint_actions_and_values(
                observations, None
            )
            
            actions = joint_output['actions']
            
            # Execute actions
            next_observations, rewards, dones, _, next_infos = env.step(actions)
            
            # Accumulate rewards
            total_step_reward = sum(rewards.values())
            episode_reward += total_step_reward
            episode_length += 1
            
            for agent_id, reward in rewards.items():
                agent_episode_rewards[agent_id] += reward
            
            observations = next_observations
            done = all(dones.values())
            
            if done:
                break
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        for agent_id, reward in agent_episode_rewards.items():
            agent_rewards[agent_id].append(reward)
        
        # Calculate coordination efficiency
        agent_reward_values = list(agent_episode_rewards.values())
        if len(agent_reward_values) > 1:
            reward_std = np.std(agent_reward_values)
            reward_mean = np.mean(agent_reward_values)
            efficiency = 1.0 - (reward_std / (reward_mean + 1e-8))
            coordination_efficiency.append(max(0, efficiency))
        else:
            coordination_efficiency.append(1.0)
        
        # Extract environment metrics
        env_metrics = env.get_evaluation_metrics() if hasattr(env, 'get_evaluation_metrics') else {}
        completion_rates.append(env_metrics.get('task_completion_rate', 0))
        load_balance_scores.append(env_metrics.get('load_balance_score', 0))
        
        if (episode + 1) % 10 == 0:
            print(f"  ✅ Completed {episode + 1}/{n_episodes} episodes")
    
    # Calculate evaluation results
    evaluation_results = {
        'mean_total_reward': np.mean(episode_rewards),
        'std_total_reward': np.std(episode_rewards),
        'min_total_reward': np.min(episode_rewards),
        'max_total_reward': np.max(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_completion_rate': np.mean(completion_rates),
        'mean_load_balance': np.mean(load_balance_scores),
        'mean_coordination_efficiency': np.mean(coordination_efficiency),
        'agent_performance': {},
        'episode_rewards': episode_rewards,
        'completion_rates': completion_rates,
        'load_balance_scores': load_balance_scores,
        'coordination_efficiency': coordination_efficiency
    }
    
    # Calculate per-agent metrics
    for agent_id, rewards in agent_rewards.items():
        evaluation_results['agent_performance'][agent_id] = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
    
    # Print evaluation summary
    print(f"\n📊 VAMAPPO Evaluation Results ({n_episodes} episodes):")
    print(f"  🎯 Mean Total Reward: {evaluation_results['mean_total_reward']:.4f} ± {evaluation_results['std_total_reward']:.4f}")
    print(f"  📏 Reward Range: [{evaluation_results['min_total_reward']:.4f}, {evaluation_results['max_total_reward']:.4f}]")
    print(f"  ⏱️  Mean Episode Length: {evaluation_results['mean_episode_length']:.2f}")
    print(f"  ✔️  Mean Completion Rate: {evaluation_results['mean_completion_rate']:.2%}")
    print(f"  ⚖️  Mean Load Balance Score: {evaluation_results['mean_load_balance']:.4f}")
    print(f"  🤝 Mean Coordination Efficiency: {evaluation_results['mean_coordination_efficiency']:.4f}")
    
    print(f"\n🤖 Per-Agent Performance:")
    for agent_id, metrics in evaluation_results['agent_performance'].items():
        print(f"  {agent_id}: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
    
    return evaluation_results

class VAMAPPOTrainingAnalyzer:
    """Enhanced training analysis for VAMAPPO multi-agent system"""
    
    def __init__(self):
        self.training_data = {
            'total_system_rewards': [],
            'agent_rewards': {f'Agent_{i}': [] for i in range(9)},  # Support up to 9 agents
            'coordination_efficiency': [],
            'load_balance_scores': [],
            'communication_costs': [],
            'collaboration_bonuses': [],
            'episode_lengths': [],
            'timestamps': [],
            'agent_uncertainties': {f'Agent_{i}': [] for i in range(9)},
            'agent_adaptation_factors': {f'Agent_{i}': [] for i in range(9)}
        }
    
    def add_data(self, analysis_data: Dict):
        """Add analysis data from VAMAPPO training"""
        self.training_data['total_system_rewards'].append(
            sum(analysis_data.get('agent_rewards', {}).values())
        )
        
        for agent_id, reward in analysis_data.get('agent_rewards', {}).items():
            if agent_id in self.training_data['agent_rewards']:
                self.training_data['agent_rewards'][agent_id].append(reward)
        
        for agent_id, uncertainty in analysis_data.get('agent_uncertainties', {}).items():
            if agent_id in self.training_data['agent_uncertainties']:
                self.training_data['agent_uncertainties'][agent_id].append(uncertainty)
        
        self.training_data['coordination_efficiency'].append(
            analysis_data.get('coordination_efficiency', 0.0)
        )
        self.training_data['load_balance_scores'].append(
            analysis_data.get('load_balance', 0.0)
        )
        self.training_data['timestamps'].append(time.time())
    
    def save_data(self, filepath: str):
        """Save VAMAPPO training data"""
        # Convert data to JSON-serializable format
        json_data = {}
        for key, value in self.training_data.items():
            if isinstance(value, dict):
                json_data[key] = {k: v for k, v in value.items() if v}  # Only non-empty lists
            else:
                json_data[key] = value
        
        # Add metadata
        json_data['metadata'] = {
            'algorithm': 'VAMAPPO',
            'total_episodes': len(json_data.get('total_system_rewards', [])),
            'training_duration': time.time() - json_data['timestamps'][0] if json_data['timestamps'] else 0,
            'active_agents': len([k for k, v in json_data.get('agent_rewards', {}).items() if v])
        }
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"VAMAPPO training data saved to: {filepath}")
    
    def generate_plots(self, save_dir: str, filename_prefix: str = "vamappo_training"):
        """Generate VAMAPPO-specific plots"""
        plt.style.use('seaborn-v0_8')
        
        # Multi-agent performance plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('VAMAPPO Multi-Agent Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Total system rewards
        if self.training_data['total_system_rewards']:
            axes[0, 0].plot(self.training_data['total_system_rewards'], 'b-', linewidth=2)
            axes[0, 0].set_title('Total System Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True)
        
        # Plot 2: Per-agent rewards
        if any(self.training_data['agent_rewards'].values()):
            for agent_id, rewards in self.training_data['agent_rewards'].items():
                if rewards:
                    axes[0, 1].plot(rewards, label=agent_id, alpha=0.7)
            axes[0, 1].set_title('Agent Rewards')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot 3: Coordination efficiency
        if self.training_data['coordination_efficiency']:
            axes[1, 0].plot(self.training_data['coordination_efficiency'], 'g-', linewidth=2)
            axes[1, 0].set_title('Coordination Efficiency')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Efficiency')
            axes[1, 0].grid(True)
        
        # Plot 4: Load balance scores
        if self.training_data['load_balance_scores']:
            axes[1, 1].plot(self.training_data['load_balance_scores'], 'r-', linewidth=2)
            axes[1, 1].set_title('Load Balance Scores')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Load Balance')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'{filename_prefix}_comprehensive.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"VAMAPPO training plots saved to: {plot_path}")

def main():
    """Main function with enhanced argument parsing for both single-agent and multi-agent modes"""
    parser = argparse.ArgumentParser(description='PPO Value Adaptive and VAMAPPO Training and Evaluation')
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='train',
                       help='Run mode: train, eval, or both')
    
    # Algorithm selection
    parser.add_argument('--algorithm', choices=['single', 'multi'], default='single',
                       help='Algorithm type: single (PPO Value Adaptive) or multi (VAMAPPO)')
    
    # Basic training parameters
    parser.add_argument('--eval-episodes', type=int, default=50, 
                       help='Number of evaluation episodes')
    parser.add_argument('--steps', type=int, required=True, help='Total training timesteps')
    parser.add_argument('--model-suffix', default='_final', 
                       help='Model file suffix for loading/saving')
    parser.add_argument('--save-freq', type=int, default=5,
                       help='Model saving frequency')
    
    # Multi-agent specific parameters
    parser.add_argument('--n-agents', type=int, default=9,
                       help='Number of agents for VAMAPPO (only used in multi mode)')
    parser.add_argument('--server-allocation', choices=['geographic', 'balanced', 'round_robin'], 
                       default='geographic',
                       help='Server allocation strategy for VAMAPPO')
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size (timesteps per batch)')
    parser.add_argument('--max-ep-len', type=int, default=20,
                       help='Maximum episode length')
    
    args = parser.parse_args()
    
    if args.algorithm == 'single':
        print(f"PPO Value Adaptive - Steps: {args.steps}, Execution Mode: {args.mode}")
    else:
        print(f"VAMAPPO Multi-Agent - Steps: {args.steps}, Agents: {args.n_agents}, Mode: {args.mode}")
        print(f"Server Allocation Strategy: {args.server_allocation}")
    
    try:
        if args.mode in ['train', 'both']:
            print("Starting training phase...")
            if args.algorithm == 'single':
                model, analyzer = train_value_adaptive(args)
            else:
                model, analyzer = train_vamappo(args)
            
        if args.mode in ['eval', 'both']:
            print("\nStarting evaluation phase...")
            
            # Determine model paths
            if args.mode == 'both':
                # Use just trained model
                results_dir = model.save_dir
            else:
                # Use existing model
                results_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    'results'
                )
            
            if args.algorithm == 'single':
                actor_path = os.path.join(results_dir, f'ppo_value_adaptive_actor{args.model_suffix}.pth')
                critic_path = os.path.join(results_dir, f'ppo_value_adaptive_critic{args.model_suffix}.pth')
                
                evaluation_results = evaluate_value_adaptive_model(
                    actor_path, critic_path, args.eval_episodes
                )
            else:
                evaluation_results = evaluate_vamappo_model(
                    results_dir, args.model_suffix, args.eval_episodes, args.n_agents
                )
            
            if evaluation_results:
                # Save evaluation results
                if args.algorithm == 'single':
                    eval_results_path = os.path.join(results_dir, 'ppo_value_adaptive_evaluation_results.json')
                else:
                    eval_results_path = os.path.join(results_dir, 'vamappo_evaluation_results.json')
                
                with open(eval_results_path, 'w') as f:
                    json.dump(evaluation_results, f, indent=2)
                print(f"Evaluation results saved to: {eval_results_path}")
    
    except Exception as e:
        print(f"Error during execution: {e}")
        raise
    
    if args.algorithm == 'single':
        print("PPO Value Adaptive execution completed successfully!")
    else:
        print("VAMAPPO execution completed successfully!")

if __name__ == "__main__":
    main() 