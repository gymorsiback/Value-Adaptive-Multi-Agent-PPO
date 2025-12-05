#!/usr/bin/env python3
"""
Distributed PPO Training Results Analysis Tool

This script provides comprehensive analysis of training results including:
- Performance trend analysis
- Convergence assessment
- Hyperparameter optimization suggestions
- Detailed visualizations
- Training efficiency metrics
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DistributedTrainingAnalyzer:
    """Comprehensive training analysis and optimization suggestions"""
    
    def __init__(self, results_dir: Optional[str] = None, data_file: Optional[str] = None):
        """
        Initialize analyzer with results directory
        
        Args:
            results_dir: Path to results directory containing training data
            data_file: Specific data file to analyze
        """
        if results_dir is None:
            # Auto-detect results directory
            current_dir = Path(__file__).parent
            results_dir = str(current_dir.parent / "results")
        
        self.results_dir = Path(results_dir)
        self.data_file = data_file
        self.training_data = None
        self.analysis_results = {}
        
        # Load training data
        self.load_training_data()
    
    def load_training_data(self):
        """Load training data from JSON files"""
        # If specific data file is provided, use it
        if self.data_file:
            file_path = self.results_dir / self.data_file
            if file_path.exists():
                print(f"Loading training data from: {file_path}")
                with open(file_path, 'r') as f:
                    self.training_data = json.load(f)
                print(f"Loaded training data with {len(self.training_data.get('total_rewards', []))} data points")
                return
            else:
                print(f"❌ Specified file not found: {file_path}")
                return
        
        # First try to find step-specific files (newest format)
        import glob
        step_specific_files = glob.glob(str(self.results_dir / "final_training_data_*.json"))
        
        if step_specific_files:
            # Sort by modification time to get the most recent
            step_specific_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_file = step_specific_files[0]
            print(f"Loading training data from: {latest_file}")
            with open(latest_file, 'r') as f:
                self.training_data = json.load(f)
            print(f"Loaded training data with {len(self.training_data.get('total_rewards', []))} data points")
            return
        
        # Fallback to current naming convention
        data_files = [
            "ppo_value_adaptive_training_data.json",
            "final_training_data.json",
            "training_data.json"
        ]
        
        for data_file in data_files:
            file_path = self.results_dir / data_file
            if file_path.exists():
                print(f"Loading training data from: {file_path}")
                with open(file_path, 'r') as f:
                    self.training_data = json.load(f)
                print(f"Loaded training data with {len(self.training_data.get('total_rewards', []))} data points")
                return
        
        if self.training_data is None:
            print("Warning: No training data found. Please run training first.")
            return
    
    def analyze_convergence(self) -> Dict:
        """Analyze training convergence patterns"""
        if not self.training_data:
            return {}
        
        rewards = self.training_data.get('total_rewards', [])
        actor_losses = self.training_data.get('actor_losses', [])
        critic_losses = self.training_data.get('critic_losses', [])
        
        if not rewards:
            return {}
        
        # Calculate convergence metrics
        convergence_analysis = {
            'reward_trend': self._analyze_trend(rewards),
            'actor_loss_trend': self._analyze_trend(actor_losses),
            'critic_loss_trend': self._analyze_trend(critic_losses),
            'stability_metrics': self._calculate_stability(rewards),
            'convergence_point': self._find_convergence_point(rewards),
            'learning_efficiency': self._calculate_learning_efficiency(rewards)
        }
        
        return convergence_analysis
    
    def _analyze_trend(self, data: List[float]) -> Dict:
        """Analyze trend in data series"""
        if len(data) < 10:
            return {'trend': 'insufficient_data'}
        
        # Calculate moving averages
        window_size = min(10, len(data) // 4)
        moving_avg = pd.Series(data).rolling(window=window_size).mean()
        moving_avg = moving_avg[~np.isnan(moving_avg)]  # 使用numpy的isnan替代dropna
        
        # Calculate trend slope
        x = np.arange(len(moving_avg))
        slope = np.polyfit(x, moving_avg, 1)[0]
        
        # Determine trend direction
        if slope > 0.01:
            trend = 'improving'
        elif slope < -0.01:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'initial_value': data[0],
            'final_value': data[-1],
            'best_value': max(data) if 'reward' in str(data) else min(data),
            'worst_value': min(data) if 'reward' in str(data) else max(data),
            'improvement_ratio': (data[-1] - data[0]) / abs(data[0]) if data[0] != 0 else 0
        }
    
    def _calculate_stability(self, rewards: List[float]) -> Dict:
        """Calculate training stability metrics"""
        if len(rewards) < 10:
            return {}
        
        # Calculate variance in different phases
        early_phase = rewards[:len(rewards)//3]
        middle_phase = rewards[len(rewards)//3:2*len(rewards)//3]
        late_phase = rewards[2*len(rewards)//3:]
        
        return {
            'overall_variance': np.var(rewards),
            'early_variance': np.var(early_phase),
            'middle_variance': np.var(middle_phase),
            'late_variance': np.var(late_phase),
            'coefficient_of_variation': np.std(rewards) / np.mean(rewards) if np.mean(rewards) != 0 else float('inf')
        }
    
    def _find_convergence_point(self, rewards: List[float]) -> Dict:
        """Find approximate convergence point"""
        if len(rewards) < 20:
            return {'converged': False}
        
        # Use moving average to smooth data
        window_size = max(5, len(rewards) // 20)
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
        moving_avg = moving_avg[~np.isnan(moving_avg)]  # 使用numpy的isnan替代dropna
        
        # Look for convergence (low variance in recent data)
        recent_data = moving_avg[-window_size*2:]
        variance_threshold = np.var(moving_avg) * 0.1  # 10% of overall variance
        
        converged = np.var(recent_data) < variance_threshold
        convergence_iteration = len(moving_avg) - window_size*2 if converged else None
        
        return {
            'converged': converged,
            'convergence_iteration': convergence_iteration,
            'convergence_value': np.mean(recent_data) if converged else None,
            'variance_at_convergence': np.var(recent_data) if converged else None
        }
    
    def _calculate_learning_efficiency(self, rewards: List[float]) -> Dict:
        """Calculate learning efficiency metrics"""
        if len(rewards) < 10:
            return {}
        
        # Calculate improvement rate
        initial_avg = np.mean(rewards[:10])
        final_avg = np.mean(rewards[-10:])
        total_improvement = final_avg - initial_avg
        
        # Find steepest learning phase
        window_size = max(5, len(rewards) // 10)
        improvements = []
        for i in range(window_size, len(rewards)):
            current_avg = np.mean(rewards[i-window_size:i])
            previous_avg = np.mean(rewards[i-2*window_size:i-window_size]) if i >= 2*window_size else initial_avg
            improvements.append(current_avg - previous_avg)
        
        max_improvement_rate = max(improvements) if improvements else 0
        max_improvement_iteration = improvements.index(max_improvement_rate) + window_size if improvements else 0
        
        return {
            'total_improvement': total_improvement,
            'improvement_rate': total_improvement / len(rewards),
            'max_improvement_rate': max_improvement_rate,
            'max_improvement_iteration': max_improvement_iteration,
            'learning_efficiency_score': total_improvement / len(rewards) * 100  # Improvement per iteration
        }
    
    def generate_optimization_suggestions(self) -> Dict:
        """Generate hyperparameter optimization suggestions based on analysis"""
        if not self.training_data:
            return {}
        
        convergence = self.analyze_convergence()
        suggestions = {
            'hyperparameter_adjustments': [],
            'training_modifications': [],
            'architecture_changes': [],
            'priority_level': 'medium'
        }
        
        # Analyze reward trends
        reward_trend = convergence.get('reward_trend', {})
        actor_loss_trend = convergence.get('actor_loss_trend', {})
        critic_loss_trend = convergence.get('critic_loss_trend', {})
        
        # Learning rate suggestions
        if reward_trend.get('trend') == 'stable' and reward_trend.get('improvement_ratio', 0) < 0.1:
            suggestions['hyperparameter_adjustments'].append({
                'parameter': 'learning_rate',
                'current_estimated': 0.0001,
                'suggested': 0.0003,
                'reason': 'Low improvement rate suggests learning rate might be too conservative'
            })
            suggestions['priority_level'] = 'high'
        
        # Batch size suggestions
        if convergence.get('stability_metrics', {}).get('coefficient_of_variation', 0) > 0.5:
            suggestions['hyperparameter_adjustments'].append({
                'parameter': 'timesteps_per_batch',
                'current_estimated': 512,
                'suggested': 1024,
                'reason': 'High variance suggests larger batch size for more stable gradients'
            })
        
        # Episode length suggestions
        avg_reward = np.mean(self.training_data.get('total_rewards', [0]))
        if avg_reward < 20:  # Low average reward
            suggestions['training_modifications'].append({
                'modification': 'increase_episode_length',
                'current_estimated': 10,
                'suggested': 15,
                'reason': 'Low average rewards suggest episodes might be too short for task completion'
            })
        
        # Critic loss analysis
        if critic_loss_trend.get('trend') == 'stable' and critic_loss_trend.get('final_value', 0) > 400:
            suggestions['hyperparameter_adjustments'].append({
                'parameter': 'value_reg_coef',
                'current_estimated': 1.0,
                'suggested': 0.5,
                'reason': 'High stable critic loss suggests value function regularization might be too strong'
            })
        
        # Actor loss analysis
        if actor_loss_trend.get('trend') == 'stable' and abs(actor_loss_trend.get('final_value', 0)) < 0.01:
            suggestions['hyperparameter_adjustments'].append({
                'parameter': 'entropy_coefficient',
                'current_estimated': 0.01,
                'suggested': 0.02,
                'reason': 'Very low actor loss suggests need for more exploration'
            })
        
        return suggestions
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive training analysis visualizations"""
        if not self.training_data:
            print("No training data available for visualization")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Training Progress Overview (2x2 grid)
        gs1 = fig.add_gridspec(2, 2, left=0.05, right=0.48, top=0.95, bottom=0.52, hspace=0.3, wspace=0.3)
        
        # Rewards over time
        ax1 = fig.add_subplot(gs1[0, 0])
        rewards = self.training_data.get('total_rewards', [])
        if rewards:
            ax1.plot(rewards, 'b-', alpha=0.6, linewidth=1, label='Raw Rewards')
            # Add moving average
            window_size = max(5, len(rewards) // 20)
            moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
            ax1.plot(moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
            ax1.set_title('Training Rewards Progress', fontweight='bold', fontsize=12)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Loss trends
        ax2 = fig.add_subplot(gs1[0, 1])
        actor_losses = self.training_data.get('actor_losses', [])
        critic_losses = self.training_data.get('critic_losses', [])
        if actor_losses and critic_losses:
            ax2.plot(actor_losses, 'g-', linewidth=1.5, label='Actor Loss', alpha=0.8)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(critic_losses, 'r-', linewidth=1.5, label='Critic Loss', alpha=0.8)
            ax2.set_title('Training Losses', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Update Step')
            ax2.set_ylabel('Actor Loss', color='g')
            ax2_twin.set_ylabel('Critic Loss', color='r')
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
        
        # Performance metrics
        ax3 = fig.add_subplot(gs1[1, 0])
        completion_rates = self.training_data.get('completion_rates', [])
        load_balance_scores = self.training_data.get('load_balance_scores', [])
        if completion_rates or load_balance_scores:
            if completion_rates:
                ax3.plot(completion_rates, 'purple', linewidth=2, label='Completion Rate', marker='o', markersize=3)
            if load_balance_scores:
                ax3_twin = ax3.twinx()
                ax3_twin.plot(load_balance_scores, 'orange', linewidth=2, label='Load Balance', marker='s', markersize=3)
                ax3_twin.set_ylabel('Load Balance Score', color='orange')
                ax3_twin.legend(loc='upper right')
            ax3.set_title('System Performance Metrics', fontweight='bold', fontsize=12)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Completion Rate', color='purple')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # Reward distribution
        ax4 = fig.add_subplot(gs1[1, 1])
        if rewards:
            ax4.hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
            ax4.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
            ax4.set_title('Reward Distribution', fontweight='bold', fontsize=12)
            ax4.set_xlabel('Reward Value')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 2. Convergence Analysis (right side)
        gs2 = fig.add_gridspec(2, 1, left=0.52, right=0.95, top=0.95, bottom=0.52, hspace=0.3)
        
        # Convergence analysis
        ax5 = fig.add_subplot(gs2[0, 0])
        if rewards:
            # Plot convergence analysis
            convergence = self.analyze_convergence()
            reward_trend = convergence.get('reward_trend', {})
            
            ax5.plot(rewards, 'b-', alpha=0.5, linewidth=1, label='Raw Data')
            
            # Add trend line
            x = np.arange(len(rewards))
            if reward_trend.get('slope'):
                trend_line = reward_trend['initial_value'] + reward_trend['slope'] * x
                ax5.plot(x, trend_line, 'r--', linewidth=2, label=f'Trend (slope: {reward_trend["slope"]:.4f})')
            
            # Mark convergence point if found
            conv_point = convergence.get('convergence_point', {})
            if conv_point.get('converged'):
                conv_iter = conv_point['convergence_iteration']
                ax5.axvline(conv_iter, color='green', linestyle=':', linewidth=2, 
                           label=f'Convergence Point: {conv_iter}')
            
            ax5.set_title('Convergence Analysis', fontweight='bold', fontsize=12)
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('Reward')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Learning efficiency
        ax6 = fig.add_subplot(gs2[1, 0])
        if rewards and len(rewards) > 20:
            # Calculate learning rate over time
            window_size = max(5, len(rewards) // 20)
            learning_rates = []
            iterations = []
            
            for i in range(window_size, len(rewards), window_size//2):
                current_avg = np.mean(rewards[i-window_size:i])
                previous_avg = np.mean(rewards[max(0, i-2*window_size):i-window_size])
                learning_rate = (current_avg - previous_avg) / window_size
                learning_rates.append(learning_rate)
                iterations.append(i)
            
            ax6.plot(iterations, learning_rates, 'purple', linewidth=2, marker='o', markersize=4)
            ax6.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax6.set_title('Learning Efficiency Over Time', fontweight='bold', fontsize=12)
            ax6.set_xlabel('Iteration')
            ax6.set_ylabel('Learning Rate (Reward/Iteration)')
            ax6.grid(True, alpha=0.3)
        
        # 3. Analysis Summary (bottom)
        gs3 = fig.add_gridspec(1, 1, left=0.05, right=0.95, top=0.45, bottom=0.05)
        ax7 = fig.add_subplot(gs3[0, 0])
        ax7.axis('off')
        
        # Generate analysis summary text
        summary_text = self._generate_analysis_summary()
        ax7.text(0.02, 0.98, summary_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Distributed PPO Training Analysis Report', fontsize=16, fontweight='bold', y=0.98)
        
        # Save the plot
        output_path = self.results_dir / 'comprehensive_training_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Comprehensive analysis saved to: {output_path}")
        
        plt.show()
    
    def _generate_analysis_summary(self) -> str:
        """Generate text summary of analysis results"""
        if not self.training_data:
            return "No training data available for analysis."
        
        convergence = self.analyze_convergence()
        suggestions = self.generate_optimization_suggestions()
        
        rewards = self.training_data.get('total_rewards', [])
        actor_losses = self.training_data.get('actor_losses', [])
        critic_losses = self.training_data.get('critic_losses', [])
        
        summary = "TRAINING ANALYSIS SUMMARY\n"
        summary += "=" * 50 + "\n\n"
        
        # Basic statistics
        if rewards:
            summary += f"Training Statistics:\n"
            summary += f"  • Total Iterations: {len(rewards)}\n"
            summary += f"  • Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n"
            summary += f"  • Best Reward: {max(rewards):.2f}\n"
            summary += f"  • Final Reward: {rewards[-1]:.2f}\n"
            summary += f"  • Improvement: {((rewards[-1] - rewards[0]) / abs(rewards[0]) * 100):.1f}%\n\n"
        
        # Convergence analysis
        reward_trend = convergence.get('reward_trend', {})
        conv_point = convergence.get('convergence_point', {})
        
        summary += f"Convergence Analysis:\n"
        summary += f"  • Trend: {reward_trend.get('trend', 'unknown').upper()}\n"
        summary += f"  • Converged: {'YES' if conv_point.get('converged') else 'NO'}\n"
        if conv_point.get('converged'):
            summary += f"  • Convergence Point: Iteration {conv_point.get('convergence_iteration', 'N/A')}\n"
        
        learning_eff = convergence.get('learning_efficiency', {})
        if learning_eff:
            summary += f"  • Learning Efficiency: {learning_eff.get('learning_efficiency_score', 0):.3f}\n\n"
        
        # Optimization suggestions
        summary += f"Optimization Suggestions ({suggestions.get('priority_level', 'medium').upper()} priority):\n"
        
        for suggestion in suggestions.get('hyperparameter_adjustments', []):
            summary += f"  • {suggestion['parameter']}: {suggestion['current_estimated']} → {suggestion['suggested']}\n"
            summary += f"    Reason: {suggestion['reason']}\n"
        
        for suggestion in suggestions.get('training_modifications', []):
            summary += f"  • {suggestion['modification']}: {suggestion['current_estimated']} → {suggestion['suggested']}\n"
            summary += f"    Reason: {suggestion['reason']}\n"
        
        if not suggestions.get('hyperparameter_adjustments') and not suggestions.get('training_modifications'):
            summary += "  • Current configuration appears well-tuned\n"
        
        return summary
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def export_analysis_report(self):
        """Export detailed analysis report to JSON"""
        if not self.training_data:
            print("No training data available for export")
            return
        
        convergence = self.analyze_convergence()
        suggestions = self.generate_optimization_suggestions()
        
        report = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'training_summary': {
                'total_iterations': len(self.training_data.get('total_rewards', [])),
                'average_reward': float(np.mean(self.training_data.get('total_rewards', [0]))),
                'final_reward': float(self.training_data.get('total_rewards', [0])[-1]) if self.training_data.get('total_rewards') else 0,
                'best_reward': float(max(self.training_data.get('total_rewards', [0]))),
                'reward_std': float(np.std(self.training_data.get('total_rewards', [0])))
            },
            'convergence_analysis': convergence,
            'optimization_suggestions': suggestions,
            'training_data_summary': {
                'data_points': len(self.training_data.get('total_rewards', [])),
                'has_completion_rates': bool(self.training_data.get('completion_rates')),
                'has_load_balance': bool(self.training_data.get('load_balance_scores')),
                'has_loss_data': bool(self.training_data.get('actor_losses'))
            }
        }
        
        # Convert to JSON serializable format
        report = self._convert_to_json_serializable(report)
        
        output_path = self.results_dir / 'training_analysis_report.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed analysis report exported to: {output_path}")
        return report

def main():
    """Main function to run comprehensive training analysis"""
    print("Distributed PPO Training Analysis Tool")
    print("=" * 50)
    
    # Check for command line arguments
    data_file = None
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        print(f"Using specified data file: {data_file}")
    
    # Initialize analyzer
    analyzer = DistributedTrainingAnalyzer(data_file=data_file)
    
    if analyzer.training_data is None:
        print("No training data found. Please run training first.")
        if data_file:
            print(f"Could not find specified file: {data_file}")
        print("\n💡 使用提示:")
        print("   python distributed_analyze.py                           # 分析最新训练数据")
        print("   python distributed_analyze.py <training_data_file>      # 分析指定训练数据文件")
        return
    
    # Run analysis
    print("\n1. Analyzing convergence patterns...")
    convergence = analyzer.analyze_convergence()
    
    print("2. Generating optimization suggestions...")
    suggestions = analyzer.generate_optimization_suggestions()
    
    print("3. Creating comprehensive visualizations...")
    analyzer.create_comprehensive_visualizations()
    
    print("4. Exporting detailed analysis report...")
    report = analyzer.export_analysis_report()
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    
    rewards = analyzer.training_data.get('total_rewards', [])
    if rewards:
        print(f"Training Performance:")
        print(f"  • Average Reward: {np.mean(rewards):.2f}")
        print(f"  • Best Reward: {max(rewards):.2f}")
        print(f"  • Improvement: {((rewards[-1] - rewards[0]) / abs(rewards[0]) * 100):.1f}%")
        
        conv_point = convergence.get('convergence_point', {})
        print(f"  • Converged: {'Yes' if conv_point.get('converged') else 'No'}")
        
        print(f"\nOptimization Priority: {suggestions.get('priority_level', 'medium').upper()}")
        print(f"Suggestions: {len(suggestions.get('hyperparameter_adjustments', []))} hyperparameter adjustments")
    
    print(f"\nFiles generated:")
    print(f"  • comprehensive_training_analysis.png")
    print(f"  • training_analysis_report.json")

if __name__ == "__main__":
    main() 