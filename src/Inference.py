#!/usr/bin/env python3
"""
PPO Value Adaptive Model Inference and Analysis System

Enhanced with communication overhead analysis for distributed systems
Supports Server3 model inference and exports data to CSV files

⚠️ ===== HOW TO USE WITH YOUR OWN MODEL =====

1. PREPARE YOUR MODEL FILES:
   - Place your model files in: vamappo/results/
   - Required files:
     * ppo_value_adaptive_training_actor_model_{steps}_{server}.pth
     * ppo_value_adaptive_training_critic_model_{steps}_{server}.pth
     * ppo_value_adaptive_training_data_{steps}_{server}.json (optional)
   
   Example: 
   - ppo_value_adaptive_training_actor_model_50000_Server1.pth
   - ppo_value_adaptive_training_critic_model_50000_Server1.pth

2. PREPARE YOUR DATASET:
   - Place your dataset in: database/{server_name}/
   - Required files:
     * {server}_topology.csv
     * {server}_information.csv
     * {server}_RandomDeployment.csv
     * train.CSV (for environment initialization)
     * test.CSV (for inference testing)

3. MODIFY THE CONFIGURATION:
   - Go to the main() function at the bottom of this file
   - Change these two parameters:
     * model_path_prefix: Your training steps (e.g., "50000")
     * server_id: Your server name (e.g., "Server1")

4. RUN THE INFERENCE:
   python vamappo/photo/InferencePhoto.py

5. OUTPUT FILES:
   - 13 CSV data files: vamappo/photo/results/inference_{server}_{steps}_*.csv
   - Inference report: vamappo/results/ppo_value_adaptive_inference_report_{steps}_{server}.json

6. GENERATE VISUALIZATIONS:
   - After running inference, use DrawPhoto.py to create charts:
     python vamappo/photo/results/DrawPhoto.py --server Server1 --steps 1000000
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

from distributed_env import DistributedModelOrchestrationEnv
from distributed_ppo import DistributedPPOValueAdaptive
from gpu_config import configure_gpu_for_training, monitor_gpu_memory, cleanup_gpu


class PPOValueAdaptiveInferenceAnalyzer:
    """PPO Value Adaptive inference analyzer with communication overhead tracking"""
    
    def __init__(self, model_path_prefix: str = "1000000", server_id: str = "Server3"):
        """
        Initialize inference analyzer
        
        Args:
            model_path_prefix: Model file prefix (training steps)
            server_id: Server identifier (Server1, Server2, Server3, or Server4)
            
        ⚠️ TO CHANGE MODEL AND DATASET:
        1. Change model_path_prefix to match your training steps (e.g., "20000", "50000", "1000000")
        2. Change server_id to match your server ("Server1", "Server2", "Server3", or "Server4")
        """
        self.model_path_prefix = model_path_prefix
        self.server_id = server_id
        self.results_dir = Path(__file__).parent.parent / "results"
        
        # Inference result storage
        self.inference_results = {
            'episode_rewards': [],
            'completion_rates': [],
            'task_success_rates': [],
            'response_times': [],
            'server_utilizations': [],
            'model_selections': [],
            'task_types': [],
            'user_locations': [],
            'data_sizes': [],
            'quality_preferences': [],
            'speed_preferences': [],
            'detailed_metrics': [],
            # Communication overhead metrics
            'communication_overhead': [],
            'transmission_times': [],
            'network_hops': [],
            'bandwidth_usage': [],
            'latency_metrics': []
        }
        
        # Performance statistics
        self.performance_stats = {}
        
    def get_model_paths(self):
        """
        集中管理所有模型文件路径
        
        这个函数的设计理念：
        1. 将所有路径配置集中在一个地方，方便修改
        2. 支持多种命名规则，自动检测可用的文件
        3. 返回完整的路径字典，包含所有需要的文件
        
        Returns:
            dict: 包含所有模型文件路径的字典
                - 'json_path': JSON数据文件路径（可选）
                - 'actor_path': Actor模型文件路径
                - 'critic_path': Critic模型文件路径
                - 'paths_found': 是否找到所有必需文件
                - 'naming_convention': 使用的命名规则
        """
        # ========== 在这里修改文件路径 ==========
        # 方案1：带服务器后缀的命名规则（新版本）
        json_path_with_server = Path("vamappo/results/ppo_value_adaptive_training_data_1000000_Server3.json")
        actor_path_with_server = Path("vamappo/results/ppo_value_adaptive_training_actor_model_1000000_Server3.pth")
        critic_path_with_server = Path("vamappo/results/ppo_value_adaptive_training_critic_model_1000000_Server3.pth")
        
        # 方案2：不带服务器后缀的命名规则（旧版本）
        json_path_no_server = self.results_dir / f"ppo_value_adaptive_training_data_{self.model_path_prefix}.json"
        actor_path_no_server = self.results_dir / f"ppo_value_adaptive_training_actor_model_{self.model_path_prefix}.pth"
        critic_path_no_server = self.results_dir / f"ppo_value_adaptive_training_critic_model_{self.model_path_prefix}.pth"
        
        # 方案3：自定义路径（如果你的文件在其他位置，可以在这里直接指定）
        # 例如：
        # custom_json_path = Path("path/to/your/custom.json")
        # custom_actor_path = Path("path/to/your/actor.pth")
        # custom_critic_path = Path("path/to/your/critic.pth")
        # ========================================
        
        # 自动检测可用的文件
        paths_result = {
            'json_path': None,
            'actor_path': None,
            'critic_path': None,
            'paths_found': False,
            'naming_convention': None
        }
        
        # 优先尝试带服务器后缀的文件
        if actor_path_with_server.exists() and critic_path_with_server.exists():
            paths_result['actor_path'] = actor_path_with_server
            paths_result['critic_path'] = critic_path_with_server
            paths_result['naming_convention'] = 'with_server_suffix'
            
            # JSON文件是可选的
            if json_path_with_server.exists():
                paths_result['json_path'] = json_path_with_server
                print(f"✅ 找到JSON数据文件: {json_path_with_server.name}")
            else:
                print(f"⚠️ JSON数据文件不存在（可选）: {json_path_with_server.name}")
            
            paths_result['paths_found'] = True
            print(f"✅ 使用带服务器后缀的命名规则")
            
        # 如果不存在，尝试不带服务器后缀的文件
        elif actor_path_no_server.exists() and critic_path_no_server.exists():
            paths_result['actor_path'] = actor_path_no_server
            paths_result['critic_path'] = critic_path_no_server
            paths_result['naming_convention'] = 'no_server_suffix'
            
            # JSON文件是可选的
            if json_path_no_server.exists():
                paths_result['json_path'] = json_path_no_server
                print(f"✅ 找到JSON数据文件: {json_path_no_server.name}")
            else:
                print(f"⚠️ JSON数据文件不存在（可选）: {json_path_no_server.name}")
            
            paths_result['paths_found'] = True
            print(f"✅ 使用不带服务器后缀的命名规则")
            
        # 如果你添加了自定义路径，可以在这里添加检测逻辑
        # elif custom_actor_path.exists() and custom_critic_path.exists():
        #     paths_result['actor_path'] = custom_actor_path
        #     paths_result['critic_path'] = custom_critic_path
        #     paths_result['naming_convention'] = 'custom'
        #     ...
        
        else:
            # 打印详细的错误信息，帮助调试
            print(f"❌ 未找到模型文件！尝试过的路径：")
            print(f"   方案1（带服务器后缀）：")
            print(f"     - Actor: {actor_path_with_server}")
            print(f"     - Critic: {critic_path_with_server}")
            print(f"   方案2（不带服务器后缀）：")
            print(f"     - Actor: {actor_path_no_server}")
            print(f"     - Critic: {critic_path_no_server}")
            
        return paths_result
        
    def load_trained_model(self, device):
        """Load trained PPO Value Adaptive model"""
        print(f"Loading PPO Value Adaptive model (training steps: {self.model_path_prefix}, server: {self.server_id})...")
        
        # 使用集中的路径管理函数获取所有文件路径
        paths = self.get_model_paths()
        
        # 检查是否找到必需的文件
        if not paths['paths_found']:
            raise FileNotFoundError(
                f"未找到模型文件！请检查文件是否存在。\n"
                f"期望的文件位置: {self.results_dir}\n"
                f"期望的文件前缀: {self.model_path_prefix}\n"
                f"期望的服务器ID: {self.server_id}"
            )
        
        # 获取文件路径
        actor_path = paths['actor_path']
        critic_path = paths['critic_path']
        json_path = paths['json_path']  # 可能为 None
        
        print(f"✅ 找到模型文件（{paths['naming_convention']}）:")
        print(f"  - Actor: {actor_path.name}")
        print(f"  - Critic: {critic_path.name}")
        if json_path:
            print(f"  - JSON: {json_path.name}")
        
        # 如果存在JSON文件，加载训练数据以获取更多信息
        if json_path and json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    training_data = json.load(f)
                
                # 显示训练数据的关键信息
                print(f"\n📊 训练数据摘要:")
                if 'metadata' in training_data:
                    metadata = training_data['metadata']
                    print(f"  - 训练时间: {metadata.get('training_start_time', 'Unknown')}")
                    print(f"  - 总训练步数: {metadata.get('total_timesteps', 'Unknown')}")
                    print(f"  - 服务器: {metadata.get('server_name', 'Unknown')}")
                
                if 'final_metrics' in training_data:
                    final_metrics = training_data['final_metrics']
                    print(f"  - 最终奖励: {final_metrics.get('final_reward', 'Unknown')}")
                    print(f"  - 最终成功率: {final_metrics.get('final_success_rate', 'Unknown')}")
                    
            except Exception as e:
                print(f"⚠️ 无法读取JSON文件: {e}")
        
        # Verify model training steps
        try:
            actor_checkpoint = torch.load(actor_path, map_location=device)
            critic_checkpoint = torch.load(critic_path, map_location=device)
            
            model_iteration = actor_checkpoint.get('iteration', 'Unknown')
            training_timesteps = actor_checkpoint.get('training_timesteps', 'Unknown')
            
            print(f"\n✅ 模型检查点信息:")
            print(f"  - 训练步数: {training_timesteps}")
            print(f"  - 迭代次数: {model_iteration}")
        except Exception as e:
            print(f"Warning: Could not verify model training steps: {e}")
        
        # ⚠️ DATASET PATHS - CHANGE HERE IF YOUR DATASET IS IN A DIFFERENT LOCATION
        # The dataset should be in: database/{server_id}/
        # Required files:
        # - {server}_topology.csv
        # - {server}_information.csv  
        # - {server}_RandomDeployment.csv
        # - train.CSV
        # - test.CSV
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir / "database" / self.server_id
        
        server_num = self.server_id.lower()  # Convert to lowercase server3
        topology_path = data_dir / f"{server_num}_topology.csv"
        servers_path = data_dir / f"{server_num}_information.csv" 
        models_path = data_dir / f"{server_num}_RandomDeployment.csv"
        train_tasks_path = data_dir / "train.CSV"
        test_tasks_path = data_dir / "test.CSV"
        
        # Verify data files
        for path in [topology_path, servers_path, models_path, train_tasks_path, test_tasks_path]:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
        
        # Create environment - using training-consistent configuration
        env = DistributedModelOrchestrationEnv(
            topology_path=str(topology_path),
            servers_path=str(servers_path),
            models_path=str(models_path),
            tasks_path=str(train_tasks_path),
            test_tasks_path=str(test_tasks_path)
        )
        
        # Use training-consistent hyperparameters to create model
        ppo_value_adaptive_hyperparameters = {
            # Standard PPO parameters - consistent with training
            'timesteps_per_batch': 1024,
            'max_timesteps_per_episode': 25,
            'n_updates_per_iteration': 6,
            'lr': 0.0002,
            'gamma': 0.99,
            'clip': 0.12,
            'ent_coef': 0.03,
            'value_clip': 0.15,
            'value_reg_coef': 0.3,
            'max_grad_norm': 0.8,
            'save_freq': 10,
            
            # Value Adaptive specific parameters - consistent with training
            'adaptive_clip_range': [0.1, 0.3],
            'uncertainty_regularization': 0.05,
            'exploration_schedule': 'exponential',
            'value_adaptation_frequency': 5,
            'adaptive_lr_bounds': [0.0001, 0.001],
            'uncertainty_threshold': 0.3,
            'exploration_decay_rate': 0.998,
            'value_history_size': 50,
        }
        
        # Create model - using complete hyperparameter configuration
        model = DistributedPPOValueAdaptive(env, device=device, **ppo_value_adaptive_hyperparameters)
        
        print(f"✅ Created PPO Value Adaptive model with training-consistent hyperparameters")
        
        # Load model weights
        # Load actor
        actor_checkpoint = torch.load(actor_path, map_location=device)
        model.actor.load_state_dict(actor_checkpoint['model_state_dict'])
        model.actor_optim.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        model.actor_scheduler.load_state_dict(actor_checkpoint['scheduler_state_dict'])
        model.current_exploration_factor = actor_checkpoint.get('exploration_factor', 1.0)
        
        # Load critic
        critic_checkpoint = torch.load(critic_path, map_location=device)
        model.critic.load_state_dict(critic_checkpoint['model_state_dict'])
        model.critic_optim.load_state_dict(critic_checkpoint['optimizer_state_dict'])
        model.critic_scheduler.load_state_dict(critic_checkpoint['scheduler_state_dict'])
        model.adaptive_lr_factor = critic_checkpoint.get('adaptive_lr_factor', 1.0)
        
        # Restore critic history
        if 'value_history' in critic_checkpoint:
            model.critic.value_history = critic_checkpoint['value_history']
        if 'uncertainty_history' in critic_checkpoint:
            model.critic.uncertainty_history = critic_checkpoint['uncertainty_history']
        
        print(f"✅ Model weights loaded successfully from:")
        print(f"  - {actor_path.name}")
        print(f"  - {critic_path.name}")
        
        # Output state space information
        if env.observation_space is not None and hasattr(env.observation_space, 'shape') and env.observation_space.shape is not None:
            state_dim = env.observation_space.shape[0]
        else:
            state_dim = "Unknown"
        print(f"Using state space: {state_dim} dimensions")
        
        return model, env
    
    def _calculate_communication_overhead(self, env, server_idx, user_location, data_size):
        """
        Calculate communication overhead for the action
        
        Args:
            env: Environment instance
            server_idx: Selected server index
            user_location: User location [lat, lon]
            data_size: Data size for transmission
            
        Returns:
            overhead_metrics: Dictionary with communication metrics
        """
        # Get server location
        server_location = [
            env.servers_df.iloc[server_idx]['Latitude'],
            env.servers_df.iloc[server_idx]['Longitude']
        ]
        
        # Calculate geographic distance
        distance = env.calculate_distance(user_location, server_location)
        
        # Estimate network hops based on distance (simplified model)
        network_hops = max(1, int(distance / 500))  # One hop per 500km
        
        # Calculate transmission time based on distance and data size
        # Assuming fiber optic speed: ~200,000 km/s
        # Bandwidth: variable based on network conditions
        base_latency = distance / 200000  # seconds
        hop_latency = network_hops * 0.001  # 1ms per hop
        
        # Bandwidth calculation (simplified model)
        # Assume bandwidth decreases with distance
        max_bandwidth = 1000  # Mbps
        distance_factor = max(0.1, 1 - distance / 10000)  # Decreases over 10000km
        effective_bandwidth = max_bandwidth * distance_factor
        
        # Transmission time for data
        data_size_mb = data_size  # Already in MB
        transmission_time = data_size_mb / effective_bandwidth + base_latency + hop_latency
        
        # Total communication overhead
        overhead = {
            'distance_km': distance,
            'network_hops': network_hops,
            'base_latency_ms': base_latency * 1000,
            'hop_latency_ms': hop_latency * 1000,
            'effective_bandwidth_mbps': effective_bandwidth,
            'transmission_time_s': transmission_time,
            'data_size_mb': data_size_mb,
            'total_overhead_s': transmission_time + base_latency + hop_latency
        }
        
        return overhead
    
    def run_inference(self, model, env, n_episodes: int = 200, deterministic: bool = True):
        """
        Run inference and collect performance data with communication overhead tracking
        
        Args:
            model: Trained PPO model
            env: Environment instance
            n_episodes: Number of test episodes
            deterministic: Whether to use deterministic inference mode
        """
        print(f"Starting inference on {n_episodes} test episodes (deterministic={deterministic})...")
        
        # Verify which dataset is being used
        print(f"\n🔍 Dataset verification:")
        print(f"  - Training dataset size: {len(env.tasks_df)} tasks")
        if hasattr(env, 'test_tasks_df') and env.test_tasks_df is not None:
            print(f"  - Test dataset size: {len(env.test_tasks_df)} tasks")
            print(f"  - Test dataset will be used (use_test_set=True)")
        else:
            print(f"  - Test dataset NOT found!")
            print(f"  - Training dataset will be used instead")
        
        # Set evaluation mode
        model.actor.eval()
        model.critic.eval()
        
        start_time = time.time()
        successful_episodes = 0
        total_completed_subtasks = 0
        total_possible_subtasks = 0
        
        for episode in range(n_episodes):
            # Reset environment using test dataset
            obs, info = env.reset(use_test_set=True)
            
            # Debug: print task info for first episode
            if episode == 0:
                task_info = info.get('task_info', {})
                print(f"\n📊 First episode task info:")
                print(f"  - Route sequence: {task_info.get('route_sequence', [])}")
                print(f"  - Data size: {task_info.get('data_size', 0)}")
                print(f"  - User location: {task_info.get('user_location', [])}")
            
            episode_reward = 0
            episode_steps = 0
            episode_start_time = time.time()
            server_visits = np.zeros(env.n_servers)
            model_usage = {}
            episode_successful = False
            
            # Initialize communication overhead tracking for episode
            episode_communication_overhead = 0
            episode_transmission_times = []
            episode_network_hops = []
            episode_bandwidth_usage = []
            episode_latency = []
            
            # Store task information
            task_info = info.get('task_info', {})
            route_sequence = task_info.get('route_sequence', [])
            total_possible_subtasks += len(route_sequence)
            
            self.inference_results['task_types'].append(route_sequence)
            self.inference_results['user_locations'].append(task_info.get('user_location', [0, 0]))
            self.inference_results['data_sizes'].append(task_info.get('data_size', 0))
            self.inference_results['quality_preferences'].append(task_info.get('quality_preference', 0))
            self.inference_results['speed_preferences'].append(task_info.get('speed_preference', 0))
            
            done = False
            step_details = []
            max_episode_steps = len(route_sequence) + 2  # Allow some extra steps
            
            while not done and episode_steps < max_episode_steps:
                # Get action from trained model
                with torch.no_grad():
                    if deterministic:
                        # Deterministic inference: select highest probability action
                        action = self._get_deterministic_action(model, obs, env)
                    else:
                        # Standard inference: use model's sampled action
                        action = model.get_action(obs)
                
                # Ensure action format is correct
                if isinstance(action, (int, float, np.integer, np.floating)):
                    # If scalar, convert to valid action
                    server_idx = int(action) % env.n_servers
                    model_idx = 0
                    action = np.array([server_idx, model_idx])
                elif isinstance(action, (list, tuple, np.ndarray)):
                    if len(action) < 2:
                        action = np.array([0, 0])
                    else:
                        action = np.array(action[:2])
                else:
                    action = np.array([0, 0])
                
                # Calculate communication overhead before executing action
                server_idx = int(action[0])
                user_location = env.current_task['user_location']
                data_size = env.current_task['data_size']
                
                comm_overhead = self._calculate_communication_overhead(
                    env, server_idx, user_location, data_size
                )
                
                # Track communication metrics
                episode_communication_overhead += comm_overhead['total_overhead_s']
                episode_transmission_times.append(comm_overhead['transmission_time_s'])
                episode_network_hops.append(comm_overhead['network_hops'])
                episode_bandwidth_usage.append(comm_overhead['effective_bandwidth_mbps'])
                episode_latency.append(comm_overhead['base_latency_ms'] + comm_overhead['hop_latency_ms'])
                
                # Execute action
                obs, reward, done, _, step_info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                # Track server and model usage
                model_idx = int(action[1])
                server_visits[server_idx] += 1
                
                if server_idx in env.server_to_models:
                    if model_idx < len(env.server_to_models[server_idx]):
                        model_info = env.server_to_models[server_idx][model_idx]
                        model_type = model_info['ModelType']
                        model_usage[model_type] = model_usage.get(model_type, 0) + 1
                
                # Store step details with communication overhead
                step_details.append({
                    'step': episode_steps,
                    'server': server_idx,
                    'model': model_idx,
                    'reward': reward,
                    'action': action,
                    'completion_rate': step_info.get('task_completion_rate', 0),
                    'completed_subtasks': step_info.get('completed_subtasks', 0),
                    'communication_overhead': comm_overhead
                })
                
                # Check successful completion
                if step_info.get('task_completed', False):
                    episode_successful = True
                    successful_episodes += 1
                    break
            
            episode_time = time.time() - episode_start_time
            
            # Calculate performance metrics
            final_completion_rate = step_info.get('task_completion_rate', 0) if step_info else 0
            completed_subtasks = step_info.get('completed_subtasks', 0) if step_info else 0
            total_completed_subtasks += completed_subtasks
            
            # Store results
            self.inference_results['episode_rewards'].append(episode_reward)
            self.inference_results['completion_rates'].append(final_completion_rate)
            self.inference_results['task_success_rates'].append(1.0 if episode_successful else 0.0)
            self.inference_results['response_times'].append(episode_time)
            self.inference_results['server_utilizations'].append(server_visits.tolist())
            self.inference_results['model_selections'].append(model_usage)
            
            # Store communication overhead metrics
            self.inference_results['communication_overhead'].append(episode_communication_overhead)
            self.inference_results['transmission_times'].append(episode_transmission_times)
            self.inference_results['network_hops'].append(episode_network_hops)
            self.inference_results['bandwidth_usage'].append(episode_bandwidth_usage)
            self.inference_results['latency_metrics'].append(episode_latency)
            
            self.inference_results['detailed_metrics'].append({
                'episode': episode,
                'steps': episode_steps,
                'total_reward': episode_reward,
                'avg_reward_per_step': episode_reward / max(episode_steps, 1),
                'completion_rate': final_completion_rate,
                'task_successful': episode_successful,
                'response_time': episode_time,
                'communication_overhead': episode_communication_overhead,
                'avg_transmission_time': np.mean(episode_transmission_times) if episode_transmission_times else 0,
                'avg_network_hops': np.mean(episode_network_hops) if episode_network_hops else 0,
                'avg_bandwidth': np.mean(episode_bandwidth_usage) if episode_bandwidth_usage else 0,
                'avg_latency': np.mean(episode_latency) if episode_latency else 0,
                'step_details': step_details
            })
            
            # Debug information - display every 50 episodes
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.inference_results['episode_rewards'][-50:])
                avg_completion = np.mean(self.inference_results['completion_rates'][-50:])
                avg_success = np.mean(self.inference_results['task_success_rates'][-50:])
                avg_comm_overhead = np.mean(self.inference_results['communication_overhead'][-50:])
                print(f"Episodes {episode+1-49}-{episode+1}: Avg Reward: {avg_reward:.2f}, "
                      f"Avg Completion: {avg_completion:.2%}, Success Rate: {avg_success:.2%}, "
                      f"Avg Comm Overhead: {avg_comm_overhead:.3f}s")
        
        total_time = time.time() - start_time
        overall_completion_rate = total_completed_subtasks / total_possible_subtasks if total_possible_subtasks > 0 else 0
        overall_success_rate = successful_episodes / n_episodes
        
        print(f"Inference completed in {total_time:.2f}s")
        print(f"Average time per episode: {total_time/n_episodes:.3f}s")
        print(f"Overall completion rate: {overall_completion_rate:.2%}")
        print(f"Overall success rate: {overall_success_rate:.2%}")
        print(f"Total completed subtasks: {total_completed_subtasks}/{total_possible_subtasks}")
        print(f"Successful episodes: {successful_episodes}/{n_episodes}")
        print(f"Average communication overhead: {np.mean(self.inference_results['communication_overhead']):.3f}s")
        
        self._calculate_performance_statistics()
    
    def _get_deterministic_action(self, model, obs, env):
        """
        Get deterministic action: select highest probability feasible action
        
        Args:
            model: PPO model
            obs: Current observation
            env: Environment instance
            
        Returns:
            action: Deterministically selected action
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        
        with torch.no_grad():
            # Get logits instead of sampled action
            server_logits, model_logits, enhancement_weights = model.actor.forward(obs_tensor)
            server_probs = torch.softmax(server_logits, dim=-1)
            model_probs = torch.softmax(model_logits, dim=-1)
            
            # Sort servers by probability
            server_indices = torch.argsort(server_probs, descending=True).squeeze()
            model_indices = torch.argsort(model_probs, descending=True).squeeze()
            
            # Try to find highest probability feasible action
            for server_idx in server_indices[:10]:  # Try top 10 servers
                server_idx = server_idx.item()
                
                for model_idx in model_indices[:5]:  # Try top 5 models
                    model_idx = model_idx.item()
                    
                    # Check action feasibility
                    if env.is_valid_action(server_idx, model_idx):
                        return np.array([server_idx, model_idx])
            
            # If no feasible action found, use greedy search
            return self._greedy_action_selection(env)
    
    def _greedy_action_selection(self, env):
        """
        Greedy action selection: select optimal action based on heuristic rules
        
        Args:
            env: Environment instance
            
        Returns:
            action: Greedily selected action
        """
        required_model_type = env.current_task['required_model_type']
        user_location = env.current_task['user_location']
        
        best_action = None
        best_score = -float('inf')
        
        # Iterate through all possible actions
        for server_idx in range(env.n_servers):
            if server_idx in env.server_to_models:
                for model_idx, model_info in enumerate(env.server_to_models[server_idx]):
                    # Check action validity
                    if env.is_valid_action(server_idx, model_idx):
                        # Calculate heuristic score
                        score = self._calculate_heuristic_score(
                            env, server_idx, model_info, user_location
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_action = np.array([server_idx, model_idx])
        
        # If still no feasible action found, return first feasible action
        if best_action is None:
            for server_idx in range(env.n_servers):
                if server_idx in env.server_to_models and len(env.server_to_models[server_idx]) > 0:
                    for model_idx in range(len(env.server_to_models[server_idx])):
                        if env.is_valid_action(server_idx, model_idx):
                            return np.array([server_idx, model_idx])
            
            # Last resort
            return np.array([0, 0])
        
        return best_action
    
    def _calculate_heuristic_score(self, env, server_idx, model_info, user_location):
        """
        Calculate heuristic score for action selection
        
        Args:
            env: Environment instance
            server_idx: Server index
            model_info: Model information
            user_location: User location
            
        Returns:
            score: Heuristic score
        """
        # Base score: model quality
        score = float(model_info['ArenaScore'])
        
        # Server load penalty (lower load is better)
        load_penalty = env.servers_computation_load[server_idx]
        score -= load_penalty * 2.0
        
        # Geographic distance reward (closer is better)
        server_location = [
            env.servers_df.iloc[server_idx]['Latitude'],
            env.servers_df.iloc[server_idx]['Longitude']
        ]
        distance = env.calculate_distance(user_location, server_location)
        distance_penalty = distance / 1000.0  # Normalize distance
        score -= distance_penalty
        
        # Load balance reward
        avg_load = np.mean(env.servers_computation_load)
        current_load = env.servers_computation_load[server_idx]
        if current_load < avg_load:
            score += 0.5  # Reward for servers with lower load
        
        return score
    
    def _calculate_performance_statistics(self):
        """Calculate comprehensive performance statistics including communication metrics"""
        rewards = self.inference_results['episode_rewards']
        completion_rates = self.inference_results['completion_rates']
        success_rates = self.inference_results['task_success_rates']
        response_times = self.inference_results['response_times']
        comm_overheads = self.inference_results['communication_overhead']
        
        self.performance_stats = {
            'reward_metrics': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'median': np.median(rewards),
                'q25': np.percentile(rewards, 25),
                'q75': np.percentile(rewards, 75)
            },
            'completion_metrics': {
                'mean_completion_rate': np.mean(completion_rates),
                'std_completion_rate': np.std(completion_rates),
                'success_rate': np.mean(success_rates),
                'total_successful_episodes': int(np.sum(success_rates)),
                'completion_consistency': 1.0 - np.std(completion_rates)
            },
            'efficiency_metrics': {
                'mean_response_time': np.mean(response_times),
                'std_response_time': np.std(response_times),
                'mean_communication_overhead': np.mean(comm_overheads),
                'std_communication_overhead': np.std(comm_overheads),
                'total_communication_time': np.sum(comm_overheads)
            },
            'model_usage': self._analyze_model_usage(),
            'server_usage': self._analyze_server_usage(),
            'communication_analysis': self._analyze_communication_patterns()
        }
    
    def _analyze_model_usage(self):
        """Analyze model type usage patterns"""
        all_model_usage = {}
        for episode_usage in self.inference_results['model_selections']:
            for model_type, count in episode_usage.items():
                all_model_usage[model_type] = all_model_usage.get(model_type, 0) + count
        
        total_usage = sum(all_model_usage.values())
        return {
            'usage_counts': all_model_usage,
            'usage_percentages': {k: v/total_usage*100 for k, v in all_model_usage.items()} if total_usage > 0 else {},
            'most_used_model': max(all_model_usage.keys(), key=lambda x: all_model_usage[x]) if all_model_usage else None,
            'model_diversity': len(all_model_usage)
        }
    
    def _analyze_server_usage(self):
        """Analyze server utilization patterns"""
        server_utilizations = np.array(self.inference_results['server_utilizations'])
        total_usage_per_server = np.sum(server_utilizations, axis=0)
        
        return {
            'total_usage_per_server': total_usage_per_server.tolist(),
            'mean_utilization_per_server': np.mean(server_utilizations, axis=0).tolist(),
            'utilization_balance_coefficient': np.std(total_usage_per_server) / np.mean(total_usage_per_server) if np.mean(total_usage_per_server) > 0 else 0,
            'most_used_server': int(np.argmax(total_usage_per_server)),
            'least_used_server': int(np.argmin(total_usage_per_server)),
            'active_servers': int(np.sum(total_usage_per_server > 0)),
            'server_diversity': np.sum(total_usage_per_server > 0) / len(total_usage_per_server)
        }
    
    def _analyze_communication_patterns(self):
        """Analyze communication overhead patterns"""
        # Flatten all transmission times
        all_transmission_times = []
        all_network_hops = []
        all_bandwidths = []
        all_latencies = []
        
        for episode_times, episode_hops, episode_bw, episode_lat in zip(
            self.inference_results['transmission_times'],
            self.inference_results['network_hops'],
            self.inference_results['bandwidth_usage'],
            self.inference_results['latency_metrics']
        ):
            all_transmission_times.extend(episode_times)
            all_network_hops.extend(episode_hops)
            all_bandwidths.extend(episode_bw)
            all_latencies.extend(episode_lat)
        
        return {
            'avg_transmission_time': np.mean(all_transmission_times) if all_transmission_times else 0,
            'avg_network_hops': np.mean(all_network_hops) if all_network_hops else 0,
            'avg_bandwidth_usage': np.mean(all_bandwidths) if all_bandwidths else 0,
            'avg_latency': np.mean(all_latencies) if all_latencies else 0,
            'max_transmission_time': np.max(all_transmission_times) if all_transmission_times else 0,
            'min_transmission_time': np.min(all_transmission_times) if all_transmission_times else 0,
            'transmission_time_std': np.std(all_transmission_times) if all_transmission_times else 0
        }
    
    def export_inference_data_to_csv(self):
        """Export inference data to CSV files for visualization"""
        print("Exporting inference data to CSV files...")
        
        # Create save directory
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        rewards = self.inference_results['episode_rewards']
        completion_rates = self.inference_results['completion_rates']
        success_rates = self.inference_results['task_success_rates']
        response_times = self.inference_results['response_times']
        comm_overheads = self.inference_results['communication_overhead']
        
        # 1. Reward Distribution
        df_reward_dist = pd.DataFrame({
            'episode_reward': rewards,
            'episode': range(len(rewards))
        })
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_reward_distribution.csv"
        df_reward_dist.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        # 2. Reward Distribution by Completion Rate
        df_reward_completion = pd.DataFrame({
            'reward': rewards,
            'completion_rate': completion_rates
        })
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_reward_distribution_by_completion_rate.csv"
        df_reward_completion.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        # 3. Task Success Rate
        success_count = int(np.sum(success_rates))
        fail_count = len(success_rates) - success_count
        df_success_rate = pd.DataFrame({
            'status': ['Successful', 'Failed'],
            'count': [success_count, fail_count],
            'percentage': [success_count/len(success_rates)*100, fail_count/len(success_rates)*100]
        })
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_task_success_rate.csv"
        df_success_rate.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        # 4. Task Completion Rate Distribution
        completion_counts = {}
        for rate in completion_rates:
            rate_key = f"{rate:.1f}"
            completion_counts[rate_key] = completion_counts.get(rate_key, 0) + 1
        
        df_completion_dist = pd.DataFrame({
            'completion_rate': list(completion_counts.keys()),
            'count': list(completion_counts.values())
        })
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_task_completion_rate_distribution.csv"
        df_completion_dist.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        # 5. Resource Utilization Over Episodes
        resource_util = [np.mean(np.array(utilization) > 0) for utilization in self.inference_results['server_utilizations']]
        df_resource_util = pd.DataFrame({
            'episode': range(len(resource_util)),
            'resource_utilization_rate': resource_util
        })
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_resource_utilization_over_episodes.csv"
        df_resource_util.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        # 6. Task Completion Rate Progress
        df_completion_progress = pd.DataFrame({
            'episode': range(len(completion_rates)),
            'completion_rate': completion_rates
        })
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_task_completion_rate_progress.csv"
        df_completion_progress.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        # 7. Active Server Usage Distribution
        server_usage = self.performance_stats['server_usage']['total_usage_per_server']
        active_servers = [(j, usage) for j, usage in enumerate(server_usage) if usage > 0]
        
        if active_servers:
            df_server_usage = pd.DataFrame({
                'server_index': [s[0] for s in active_servers],
                'usage_count': [s[1] for s in active_servers]
            })
        else:
            df_server_usage = pd.DataFrame({'server_index': [], 'usage_count': []})
        
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_active_server_usage_distribution.csv"
        df_server_usage.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        # 8. Model Type Usage
        model_usage = self.performance_stats['model_usage']['usage_percentages']
        if model_usage:
            df_model_usage = pd.DataFrame({
                'model_type': list(model_usage.keys()),
                'usage_percentage': list(model_usage.values())
            })
        else:
            df_model_usage = pd.DataFrame({'model_type': [], 'usage_percentage': []})
        
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_model_type_usage.csv"
        df_model_usage.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        # 9. Reward vs Response Time
        df_reward_response = pd.DataFrame({
            'response_time': response_times,
            'reward': rewards,
            'completion_rate': completion_rates
        })
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_reward_vs_response_time.csv"
        df_reward_response.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        # 10. Key Performance Metrics
        resource_util_mean = np.mean(resource_util)
        action_efficiency = np.mean([len(task_route) / max(metric['steps'], 1) 
                                   for task_route, metric in zip(self.inference_results['task_types'], 
                                                         self.inference_results['detailed_metrics'])])
        
        df_key_metrics = pd.DataFrame({
            'metric': ['Action Efficiency', 'Resource Utilization', 'Success Rate', 'Avg Completion'],
            'value': [
                action_efficiency,
                resource_util_mean, 
                np.mean(success_rates),
                np.mean(completion_rates)
            ]
        })
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_key_performance_metrics.csv"
        df_key_metrics.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        # 11. Communication Overhead Analysis
        comm_analysis = self.performance_stats['communication_analysis']
        
        # 11a. Communication overhead distribution and episodes
        df_comm_overhead = pd.DataFrame({
            'episode': range(len(comm_overheads)),
            'communication_overhead': comm_overheads
        })
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_communication_overhead_episodes.csv"
        df_comm_overhead.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        # 11b. Network performance metrics
        df_network_metrics = pd.DataFrame({
            'metric': ['Avg Transmission', 'Avg Latency', 'Avg Hops', 'Avg Bandwidth'],
            'value': [
                comm_analysis['avg_transmission_time'] * 1000,  # Convert to ms
                comm_analysis['avg_latency'],
                comm_analysis['avg_network_hops'],
                comm_analysis['avg_bandwidth_usage']
            ],
            'unit': ['ms', 'ms', 'hops', 'Mbps']
        })
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_network_performance_metrics.csv"
        df_network_metrics.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        # 11c. Communication overhead vs reward correlation
        df_comm_reward = pd.DataFrame({
            'communication_overhead': comm_overheads,
            'reward': rewards,
            'completion_rate': completion_rates
        })
        csv_path = results_dir / f"inference_{self.server_id.lower()}_{self.model_path_prefix}_communication_vs_reward.csv"
        df_comm_reward.to_csv(csv_path, index=False)
        print(f"  ✅ Exported: {csv_path.name}")
        
        print(f"✅ All CSV files exported to: {results_dir}")
        return results_dir
    



    
    def export_inference_report(self):
        """Export detailed inference report to JSON including communication metrics"""
        print("Exporting inference report...")
        
        report = {
            'inference_metadata': {
                'model_version': self.model_path_prefix,
                'server_id': self.server_id,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'total_episodes': len(self.inference_results['episode_rewards']),
                'test_dataset': 'test.CSV',
                'analysis_type': 'PPO_Value_Adaptive_Inference_With_Communication_Analysis'
            },
            'performance_statistics': self.performance_stats,
            'inference_results_summary': {
                'episode_count': len(self.inference_results['episode_rewards']),
                'average_reward': float(np.mean(self.inference_results['episode_rewards'])),
                'average_completion_rate': float(np.mean(self.inference_results['completion_rates'])),
                'task_success_rate': float(np.mean(self.inference_results['task_success_rates'])),
                'average_response_time': float(np.mean(self.inference_results['response_times'])),
                'average_communication_overhead': float(np.mean(self.inference_results['communication_overhead'])),
                'total_communication_time': float(np.sum(self.inference_results['communication_overhead']))
            },
            'detailed_results': self.inference_results['detailed_metrics']
        }
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        report = convert_numpy_types(report)
        
        output_path = self.results_dir / f'ppo_value_adaptive_inference_report_{self.model_path_prefix}_{self.server_id}.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Inference report exported to: {output_path}")
        return report


def main():
    """Main inference function for Server1 model"""
    print("PPO Value Adaptive Model Inference and Analysis System")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print("Initial GPU memory status:")
        monitor_gpu_memory()
    
    # ⚠️ ===== CONFIGURATION - CHANGE HERE FOR YOUR MODEL =====
    # 1. model_path_prefix: Your training steps (e.g., "20000", "50000", "1000000")
    # 2. server_id: Your server name ("Server1", "Server2", "Server3", or "Server4")
    # 
    # EXAMPLES:
    # - For Server1 model trained for 50000 steps:
    #   analyzer = PPOValueAdaptiveInferenceAnalyzer(model_path_prefix="50000", server_id="Server1")
    # - For Server2 model trained for 1000000 steps:
    #   analyzer = PPOValueAdaptiveInferenceAnalyzer(model_path_prefix="1000000", server_id="Server2")
    # - For Server4 model trained for 20000 steps:
    #   analyzer = PPOValueAdaptiveInferenceAnalyzer(model_path_prefix="20000", server_id="Server4")
    
    # Initialize analyzer for Server1 model with 1000000 training steps
    analyzer = PPOValueAdaptiveInferenceAnalyzer(model_path_prefix="1000000", server_id="Server3")
    
    # ⚠️ ===== INFERENCE PARAMETERS - OPTIONAL CHANGES =====
    # n_episodes: Number of test episodes to run (default: 200)
    # deterministic: Whether to use deterministic action selection (default: True)
    
    try:
        # Load trained model
        print("\n1. Loading trained PPO Value Adaptive model...")
        model, env = analyzer.load_trained_model(device)
        
        # Run inference
        print("\n2. Running inference on test dataset...")
        # ⚠️ You can change n_episodes here if needed (e.g., n_episodes=100 for faster testing)
        analyzer.run_inference(model, env, n_episodes=200, deterministic=True)
        
        # Export inference data to CSV
        print("\n3. Exporting inference data to CSV files...")
        analyzer.export_inference_data_to_csv()
        
        # Export detailed report
        print("\n4. Exporting detailed inference report...")
        report = analyzer.export_inference_report()
        
        # Print final summary
        print("\n" + "="*60)
        print("PPO VALUE ADAPTIVE INFERENCE ANALYSIS COMPLETE")
        print("="*60)
        
        stats = analyzer.performance_stats
        print(f"Performance Summary:")
        print(f"  • Average Reward: {stats['reward_metrics']['mean']:.3f}")
        print(f"  • Task Success Rate: {stats['completion_metrics']['success_rate']:.1%}")
        print(f"  • Average Completion Rate: {stats['completion_metrics']['mean_completion_rate']:.1%}")
        print(f"  • Average Response Time: {stats['efficiency_metrics']['mean_response_time']:.3f}s")
        print(f"  • Average Communication Overhead: {stats['efficiency_metrics']['mean_communication_overhead']:.3f}s")
        
        comm_analysis = stats['communication_analysis']
        print(f"\nCommunication Analysis:")
        print(f"  • Average Transmission Time: {comm_analysis['avg_transmission_time']:.4f}s")
        print(f"  • Average Network Hops: {comm_analysis['avg_network_hops']:.1f}")
        print(f"  • Average Bandwidth Usage: {comm_analysis['avg_bandwidth_usage']:.1f} Mbps")
        print(f"  • Average Latency: {comm_analysis['avg_latency']:.1f} ms")
        
        print(f"\nGenerated Files:")
        print(f"  • 13 CSV data files: inference_{analyzer.server_id.lower()}_{analyzer.model_path_prefix}_*.csv")
        print(f"  • ppo_value_adaptive_inference_report_{analyzer.model_path_prefix}_{analyzer.server_id}.json")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up GPU resources
        if device.type == 'cuda':
            print("\nCleaning up GPU resources...")
            cleanup_gpu()


if __name__ == "__main__":
    main()
