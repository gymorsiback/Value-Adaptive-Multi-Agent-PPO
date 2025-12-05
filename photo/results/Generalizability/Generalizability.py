#!/usr/bin/env python3
"""
PPO Value Adaptive Model Cross-Server Generalization Analysis System

Enhanced generalization testing: Server1/Server2/Server3 trained models -> Server4 testing
Exports inference results to CSV files for visualization

Cross-Server Generalization Testing:
- Server1 trained model -> Server4 environment testing
- Server2 trained model -> Server4 environment testing  
- Server3 trained model -> Server4 environment testing
- All tests use Server4 dataset and network topology

Output: CSV files with inference results for visualization
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
current_dir = os.path.dirname(os.path.abspath(__file__))
# GeneralizabilityPhoto.py -> results -> photo -> vamappo -> src
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'src')
sys.path.append(src_dir)

from distributed_env import DistributedModelOrchestrationEnv
from distributed_ppo import DistributedPPOValueAdaptive
from gpu_config import configure_gpu_for_training, monitor_gpu_memory, cleanup_gpu


class PPOValueAdaptiveGeneralizationAnalyzer:
    """PPO Value Adaptive cross-server generalization analyzer"""
    
    def __init__(self, model_path_prefix: str = "1000000", target_server: str = "Server4"):
        """
        Initialize generalization analyzer
        
        Args:
            model_path_prefix: Model file prefix (training steps)
            target_server: Target testing server (Server4 for generalization)
        """
        self.model_path_prefix = model_path_prefix
        self.target_server = target_server
        self.training_servers = ["Server1", "Server2", "Server3"]
        self.results_dir = Path(__file__).parent.parent.parent.parent / "results"
        self.save_dir = Path(__file__).parent
        
        # Store inference results for each training server
        self.all_inference_results = {}
        self.all_performance_stats = {}
        
        # Initialize results storage for each training server
        for train_server in self.training_servers:
            self.all_inference_results[train_server] = {
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
                'communication_overhead': [],
                'transmission_times': [],
                'network_hops': [],
                'bandwidth_usage': [],
                'latency_metrics': []
            }
            self.all_performance_stats[train_server] = {}
        
    def load_trained_model(self, train_server: str, device):
        """
        Load trained PPO Value Adaptive model from specific training server with cross-environment adaptation
        
        Args:
            train_server: Training server ID (Server1, Server2, Server3)
            device: Computing device
            
        Returns:
            model: Loaded and adapted PPO model
            env: Server4 testing environment
        """
        print(f"Loading model trained on {train_server} for testing on {self.target_server}...")
        
        # Model file paths - trained on train_server
        if train_server == "Server1":
            # Server1 models don't have server suffix
            actor_path = self.results_dir / f"ppo_value_adaptive_training_actor_model_{self.model_path_prefix}.pth"
            critic_path = self.results_dir / f"ppo_value_adaptive_training_critic_model_{self.model_path_prefix}.pth"
        else:
            # Server2 and Server3 models have server suffix
            actor_path = self.results_dir / f"ppo_value_adaptive_training_actor_model_{self.model_path_prefix}_{train_server}.pth"
            critic_path = self.results_dir / f"ppo_value_adaptive_training_critic_model_{self.model_path_prefix}_{train_server}.pth"
        
        # Verify model files exist
        if not actor_path.exists() or not critic_path.exists():
            raise FileNotFoundError(f"Model files not found for {train_server}: {actor_path}, {critic_path}")
        
        # Load checkpoints to get training environment specs
        try:
            actor_checkpoint = torch.load(actor_path, map_location=device)
            critic_checkpoint = torch.load(critic_path, map_location=device)
            
            model_iteration = actor_checkpoint.get('iteration', 'Unknown')
            training_timesteps = actor_checkpoint.get('training_timesteps', 'Unknown')
            
            print(f"✅ {train_server} model loaded: {training_timesteps} training steps, iteration {model_iteration}")
        except Exception as e:
            print(f"Warning: Could not verify {train_server} model training steps: {e}")
        
        # Dataset paths - always use Server4 for testing environment
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        data_dir = base_dir / "database" / self.target_server
        
        target_server_lower = self.target_server.lower()  # server4
        topology_path = data_dir / f"{target_server_lower}_topology.csv"
        servers_path = data_dir / f"{target_server_lower}_information.csv" 
        # Use SingleDeployment for Server4 as specified by user
        models_path = data_dir / f"{target_server_lower}_SingleDeployment.csv"
        train_tasks_path = data_dir / "train.CSV"
        test_tasks_path = data_dir / "train.CSV"  # Use train.CSV for generalization testing
        
        # Verify data files
        for path in [topology_path, servers_path, models_path, train_tasks_path, test_tasks_path]:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
        
        print(f"✅ Testing environment: {self.target_server} dataset")
        
        # Create environment - using Server4 environment for testing
        env = DistributedModelOrchestrationEnv(
            topology_path=str(topology_path),
            servers_path=str(servers_path),
            models_path=str(models_path),
            tasks_path=str(train_tasks_path),
            test_tasks_path=str(test_tasks_path),
            use_full_state_space=False  # Use compact state space for cross-environment compatibility
        )
        
        # Extract training environment parameters from checkpoint
        training_state_dict = actor_checkpoint['model_state_dict']
        training_n_servers = None
        training_max_models = None
        training_state_dim = None
        
        # Infer training environment parameters from model weights
        for key, tensor in training_state_dict.items():
            if 'shared_layers.0.weight' in key:
                training_state_dim = tensor.shape[1]  # Input dimension
            elif 'server_head.2.weight' in key:
                training_n_servers = tensor.shape[0]  # Output dimension
            elif 'model_head.2.weight' in key:
                training_max_models = tensor.shape[0]  # Output dimension
        
        print(f"Training environment: state_dim={training_state_dim}, n_servers={training_n_servers}, max_models={training_max_models}")
        print(f"Testing environment: state_dim={env.observation_space.shape[0]}, n_servers={env.n_servers}, max_models={env.max_models_per_server}")
        
        # Use training-consistent hyperparameters
        ppo_value_adaptive_hyperparameters = {
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
            'adaptive_clip_range': [0.1, 0.3],
            'uncertainty_regularization': 0.05,
            'exploration_schedule': 'exponential',
            'value_adaptation_frequency': 5,
            'adaptive_lr_bounds': [0.0001, 0.001],
            'uncertainty_threshold': 0.3,
            'exploration_decay_rate': 0.998,
            'value_history_size': 50,
        }
        
        # Create model with testing environment dimensions
        model = DistributedPPOValueAdaptive(env, device=device, **ppo_value_adaptive_hyperparameters)
        
        # Apply cross-environment model adaptation
        self._adapt_model_weights(model, actor_checkpoint, critic_checkpoint, training_state_dim, training_n_servers, training_max_models, env)
        
        print(f"✅ {train_server} model weights adapted and loaded successfully")
        
        return model, env
    
    def _adapt_model_weights(self, model, actor_checkpoint, critic_checkpoint, training_state_dim, training_n_servers, training_max_models, env):
        """
        Adapt model weights from training environment to testing environment
        
        Args:
            model: Target model (testing environment)
            actor_checkpoint: Training actor checkpoint
            critic_checkpoint: Training critic checkpoint  
            training_state_dim: Training environment state dimension
            training_n_servers: Training environment server count
            training_max_models: Training environment max models per server
            env: Testing environment
        """
        print("Applying cross-environment model adaptation...")
        
        # Adapt Actor Network
        self._adapt_actor_weights(model.actor, actor_checkpoint['model_state_dict'], 
                                training_state_dim, training_n_servers, training_max_models, env)
        
        # Adapt Critic Network (only state dimension matters for critic)
        self._adapt_critic_weights(model.critic, critic_checkpoint['model_state_dict'], 
                                 training_state_dim, env.observation_space.shape[0])
        
        # Load optimizer states with adapted learning rates
        try:
            model.actor_optim.load_state_dict(actor_checkpoint['optimizer_state_dict'])
            model.critic_optim.load_state_dict(critic_checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load optimizer states: {e}")
            
        # Load scheduler states
        try:
            model.actor_scheduler.load_state_dict(actor_checkpoint['scheduler_state_dict'])
            model.critic_scheduler.load_state_dict(critic_checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load scheduler states: {e}")
        
        # Load additional model states
        model.current_exploration_factor = actor_checkpoint.get('exploration_factor', 1.0)
        model.adaptive_lr_factor = critic_checkpoint.get('adaptive_lr_factor', 1.0)
        
        # Restore critic history
        if 'value_history' in critic_checkpoint:
            model.critic.value_history = critic_checkpoint['value_history']
        if 'uncertainty_history' in critic_checkpoint:
            model.critic.uncertainty_history = critic_checkpoint['uncertainty_history']
        
    def _adapt_actor_weights(self, target_actor, source_state_dict, training_state_dim, training_n_servers, training_max_models, env):
        """
        Adapt actor network weights for cross-environment compatibility
        """
        target_state_dict = target_actor.state_dict()
        adapted_state_dict = {}
        
        test_state_dim = env.observation_space.shape[0]
        test_n_servers = env.n_servers
        test_max_models = env.max_models_per_server
        
        for name, param in target_state_dict.items():
            if name in source_state_dict:
                source_param = source_state_dict[name]
                
                # Adapt shared layers (state dimension change)
                if 'shared_layers.0.weight' in name:
                    # Input dimension adaptation
                    if source_param.shape[1] != test_state_dim:
                        adapted_param = self._adapt_input_dimension(source_param, training_state_dim, test_state_dim)
                        adapted_state_dict[name] = adapted_param
                    else:
                        adapted_state_dict[name] = source_param
                        
                # Adapt server head (server count change)
                elif 'server_head.2.weight' in name:
                    if source_param.shape[0] != test_n_servers:
                        adapted_param = self._adapt_output_dimension(source_param, training_n_servers, test_n_servers)
                        adapted_state_dict[name] = adapted_param
                    else:
                        adapted_state_dict[name] = source_param
                        
                elif 'server_head.2.bias' in name:
                    if source_param.shape[0] != test_n_servers:
                        adapted_param = self._adapt_bias_dimension(source_param, training_n_servers, test_n_servers)
                        adapted_state_dict[name] = adapted_param
                    else:
                        adapted_state_dict[name] = source_param
                        
                # Adapt model head (max models change)
                elif 'model_head.2.weight' in name:
                    if source_param.shape[0] != test_max_models:
                        adapted_param = self._adapt_output_dimension(source_param, training_max_models, test_max_models)
                        adapted_state_dict[name] = adapted_param
                    else:
                        adapted_state_dict[name] = source_param
                        
                elif 'model_head.2.bias' in name:
                    if source_param.shape[0] != test_max_models:
                        adapted_param = self._adapt_bias_dimension(source_param, training_max_models, test_max_models)
                        adapted_state_dict[name] = adapted_param
                    else:
                        adapted_state_dict[name] = source_param
                        
                # Copy other layers directly
                else:
                    adapted_state_dict[name] = source_param
            else:
                # Initialize missing parameters with Xavier uniform
                adapted_state_dict[name] = param
                torch.nn.init.xavier_uniform_(adapted_state_dict[name])
        
        target_actor.load_state_dict(adapted_state_dict)
    
    def _adapt_critic_weights(self, target_critic, source_state_dict, training_state_dim, test_state_dim):
        """
        Adapt critic network weights for cross-environment compatibility
        """
        target_state_dict = target_critic.state_dict()
        adapted_state_dict = {}
        
        for name, param in target_state_dict.items():
            if name in source_state_dict:
                source_param = source_state_dict[name]
                
                # Adapt value_net first layer (state_dim input)
                if 'value_net.0.weight' in name:
                    if source_param.shape[1] != test_state_dim:
                        adapted_param = self._adapt_input_dimension(source_param, training_state_dim, test_state_dim)
                        adapted_state_dict[name] = adapted_param
                    else:
                        adapted_state_dict[name] = source_param
                        
                # Adapt uncertainty_net first layer (state_dim input)        
                elif 'uncertainty_net.0.weight' in name:
                    if source_param.shape[1] != test_state_dim:
                        adapted_param = self._adapt_input_dimension(source_param, training_state_dim, test_state_dim)
                        adapted_state_dict[name] = adapted_param
                    else:
                        adapted_state_dict[name] = source_param
                        
                # Adapt adaptation_net first layer (state_dim + 1 input)
                elif 'adaptation_net.0.weight' in name:
                    expected_input_dim = test_state_dim + 1
                    source_input_dim = training_state_dim + 1
                    if source_param.shape[1] != expected_input_dim:
                        adapted_param = self._adapt_input_dimension(source_param, source_input_dim, expected_input_dim)
                        adapted_state_dict[name] = adapted_param
                    else:
                        adapted_state_dict[name] = source_param
                        
                # Adapt td_net first layer (state_dim * 2 input)
                elif 'td_net.0.weight' in name:
                    expected_input_dim = test_state_dim * 2
                    source_input_dim = training_state_dim * 2
                    if source_param.shape[1] != expected_input_dim:
                        adapted_param = self._adapt_input_dimension(source_param, source_input_dim, expected_input_dim)
                        adapted_state_dict[name] = adapted_param
                    else:
                        adapted_state_dict[name] = source_param
                        
                # Copy other layers directly  
                else:
                    adapted_state_dict[name] = source_param
            else:
                # Initialize missing parameters
                adapted_state_dict[name] = param
                if param.dim() > 1:  # Only initialize weight matrices, not biases or buffers
                    torch.nn.init.xavier_uniform_(adapted_state_dict[name])
        
        target_critic.load_state_dict(adapted_state_dict)
    
    def _adapt_input_dimension(self, source_param, source_dim, target_dim):
        """
        Adapt input dimension of weight matrix
        """
        if source_dim == target_dim:
            return source_param
            
        target_param = torch.zeros((source_param.shape[0], target_dim), dtype=source_param.dtype, device=source_param.device)
        
        if target_dim > source_dim:
            # Expand: copy existing weights and initialize new ones
            target_param[:, :source_dim] = source_param
            # Initialize additional dimensions with Xavier uniform
            torch.nn.init.xavier_uniform_(target_param[:, source_dim:])
        else:
            # Compress: select most important features (first target_dim dimensions)
            target_param = source_param[:, :target_dim]
            
        return target_param
    
    def _adapt_output_dimension(self, source_param, source_dim, target_dim):
        """
        Adapt output dimension of weight matrix
        """
        if source_dim == target_dim:
            return source_param
            
        if target_dim > source_dim:
            # Expand: copy existing weights and initialize new ones
            target_param = torch.zeros((target_dim, source_param.shape[1]), dtype=source_param.dtype, device=source_param.device)
            target_param[:source_dim, :] = source_param
            # Initialize additional dimensions with Xavier uniform
            torch.nn.init.xavier_uniform_(target_param[source_dim:, :])
        else:
            # Compress: select first target_dim outputs
            target_param = source_param[:target_dim, :]
            
        return target_param
    
    def _adapt_bias_dimension(self, source_param, source_dim, target_dim):
        """
        Adapt bias dimension
        """
        if source_dim == target_dim:
            return source_param
            
        if target_dim > source_dim:
            # Expand: copy existing biases and initialize new ones to zero
            target_param = torch.zeros(target_dim, dtype=source_param.dtype, device=source_param.device)
            target_param[:source_dim] = source_param
        else:
            # Compress: select first target_dim biases
            target_param = source_param[:target_dim]
            
        return target_param
    
    def _calculate_communication_overhead(self, env, server_idx, user_location, data_size):
        """Calculate communication overhead for the action"""
        server_location = [
            env.servers_df.iloc[server_idx]['Latitude'],
            env.servers_df.iloc[server_idx]['Longitude']
        ]
        
        distance = env.calculate_distance(user_location, server_location)
        network_hops = max(1, int(distance / 500))
        
        base_latency = distance / 200000
        hop_latency = network_hops * 0.001
        
        max_bandwidth = 1000
        distance_factor = max(0.1, 1 - distance / 10000)
        effective_bandwidth = max_bandwidth * distance_factor
        
        data_size_mb = data_size
        transmission_time = data_size_mb / effective_bandwidth + base_latency + hop_latency
        
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
    
    def run_inference(self, train_server: str, model, env, n_episodes: int = 200):
        """
        Run inference for specific training server model on Server4 environment
        
        Args:
            train_server: Training server ID
            model: Trained PPO model
            env: Server4 environment
            n_episodes: Number of test episodes
        """
        print(f"Running {train_server} model inference on {self.target_server} (episodes: {n_episodes})...")
        
        # Get results storage for this training server
        inference_results = self.all_inference_results[train_server]
        
        model.actor.eval()
        model.critic.eval()
        
        start_time = time.time()
        successful_episodes = 0
        total_completed_subtasks = 0
        total_possible_subtasks = 0
        
        for episode in range(n_episodes):
            obs, info = env.reset(use_test_set=True)
            
            episode_reward = 0
            episode_steps = 0
            episode_start_time = time.time()
            server_visits = np.zeros(env.n_servers)
            model_usage = {}
            episode_successful = False
            
            episode_communication_overhead = 0
            episode_transmission_times = []
            episode_network_hops = []
            episode_bandwidth_usage = []
            episode_latency = []
            
            task_info = info.get('task_info', {})
            route_sequence = task_info.get('route_sequence', [])
            total_possible_subtasks += len(route_sequence)
            
            inference_results['task_types'].append(route_sequence)
            inference_results['user_locations'].append(task_info.get('user_location', [0, 0]))
            inference_results['data_sizes'].append(task_info.get('data_size', 0))
            inference_results['quality_preferences'].append(task_info.get('quality_preference', 0))
            inference_results['speed_preferences'].append(task_info.get('speed_preference', 0))
            
            done = False
            step_details = []
            max_episode_steps = len(route_sequence) + 2
            
            while not done and episode_steps < max_episode_steps:
                with torch.no_grad():
                    action = self._get_deterministic_action(model, obs, env)
                
                if isinstance(action, (int, float, np.integer, np.floating)):
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
                
                server_idx = int(action[0])
                user_location = env.current_task['user_location']
                data_size = env.current_task['data_size']
                
                comm_overhead = self._calculate_communication_overhead(
                    env, server_idx, user_location, data_size
                )
                
                episode_communication_overhead += comm_overhead['total_overhead_s']
                episode_transmission_times.append(comm_overhead['transmission_time_s'])
                episode_network_hops.append(comm_overhead['network_hops'])
                episode_bandwidth_usage.append(comm_overhead['effective_bandwidth_mbps'])
                episode_latency.append(comm_overhead['base_latency_ms'] + comm_overhead['hop_latency_ms'])
                
                obs, reward, done, _, step_info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                model_idx = int(action[1])
                server_visits[server_idx] += 1
                
                if server_idx in env.server_to_models:
                    if model_idx < len(env.server_to_models[server_idx]):
                        model_info = env.server_to_models[server_idx][model_idx]
                        model_type = model_info['ModelType']
                        model_usage[model_type] = model_usage.get(model_type, 0) + 1
                
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
                
                if step_info.get('task_completed', False):
                    episode_successful = True
                    successful_episodes += 1
                    break
            
            episode_time = time.time() - episode_start_time
            
            final_completion_rate = step_info.get('task_completion_rate', 0) if step_info else 0
            completed_subtasks = step_info.get('completed_subtasks', 0) if step_info else 0
            total_completed_subtasks += completed_subtasks
            
            # Store results for this training server
            inference_results['episode_rewards'].append(episode_reward)
            inference_results['completion_rates'].append(final_completion_rate)
            inference_results['task_success_rates'].append(1.0 if episode_successful else 0.0)
            inference_results['response_times'].append(episode_time)
            inference_results['server_utilizations'].append(server_visits.tolist())
            inference_results['model_selections'].append(model_usage)
            inference_results['communication_overhead'].append(episode_communication_overhead)
            inference_results['transmission_times'].append(episode_transmission_times)
            inference_results['network_hops'].append(episode_network_hops)
            inference_results['bandwidth_usage'].append(episode_bandwidth_usage)
            inference_results['latency_metrics'].append(episode_latency)
            
            inference_results['detailed_metrics'].append({
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
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(inference_results['episode_rewards'][-50:])
                avg_completion = np.mean(inference_results['completion_rates'][-50:])
                avg_success = np.mean(inference_results['task_success_rates'][-50:])
                print(f"{train_server} Episodes {episode+1-49}-{episode+1}: "
                      f"Avg Reward: {avg_reward:.2f}, Avg Completion: {avg_completion:.2%}, "
                      f"Success Rate: {avg_success:.2%}")
        
        total_time = time.time() - start_time
        overall_completion_rate = total_completed_subtasks / total_possible_subtasks if total_possible_subtasks > 0 else 0
        overall_success_rate = successful_episodes / n_episodes
        
        print(f"{train_server} inference completed in {total_time:.2f}s")
        print(f"{train_server} overall completion rate: {overall_completion_rate:.2%}")
        print(f"{train_server} overall success rate: {overall_success_rate:.2%}")
        
        self._calculate_performance_statistics(train_server)
    
    def _get_deterministic_action(self, model, obs, env):
        """Get deterministic action: select highest probability feasible action"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        
        with torch.no_grad():
            server_logits, model_logits, enhancement_weights = model.actor.forward(obs_tensor)
            server_probs = torch.softmax(server_logits, dim=-1)
            model_probs = torch.softmax(model_logits, dim=-1)
            
            server_indices = torch.argsort(server_probs, descending=True).squeeze()
            model_indices = torch.argsort(model_probs, descending=True).squeeze()
            
            for server_idx in server_indices[:10]:
                server_idx = server_idx.item()
                
                for model_idx in model_indices[:5]:
                    model_idx = model_idx.item()
                    
                    if env.is_valid_action(server_idx, model_idx):
                        return np.array([server_idx, model_idx])
            
            return self._greedy_action_selection(env)
    
    def _greedy_action_selection(self, env):
        """Greedy action selection: select optimal action based on heuristic rules"""
        required_model_type = env.current_task['required_model_type']
        user_location = env.current_task['user_location']
        
        best_action = None
        best_score = -float('inf')
        
        for server_idx in range(env.n_servers):
            if server_idx in env.server_to_models:
                for model_idx, model_info in enumerate(env.server_to_models[server_idx]):
                    if env.is_valid_action(server_idx, model_idx):
                        score = self._calculate_heuristic_score(
                            env, server_idx, model_info, user_location
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_action = np.array([server_idx, model_idx])
        
        if best_action is None:
            for server_idx in range(env.n_servers):
                if server_idx in env.server_to_models and len(env.server_to_models[server_idx]) > 0:
                    for model_idx in range(len(env.server_to_models[server_idx])):
                        if env.is_valid_action(server_idx, model_idx):
                            return np.array([server_idx, model_idx])
            
            return np.array([0, 0])
        
        return best_action
    
    def _calculate_heuristic_score(self, env, server_idx, model_info, user_location):
        """Calculate heuristic score for action selection"""
        score = float(model_info['ArenaScore'])
        
        load_penalty = env.servers_computation_load[server_idx]
        score -= load_penalty * 2.0
        
        server_location = [
            env.servers_df.iloc[server_idx]['Latitude'],
            env.servers_df.iloc[server_idx]['Longitude']
        ]
        distance = env.calculate_distance(user_location, server_location)
        distance_penalty = distance / 1000.0
        score -= distance_penalty
        
        avg_load = np.mean(env.servers_computation_load)
        current_load = env.servers_computation_load[server_idx]
        if current_load < avg_load:
            score += 0.5
        
        return score
    
    def _calculate_performance_statistics(self, train_server: str):
        """Calculate comprehensive performance statistics for specific training server"""
        inference_results = self.all_inference_results[train_server]
        
        rewards = inference_results['episode_rewards']
        completion_rates = inference_results['completion_rates']
        success_rates = inference_results['task_success_rates']
        response_times = inference_results['response_times']
        comm_overheads = inference_results['communication_overhead']
        
        self.all_performance_stats[train_server] = {
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
            'model_usage': self._analyze_model_usage(train_server),
            'server_usage': self._analyze_server_usage(train_server),
            'communication_analysis': self._analyze_communication_patterns(train_server)
        }
    
    def _analyze_model_usage(self, train_server: str):
        """Analyze model type usage patterns for specific training server"""
        inference_results = self.all_inference_results[train_server]
        all_model_usage = {}
        for episode_usage in inference_results['model_selections']:
            for model_type, count in episode_usage.items():
                all_model_usage[model_type] = all_model_usage.get(model_type, 0) + count
        
        total_usage = sum(all_model_usage.values())
        return {
            'usage_counts': all_model_usage,
            'usage_percentages': {k: v/total_usage*100 for k, v in all_model_usage.items()} if total_usage > 0 else {},
            'most_used_model': max(all_model_usage.keys(), key=lambda x: all_model_usage[x]) if all_model_usage else None,
            'model_diversity': len(all_model_usage)
        }
    
    def _analyze_server_usage(self, train_server: str):
        """Analyze server utilization patterns for specific training server"""
        inference_results = self.all_inference_results[train_server]
        server_utilizations = np.array(inference_results['server_utilizations'])
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
    
    def _analyze_communication_patterns(self, train_server: str):
        """Analyze communication overhead patterns for specific training server"""
        inference_results = self.all_inference_results[train_server]
        
        all_transmission_times = []
        all_network_hops = []
        all_bandwidths = []
        all_latencies = []
        
        for episode_times, episode_hops, episode_bw, episode_lat in zip(
            inference_results['transmission_times'],
            inference_results['network_hops'],
            inference_results['bandwidth_usage'],
            inference_results['latency_metrics']
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
        """Export inference results to CSV files for each chart"""
        print("Exporting inference data to CSV files...")
        
        # Create save directory
        save_dir = self.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart configurations
        chart_configs = [
            ("Active Server Usage Distribution Comparison", "active_server_usage_distribution"),
            ("Model Type Usage Comparison", "model_type_usage"),
            ("Reward vs Response Time Comparison", "reward_vs_response_time"),
            ("Key Performance Metrics Comparison", "key_performance_metrics"),
            ("Communication Overhead by Episodes", "communication_overhead_episodes")
        ]
        
        # Export data for each chart
        for i, (chart_title, file_suffix) in enumerate(chart_configs, 1):
            
            if i == 1:  # Active Server Usage Distribution
                for train_server in self.training_servers:
                    server_usage = self.all_performance_stats[train_server]['server_usage']['total_usage_per_server']
                    data = {
                        'server_index': list(range(len(server_usage))),
                        'usage_count': server_usage
                    }
                    df = pd.DataFrame(data)
                    csv_filename = f"inference_generalization_{self.target_server.lower()}_{file_suffix}_{train_server}.csv"
                    df.to_csv(save_dir / csv_filename, index=False)
                    print(f"✅ Exported: {csv_filename}")
                
            elif i == 2:  # Model Type Usage
                data = {
                    'model_type': [],
                    'Server1_usage': [],
                    'Server2_usage': [],
                    'Server3_usage': []
                }
                
                # Collect all model types
                all_model_types = set()
                for train_server in self.training_servers:
                    model_usage = self.all_performance_stats[train_server]['model_usage']['usage_percentages']
                    all_model_types.update(model_usage.keys())
                all_model_types = sorted(list(all_model_types))
                
                for model_type in all_model_types:
                    data['model_type'].append(model_type)
                    for train_server in self.training_servers:
                        model_usage = self.all_performance_stats[train_server]['model_usage']['usage_percentages']
                        data[f'{train_server}_usage'].append(model_usage.get(model_type, 0))
                
                df = pd.DataFrame(data)
                csv_filename = f"inference_generalization_{self.target_server.lower()}_{file_suffix}.csv"
                df.to_csv(save_dir / csv_filename, index=False)
                print(f"✅ Exported: {csv_filename}")
                
            elif i == 3:  # Reward vs Response Time
                for train_server in self.training_servers:
                    rewards = self.all_inference_results[train_server]['episode_rewards']
                    response_times = self.all_inference_results[train_server]['response_times']
                    data = {
                        'response_time': response_times,
                        'reward': rewards
                    }
                    df = pd.DataFrame(data)
                    csv_filename = f"inference_generalization_{self.target_server.lower()}_{file_suffix}_{train_server}.csv"
                    df.to_csv(save_dir / csv_filename, index=False)
                    print(f"✅ Exported: {csv_filename}")
                
            elif i == 4:  # Key Performance Metrics (去掉Completion Rate)
                data = {
                    'metric': ['Avg_Reward', 'Success_Rate', 'Response_Time'],
                    'Server1': [],
                    'Server2': [],
                    'Server3': []
                }
                
                for train_server in self.training_servers:
                    stats = self.all_performance_stats[train_server]
                    metrics = [
                        stats['reward_metrics']['mean'],
                        stats['completion_metrics']['success_rate'],
                        stats['efficiency_metrics']['mean_response_time']
                    ]
                    data[train_server] = metrics
                
                df = pd.DataFrame(data)
                csv_filename = f"inference_generalization_{self.target_server.lower()}_{file_suffix}.csv"
                df.to_csv(save_dir / csv_filename, index=False)
                print(f"✅ Exported: {csv_filename}")
                
            elif i == 5:  # Communication Overhead by Episodes
                data = {
                    'server': [],
                    'episode': [],
                    'communication_overhead': []
                }
                for train_server in self.training_servers:
                    overheads = self.all_inference_results[train_server]['communication_overhead']
                    for idx, overhead in enumerate(overheads):
                        data['server'].append(train_server)
                        data['episode'].append(idx)
                        data['communication_overhead'].append(overhead)
                
                df = pd.DataFrame(data)
                csv_filename = f"inference_generalization_{self.target_server.lower()}_{file_suffix}.csv"
                df.to_csv(save_dir / csv_filename, index=False)
                print(f"✅ Exported: {csv_filename}")
        
        print(f"✅ All inference data exported to CSV files in: {save_dir}")
        print(f"📁 CSV naming pattern: inference_generalization_{self.target_server.lower()}_[chart_name].csv")
    
    def export_generalization_report(self):
        """Export comprehensive generalization analysis report to JSON"""
        print("Exporting generalization analysis report...")
        
        # Prepare comprehensive report
        report = {
            'generalization_metadata': {
                'model_version': self.model_path_prefix,
                'training_servers': self.training_servers,
                'target_testing_server': self.target_server,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'test_dataset': f'{self.target_server}/train.CSV',
                'analysis_type': 'PPO_Value_Adaptive_Cross_Server_Generalization_Analysis'
            },
            'comparative_performance_summary': {},
            'detailed_server_analysis': {},
            'generalization_insights': {}
        }
        
        # Calculate comparative metrics
        for train_server in self.training_servers:
            inference_results = self.all_inference_results[train_server]
            stats = self.all_performance_stats[train_server]
            
            # Summary metrics for comparison
            report['comparative_performance_summary'][train_server] = {
                'average_reward': float(np.mean(inference_results['episode_rewards'])),
                'reward_std': float(np.std(inference_results['episode_rewards'])),
                'success_rate': float(np.mean(inference_results['task_success_rates'])),
                'completion_rate': float(np.mean(inference_results['completion_rates'])),
                'response_time': float(np.mean(inference_results['response_times'])),
                'communication_overhead': float(np.mean(inference_results['communication_overhead'])),
                'episode_count': len(inference_results['episode_rewards'])
            }
            
            # Detailed analysis
            report['detailed_server_analysis'][train_server] = {
                'performance_statistics': stats,
                'inference_results_summary': {
                    'total_episodes': len(inference_results['episode_rewards']),
                    'successful_episodes': int(np.sum(inference_results['task_success_rates'])),
                    'average_steps_per_episode': float(np.mean([m['steps'] for m in inference_results['detailed_metrics']])),
                    'max_reward': float(np.max(inference_results['episode_rewards'])),
                    'min_reward': float(np.min(inference_results['episode_rewards'])),
                    'reward_variance': float(np.var(inference_results['episode_rewards']))
                }
            }
        
        # Calculate generalization insights
        summary_data = report['comparative_performance_summary']
        
        # Best performing server analysis
        best_reward_server = max(summary_data.keys(), key=lambda x: summary_data[x]['average_reward'])
        best_success_server = max(summary_data.keys(), key=lambda x: summary_data[x]['success_rate'])
        best_completion_server = max(summary_data.keys(), key=lambda x: summary_data[x]['completion_rate'])
        most_efficient_server = min(summary_data.keys(), key=lambda x: summary_data[x]['response_time'])
        
        report['generalization_insights'] = {
            'best_performers': {
                'highest_reward': best_reward_server,
                'highest_success_rate': best_success_server,
                'highest_completion_rate': best_completion_server,
                'most_efficient': most_efficient_server
            },
            'performance_comparison': {
                'reward_range': {
                    'max': max([summary_data[s]['average_reward'] for s in self.training_servers]),
                    'min': min([summary_data[s]['average_reward'] for s in self.training_servers]),
                    'spread': max([summary_data[s]['average_reward'] for s in self.training_servers]) - 
                             min([summary_data[s]['average_reward'] for s in self.training_servers])
                },
                'success_rate_range': {
                    'max': max([summary_data[s]['success_rate'] for s in self.training_servers]),
                    'min': min([summary_data[s]['success_rate'] for s in self.training_servers]),
                    'spread': max([summary_data[s]['success_rate'] for s in self.training_servers]) - 
                             min([summary_data[s]['success_rate'] for s in self.training_servers])
                }
            },
            'generalization_quality': {
                'performance_consistency': 1.0 - np.std([summary_data[s]['average_reward'] for s in self.training_servers]) / np.mean([summary_data[s]['average_reward'] for s in self.training_servers]),
                'cross_server_adaptability': np.mean([summary_data[s]['success_rate'] for s in self.training_servers])
            }
        }
        
        # Convert numpy types for JSON serialization
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
        
        # Save report
        output_path = self.results_dir / f'ppo_value_adaptive_generalization_report_{self.model_path_prefix}_{self.target_server}.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Generalization report exported to: {output_path}")
        return report
    
    def run_complete_generalization_analysis(self, n_episodes: int = 200):
        """
        Run complete cross-server generalization analysis
        
        Args:
            n_episodes: Number of test episodes per training server
        """
        print("="*80)
        print("PPO VALUE ADAPTIVE CROSS-SERVER GENERALIZATION ANALYSIS")
        print("="*80)
        print(f"Training servers: {', '.join(self.training_servers)}")
        print(f"Testing environment: {self.target_server}")
        print(f"Episodes per model: {n_episodes}")
        print("="*80)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        if device.type == 'cuda':
            print("Initial GPU memory status:")
            monitor_gpu_memory()
        
        try:
            # Run inference for each training server
            for train_server in self.training_servers:
                print(f"\n{'='*60}")
                print(f"TESTING {train_server} MODEL ON {self.target_server}")
                print(f"{'='*60}")
                
                # Load model and environment
                model, env = self.load_trained_model(train_server, device)
                
                # Run inference
                self.run_inference(train_server, model, env, n_episodes)
                
                # Clean up GPU memory after each model
                if device.type == 'cuda':
                    del model
                    torch.cuda.empty_cache()
            
            # Export CSV data files
            print(f"\n{'='*60}")
            print("EXPORTING CSV DATA FILES")
            print(f"{'='*60}")
            self.export_inference_data_to_csv()
            
            # Export comprehensive report
            print(f"\n{'='*60}")
            print("EXPORTING GENERALIZATION REPORT")
            print(f"{'='*60}")
            report = self.export_generalization_report()
            
            # Print summary
            print(f"\n{'='*80}")
            print("GENERALIZATION ANALYSIS COMPLETE")
            print(f"{'='*80}")
            
            # Ensure report is a dictionary before accessing
            if isinstance(report, dict):
                summary_data = report.get('comparative_performance_summary', {})
                print(f"Performance Summary on {self.target_server}:")
                for train_server in self.training_servers:
                    if train_server in summary_data:
                        data = summary_data[train_server]
                        print(f"  {train_server}:")
                        print(f"    • Average Reward: {data.get('average_reward', 0):.3f}")
                        print(f"    • Success Rate: {data.get('success_rate', 0):.1%}")
                        print(f"    • Completion Rate: {data.get('completion_rate', 0):.1%}")
                        print(f"    • Response Time: {data.get('response_time', 0):.3f}s")
                        print(f"    • Communication Overhead: {data.get('communication_overhead', 0):.3f}s")
                
                insights = report.get('generalization_insights', {})
                best_performers = insights.get('best_performers', {})
                generalization_quality = insights.get('generalization_quality', {})
                
                print(f"\nGeneralization Insights:")
                print(f"  • Best Reward: {best_performers.get('highest_reward', 'N/A')}")
                print(f"  • Best Success Rate: {best_performers.get('highest_success_rate', 'N/A')}")
                print(f"  • Most Efficient: {best_performers.get('most_efficient', 'N/A')}")
                print(f"  • Performance Consistency: {generalization_quality.get('performance_consistency', 0):.2%}")
                print(f"  • Cross-Server Adaptability: {generalization_quality.get('cross_server_adaptability', 0):.2%}")
            else:
                print("Report format error: Expected dictionary but got different type")
            
            print(f"\nGenerated Files:")
            print(f"  • CSV data files: inference_generalization_{self.target_server.lower()}_*.csv")
            print(f"  • Generalization report: ppo_value_adaptive_generalization_report_{self.model_path_prefix}_{self.target_server}.json")
            print(f"\nNote: To generate visualization charts, run InferenceDraw.py after this completes.")
            
        except Exception as e:
            print(f"Error during generalization analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            if device.type == 'cuda':
                print("\nCleaning up GPU resources...")
                cleanup_gpu()


def main():
    """Main function for cross-server generalization analysis"""
    
    # Initialize generalization analyzer
    analyzer = PPOValueAdaptiveGeneralizationAnalyzer(
        model_path_prefix="1000000", 
        target_server="Server4"
    )
    
    # Run complete generalization analysis
    # Change n_episodes if needed (e.g., n_episodes=100 for faster testing)
    analyzer.run_complete_generalization_analysis(n_episodes=200)


if __name__ == "__main__":
    main() 