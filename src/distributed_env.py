"""
Distributed Multi-Modal Model Orchestration Environment Class
Supports both single-agent and multi-agent PPO reinforcement learning
Extended for VAMAPPO (Value-Adaptive Multi-Agent PPO)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import random
import math
import ast
import torch
from typing import Dict, List, Tuple, Any, Optional
from gymnasium import spaces
from sklearn.cluster import KMeans

class DistributedModelOrchestrationEnv(gym.Env):
    """
    Distributed Multi-Modal Model Orchestration Environment
    Supports both single-agent and VAMAPPO multi-agent modes
    """
    
    def __init__(self, 
                 topology_path: str,
                 servers_path: str, 
                 models_path: str,
                 tasks_path: str,
                 test_tasks_path: Optional[str] = None,
                 use_full_state_space: bool = True,
                 # VAMAPPO parameters
                 multi_agent_mode: bool = False,
                 n_agents: int = 1,
                 server_allocation_strategy: str = 'geographic'):
        """
        Initialize environment
        
        Args:
            topology_path: Network topology file path
            servers_path: Server information file path
            models_path: Model deployment file path
            tasks_path: Training tasks file path
            test_tasks_path: Test tasks file path (optional)
            use_full_state_space: Whether to use full 735-dim state space (True) or compact 107-dim (False)
            multi_agent_mode: Whether to use VAMAPPO multi-agent mode
            n_agents: Number of agents for multi-agent mode
            server_allocation_strategy: Strategy for allocating servers to agents ('geographic' or 'balanced')
        """
        super().__init__()
        
        # Store configuration
        self.use_full_state_space = use_full_state_space
        self.multi_agent_mode = multi_agent_mode
        self.n_agents = n_agents
        self.server_allocation_strategy = server_allocation_strategy
        
        # Load datasets
        self.load_datasets(topology_path, servers_path, models_path, tasks_path, test_tasks_path)
        
        # Environment parameters
        self.max_episode_steps = 20
        self.current_step = 0
        self.gamma = 0.98
        self.max_expected_time = 1000.0  # Maximum expected time
        
        # Current task and state
        self.current_task = None
        self.current_subtask_index = 0
        self.servers_computation_load = np.zeros(self.n_servers)
        self.servers_storage_load = np.zeros(self.n_servers)
        self.episode_rewards = []
        
        # Define action and state spaces
        self.action_space = self.define_action_space()
        self.observation_space = self.define_observation_space()
        
        # VAMAPPO multi-agent setup
        if self.multi_agent_mode:
            self.server_allocation = self._allocate_servers_to_agents()
            self.agent_states = {}
            print(f"VAMAPPO mode: {self.n_agents} agents managing {self.n_servers} servers")
            for agent_id, servers in self.server_allocation.items():
                print(f"  {agent_id}: {servers}")
        
        # Initialize environment
        self.reset()
    
    def load_datasets(self, topology_path, servers_path, models_path, tasks_path, test_tasks_path):
        """Load all datasets and establish mapping relationships"""
        
        # Load network topology
        try:
            self.topology_df = pd.read_csv(topology_path, index_col=0, encoding='utf-8')
        except UnicodeDecodeError:
            self.topology_df = pd.read_csv(topology_path, index_col=0, encoding='gbk')
        self.topology_matrix = self.topology_df.values.astype(float)
        self.n_servers = len(self.topology_matrix)
        
        # Load server information
        try:
            self.servers_df = pd.read_csv(servers_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.servers_df = pd.read_csv(servers_path, encoding='gbk')
        self.servers_df = self.servers_df.sort_values('ServerID').reset_index(drop=True)
        
        # Build server ID to index mapping
        self.server_id_to_idx = {int(server_id): idx for idx, server_id in enumerate(self.servers_df['ServerID'])}
        self.idx_to_server_id = {idx: int(server_id) for idx, server_id in enumerate(self.servers_df['ServerID'])}
        
        # Load model deployment information
        try:
            self.models_df = pd.read_csv(models_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.models_df = pd.read_csv(models_path, encoding='gbk')
        
        # Build server to models mapping
        self.server_to_models = {}
        self.model_type_to_models = {}
        
        for _, model in self.models_df.iterrows():
            server_idx = self.server_id_to_idx[int(model['ServerID'])]
            model_type = int(model['ModelType'])
            
            if server_idx not in self.server_to_models:
                self.server_to_models[server_idx] = []
            self.server_to_models[server_idx].append(model)
            
            if model_type not in self.model_type_to_models:
                self.model_type_to_models[model_type] = []
            self.model_type_to_models[model_type].append(model)
        
        # Load task data
        try:
            self.tasks_df = pd.read_csv(tasks_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.tasks_df = pd.read_csv(tasks_path, encoding='gbk')
            
        if test_tasks_path:
            try:
                self.test_tasks_df = pd.read_csv(test_tasks_path, encoding='utf-8')
            except UnicodeDecodeError:
                self.test_tasks_df = pd.read_csv(test_tasks_path, encoding='gbk')
        else:
            self.test_tasks_df = None
        
        # Process route strings
        self.tasks_df['route_parsed'] = self.tasks_df['route'].apply(ast.literal_eval)
        if self.test_tasks_df is not None:
            self.test_tasks_df['route_parsed'] = self.test_tasks_df['route'].apply(ast.literal_eval)
        
        # Extract unique model types
        all_model_types = set()
        for _, model in self.models_df.iterrows():
            all_model_types.add(int(model['ModelType']))
        self.model_types = sorted(list(all_model_types))
        
        # Calculate maximum models per server
        max_models = 0
        for server_idx in range(self.n_servers):
            if server_idx in self.server_to_models:
                max_models = max(max_models, len(self.server_to_models[server_idx]))
        self.max_models_per_server = max_models
        
        print(f"Environment loaded: {self.n_servers} servers, {len(self.models_df)} models, {len(self.tasks_df)} tasks")
        print(f"Model types: {self.model_types}")
        print(f"Max models per server: {self.max_models_per_server}")
    
    def _allocate_servers_to_agents(self) -> Dict[str, List[str]]:
        """
        Allocate servers to agents based on the selected strategy
        """
        if self.server_allocation_strategy == 'geographic':
            return self._allocate_servers_geographic()
        elif self.server_allocation_strategy == 'balanced':
            return self._allocate_servers_balanced()
        else:
            # Default: simple round-robin allocation
            return self._allocate_servers_round_robin()
    
    def _allocate_servers_geographic(self) -> Dict[str, List[str]]:
        """
        Allocate servers to agents based on geographic clustering
        """
        # Extract server coordinates
        coordinates = []
        server_ids = []
        for idx, server in self.servers_df.iterrows():
            coordinates.append([server['Latitude'], server['Longitude']])
            server_ids.append(idx)
        
        # Use K-means clustering
        kmeans = KMeans(n_clusters=self.n_agents, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coordinates)
        
        # Create allocation dictionary
        allocation = {f'Agent_{i}': [] for i in range(self.n_agents)}
        for server_idx, cluster_id in enumerate(clusters):
            allocation[f'Agent_{cluster_id}'].append(f'Server_{server_idx}')
        
        return allocation
    
    def _allocate_servers_balanced(self) -> Dict[str, List[str]]:
        """
        Allocate servers to agents based on computational load balancing
        """
        # Calculate server capacities
        server_capacities = []
        for idx, server in self.servers_df.iterrows():
            capacity = server['ComputationCapacity']
            server_capacities.append((f'Server_{idx}', capacity))
        
        # Sort by capacity (descending)
        server_capacities.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy allocation to balance total capacity
        agent_loads = [0.0] * self.n_agents
        allocation = {f'Agent_{i}': [] for i in range(self.n_agents)}
        
        for server_id, capacity in server_capacities:
            # Assign to agent with minimum current load
            min_load_agent = np.argmin(agent_loads)
            allocation[f'Agent_{min_load_agent}'].append(server_id)
            agent_loads[min_load_agent] += capacity
        
        return allocation
    
    def _allocate_servers_round_robin(self) -> Dict[str, List[str]]:
        """
        Simple round-robin allocation of servers to agents
        """
        allocation = {f'Agent_{i}': [] for i in range(self.n_agents)}
        
        for server_idx in range(self.n_servers):
            agent_id = f'Agent_{server_idx % self.n_agents}'
            allocation[agent_id].append(f'Server_{server_idx}')
        
        return allocation
    
    def define_action_space(self):
        """Define action space - server selection and model selection"""
        if self.multi_agent_mode:
            # In multi-agent mode, each agent has a different action space
            # based on the servers they manage
            return {f'Agent_{i}': self._get_agent_action_space(f'Agent_{i}') 
                    for i in range(self.n_agents)}
        else:
            # Single agent mode
            return spaces.MultiDiscrete([self.n_servers, self.max_models_per_server])
    
    def _get_agent_action_space(self, agent_id: str):
        """Get action space for a specific agent"""
        if not hasattr(self, 'server_allocation'):
            # Fallback for single agent mode
            return spaces.MultiDiscrete([self.n_servers, self.max_models_per_server])
        
        managed_servers = self.server_allocation[agent_id]
        n_managed_servers = len(managed_servers)
        
        # Agent can only select from servers they manage
        return spaces.MultiDiscrete([n_managed_servers, self.max_models_per_server])
    
    def define_observation_space(self):
        """Define observation space"""
        if self.multi_agent_mode:
            # Multi-agent mode: each agent has local observation space
            return {f'Agent_{i}': self._get_agent_observation_space(f'Agent_{i}') 
                    for i in range(self.n_agents)}
        else:
            # Single agent mode: use original observation space
            if self.use_full_state_space:
                # Full state space (735 dimensions) - Compatible with trained models
                state_dim = (
                    self.n_servers * 4 +          # Server states: 25 * 4 = 100
                    7 +                           # Task information: 7
                    len(self.model_types) +       # Model type availability: 3
                    self.n_servers * self.n_servers  # Network topology: 25 * 25 = 625
                )
            else:
                # Compact state space (107 dimensions)
                state_dim = self.n_servers * 4 + 7
            
            return spaces.Box(low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32)
    
    def _get_agent_observation_space(self, agent_id: str):
        """Get observation space for a specific agent"""
        if not hasattr(self, 'server_allocation'):
            # Fallback for single agent mode
            return spaces.Box(low=-1.0, high=1.0, shape=(735,), dtype=np.float32)
        
        managed_servers = self.server_allocation[agent_id]
        n_managed_servers = len(managed_servers)
        
        # Local observation space for multi-agent:
        # - Managed server states: n_managed_servers * 4
        # - Task information: 7 dimensions
        # - Other agents summary: (n_agents - 1) * 3 (load, availability, performance)
        # - Communication info: 16 dimensions
        local_state_dim = (
            n_managed_servers * 4 +    # Local server states
            7 +                        # Task information
            (self.n_agents - 1) * 3 + # Other agents summary
            16                         # Communication/coordination info
        )
        
        return spaces.Box(low=-1.0, high=1.0, shape=(local_state_dim,), dtype=np.float32)
    
    def reset(self, seed=None, use_test_set=False):
        """Reset environment, start new episode"""
        super().reset(seed=seed)
        
        # Select task dataset
        tasks_df = self.test_tasks_df if (use_test_set and self.test_tasks_df is not None) else self.tasks_df
        
        # Randomly select a task
        task_idx = random.randint(0, len(tasks_df) - 1)
        task_row = tasks_df.iloc[task_idx]
        
        # Set current task with flexible column name handling
        # Check for different column name formats (La/La, la/lo)
        lat_col = 'La' if 'La' in task_row.index else 'la'
        lon_col = 'Lo' if 'Lo' in task_row.index else 'lo'
        
        self.current_task = {
            'user_location': [float(task_row[lat_col]), float(task_row[lon_col])],
            'quality_preference': float(task_row['w1']),
            'speed_preference': float(task_row['w2']),
            'data_size': float(task_row['size']),
            'route_sequence': task_row['route_parsed'],
            'current_subtask_index': 0,
            'required_model_type': task_row['route_parsed'][0],
        }
        
        # Reset environment state
        self.current_subtask_index = 0
        self.current_step = 0
        self.servers_computation_load = np.random.uniform(0.1, 0.3, self.n_servers)  # Initial load
        self.servers_storage_load = np.random.uniform(0.1, 0.3, self.n_servers)
        self.episode_rewards = []
        
        # Return initial state
        if self.multi_agent_mode:
            # Multi-agent mode: return dictionary of observations
            observations = {}
            infos = {}
            for agent_id in [f'Agent_{i}' for i in range(self.n_agents)]:
                observations[agent_id] = self.get_agent_state(agent_id)
                infos[agent_id] = {'task_info': self.current_task, 'managed_servers': self.server_allocation[agent_id]}
            return observations, infos
        else:
            # Single agent mode
            state = self.get_current_state()
            info = {'task_info': self.current_task}
            return state, info
    
    def step(self, action):
        """Execute action, update environment state"""
        self.current_step += 1
        
        if self.multi_agent_mode:
            # Multi-agent step
            return self._multi_agent_step(action)
        else:
            # Single agent step (original logic)
            return self._single_agent_step(action)
    
    def _single_agent_step(self, action):
        """Original single agent step logic"""
        # Parse action
        server_idx = int(action[0])
        model_idx = int(action[1])
        
        # Validate action validity
        if not self.is_valid_action(server_idx, model_idx):
            reward = 1.0
            done = True
            info = {'error': 'invalid_action', 'reason': 'Action validation failed'}
            return self.get_current_state(), reward, done, False, info
        
        # Get selected model
        selected_model = self.server_to_models[server_idx][model_idx]
        
        # Calculate reward
        reward = self.calculate_reward(server_idx, selected_model)
        self.episode_rewards.append(reward)
        
        # Update resource state
        computation_cost = self.current_task['data_size'] / self.servers_df.iloc[server_idx]['ComputationCapacity']
        self.servers_computation_load[server_idx] += computation_cost * 0.1
        self.servers_storage_load[server_idx] += 0.05
        
        # Update task progress
        self.current_subtask_index += 1
        
        # Calculate task completion rate
        if self.current_task and 'route_sequence' in self.current_task:
            task_completion_rate = self.current_subtask_index / len(self.current_task['route_sequence'])
        else:
            task_completion_rate = 0.0
        
        # Calculate load balance score
        load_balance_score = 1.0 - np.std(self.servers_computation_load)
        
        # Check if task is completed
        if (self.current_task and 'route_sequence' in self.current_task and 
            self.current_subtask_index >= len(self.current_task['route_sequence'])):
            done = True
            # Task completed, release all resources
            self.release_all_resources()
            info = {
                'task_completed': True, 
                'total_reward': sum(self.episode_rewards) if self.episode_rewards else 0.0,
                'task_completion_rate': task_completion_rate,
                'load_balance_score': load_balance_score
            }
        else:
            done = False
            # Update next subtask
            if self.current_task and 'route_sequence' in self.current_task:
                self.current_task['current_subtask_index'] = self.current_subtask_index
                if self.current_subtask_index < len(self.current_task['route_sequence']):
                    self.current_task['required_model_type'] = self.current_task['route_sequence'][self.current_subtask_index]
            # Partially release resources
            self.servers_computation_load[server_idx] *= 0.9
            info = {
                'subtask_completed': True,
                'task_completion_rate': task_completion_rate,
                'load_balance_score': load_balance_score
            }
        
        # Check if maximum steps reached
        if self.current_step >= self.max_episode_steps:
            done = True
            if 'max_steps_reached' not in info:
                info['max_steps_reached'] = True
            info['task_completion_rate'] = task_completion_rate
            info['load_balance_score'] = load_balance_score
        
        # Get next state
        next_state = self.get_current_state()
        
        return next_state, reward, done, False, info
    
    def _multi_agent_step(self, joint_actions):
        """Multi-agent step logic for VAMAPPO"""
        # joint_actions is a dictionary: {agent_id: action}
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        # Track global metrics
        global_success = True
        total_collaboration_bonus = 0.0
        
        # Process each agent's action
        for agent_id, action in joint_actions.items():
            # Parse agent action (local server index, model index)
            # Handle different action formats with more robust tensor handling
            try:
                if torch.is_tensor(action):
                    # Convert tensor to numpy first to avoid tensor scalar issues
                    action_np = action.detach().cpu().numpy()
                    if action_np.ndim == 0:
                        # Scalar tensor
                        local_server_idx = int(action_np.item())
                        model_idx = 0
                    elif action_np.ndim == 1 and len(action_np) >= 2:
                        # 1D Vector tensor [server_idx, model_idx]
                        local_server_idx = int(action_np[0])
                        model_idx = int(action_np[1])
                    elif action_np.ndim == 2 and action_np.shape[0] >= 1 and action_np.shape[1] >= 2:
                        # 2D tensor [[server_idx, model_idx]] - flatten first
                        local_server_idx = int(action_np[0, 0])
                        model_idx = int(action_np[0, 1])
                    else:
                        # Single element - flatten and use first element
                        action_flat = action_np.flatten()
                        local_server_idx = int(action_flat[0])
                        model_idx = int(action_flat[1]) if len(action_flat) > 1 else 0
                elif hasattr(action, '__len__') and len(action) >= 2:
                    # List or array
                    local_server_idx = int(action[0])
                    model_idx = int(action[1])
                else:
                    # Single scalar
                    local_server_idx = int(action)
                    model_idx = 0
            except Exception as e:
                print(f"Error parsing action for {agent_id}: {action}, type: {type(action)}, error: {e}")
                # Fallback: use default values
                local_server_idx = 0
                model_idx = 0
            
            # Convert local server index to global server index
            managed_servers = self.server_allocation[agent_id]
            if local_server_idx >= len(managed_servers):
                # Invalid local server index
                rewards[agent_id] = -2.0
                dones[agent_id] = True
                infos[agent_id] = {'error': 'invalid_local_server_index'}
                observations[agent_id] = self.get_agent_state(agent_id)
                global_success = False
                continue
            
            # Get global server index
            server_name = managed_servers[local_server_idx]
            global_server_idx = int(server_name.split('_')[1])  # Extract index from "Server_X"
            
            # Validate action
            if not self.is_valid_action(global_server_idx, model_idx):
                rewards[agent_id] = -1.0
                dones[agent_id] = True
                infos[agent_id] = {'error': 'invalid_action', 'agent_id': agent_id}
                observations[agent_id] = self.get_agent_state(agent_id)
                global_success = False
                continue
            
            # Get selected model
            selected_model = self.server_to_models[global_server_idx][model_idx]
            
            # Calculate individual reward
            individual_reward = self.calculate_reward(global_server_idx, selected_model)
            
            # Update resource state
            computation_cost = self.current_task['data_size'] / self.servers_df.iloc[global_server_idx]['ComputationCapacity']
            self.servers_computation_load[global_server_idx] += computation_cost * 0.1
            self.servers_storage_load[global_server_idx] += 0.05
            
            # Store individual results
            rewards[agent_id] = individual_reward
            dones[agent_id] = False
            infos[agent_id] = {
                'agent_id': agent_id,
                'selected_server': global_server_idx,
                'selected_model': model_idx,
                'individual_reward': individual_reward
            }
        
        # Calculate collaboration bonus
        if global_success and len(joint_actions) > 1:
            collaboration_bonus = self._calculate_collaboration_bonus(joint_actions)
            total_collaboration_bonus = collaboration_bonus
        
        # Add collaboration bonus to all agents
        for agent_id in rewards:
            if agent_id in joint_actions and not dones[agent_id]:
                collaboration_reward = total_collaboration_bonus / len([a for a in dones.values() if not a])
                rewards[agent_id] = 0.8 * rewards[agent_id] + 0.2 * collaboration_reward
                infos[agent_id]['collaboration_bonus'] = collaboration_reward
        
        # Update task progress
        self.current_subtask_index += 1
        self.episode_rewards.extend(list(rewards.values()))
        
        # Check task completion
        task_completion_rate = 0.0
        if self.current_task and 'route_sequence' in self.current_task:
            task_completion_rate = self.current_subtask_index / len(self.current_task['route_sequence'])
            
            if self.current_subtask_index >= len(self.current_task['route_sequence']):
                # Task completed
                for agent_id in rewards:
                    dones[agent_id] = True
                    infos[agent_id]['task_completed'] = True
                self.release_all_resources()
            else:
                # Update next subtask
                self.current_task['current_subtask_index'] = self.current_subtask_index
                self.current_task['required_model_type'] = self.current_task['route_sequence'][self.current_subtask_index]
        
        # Check maximum steps
        if self.current_step >= self.max_episode_steps:
            for agent_id in rewards:
                dones[agent_id] = True
                infos[agent_id]['max_steps_reached'] = True
        
        # Get observations for all agents
        for agent_id in [f'Agent_{i}' for i in range(self.n_agents)]:
            if agent_id not in observations:  # Only if not already set due to error
                observations[agent_id] = self.get_agent_state(agent_id)
            
            # Add common info
            if agent_id in infos:
                infos[agent_id].update({
                    'task_completion_rate': task_completion_rate,
                    'global_load_balance': 1.0 - np.std(self.servers_computation_load),
                    'step': self.current_step
                })
        
        return observations, rewards, dones, False, infos
    
    def _calculate_collaboration_bonus(self, joint_actions):
        """Calculate collaboration bonus based on joint actions"""
        if len(joint_actions) <= 1:
            return 0.0
        
        # Bonus for load balancing across agents
        agent_loads = []
        for agent_id, action in joint_actions.items():
            # Handle different action formats with robust tensor handling
            try:
                if torch.is_tensor(action):
                    action_np = action.detach().cpu().numpy()
                    if action_np.ndim == 0:
                        local_server_idx = int(action_np.item())
                    elif action_np.ndim == 1 and len(action_np) >= 1:
                        local_server_idx = int(action_np[0])
                    elif action_np.ndim == 2 and action_np.shape[0] >= 1 and action_np.shape[1] >= 1:
                        local_server_idx = int(action_np[0, 0])
                    else:
                        action_flat = action_np.flatten()
                        local_server_idx = int(action_flat[0])
                elif hasattr(action, '__len__') and len(action) >= 1:
                    local_server_idx = int(action[0])
                else:
                    local_server_idx = int(action)
            except Exception:
                local_server_idx = 0  # Fallback
                
            managed_servers = self.server_allocation[agent_id]
            if local_server_idx < len(managed_servers):
                server_name = managed_servers[local_server_idx]
                global_server_idx = int(server_name.split('_')[1])
                agent_loads.append(self.servers_computation_load[global_server_idx])
        
        if len(agent_loads) > 1:
            load_balance_bonus = max(0, 1.0 - np.std(agent_loads) * 2.0)
        else:
            load_balance_bonus = 0.0
        
        # Bonus for task completion efficiency
        efficiency_bonus = 0.5 if self.current_step < self.max_episode_steps * 0.7 else 0.0
        
        return load_balance_bonus + efficiency_bonus
    
    def get_current_state(self):
        """Get current environment state"""
        state_components = []
        
        # 1. Server states (n_servers * 4) - Always included
        for i in range(self.n_servers):
            server_info = self.servers_df.iloc[i]
            state_components.extend([
                self.servers_computation_load[i],  # Computation load
                self.servers_storage_load[i],      # Storage load
                server_info['Longitude'] / 180.0,  # Normalized longitude
                server_info['Latitude'] / 90.0,    # Normalized latitude
            ])
        
        # 2. Current task information (7 dimensions) - Always included
        if self.current_task:
            state_components.extend([
                self.current_task['user_location'][1] / 90.0,     # Normalized user latitude
                self.current_task['user_location'][0] / 180.0,    # Normalized user longitude
                self.current_task['quality_preference'],          # Quality preference
                self.current_task['speed_preference'],            # Speed preference
                self.current_task['data_size'] / 1000.0,          # Normalized task size
                self.current_subtask_index / len(self.current_task['route_sequence']),  # Task progress
                self.current_task['required_model_type'] / max(self.model_types),  # Normalized model type
            ])
        else:
            state_components.extend([0.0] * 7)
        
        # 3. Additional components for full state space
        if self.use_full_state_space:
            # 3a. Model type availability (n_model_types dimensions)
            for model_type in self.model_types:
                available_count = 0
                if self.current_task and model_type == self.current_task['required_model_type']:
                    # Count available models of the required type
                    for server_idx in range(self.n_servers):
                        if server_idx in self.server_to_models:
                            for model in self.server_to_models[server_idx]:
                                if int(model['ModelType']) == model_type and self.servers_computation_load[server_idx] < 0.8:
                                    available_count += 1
                state_components.append(available_count / 10.0)  # Normalize
            
            # 3b. Network topology matrix flattened (n_servers * n_servers)
            state_components.extend(self.topology_matrix.flatten())
        
        return np.array(state_components, dtype=np.float32)
    
    def get_agent_state(self, agent_id: str) -> np.ndarray:
        """
        Get local state for a specific agent in multi-agent mode
        
        Args:
            agent_id: Agent identifier (e.g., 'Agent_0')
            
        Returns:
            Local state vector for the agent
        """
        if not self.multi_agent_mode:
            # Fallback to global state for single agent mode
            return self.get_current_state()
        
        if agent_id not in self.server_allocation:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        
        state_components = []
        managed_servers = self.server_allocation[agent_id]
        
        # 1. Managed server states (n_managed_servers * 4)
        for server_name in managed_servers:
            server_idx = int(server_name.split('_')[1])  # Extract index from "Server_X"
            server_info = self.servers_df.iloc[server_idx]
            state_components.extend([
                self.servers_computation_load[server_idx],  # Computation load
                self.servers_storage_load[server_idx],      # Storage load
                server_info['Longitude'] / 180.0,          # Normalized longitude
                server_info['Latitude'] / 90.0,            # Normalized latitude
            ])
        
        # 2. Current task information (7 dimensions)
        if self.current_task:
            state_components.extend([
                self.current_task['user_location'][1] / 90.0,     # Normalized user latitude
                self.current_task['user_location'][0] / 180.0,    # Normalized user longitude
                self.current_task['quality_preference'],          # Quality preference
                self.current_task['speed_preference'],            # Speed preference
                self.current_task['data_size'] / 1000.0,          # Normalized task size
                self.current_subtask_index / len(self.current_task['route_sequence']) if self.current_task['route_sequence'] else 0.0,
                self.current_task['required_model_type'] / max(self.model_types) if self.model_types else 0.0,
            ])
        else:
            state_components.extend([0.0] * 7)
        
        # 3. Other agents summary information ((n_agents - 1) * 3)
        for other_agent_id in [f'Agent_{i}' for i in range(self.n_agents)]:
            if other_agent_id != agent_id:
                # Calculate other agent's summary metrics
                other_managed_servers = self.server_allocation[other_agent_id]
                
                # Average load of other agent's servers
                avg_load = 0.0
                availability = 0.0
                if other_managed_servers:
                    loads = []
                    available_count = 0
                    for server_name in other_managed_servers:
                        server_idx = int(server_name.split('_')[1])
                        loads.append(self.servers_computation_load[server_idx])
                        if self.servers_computation_load[server_idx] < 0.8:
                            available_count += 1
                    avg_load = np.mean(loads)
                    availability = available_count / len(other_managed_servers)
                
                # Performance proxy (inverse of load)
                performance = max(0.0, 1.0 - avg_load)
                
                state_components.extend([avg_load, availability, performance])
        
        # 4. Communication/coordination information (16 dimensions)
        # This represents information exchange and coordination signals
        coordination_info = self._get_coordination_info(agent_id)
        state_components.extend(coordination_info)
        
        return np.array(state_components, dtype=np.float32)
    
    def _get_coordination_info(self, agent_id: str) -> List[float]:
        """
        Generate coordination information for multi-agent communication
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of coordination features (16 dimensions)
        """
        coordination_features = []
        
        # Global system metrics (4 dimensions)
        coordination_features.extend([
            np.mean(self.servers_computation_load),     # Global average load
            np.std(self.servers_computation_load),      # Global load variance
            len(self.episode_rewards) / 100.0,          # Episode progress proxy
            self.current_step / self.max_episode_steps,  # Step progress
        ])
        
        # Task-related coordination (4 dimensions)
        if self.current_task:
            coordination_features.extend([
                self.current_task['quality_preference'],
                self.current_task['speed_preference'],
                self.current_task['data_size'] / 1000.0,
                self.current_subtask_index / len(self.current_task['route_sequence']) if self.current_task['route_sequence'] else 0.0,
            ])
        else:
            coordination_features.extend([0.0] * 4)
        
        # Agent-specific coordination signals (8 dimensions)
        managed_servers = self.server_allocation[agent_id]
        
        # Local metrics
        local_avg_load = 0.0
        local_capacity = 0.0
        if managed_servers:
            loads = []
            capacities = []
            for server_name in managed_servers:
                server_idx = int(server_name.split('_')[1])
                loads.append(self.servers_computation_load[server_idx])
                capacities.append(self.servers_df.iloc[server_idx]['ComputationCapacity'])
            local_avg_load = np.mean(loads)
            local_capacity = np.sum(capacities)
        
        # Coordination signals
        coordination_features.extend([
            local_avg_load,                             # Local average load
            local_capacity / 1000.0,                    # Normalized local capacity
            len(managed_servers) / self.n_servers,      # Server allocation ratio
            0.0,  # Reserved for future use
            0.0,  # Reserved for future use
            0.0,  # Reserved for future use
            0.0,  # Reserved for future use
            0.0,  # Reserved for future use
        ])
        
        return coordination_features
    
    def is_valid_action(self, server_idx: int, model_idx: int) -> bool:
        """Check if action is valid"""
        # Check if current task exists
        if not self.current_task:
            return False
        
        # Check server index
        if not (0 <= server_idx < self.n_servers):
            return False
        
        # Check if server has models
        if server_idx not in self.server_to_models:
            return False
        
        # Check model index
        if not (0 <= model_idx < len(self.server_to_models[server_idx])):
            return False
        
        # Check if model type matches requirement
        selected_model = self.server_to_models[server_idx][model_idx]
        required_model_type = self.current_task.get('required_model_type')
        
        if required_model_type is not None and int(selected_model['ModelType']) != required_model_type:
            return False
        
        # Check server load constraint
        if self.servers_computation_load is not None and self.servers_computation_load[server_idx] > 0.8:
            return False
        
        return True
    
    def calculate_reward(self, server_idx: int, selected_model: pd.Series) -> float:
        """
        Calculate reward for current action
        
        Args:
            server_idx: Selected server index
            selected_model: Selected model information
            
        Returns:
            reward: Calculated reward value
        """
        # Check if current task exists
        if not self.current_task:
            return 0.0
        
        # Get basic metrics
        quality = self._calculate_quality_factor(selected_model)
        speed = self._calculate_speed_factor(server_idx, selected_model)
        
        # Simple linear weighted sum
        # This simplifies the complex utility function described in the paper
        w1 = self.current_task.get('quality_preference', 0.5)
        w2 = self.current_task.get('speed_preference', 0.5)
        
        reward = w1 * quality + w2 * speed
        
        return float(reward)
    
    def _calculate_quality_factor(self, selected_model: pd.Series) -> float:
        """Calculate model quality factor"""
        arena_score = float(selected_model['ArenaScore'])
        return arena_score  # ArenaScore is already in [0, 1] range
    
    def _calculate_speed_factor(self, server_idx: int, selected_model: pd.Series) -> float:
        """Calculate speed factor"""
        # Check if current task exists
        if not self.current_task:
            return 0.5  # Default speed factor
        
        # Calculate processing time
        data_size = self.current_task.get('data_size', 1.0)
        computation_time = data_size / self.servers_df.iloc[server_idx]['ComputationCapacity']
        
        # Use ArenaScore as proxy for inference speed (higher score = faster inference)
        try:
            arena_score = float(selected_model['ArenaScore'])
            inference_time = (1.0 - arena_score) * 2.0  # Convert to reasonable inference time range
        except KeyError:
            inference_time = 1.0  # Default inference time
        
        # Calculate transmission time
        user_pos = self.current_task.get('user_location', [0.0, 0.0])
        server_pos = [self.servers_df.iloc[server_idx]['Latitude'], self.servers_df.iloc[server_idx]['Longitude']]
        distance = self.calculate_distance(user_pos, server_pos)
        transmission_time = distance * 0.1 + data_size * 0.01
        
        total_time = computation_time + inference_time + transmission_time
        
        # Convert to speed factor (smaller time = higher factor)
        speed_factor = max(0.1, 1.0 - total_time / self.max_expected_time)
        return speed_factor
    
    def _calculate_load_balance_factor(self, server_idx: int) -> float:
        """Calculate load balance factor"""
        if self.servers_computation_load is None:
            return 1.0
        
        current_load = self.servers_computation_load[server_idx]
        avg_load = np.mean(self.servers_computation_load)
        
        # Prefer servers with lower load
        if avg_load > 0:
            balance_factor = max(0.1, 1.0 - (current_load - avg_load) / avg_load)
        else:
            balance_factor = 1.0
        
        return balance_factor
    
    def _calculate_resource_efficiency_factor(self, server_idx: int) -> float:
        """Calculate resource efficiency factor"""
        if not self.current_task:
            return 0.5
        
        server_info = self.servers_df.iloc[server_idx]
        
        # Consider computation and storage efficiency
        comp_efficiency = 1.0 - self.servers_computation_load[server_idx]
        storage_efficiency = 1.0 - self.servers_storage_load[server_idx]
        
        # Consider server capacity utilization
        data_size = self.current_task.get('data_size', 1.0)
        capacity_utilization = min(1.0, data_size / server_info['ComputationCapacity'])
        
        efficiency = (comp_efficiency + storage_efficiency + capacity_utilization) / 3.0
        return max(0.1, efficiency)
    
    def _calculate_geographical_factor(self, server_idx: int) -> float:
        """Calculate geographical proximity factor"""
        if not self.current_task:
            return 0.5
        
        user_pos = self.current_task.get('user_location', [0.0, 0.0])
        server_pos = [self.servers_df.iloc[server_idx]['Latitude'], self.servers_df.iloc[server_idx]['Longitude']]
        distance = self.calculate_distance(user_pos, server_pos)
        
        # Prefer closer servers
        max_distance = 1000.0  # km
        geo_factor = max(0.1, 1.0 - distance / max_distance)
        return geo_factor
    
    def _apply_reward_shaping(self, base_reward: float, server_idx: int, selected_model: pd.Series) -> float:
        """Apply reward shaping techniques"""
        shaped_reward = base_reward
        
        # Progress bonus
        if hasattr(self, 'current_subtask_index'):
            progress_bonus = self.current_subtask_index * 0.5
            shaped_reward += progress_bonus
        
        # Diversity bonus
        if hasattr(self, 'previous_server_selections'):
            if server_idx not in self.previous_server_selections[-3:]:  # Recent 3 selections
                shaped_reward += 1.0  # Diversity bonus
        else:
            self.previous_server_selections = []
        self.previous_server_selections.append(server_idx)
        
        # Early completion bonus
        if self.current_step < self.max_episode_steps * 0.7:
            shaped_reward += 2.0
        
        return shaped_reward
    
    def _apply_gradient_enhancement(self, shaped_reward: float, server_idx: int) -> float:
        """Apply gradient enhancement for accelerated learning"""
        enhanced_reward = shaped_reward
        
        # Add gradient signals for better learning
        if self.episode_rewards and len(self.episode_rewards) > 0:
            # Improvement bonus
            if shaped_reward > np.mean(self.episode_rewards):
                enhanced_reward += 1.0  # Improvement bonus
            
            # Consistency bonus
            if len(self.episode_rewards) >= 3:
                recent_rewards = self.episode_rewards[-3:]
                if np.std(recent_rewards) < 0.5:  # Stable performance
                    enhanced_reward += 0.5
        
        # Load balancing gradient
        if self.servers_computation_load is not None:
            load_variance = np.var(self.servers_computation_load)
            if load_variance < 0.1:  # Good load balancing
                enhanced_reward += 1.5
        
        return enhanced_reward
    
    def calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate geographical distance between two points (km)"""
        lat1, lon1 = pos1[0], pos1[1]
        lat2, lon2 = pos2[0], pos2[1]
        
        # Use Haversine formula
        R = 6371  # Earth radius (km)
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def release_all_resources(self):
        """Release all occupied resources"""
        self.servers_computation_load *= 0.7  # Retain 30% base load
        self.servers_storage_load *= 0.8      # Retain 20% base load
    
    def render(self, mode='human'):
        """Render environment state (optional implementation)"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Current Task: Subtask {self.current_subtask_index + 1}/{len(self.current_task['route_sequence'])}")
            print(f"Required Model Type: {self.current_task['required_model_type']}")
            print(f"Server Loads: {self.servers_computation_load[:5]}")  # Show only first 5
            print("---")
    
    def close(self):
        """Close environment"""
        pass

    def get_evaluation_metrics(self):
        """Get evaluation metrics"""
        if not self.episode_rewards:
            return {}
        
        # Basic safety checks
        if self.servers_computation_load is None:
            return {'error': 'Environment not properly initialized'}
        
        metrics = {
            'average_reward': np.mean(self.episode_rewards),
            'total_reward': sum(self.episode_rewards),
            'average_computation_load': np.mean(self.servers_computation_load),
            'max_computation_load': np.max(self.servers_computation_load),
            'load_balance_score': 1.0 - np.std(self.servers_computation_load),
        }
        
        # Add task completion rate if current task exists
        if self.current_task and 'route_sequence' in self.current_task:
            metrics['task_completion_rate'] = self.current_subtask_index / len(self.current_task['route_sequence'])
        else:
            metrics['task_completion_rate'] = 0.0
        
        return metrics 