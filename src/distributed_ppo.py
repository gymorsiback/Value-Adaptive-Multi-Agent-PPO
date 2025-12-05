"""
Distributed Multi-Modal Model Orchestration PPO Value Adaptive Algorithm Implementation
Extended for VAMAPPO (Value-Adaptive Multi-Agent PPO)
Enhanced with adaptive value function learning and dynamic optimization strategies
Supports both single-agent and multi-agent modes
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import sys
import os
from typing import Tuple, List, Dict, Any, Optional
from collections import defaultdict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distributed_network import (
    DistributedActorNetwork, 
    AdaptiveValueCriticNetwork, 
    DistributedPolicyNetwork,
    VAMAPPOAgentNetwork,
    VAMAPPOMultiAgentSystem,
    CoordinationModule
)

class DistributedPPOValueAdaptive:
    """
    Distributed Multi-Modal Model Orchestration PPO Value Adaptive Algorithm
    Incorporates adaptive value function learning and dynamic optimization
    """
    
    def __init__(self, env, **hyperparameters):
        """
        Initialize PPO Value Adaptive model
        
        Args:
            env: Distributed environment instance
            hyperparameters: Hyperparameter dictionary
        """
        # Initialize hyperparameters
        self._init_hyperparameters(hyperparameters)
        
        # Set device
        device_input = hyperparameters.get('device', torch.device('cpu'))
        if isinstance(device_input, tuple):
            self.device = device_input[0]  # Extract device from tuple if needed
        else:
            self.device = device_input
        print(f"PPO Value Adaptive model will use device: {self.device}")
        
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.n_servers = env.n_servers
        self.max_models_per_server = env.max_models_per_server
        
        print(f"PPO Value Adaptive initialization - Obs dim: {self.obs_dim}, Servers: {self.n_servers}, Max models: {self.max_models_per_server}")
        
        # Create enhanced Actor and Adaptive Critic networks
        self.actor = DistributedActorNetwork(
            self.obs_dim, 
            self.n_servers, 
            self.max_models_per_server
        ).to(self.device)
        
        self.critic = AdaptiveValueCriticNetwork(self.obs_dim).to(self.device)
        
        # Create adaptive optimizers with dynamic learning rates
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        # Value adaptive specific parameters
        self.base_lr = self.lr
        self.adaptive_lr_factor = 1.0
        self.current_adaptive_lr = 0.5  # Initialize current adaptive LR
        self.value_uncertainty_threshold = 0.5
        self.exploration_decay = 0.995
        self.current_exploration_factor = 1.0
        
        # Enhanced learning rate schedulers
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(
            self.actor_optim, step_size=50, gamma=0.95
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optim, mode='min', factor=0.8, patience=10
        )
        
        # Create save directory
        self.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'results'
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Enhanced logger for value adaptive metrics
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rews': [],
            'actor_losses': [],
            'critic_losses': [],
            'value_errors': [],
            'policy_entropy': [],
            'total_rewards': [],
            'completion_rates': [],
            'load_balance_scores': [],
            'episode_lengths': [],
            # Value adaptive specific metrics
            'value_uncertainties': [],
            'adaptive_lr_factors': [],
            'td_errors': [],
            'exploration_factors': [],
            'value_adaptation_rates': []
        }
    
    def learn(self, total_timesteps, analyzer=None):
        """
        Train Actor and Critic networks with value adaptive enhancements
        
        Args:
            total_timesteps: Total training timesteps
            analyzer: Optional training analyzer for metrics collection
        """
        self._training_mode = True
        print(f"Starting PPO Value Adaptive learning... Max {self.max_timesteps_per_episode} steps per episode")
        print(f"{self.timesteps_per_batch} steps per batch, {total_timesteps} total steps")
        
        t_so_far = 0
        i_so_far = 0
        
        while t_so_far < total_timesteps:
            # Collect batch data with adaptive exploration
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_info = self.rollout()
            
            # Calculate timesteps collected in this batch
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            
            # Update logger
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far
            
            # Adaptive value function learning
            self._adaptive_value_learning(batch_obs, batch_rtgs, i_so_far)
            
            # Calculate initial advantage function with uncertainty weighting (detached)
            with torch.no_grad():
                V_initial, uncertainties_initial, _ = self.critic(batch_obs)
                V_initial = V_initial.squeeze()
                uncertainties_initial = uncertainties_initial.squeeze()
                
                # Uncertainty-weighted advantage calculation
                A_k_base = batch_rtgs - V_initial
                uncertainty_weights = torch.clamp(1.0 / (uncertainties_initial + 1e-8), 0.1, 3.0)
                A_k_base = A_k_base * uncertainty_weights
                
                # Normalize advantage function
                A_k_mean = A_k_base.mean()
                A_k_std = A_k_base.std()
                A_k_normalized = (A_k_base - A_k_mean) / (A_k_std + 1e-10)
            
            # Update networks with adaptive learning
            for update_idx in range(self.n_updates_per_iteration):
                # Calculate current policy values and log probabilities with fresh forward pass
                with torch.no_grad():
                    # Get fresh values for adaptation factors (no gradients needed for these)
                    _, uncertainties_ref, adaptation_factors_ref = self.critic(batch_obs)
                    adaptation_factors_ref = adaptation_factors_ref.squeeze()
                    adaptive_clip = self.clip * adaptation_factors_ref
                
                # Actor forward pass
                curr_log_probs = self.actor.get_log_prob(batch_obs, batch_acts)
                
                # Calculate ratios with exploration bonus - avoid inplace operations
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                exploration_bonus = self.current_exploration_factor * 0.01
                ratios = ratios + exploration_bonus
                
                # Calculate surrogate loss with adaptive clipping - avoid inplace operations
                surr1 = ratios * A_k_normalized
                ratios_clipped = torch.clamp(ratios, 1 - adaptive_clip, 1 + adaptive_clip)
                surr2 = ratios_clipped * A_k_normalized
                
                # Actor loss with enhanced entropy regularization
                actor_loss = (-torch.min(surr1, surr2)).mean()
                
                # Calculate enhanced policy entropy
                server_logits, model_logits, enhancement_weights = self.actor(batch_obs)
                server_dist = Categorical(logits=server_logits)
                model_dist = Categorical(logits=model_logits)
                entropy = (server_dist.entropy() + model_dist.entropy()).mean()
                
                # Adaptive entropy coefficient based on exploration factor
                adaptive_ent_coef = self.ent_coef * self.current_exploration_factor
                actor_loss = actor_loss - adaptive_ent_coef * entropy
                
                # Update Actor with gradient clipping
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                
                # Critic forward pass (separate from actor)
                V_critic, uncertainties_critic, adaptation_factors_critic = self.critic(batch_obs)
                V_critic = V_critic.squeeze()
                uncertainties_critic = uncertainties_critic.squeeze()
                
                # Enhanced critic loss with uncertainty regularization
                value_pred = V_critic
                value_target = batch_rtgs
                
                # Uncertainty-aware value function loss
                prediction_error = (value_pred - value_target).pow(2)
                uncertainty_penalty = uncertainties_critic.mean() * 0.1
                
                # Adaptive value function clipping - avoid inplace operations
                adaptive_value_clip = self.value_clip * adaptation_factors_critic.squeeze()
                value_diff = value_pred - value_target
                value_diff_clipped = torch.clamp(value_diff, -adaptive_value_clip, adaptive_value_clip)
                value_pred_clipped = value_pred + value_diff_clipped
                
                value_loss = torch.max(
                    prediction_error,
                    (value_pred_clipped - value_target).pow(2)
                ).mean()
                
                # Enhanced value regularization
                value_reg = self.value_reg_coef * prediction_error.mean()
                critic_loss = value_loss + value_reg + uncertainty_penalty
                
                # Adaptive learning rate adjustment
                current_adaptive_factor = adaptation_factors_critic.mean().item()
                self._update_adaptive_learning_rates(current_adaptive_factor, uncertainties_critic.mean().item())
                
                # Update Critic with adaptive learning
                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()
                
                # Update critic's historical data
                with torch.no_grad():
                    self.critic.update_history(value_pred.detach(), uncertainties_critic.detach())
                
                # Record enhanced metrics
                self.logger['actor_losses'].append(actor_loss.detach().item())
                self.logger['critic_losses'].append(critic_loss.detach().item())
                self.logger['value_errors'].append((value_pred - value_target).abs().mean().detach().item())
                self.logger['policy_entropy'].append(entropy.detach().item())
                self.logger['value_uncertainties'].append(uncertainties_critic.mean().detach().item())
                # Record the actual adaptive LR factor computed in _adaptive_value_learning
                self.logger['adaptive_lr_factors'].append(getattr(self, 'current_adaptive_lr', 0.5))
                self.logger['exploration_factors'].append(self.current_exploration_factor)
            
            # Update learning rate schedulers
            self.actor_scheduler.step()
            avg_value_error = np.mean(self.logger['value_errors'][-self.n_updates_per_iteration:])
            self.critic_scheduler.step(avg_value_error)
            
            # Decay exploration factor
            self.current_exploration_factor *= self.exploration_decay
            self.current_exploration_factor = max(self.current_exploration_factor, 0.1)
            
            # Record batch information with enhanced metrics
            if batch_info:
                avg_completion_rate = np.mean([info.get('task_completion_rate', 0) for info in batch_info])
                avg_load_balance = np.mean([info.get('load_balance_score', 0) for info in batch_info])
                avg_episode_reward = np.mean(self.logger['batch_rews'][-len(batch_lens):]) if self.logger['batch_rews'] else 0
                
                self.logger['completion_rates'].append(avg_completion_rate)
                self.logger['load_balance_scores'].append(avg_load_balance)
                self.logger['total_rewards'].append(avg_episode_reward)
                self.logger['episode_lengths'].append(np.mean(batch_lens))
                
                # Update analyzer if provided
                if analyzer:
                    enhanced_metrics = {
                        'total_rewards': avg_episode_reward,
                        'completion_rates': avg_completion_rate,
                        'load_balance_scores': avg_load_balance,
                        'value_uncertainties': self.logger['value_uncertainties'][-1] if self.logger['value_uncertainties'] else 0,
                        'adaptive_lr_factors': self.logger['adaptive_lr_factors'][-1] if self.logger['adaptive_lr_factors'] else 0.5,
                        'exploration_factors': self.current_exploration_factor,
                        # Add loss data
                        'actor_losses': self.logger['actor_losses'][-1] if self.logger['actor_losses'] else 0,
                        'critic_losses': self.logger['critic_losses'][-1] if self.logger['critic_losses'] else 0,
                        'episode_lengths': np.mean(batch_lens)
                    }
                    analyzer.update_metrics(enhanced_metrics)
            
            # Enhanced logging
            if i_so_far % self.save_freq == 0:
                self._log_summary()
        
        print("PPO Value Adaptive training completed!")
    
    def _adaptive_value_learning(self, batch_obs, batch_rtgs, iteration):
        """
        Implement adaptive value function learning with enhanced smoothing
        
        Args:
            batch_obs: Batch observations
            batch_rtgs: Batch returns to go
            iteration: Current iteration number
        """
        # Calculate adaptive learning rate factor with smoothing
        raw_adaptive_lr_factor = self.critic.get_adaptive_lr_factor()
        
        # Apply additional smoothing at the PPO level
        if not hasattr(self, 'ppo_lr_smoothing_buffer'):
            self.ppo_lr_smoothing_buffer = []
            self.ppo_lr_smoothing_size = 5  # Buffer size for smoothing
        
        # Add to smoothing buffer
        lr_value = raw_adaptive_lr_factor.item() if torch.is_tensor(raw_adaptive_lr_factor) else raw_adaptive_lr_factor
        self.ppo_lr_smoothing_buffer.append(lr_value)
        
        # Keep buffer size limited
        if len(self.ppo_lr_smoothing_buffer) > self.ppo_lr_smoothing_size:
            self.ppo_lr_smoothing_buffer.pop(0)
        
        # Calculate smoothed adaptive LR factor
        if len(self.ppo_lr_smoothing_buffer) >= 2:
            # Use weighted average with more weight on recent values
            weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5][-len(self.ppo_lr_smoothing_buffer):])
            weights = weights / weights.sum()  # Normalize weights
            smoothed_lr = sum(w * v for w, v in zip(weights, self.ppo_lr_smoothing_buffer))
            self.adaptive_lr_factor = torch.tensor(smoothed_lr, dtype=torch.float32)
        else:
            self.adaptive_lr_factor = raw_adaptive_lr_factor
        
        # Apply rate limiting to prevent sudden changes
        if hasattr(self, 'prev_adaptive_lr'):
            max_change = 0.1  # Maximum change per iteration (10%)
            lr_change = abs(self.adaptive_lr_factor.item() - self.prev_adaptive_lr)
            if lr_change > max_change:
                # Limit the change
                direction = 1 if self.adaptive_lr_factor.item() > self.prev_adaptive_lr else -1
                self.adaptive_lr_factor = torch.tensor(self.prev_adaptive_lr + direction * max_change, dtype=torch.float32)
        
        # Store current LR for next iteration
        self.prev_adaptive_lr = self.adaptive_lr_factor.item()
        
        # Update critic learning rate based on smoothed adaptation factor
        current_lr = self.base_lr * self.adaptive_lr_factor
        # Also apply minimum and maximum LR bounds
        current_lr = torch.clamp(current_lr, min=1e-5, max=1e-2)
        
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = current_lr.item()
        
        # Log adaptation rate - store for use in updates
        adaptive_lr_value = self.adaptive_lr_factor.item()
        self.current_adaptive_lr = adaptive_lr_value  # Store for use in training loop
        self.logger['value_adaptation_rates'].append(adaptive_lr_value)
        
        # Update critic history with current batch statistics
        with torch.no_grad():
            current_values, current_uncertainties, _ = self.critic(batch_obs)
            self.critic.update_history(current_values.mean(), current_uncertainties.mean())
        
        # Periodic value function validation
        if iteration % 10 == 0:
            self._validate_value_function(batch_obs, batch_rtgs)
    
    def _validate_value_function(self, batch_obs, batch_rtgs):
        """
        Validate value function accuracy and adjust if necessary
        
        Args:
            batch_obs: Batch observations
            batch_rtgs: Batch returns to go
        """
        with torch.no_grad():
            predictions, uncertainties, _ = self.critic(batch_obs)
            predictions = predictions.squeeze()
            
            # Calculate validation metrics
            mse = ((predictions - batch_rtgs) ** 2).mean()
            mae = (predictions - batch_rtgs).abs().mean()
            avg_uncertainty = uncertainties.mean()
            
            # Log validation metrics
            if len(self.logger['td_errors']) < 1000:  # Limit log size
                self.logger['td_errors'].append(mse.item())
    
    def _update_adaptive_learning_rates(self, adaptation_factor, uncertainty_level):
        """
        Update learning rates based on adaptation factor and uncertainty with smoothing
        
        Args:
            adaptation_factor: Current adaptation factor
            uncertainty_level: Current uncertainty level
        """
        # Calculate actor learning rate factor based on uncertainty (more conservative)
        if uncertainty_level > self.value_uncertainty_threshold:
            raw_actor_lr_factor = 1.1  # Slightly increase learning rate when uncertain
        else:
            raw_actor_lr_factor = 0.95  # Slightly decrease learning rate when confident
        
        # Apply smoothing to actor LR factor
        if not hasattr(self, 'actor_lr_smoothing_buffer'):
            self.actor_lr_smoothing_buffer = []
            self.actor_lr_smoothing_size = 3  # Smaller buffer for actor
        
        # Add to smoothing buffer
        self.actor_lr_smoothing_buffer.append(raw_actor_lr_factor)
        
        # Keep buffer size limited
        if len(self.actor_lr_smoothing_buffer) > self.actor_lr_smoothing_size:
            self.actor_lr_smoothing_buffer.pop(0)
        
        # Calculate smoothed actor LR factor
        if len(self.actor_lr_smoothing_buffer) >= 2:
            smoothed_actor_lr_factor = sum(self.actor_lr_smoothing_buffer) / len(self.actor_lr_smoothing_buffer)
        else:
            smoothed_actor_lr_factor = raw_actor_lr_factor
        
        # Apply rate limiting for actor LR
        if hasattr(self, 'prev_actor_lr_factor'):
            max_actor_change = 0.05  # Maximum change per iteration (5%)
            actor_lr_change = abs(smoothed_actor_lr_factor - self.prev_actor_lr_factor)
            if actor_lr_change > max_actor_change:
                direction = 1 if smoothed_actor_lr_factor > self.prev_actor_lr_factor else -1
                smoothed_actor_lr_factor = self.prev_actor_lr_factor + direction * max_actor_change
        
        # Store current actor LR factor for next iteration
        self.prev_actor_lr_factor = smoothed_actor_lr_factor
        
        # Update actor learning rate with bounds
        actor_lr = self.base_lr * smoothed_actor_lr_factor * adaptation_factor
        actor_lr = max(min(actor_lr, 5e-3), 1e-6)  # Clamp between 1e-6 and 5e-3
        
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = actor_lr
    
    def rollout(self):
        """
        Enhanced rollout with adaptive exploration and uncertainty tracking
        """
        # Batch data storage
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_info = []
        
        # Episode tracking
        ep_rews = []
        
        t = 0  # Tracks when we reach timesteps_per_batch timesteps
        
        while t < self.timesteps_per_batch:
            ep_rews = []  # Episode rewards
            ep_info = []  # Episode information
            
            # Reset environment
            obs, info = self.env.reset()
            ep_info.append(info)
            done = False
            
            # Episode rollout
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1  # Increment global timestep counter
                
                # Collect observation
                batch_obs.append(obs)
                
                # Get action with enhanced exploration
                action, log_prob, enhancement_weights = self.get_action_with_exploration(obs)
                
                # Execute action
                obs, rew, done, truncated, info = self.env.step(action)
                
                # Collect experience
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                ep_rews.append(rew)
                ep_info.append(info)
                
                # Check termination
                if done or truncated:
                    break
            
            # Collect episode data
            batch_lens.append(ep_t + 1)
            batch_rews.extend(ep_rews)
            batch_info.extend(ep_info)
        
        # Convert to tensors
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32, device=self.device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.long, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32, device=self.device)
        
        # Calculate returns-to-go with adaptive discounting
        batch_rtgs = self.compute_adaptive_rtgs(batch_rews, batch_lens)
        
        # Update logger
        self.logger['batch_lens'].extend(batch_lens)
        self.logger['batch_rews'].extend(batch_rews)
        
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_info
    
    def compute_adaptive_rtgs(self, batch_rews, batch_lens):
        """
        Compute returns-to-go with adaptive discounting based on uncertainty
        
        Args:
            batch_rews: Batch rewards
            batch_lens: Batch episode lengths
            
        Returns:
            batch_rtgs: Batch returns-to-go
        """
        batch_rtgs = []
        
        for ep_rews in np.split(batch_rews, np.cumsum(batch_lens)[:-1]):
            discounted_rewards = []
            
            # Adaptive discount factor based on episode performance
            ep_mean_reward = np.mean(ep_rews)
            adaptive_gamma = self.gamma
            
            if ep_mean_reward > 0:
                adaptive_gamma = min(self.gamma * 1.05, 0.99)  # Increase gamma for good episodes
            else:
                adaptive_gamma = max(self.gamma * 0.95, 0.85)  # Decrease gamma for poor episodes
            
            # Calculate discounted rewards
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + adaptive_gamma * discounted_reward
                discounted_rewards.insert(0, discounted_reward)
            
            batch_rtgs.extend(discounted_rewards)
        
        return torch.tensor(batch_rtgs, dtype=torch.float32, device=self.device)
    
    def get_action_with_exploration(self, obs):
        """
        Get action with enhanced exploration based on value uncertainty
        
        Args:
            obs: Current observation
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            enhancement_weights: Policy enhancement weights
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get value uncertainty for exploration adjustment
        with torch.no_grad():
            _, uncertainty, _ = self.critic(obs_tensor)
            # Handle both single values and batch tensors
            if uncertainty.numel() == 1:
                uncertainty_factor = uncertainty.item()
            else:
                uncertainty_factor = uncertainty.mean().item()
        
        # Adjust exploration based on uncertainty
        if uncertainty_factor > self.value_uncertainty_threshold:
            # Higher uncertainty -> more exploration
            exploration_bonus = 0.1 * self.current_exploration_factor
        else:
            # Lower uncertainty -> less exploration
            exploration_bonus = 0.01 * self.current_exploration_factor
        
        # Get action with exploration bonus
        action, log_prob, enhancement_weights = self.actor.get_action_and_log_prob(obs_tensor)
        
        # Apply exploration bonus to log probability
        log_prob = log_prob + exploration_bonus
        
        return action.cpu().numpy().flatten(), log_prob.item(), enhancement_weights
    
    def get_action(self, obs):
        """
        Get action for inference (without exploration)
        
        Args:
            obs: Current observation
            
        Returns:
            action: Selected action
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            action, _, _ = self.actor.get_action_and_log_prob(obs_tensor)
        
        return action.cpu().numpy().flatten()
    
    def evaluate(self, batch_obs, batch_acts):
        """
        Evaluate batch of observations and actions with uncertainty
        
        Args:
            batch_obs: Batch observations
            batch_acts: Batch actions
            
        Returns:
            values: Value estimates
            log_probs: Log probabilities
            uncertainties: Value uncertainties
        """
        values, uncertainties, _ = self.critic(batch_obs)
        log_probs = self.actor.get_log_prob(batch_obs, batch_acts)
        
        return values.squeeze(), log_probs, uncertainties.squeeze()
    
    def save_models(self, suffix=''):
        """
        Save actor and critic models with enhanced metadata
        
        Args:
            suffix: Filename suffix
        """
        actor_path = os.path.join(self.save_dir, f'ppo_value_adaptive_actor{suffix}.pth')
        critic_path = os.path.join(self.save_dir, f'ppo_value_adaptive_critic{suffix}.pth')
        
        # Save models with training metadata
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optim.state_dict(),
            'scheduler_state_dict': self.actor_scheduler.state_dict(),
            'exploration_factor': self.current_exploration_factor,
            'iteration': self.logger['i_so_far']
        }, actor_path)
        
        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optim.state_dict(),
            'scheduler_state_dict': self.critic_scheduler.state_dict(),
            'adaptive_lr_factor': self.adaptive_lr_factor,
            'value_history': self.critic.value_history,
            'uncertainty_history': self.critic.uncertainty_history,
            'iteration': self.logger['i_so_far']
        }, critic_path)
        
        print(f"PPO Value Adaptive models saved: {actor_path}, {critic_path}")
    
    def load_models(self, suffix=''):
        """
        Load actor and critic models with metadata restoration
        
        Args:
            suffix: Filename suffix
        """
        actor_path = os.path.join(self.save_dir, f'ppo_value_adaptive_actor{suffix}.pth')
        critic_path = os.path.join(self.save_dir, f'ppo_value_adaptive_critic{suffix}.pth')
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            # Load actor
            actor_checkpoint = torch.load(actor_path, map_location=self.device)
            self.actor.load_state_dict(actor_checkpoint['model_state_dict'])
            self.actor_optim.load_state_dict(actor_checkpoint['optimizer_state_dict'])
            self.actor_scheduler.load_state_dict(actor_checkpoint['scheduler_state_dict'])
            self.current_exploration_factor = actor_checkpoint.get('exploration_factor', 1.0)
            
            # Load critic
            critic_checkpoint = torch.load(critic_path, map_location=self.device)
            self.critic.load_state_dict(critic_checkpoint['model_state_dict'])
            self.critic_optim.load_state_dict(critic_checkpoint['optimizer_state_dict'])
            self.critic_scheduler.load_state_dict(critic_checkpoint['scheduler_state_dict'])
            self.adaptive_lr_factor = critic_checkpoint.get('adaptive_lr_factor', 1.0)
            
            # Restore critic history
            if 'value_history' in critic_checkpoint:
                self.critic.value_history = critic_checkpoint['value_history']
            if 'uncertainty_history' in critic_checkpoint:
                self.critic.uncertainty_history = critic_checkpoint['uncertainty_history']
            
            print(f"PPO Value Adaptive models loaded: {actor_path}, {critic_path}")
        else:
            print(f"Model files not found: {actor_path}, {critic_path}")
    
    def plot_training_curves(self):
        """
        Plot enhanced training curves including value adaptive metrics
        """
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('PPO Value Adaptive Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Total Rewards
        if self.logger['total_rewards']:
            axes[0, 0].plot(self.logger['total_rewards'], 'b-', linewidth=2)
            axes[0, 0].set_title('Total Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # Plot 2: Actor and Critic Losses
        if self.logger['actor_losses'] and self.logger['critic_losses']:
            axes[0, 1].plot(self.logger['actor_losses'], 'r-', label='Actor Loss', alpha=0.7)
            axes[0, 1].plot(self.logger['critic_losses'], 'b-', label='Critic Loss', alpha=0.7)
            axes[0, 1].set_title('Training Losses')
            axes[0, 1].set_xlabel('Update')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot 3: Value Uncertainties
        if self.logger['value_uncertainties']:
            axes[0, 2].plot(self.logger['value_uncertainties'], 'g-', linewidth=2)
            axes[0, 2].set_title('Value Uncertainties')
            axes[0, 2].set_xlabel('Update')
            axes[0, 2].set_ylabel('Uncertainty')
            axes[0, 2].grid(True)
        
        # Plot 4: Adaptive Learning Rate Factors
        if self.logger['adaptive_lr_factors']:
            axes[1, 0].plot(self.logger['adaptive_lr_factors'], 'orange', linewidth=2)
            axes[1, 0].set_title('Adaptive LR Factors')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('LR Factor')
            axes[1, 0].grid(True)
        
        # Plot 5: Exploration Factors
        if self.logger['exploration_factors']:
            axes[1, 1].plot(self.logger['exploration_factors'], 'purple', linewidth=2)
            axes[1, 1].set_title('Exploration Factors')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Exploration Factor')
            axes[1, 1].grid(True)
        
        # Plot 6: Completion Rates
        if self.logger['completion_rates']:
            axes[1, 2].plot(self.logger['completion_rates'], 'brown', linewidth=2)
            axes[1, 2].set_title('Task Completion Rates')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Completion Rate')
            axes[1, 2].grid(True)
        
        # Plot 7: Value Errors
        if self.logger['value_errors']:
            axes[2, 0].plot(self.logger['value_errors'], 'red', alpha=0.7)
            axes[2, 0].set_title('Value Prediction Errors')
            axes[2, 0].set_xlabel('Update')
            axes[2, 0].set_ylabel('MAE')
            axes[2, 0].grid(True)
        
        # Plot 8: Policy Entropy
        if self.logger['policy_entropy']:
            axes[2, 1].plot(self.logger['policy_entropy'], 'teal', linewidth=2)
            axes[2, 1].set_title('Policy Entropy')
            axes[2, 1].set_xlabel('Update')
            axes[2, 1].set_ylabel('Entropy')
            axes[2, 1].grid(True)
        
        # Plot 9: Value Adaptation Rates
        if self.logger['value_adaptation_rates']:
            axes[2, 2].plot(self.logger['value_adaptation_rates'], 'navy', linewidth=2)
            axes[2, 2].set_title('Value Adaptation Rates')
            axes[2, 2].set_xlabel('Iteration')
            axes[2, 2].set_ylabel('Adaptation Rate')
            axes[2, 2].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'ppo_value_adaptive_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {plot_path}")
    
    def _init_hyperparameters(self, hyperparameters):
        """
        Initialize enhanced hyperparameters for value adaptive learning
        """
        # Standard PPO hyperparameters
        self.timesteps_per_batch = hyperparameters.get('timesteps_per_batch', 512)
        self.max_timesteps_per_episode = hyperparameters.get('max_timesteps_per_episode', 10)
        self.n_updates_per_iteration = hyperparameters.get('n_updates_per_iteration', 2)
        self.lr = hyperparameters.get('lr', 0.0001)
        self.gamma = hyperparameters.get('gamma', 0.95)
        self.clip = hyperparameters.get('clip', 0.1)
        self.ent_coef = hyperparameters.get('ent_coef', 0.01)
        self.value_clip = hyperparameters.get('value_clip', 0.1)
        self.value_reg_coef = hyperparameters.get('value_reg_coef', 1.0)
        self.max_grad_norm = hyperparameters.get('max_grad_norm', 0.5)
        self.save_freq = hyperparameters.get('save_freq', 5)
        
        # Value adaptive specific hyperparameters
        self.adaptive_clip_range = hyperparameters.get('adaptive_clip_range', [0.05, 0.2])
        self.uncertainty_regularization = hyperparameters.get('uncertainty_regularization', 0.1)
        self.exploration_schedule = hyperparameters.get('exploration_schedule', 'exponential')
        self.value_adaptation_frequency = hyperparameters.get('value_adaptation_frequency', 10)
    
    def _log_summary(self):
        """
        Enhanced logging with value adaptive metrics
        """
        print("=" * 60)
        print(f"Iteration: {self.logger['i_so_far']}")
        print(f"Timesteps: {self.logger['t_so_far']}")
        
        if self.logger['total_rewards']:
            avg_reward = np.mean(self.logger['total_rewards'][-10:])
            print(f"Average Reward (last 10): {avg_reward:.4f}")
        
        if self.logger['completion_rates']:
            avg_completion = np.mean(self.logger['completion_rates'][-10:])
            print(f"Average Completion Rate: {avg_completion:.4f}")
        
        if self.logger['value_uncertainties']:
            avg_uncertainty = np.mean(self.logger['value_uncertainties'][-10:])
            print(f"Average Value Uncertainty: {avg_uncertainty:.4f}")
        
        if self.logger['adaptive_lr_factors']:
            current_lr_factor = self.logger['adaptive_lr_factors'][-1]
            print(f"Current Adaptive LR Factor: {current_lr_factor:.4f}")
        
        print(f"Current Exploration Factor: {self.current_exploration_factor:.4f}")
        
        current_lr = self.actor_optim.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
        
        print("=" * 60)

# Maintain backward compatibility
DistributedPPO = DistributedPPOValueAdaptive 


# ============================================================================
# VAMAPPO Multi-Agent Implementation
# ============================================================================

class VAMAPPO:
    """
    VAMAPPO (Value-Adaptive Multi-Agent PPO) Algorithm Implementation
    
    Extends the single-agent PPO Value Adaptive algorithm to support multiple agents
    with coordination mechanisms and distributed learning.
    """
    
    def __init__(self, env, agent_configs=None, **hyperparameters):
        """
        Initialize VAMAPPO Multi-Agent System
        
        Args:
            env: Multi-agent environment instance
            agent_configs: List of agent configurations (auto-generated if None)
            hyperparameters: Hyperparameter dictionary
        """
        # Initialize hyperparameters
        self._init_hyperparameters(hyperparameters)
        
        # Set device
        device_input = hyperparameters.get('device', torch.device('cpu'))
        if isinstance(device_input, tuple):
            self.device = device_input[0]  # Extract device from tuple if needed
        else:
            self.device = device_input
        print(f"VAMAPPO will use device: {self.device}")
        
        # Environment setup
        self.env = env
        self.multi_agent_mode = env.multi_agent_mode
        self.n_agents = env.n_agents if self.multi_agent_mode else 1
        
        if not self.multi_agent_mode:
            raise ValueError("VAMAPPO requires multi_agent_mode=True in environment")
        
        # Agent configuration setup (delayed until first learn call)
        self.agent_configs = agent_configs
        self.multi_agent_system = None
        self.optimizers = {}
        self.schedulers = {}
        self._networks_initialized = False
        
        # Value adaptive specific parameters
        self.base_lr = self.lr
        self.adaptive_lr_factors = {f'Agent_{i}': 1.0 for i in range(self.n_agents)}
        self.current_adaptive_lrs = {f'Agent_{i}': 0.5 for i in range(self.n_agents)}
        self.exploration_factors = {f'Agent_{i}': 1.0 for i in range(self.n_agents)}
        self.exploration_decay = 0.995
        
        # Create save directory
        self.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'results'
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Enhanced logger for multi-agent metrics
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rews': [],
            'agent_losses': defaultdict(list),
            'agent_values': defaultdict(list),
            'agent_uncertainties': defaultdict(list),
            'agent_rewards': defaultdict(list),
            'collaboration_bonuses': [],
            'coordination_efficiency': [],
            'load_balance_scores': [],
            'total_system_reward': [],
            'episode_lengths': [],
            'task_completion_rates': [],
            'communication_costs': [],
            'adaptation_factors': defaultdict(list)
        }
        
        print(f"VAMAPPO initialized with {self.n_agents} agents")
    
    def _initialize_networks(self):
        """Initialize networks after environment setup"""
        if self._networks_initialized:
            return
            
        # Generate agent configurations
        if self.agent_configs is None:
            self.agent_configs = self._generate_agent_configs()
        
        # Create multi-agent system
        self.multi_agent_system = VAMAPPOMultiAgentSystem(self.agent_configs).to(self.device)
        
        # Create optimizers for all agents
        for agent_id in [f'Agent_{i}' for i in range(self.n_agents)]:
            agent_network = self.multi_agent_system.agents[agent_id]
            self.optimizers[agent_id] = Adam(agent_network.parameters(), lr=self.lr)
            self.schedulers[agent_id] = torch.optim.lr_scheduler.StepLR(
                self.optimizers[agent_id], step_size=50, gamma=0.95
            )
        
        self._networks_initialized = True
        print("VAMAPPO networks initialized successfully")
        
    def _generate_agent_configs(self):
        """Generate agent configurations automatically"""
        agent_configs = []
        
        # Get a sample observation to determine actual dimensions
        sample_obs, _ = self.env.reset()
        
        for i in range(self.n_agents):
            agent_id = f'Agent_{i}'
            
            # Get actual local state dimension from sample observation
            if agent_id in sample_obs:
                local_state_dim = len(sample_obs[agent_id])
            else:
                # Fallback: calculate based on environment configuration
                managed_servers = self.env.server_allocation[agent_id]
                n_managed_servers = len(managed_servers)
                local_state_dim = (
                    n_managed_servers * 4 +    # Local server states
                    7 +                        # Task information
                    (self.n_agents - 1) * 3 + # Other agents summary
                    16                         # Communication/coordination info
                )
            
            # Get agent's action space based on managed servers
            managed_servers = self.env.server_allocation[agent_id]
            n_local_servers = len(managed_servers)
            max_models_per_server = self.env.max_models_per_server
            
            print(f"Agent {agent_id} config: local_state_dim={local_state_dim}, n_local_servers={n_local_servers}")
            
            agent_configs.append({
                'agent_id': agent_id,
                'local_state_dim': local_state_dim,
                'n_local_servers': n_local_servers,
                'max_models_per_server': max_models_per_server,
                'coordination_dim': 16
            })
        
        return agent_configs
    
    def learn(self, total_timesteps, analyzer=None):
        """
        Train Multi-Agent System with VAMAPPO
        
        Args:
            total_timesteps: Total training timesteps
            analyzer: Optional training analyzer for metrics collection
        """
        # Initialize networks if not already done
        self._initialize_networks()
        
        self._training_mode = True
        print(f"Starting VAMAPPO learning... Max {self.max_timesteps_per_episode} steps per episode")
        print(f"{self.timesteps_per_batch} steps per batch, {total_timesteps} total steps")
        
        t_so_far = 0
        i_so_far = 0
        
        while t_so_far < total_timesteps:
            # Collect multi-agent batch data
            batch_data = self.multi_agent_rollout()
            
            # Calculate timesteps collected in this batch
            t_so_far += sum(len(batch_data[agent_id]['rewards']) for agent_id in batch_data)
            i_so_far += 1
            
            # Update logger
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far
            
            # Multi-agent learning step
            self._multi_agent_learning_step(batch_data, i_so_far)
            
            # Update exploration factors
            for agent_id in self.exploration_factors:
                self.exploration_factors[agent_id] *= self.exploration_decay
            
            # Update schedulers
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            # Logging and analysis - collect data every iteration
            if analyzer:
                self._collect_analysis_data(batch_data, analyzer)
            
            # Save models periodically
            if i_so_far % self.save_freq == 0:
                self.save_models(suffix=f'_iteration_{i_so_far}')
            
            # Print progress every 5 iterations (more frequent for debugging)
            if i_so_far % 5 == 0:
                self._log_summary()
    
    def multi_agent_rollout(self):
        """
        Collect experience data from multi-agent environment
        
        Returns:
            Dictionary of batch data for each agent
        """
        batch_data = defaultdict(lambda: {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'uncertainties': [],
            'coordination_signals': [],
            'episode_lengths': []
        })
        
        # Collect experience
        t = 0
        while t < self.timesteps_per_batch:
            # Start new episode
            observations, infos = self.env.reset()
            coordination_info = None
            
            ep_rews = defaultdict(list)
            ep_lens = defaultdict(int)
            
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                
                # Get joint actions from all agents
                joint_output = self.multi_agent_system.get_joint_actions_and_values(
                    observations, coordination_info
                )
                
                actions = joint_output['actions']
                log_probs = joint_output['log_probs']
                values = joint_output['values']
                uncertainties = joint_output['uncertainties']
                coordination_signals = joint_output['coordination_signals']
                
                # Environment step
                next_observations, rewards, dones, _, next_infos = self.env.step(actions)
                
                # Store experience for each agent
                for agent_id in actions:
                    batch_data[agent_id]['observations'].append(observations[agent_id])
                    batch_data[agent_id]['actions'].append(actions[agent_id])
                    batch_data[agent_id]['log_probs'].append(log_probs[agent_id])
                    batch_data[agent_id]['rewards'].append(rewards[agent_id])
                    batch_data[agent_id]['values'].append(values[agent_id])
                    batch_data[agent_id]['uncertainties'].append(uncertainties[agent_id])
                    batch_data[agent_id]['coordination_signals'].append(coordination_signals[agent_id])
                    
                    ep_rews[agent_id].append(rewards[agent_id])
                    ep_lens[agent_id] += 1
                
                # Update for next step
                observations = next_observations
                coordination_info = coordination_signals
                
                # Check if episode is done
                if all(dones.values()):
                    break
                
                if t >= self.timesteps_per_batch:
                    break
            
            # Record episode statistics
            for agent_id in ep_rews:
                batch_data[agent_id]['episode_lengths'].append(ep_lens[agent_id])
                self.logger['agent_rewards'][agent_id].extend(ep_rews[agent_id])
        
        # Convert to tensors and compute returns
        for agent_id in batch_data:
            agent_data = batch_data[agent_id]
            
            # Convert to tensors (handle both tensor and non-tensor inputs)
            obs_tensors = []
            for obs in agent_data['observations']:
                if torch.is_tensor(obs):
                    obs_tensors.append(obs.detach().clone().float())
                else:
                    obs_tensors.append(torch.tensor(obs, dtype=torch.float32))
            agent_data['observations'] = torch.stack(obs_tensors).to(self.device)
            
            act_tensors = []
            for act in agent_data['actions']:
                if torch.is_tensor(act):
                    # Ensure actions are properly shaped
                    act_flat = act.detach().clone().long()
                    if act_flat.dim() > 1:
                        act_flat = act_flat.view(-1)  # Flatten if needed
                    act_tensors.append(act_flat)
                else:
                    act_tensors.append(torch.tensor(act, dtype=torch.long))
            agent_data['actions'] = torch.stack(act_tensors).to(self.device)
            
            log_prob_tensors = []
            for lp in agent_data['log_probs']:
                if torch.is_tensor(lp):
                    lp_flat = lp.detach().clone().float()
                    if lp_flat.dim() > 0:
                        lp_flat = lp_flat.view(-1)[0] if lp_flat.numel() > 0 else lp_flat
                    log_prob_tensors.append(lp_flat)
                else:
                    log_prob_tensors.append(torch.tensor(lp, dtype=torch.float32))
            agent_data['log_probs'] = torch.stack(log_prob_tensors).to(self.device)
            
            agent_data['rewards'] = torch.tensor(agent_data['rewards'], dtype=torch.float32).to(self.device)
            
            # Handle values and uncertainties (already tensors)
            values_tensors = []
            for val in agent_data['values']:
                if torch.is_tensor(val):
                    val_flat = val.detach().clone().float()
                    if val_flat.dim() > 0:
                        val_flat = val_flat.view(-1)[0] if val_flat.numel() > 0 else val_flat
                    values_tensors.append(val_flat)
                else:
                    values_tensors.append(torch.tensor(val, dtype=torch.float32))
            agent_data['values'] = torch.stack(values_tensors).to(self.device)
            
            uncertainties_tensors = []
            for unc in agent_data['uncertainties']:
                if torch.is_tensor(unc):
                    unc_flat = unc.detach().clone().float()
                    if unc_flat.dim() > 0:
                        unc_flat = unc_flat.view(-1)[0] if unc_flat.numel() > 0 else unc_flat
                    uncertainties_tensors.append(unc_flat)
                else:
                    uncertainties_tensors.append(torch.tensor(unc, dtype=torch.float32))
            agent_data['uncertainties'] = torch.stack(uncertainties_tensors).to(self.device)
            
            # Compute returns-to-go
            agent_data['rtgs'] = self._compute_rtgs(agent_data['rewards'])
        
        return dict(batch_data)
    
    def _multi_agent_learning_step(self, batch_data, iteration):
        """
        Perform multi-agent learning step with coordination
        
        Args:
            batch_data: Dictionary of batch data for each agent
            iteration: Current iteration number
        """
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        # Collect coordination information
        coordination_info = {}
        for agent_id in batch_data:
            if batch_data[agent_id]['coordination_signals']:
                coordination_info[agent_id] = torch.stack(batch_data[agent_id]['coordination_signals']).mean(dim=0)
        
        # Update each agent
        for agent_id in batch_data:
            agent_data = batch_data[agent_id]
            agent_network = self.multi_agent_system.agents[agent_id]
            optimizer = self.optimizers[agent_id]
            
            # Compute advantages
            with torch.no_grad():
                advantages = agent_data['rtgs'] - agent_data['values']
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Multiple update steps
            for _ in range(self.n_updates_per_iteration):
                # Clear gradients first
                optimizer.zero_grad()
                
                # Evaluate current policy (fresh forward pass)
                eval_output = agent_network.evaluate_action(
                    agent_data['observations'].detach(), 
                    agent_data['actions'].detach(),
                    coordination_info.get(agent_id).detach() if coordination_info.get(agent_id) is not None else None
                )
                
                curr_log_probs = eval_output[0]
                curr_values = eval_output[1]
                entropy = eval_output[2]
                uncertainties = eval_output[3]
                
                # Calculate ratios and losses (use detached old log probs)
                ratios = torch.exp(curr_log_probs - agent_data['log_probs'].detach())
                surr1 = ratios * advantages.detach()
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages.detach()
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = actor_loss - self.ent_coef * entropy.mean()
                
                # Critic loss (use detached targets)
                critic_loss = nn.MSELoss()(curr_values, agent_data['rtgs'].detach())
                
                # Total loss
                total_loss = actor_loss + self.value_reg_coef * critic_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent_network.parameters(), self.max_grad_norm)
                optimizer.step()
                
                # Logging
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                
                self.logger['agent_losses'][agent_id].append(total_loss.item())
                self.logger['agent_values'][agent_id].append(curr_values.mean().item())
                self.logger['agent_uncertainties'][agent_id].append(uncertainties.mean().item())
        
        # Global coordination bonus calculation
        collaboration_bonus = self._calculate_global_collaboration_bonus(batch_data)
        self.logger['collaboration_bonuses'].append(collaboration_bonus)
        
        # Update adaptive learning rates
        self._update_multi_agent_adaptive_rates(batch_data)
    
    def _compute_rtgs(self, rewards):
        """Compute returns-to-go for a single agent"""
        rtgs = []
        discounted_reward = 0
        
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            rtgs.insert(0, discounted_reward)
        
        return torch.tensor(rtgs, dtype=torch.float32).to(self.device)
    
    def _calculate_global_collaboration_bonus(self, batch_data):
        """Calculate global collaboration bonus across all agents"""
        if len(batch_data) <= 1:
            return 0.0
        
        # Collect all rewards
        all_rewards = []
        for agent_data in batch_data.values():
            # Handle tensor/numpy conversion safely
            rewards = agent_data['rewards']
            if torch.is_tensor(rewards):
                rewards = rewards.cpu().numpy()
            all_rewards.extend(rewards)
            
        if not all_rewards:
            return 0.0
            
        # Simple mean-variance calculation
        # This is a simplified implementation for demonstration
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        
        # Linear combination
        bonus = max(0.0, avg_reward * 0.1 - std_reward * 0.05)
        
        return float(bonus)
    
    def _update_multi_agent_adaptive_rates(self, batch_data):
        """Update adaptive learning rates for all agents"""
        for agent_id in batch_data:
            agent_data = batch_data[agent_id]
            
            # Calculate adaptation factor based on uncertainty
            avg_uncertainty = torch.mean(agent_data['uncertainties']).item()
            adaptation_factor = min(2.0, max(0.5, 1.0 / (avg_uncertainty + 0.1)))
            
            self.adaptive_lr_factors[agent_id] = adaptation_factor
            self.current_adaptive_lrs[agent_id] = self.base_lr * adaptation_factor
            
            # Update optimizer learning rate
            for param_group in self.optimizers[agent_id].param_groups:
                param_group['lr'] = self.current_adaptive_lrs[agent_id]
            
            self.logger['adaptation_factors'][agent_id].append(adaptation_factor)
    
    def _collect_analysis_data(self, batch_data, analyzer):
        """Collect data for analysis"""
        analysis_data = {
            'iteration': self.logger['i_so_far'],
            'timesteps': self.logger['t_so_far'],
            'agent_rewards': {},
            'agent_uncertainties': {},
            'coordination_efficiency': 0.0,
            'load_balance': 0.0
        }
        
        for agent_id in batch_data:
            agent_data = batch_data[agent_id]
            analysis_data['agent_rewards'][agent_id] = torch.mean(agent_data['rewards']).item()
            analysis_data['agent_uncertainties'][agent_id] = torch.mean(agent_data['uncertainties']).item()
        
        # Calculate coordination efficiency
        if len(batch_data) > 1:
            reward_values = list(analysis_data['agent_rewards'].values())
            analysis_data['coordination_efficiency'] = 1.0 - np.std(reward_values) / (np.mean(reward_values) + 1e-8)
            analysis_data['load_balance'] = 1.0 - np.std(reward_values)
        
        analyzer.add_data(analysis_data)
    
    def save_models(self, suffix=''):
        """Save all agent models"""
        for agent_id in self.multi_agent_system.agents:
            agent_network = self.multi_agent_system.agents[agent_id]
            optimizer = self.optimizers[agent_id]
            scheduler = self.schedulers[agent_id]
            
            model_path = os.path.join(
                self.save_dir, 
                f'vamappo_{agent_id}_model{suffix}.pth'
            )
            
            torch.save({
                'model_state_dict': agent_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'adaptive_lr_factor': self.adaptive_lr_factors[agent_id],
                'exploration_factor': self.exploration_factors[agent_id],
                'iteration': self.logger['i_so_far']
            }, model_path)
        
        print(f"VAMAPPO models saved with suffix: {suffix}")
    
    def load_models(self, suffix=''):
        """Load all agent models"""
        for agent_id in self.multi_agent_system.agents:
            model_path = os.path.join(
                self.save_dir, 
                f'vamappo_{agent_id}_model{suffix}.pth'
            )
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                agent_network = self.multi_agent_system.agents[agent_id]
                agent_network.load_state_dict(checkpoint['model_state_dict'])
                
                self.optimizers[agent_id].load_state_dict(checkpoint['optimizer_state_dict'])
                self.schedulers[agent_id].load_state_dict(checkpoint['scheduler_state_dict'])
                
                self.adaptive_lr_factors[agent_id] = checkpoint.get('adaptive_lr_factor', 1.0)
                self.exploration_factors[agent_id] = checkpoint.get('exploration_factor', 1.0)
                
                print(f"Loaded model for {agent_id}")
            else:
                print(f"Model file not found: {model_path}")
    
    def _init_hyperparameters(self, hyperparameters):
        """Initialize hyperparameters for VAMAPPO"""
        # Standard PPO hyperparameters
        self.timesteps_per_batch = hyperparameters.get('timesteps_per_batch', 1024)
        self.max_timesteps_per_episode = hyperparameters.get('max_timesteps_per_episode', 20)
        self.n_updates_per_iteration = hyperparameters.get('n_updates_per_iteration', 4)
        self.lr = hyperparameters.get('lr', 0.0003)
        self.gamma = hyperparameters.get('gamma', 0.99)
        self.clip = hyperparameters.get('clip', 0.2)
        self.ent_coef = hyperparameters.get('ent_coef', 0.01)
        self.value_reg_coef = hyperparameters.get('value_reg_coef', 1.0)
        self.max_grad_norm = hyperparameters.get('max_grad_norm', 0.5)
        self.save_freq = hyperparameters.get('save_freq', 10)
        
        # Multi-agent specific hyperparameters
        self.coordination_coef = hyperparameters.get('coordination_coef', 0.1)
        self.collaboration_bonus_coef = hyperparameters.get('collaboration_bonus_coef', 0.05)
        self.communication_cost = hyperparameters.get('communication_cost', 0.001)
    
    def _log_summary(self):
        """Enhanced logging for multi-agent system"""
        print("=" * 80)
        print(f"VAMAPPO - Iteration: {self.logger['i_so_far']}")
        print(f"Timesteps: {self.logger['t_so_far']}")
        
        # Agent-specific metrics
        for agent_id in [f'Agent_{i}' for i in range(self.n_agents)]:
            if agent_id in self.logger['agent_rewards'] and self.logger['agent_rewards'][agent_id]:
                avg_reward = np.mean(self.logger['agent_rewards'][agent_id][-10:])
                print(f"{agent_id} - Avg Reward: {avg_reward:.4f}")
                
                if agent_id in self.logger['agent_uncertainties'] and self.logger['agent_uncertainties'][agent_id]:
                    avg_uncertainty = np.mean(self.logger['agent_uncertainties'][agent_id][-10:])
                    print(f"{agent_id} - Avg Uncertainty: {avg_uncertainty:.4f}")
                
                if agent_id in self.adaptive_lr_factors:
                    print(f"{agent_id} - Adaptive LR Factor: {self.adaptive_lr_factors[agent_id]:.4f}")
                
                if agent_id in self.exploration_factors:
                    print(f"{agent_id} - Exploration Factor: {self.exploration_factors[agent_id]:.4f}")
        
        # Global metrics
        if self.logger['collaboration_bonuses']:
            avg_collaboration = np.mean(self.logger['collaboration_bonuses'][-10:])
            print(f"Average Collaboration Bonus: {avg_collaboration:.4f}")
        
        if self.logger['total_system_reward']:
            avg_system_reward = np.mean(self.logger['total_system_reward'][-10:])
            print(f"Average System Reward: {avg_system_reward:.4f}")
        
        print("=" * 80)