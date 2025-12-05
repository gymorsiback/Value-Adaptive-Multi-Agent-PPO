"""
Distributed Multi-Modal Model Orchestration System Neural Network Architectures
Extended for VAMAPPO (Value-Adaptive Multi-Agent PPO)
Supports both single-agent and multi-agent modes with coordination mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

class DistributedActorNetwork(nn.Module):
    """
    Actor Network - Enhanced for Value Adaptive PPO
    Outputs two independent action distributions: server selection + model selection
    """
    
    def __init__(self, state_dim, n_servers, max_models_per_server):
        """
        Initialize Actor network
        
        Args:
            state_dim: State space dimension
            n_servers: Number of servers
            max_models_per_server: Maximum models per server
        """
        super(DistributedActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.n_servers = n_servers
        self.max_models_per_server = max_models_per_server
        
        # Enhanced shared feature extraction layers for value adaptive
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Server selection head with attention mechanism
        self.server_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.server_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_servers)
        )
        
        # Model selection head with enhanced capacity
        self.model_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_models_per_server)
        )
        
        # Value-adaptive policy enhancement layers
        self.policy_enhancement = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()  # Adaptive weighting factors
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights with Xavier uniform"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.MultiheadAttention):
            torch.nn.init.xavier_uniform_(module.in_proj_weight)
            torch.nn.init.constant_(module.in_proj_bias, 0)
    
    def forward(self, state):
        """
        Forward propagation with value-adaptive enhancements
        
        Args:
            state: State vector
            
        Returns:
            server_logits: Server selection logits
            model_logits: Model selection logits
            enhancement_weights: Adaptive enhancement weights
        """
        # Process input
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Ensure input and network are on the same device
        state = state.to(next(self.parameters()).device)
        
        # Shared feature extraction
        shared_features = self.shared_layers(state)
        
        # Apply attention mechanism for server selection
        if shared_features.dim() == 1:
            shared_features = shared_features.unsqueeze(0).unsqueeze(0)
        elif shared_features.dim() == 2:
            shared_features = shared_features.unsqueeze(1)
            
        attended_features, _ = self.server_attention(shared_features, shared_features, shared_features)
        attended_features = attended_features.squeeze(1) if attended_features.dim() == 3 else attended_features.squeeze(0)
        
        # Calculate logits for both actions
        server_logits = self.server_head(attended_features)
        model_logits = self.model_head(attended_features)
        
        # Generate adaptive enhancement weights
        enhancement_weights = self.policy_enhancement(attended_features)
        
        return server_logits, model_logits, enhancement_weights
    
    def get_action_and_log_prob(self, state, action_mask=None):
        """
        Get action and log probability with value-adaptive enhancement
        
        Args:
            state: State vector
            action_mask: Action mask (optional)
            
        Returns:
            action: Selected action [server_idx, model_idx]
            log_prob: Log probability of action
            enhancement_weights: Adaptive enhancement weights
        """
        server_logits, model_logits, enhancement_weights = self.forward(state)
        
        # Apply action mask (if provided)
        if action_mask is not None:
            server_mask, model_mask = action_mask
            server_logits = server_logits.masked_fill(~server_mask, -float('inf'))
            model_logits = model_logits.masked_fill(~model_mask, -float('inf'))
        
        # Create distributions
        server_dist = torch.distributions.Categorical(logits=server_logits)
        model_dist = torch.distributions.Categorical(logits=model_logits)
        
        # Sample actions
        server_action = server_dist.sample()
        model_action = model_dist.sample()
        
        # Calculate log probabilities
        server_log_prob = server_dist.log_prob(server_action)
        model_log_prob = model_dist.log_prob(model_action)
        total_log_prob = server_log_prob + model_log_prob
        
        action = torch.stack([server_action, model_action], dim=-1)
        
        return action, total_log_prob, enhancement_weights
    
    def get_log_prob(self, state, action):
        """
        Get log probability of given action
        
        Args:
            state: State vector
            action: Action [server_idx, model_idx]
            
        Returns:
            log_prob: Log probability of action
        """
        server_logits, model_logits, _ = self.forward(state)
        
        # Create distributions
        server_dist = torch.distributions.Categorical(logits=server_logits)
        model_dist = torch.distributions.Categorical(logits=model_logits)
        
        # Extract actions
        server_action = action[:, 0] if action.dim() > 1 else action[0]
        model_action = action[:, 1] if action.dim() > 1 else action[1]
        
        # Calculate log probabilities
        server_log_prob = server_dist.log_prob(server_action)
        model_log_prob = model_dist.log_prob(model_action)
        total_log_prob = server_log_prob + model_log_prob
        
        return total_log_prob

class AdaptiveValueCriticNetwork(nn.Module):
    """
    Adaptive Value Critic Network - Multi-scale value function estimation
    Implements dynamic learning rate adjustment and uncertainty quantification
    """
    
    def __init__(self, state_dim):
        """
        Initialize Adaptive Value Critic network
        
        Args:
            state_dim: State space dimension
        """
        super(AdaptiveValueCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Primary value function network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Enhanced uncertainty estimation network with better architecture
        self.uncertainty_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),  # Higher dropout for better MC sampling
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # Ensure positive uncertainty values
        )
        
        # Value adaptation network for dynamic learning rate
        self.adaptation_net = nn.Sequential(
            nn.Linear(state_dim + 1, 64),  # state + previous value estimate
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1 for adaptation factor
        )
        
        # Temporal difference estimation network
        self.td_net = nn.Sequential(
            nn.Linear(state_dim * 2, 128),  # current state + next state
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Adaptive learning parameters
        self.register_buffer('value_history', torch.zeros(100))  # Rolling history for adaptation
        self.register_buffer('uncertainty_history', torch.zeros(100))
        self.history_index = 0
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state, prev_value=None):
        """
        Forward propagation with adaptive value estimation
        
        Args:
            state: State vector
            prev_value: Previous value estimate for adaptation
            
        Returns:
            value: Primary value estimate
            uncertainty: Value uncertainty estimate
            adaptation_factor: Learning rate adaptation factor
        """
        # Process input
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Ensure input and network are on the same device
        state = state.to(next(self.parameters()).device)
        
        # Primary value estimation
        value = self.value_net(state)
        
        # Uncertainty estimation with Monte Carlo dropout (keep training mode for uncertainty)
        self.uncertainty_net.train()  # Always use training mode for uncertainty estimation
        
        # Multiple forward passes for Monte Carlo estimation
        uncertainty_samples = []
        for _ in range(3):  # 3 MC samples for better uncertainty estimation
            sample = self.uncertainty_net(state)
            uncertainty_samples.append(sample)
        
        # Calculate uncertainty as variance across MC samples plus network output
        mc_uncertainty = torch.var(torch.stack(uncertainty_samples), dim=0)
        base_uncertainty = torch.mean(torch.stack(uncertainty_samples), dim=0)
        
        # Combine MC uncertainty with base uncertainty
        uncertainty = base_uncertainty + 0.5 * mc_uncertainty
        
        # Progressive minimum uncertainty (starts high, gradually decreases)
        if not hasattr(self, 'uncertainty_min_schedule'):
            self.uncertainty_min_schedule = 0.08  # Start with higher minimum
            self.uncertainty_decay_rate = 0.999  # Very slow decay
        
        # Update minimum uncertainty schedule
        self.uncertainty_min_schedule *= self.uncertainty_decay_rate
        self.uncertainty_min_schedule = max(self.uncertainty_min_schedule, 0.02)  # Floor at 0.02
        
        # Add scheduled minimum uncertainty with noise
        uncertainty = uncertainty + self.uncertainty_min_schedule + 0.005 * torch.randn_like(uncertainty)
        
        # Ensure uncertainty doesn't drop too quickly by adding temporal regularization
        if hasattr(self, 'prev_uncertainty_mean'):
            # Use mean of previous uncertainty for comparison (shape-independent)
            current_uncertainty_mean = uncertainty.mean()
            max_decrease = 0.05 * self.prev_uncertainty_mean
            min_allowed_uncertainty = self.prev_uncertainty_mean - max_decrease
            
            # Apply constraint only if current mean is below threshold
            if current_uncertainty_mean < min_allowed_uncertainty:
                # Scale up all uncertainties proportionally
                scale_factor = min_allowed_uncertainty / current_uncertainty_mean
                uncertainty = uncertainty * scale_factor
        
        # Store mean uncertainty for next iteration
        self.prev_uncertainty_mean = uncertainty.mean().detach().clone()
        
        # Adaptation factor calculation
        if prev_value is not None:
            if isinstance(prev_value, (int, float)):
                prev_value = torch.tensor([prev_value], dtype=torch.float32, device=state.device)
            elif prev_value.dim() == 0:
                prev_value = prev_value.unsqueeze(0)
                
            adaptation_input = torch.cat([state, prev_value], dim=-1)
            adaptation_factor = self.adaptation_net(adaptation_input)
        else:
            # Default adaptation factor when no previous value
            adaptation_factor = torch.ones_like(value) * 0.5
        
        return value, uncertainty, adaptation_factor
    
    def compute_td_error(self, current_state, next_state, reward, gamma=0.95):
        """
        Compute temporal difference error for adaptive learning
        
        Args:
            current_state: Current state
            next_state: Next state
            reward: Immediate reward
            gamma: Discount factor
            
        Returns:
            td_error: Temporal difference error
        """
        # Concatenate states for TD network
        combined_states = torch.cat([current_state, next_state], dim=-1)
        
        # Estimate TD error directly
        td_estimate = self.td_net(combined_states)
        
        # Get value estimates
        current_value, _, _ = self.forward(current_state)
        next_value, _, _ = self.forward(next_state)
        
        # Calculate actual TD error
        target = reward + gamma * next_value
        actual_td_error = target - current_value
        
        # Combine network estimate with actual calculation
        td_error = 0.7 * actual_td_error + 0.3 * td_estimate
        
        return td_error
    
    def update_history(self, value, uncertainty):
        """
        Update rolling history for adaptive learning
        
        Args:
            value: Current value estimate
            uncertainty: Current uncertainty estimate
        """
        # Handle both tensor and scalar inputs
        if torch.is_tensor(value):
            value_scalar = value.detach().mean().item() if value.numel() > 1 else value.detach().item()
        else:
            value_scalar = float(value)
            
        if torch.is_tensor(uncertainty):
            uncertainty_scalar = uncertainty.detach().mean().item() if uncertainty.numel() > 1 else uncertainty.detach().item()
        else:
            uncertainty_scalar = float(uncertainty)
        
        self.value_history[self.history_index] = value_scalar
        self.uncertainty_history[self.history_index] = uncertainty_scalar
        self.history_index = (self.history_index + 1) % 100
    
    def get_adaptive_lr_factor(self):
        """
        Calculate adaptive learning rate factor with enhanced smoothing
        
        Returns:
            lr_factor: Smoothed learning rate adjustment factor
        """
        # Calculate variance in recent value estimates
        value_variance = torch.var(self.value_history)
        uncertainty_mean = torch.mean(self.uncertainty_history)
        
        # Prevent division by zero and ensure meaningful adaptation
        value_variance = torch.clamp(value_variance, min=0.01)
        uncertainty_mean = torch.clamp(uncertainty_mean, min=0.001)
        
        # Enhanced adaptive calculation with reduced sensitivity
        # Higher variance suggests instability -> reduce LR slightly (less aggressive)
        variance_factor = torch.clamp(1.0 / (1.0 + value_variance * 0.5), 0.6, 1.2)
        
        # Higher uncertainty suggests need for more learning -> increase LR (more conservative)
        uncertainty_factor = torch.clamp(1.0 + uncertainty_mean * 1.0, 0.7, 1.5)
        
        # Time-based decay for stability (smoother decay)
        time_factor = 0.9 + 0.2 * torch.exp(torch.tensor(-self.history_index / 100.0))
        
        # Calculate raw LR factor
        raw_lr_factor = variance_factor * uncertainty_factor * time_factor
        
        # Apply exponential moving average for smoothing
        if not hasattr(self, 'lr_factor_ema'):
            self.lr_factor_ema = raw_lr_factor.clone()
            self.lr_smoothing_alpha = 0.15  # Smoothing factor (lower = more smoothing)
        else:
            # EMA smoothing: new_value = alpha * raw + (1-alpha) * old
            self.lr_factor_ema = (self.lr_smoothing_alpha * raw_lr_factor + 
                                 (1 - self.lr_smoothing_alpha) * self.lr_factor_ema)
        
        # Additional trend-based smoothing
        if not hasattr(self, 'lr_factor_history'):
            self.lr_factor_history = [self.lr_factor_ema.item()]
        else:
            self.lr_factor_history.append(self.lr_factor_ema.item())
            # Keep only recent history (last 10 values)
            if len(self.lr_factor_history) > 10:
                self.lr_factor_history.pop(0)
            
            # Apply moving average over recent history
            if len(self.lr_factor_history) >= 3:
                recent_mean = torch.tensor(self.lr_factor_history[-3:]).mean()
                # Blend EMA with recent moving average
                self.lr_factor_ema = 0.7 * self.lr_factor_ema + 0.3 * recent_mean
        
        # Add convergence mechanism for long-term stability
        if not hasattr(self, 'convergence_target'):
            self.convergence_target = 0.8  # Target convergence value
            self.convergence_strength = 0.0  # Start with no convergence pressure
            self.convergence_threshold = 50  # Start convergence after 50 updates
        
        # Increase convergence strength over time
        if self.history_index > self.convergence_threshold:
            # Gradually increase convergence strength
            self.convergence_strength = min(0.3, (self.history_index - self.convergence_threshold) * 0.005)
            
            # Apply convergence pressure towards target
            convergence_adjustment = self.convergence_strength * (self.convergence_target - self.lr_factor_ema)
            self.lr_factor_ema = self.lr_factor_ema + convergence_adjustment
        
        # Final clamping with tighter bounds for stability
        smoothed_lr_factor = torch.clamp(self.lr_factor_ema, 0.4, 1.8)
        
        return smoothed_lr_factor

# Maintain backward compatibility
DistributedCriticNetwork = AdaptiveValueCriticNetwork

class DistributedPolicyNetwork(nn.Module):
    """
    Combined Policy Network - Enhanced for Value Adaptive PPO
    Combines actor and critic into single network for shared feature learning
    """
    
    def __init__(self, state_dim, n_servers, max_models_per_server):
        """
        Initialize combined policy network
        
        Args:
            state_dim: State space dimension
            n_servers: Number of servers
            max_models_per_server: Maximum models per server
        """
        super(DistributedPolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.n_servers = n_servers
        self.max_models_per_server = max_models_per_server
        
        # Enhanced shared backbone with residual connections
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Residual connection layer
        self.residual_layer = nn.Linear(256, 256)
        
        # Actor head with value-adaptive enhancement
        self.actor_head = DistributedActorNetwork(256, n_servers, max_models_per_server)
        
        # Critic head with adaptive value learning
        self.critic_head = AdaptiveValueCriticNetwork(256)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward propagation through combined network
        
        Args:
            state: State vector
            
        Returns:
            server_logits: Server selection logits
            model_logits: Model selection logits  
            value: Value estimate
            uncertainty: Value uncertainty
            adaptation_factor: Learning rate adaptation factor
        """
        # Shared feature extraction with residual connection
        backbone_features = self.backbone(state)
        residual_features = self.residual_layer(backbone_features)
        enhanced_features = backbone_features + residual_features
        enhanced_features = F.relu(enhanced_features)
        
        # Actor forward pass
        server_logits, model_logits, enhancement_weights = self.actor_head(enhanced_features)
        
        # Critic forward pass
        value, uncertainty, adaptation_factor = self.critic_head(enhanced_features)
        
        return server_logits, model_logits, value, uncertainty, adaptation_factor
    
    def get_action_and_value(self, state, action_mask=None):
        """
        Get action and value estimate simultaneously
        
        Args:
            state: State vector
            action_mask: Action mask (optional)
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate
            uncertainty: Value uncertainty
            adaptation_factor: Learning rate adaptation factor
        """
        server_logits, model_logits, value, uncertainty, adaptation_factor = self.forward(state)
        
        # Apply action mask (if provided)
        if action_mask is not None:
            server_mask, model_mask = action_mask
            server_logits = server_logits.masked_fill(~server_mask, -float('inf'))
            model_logits = model_logits.masked_fill(~model_mask, -float('inf'))
        
        # Create distributions and sample actions
        server_dist = torch.distributions.Categorical(logits=server_logits)
        model_dist = torch.distributions.Categorical(logits=model_logits)
        
        server_action = server_dist.sample()
        model_action = model_dist.sample()
        
        # Calculate log probabilities
        server_log_prob = server_dist.log_prob(server_action)
        model_log_prob = model_dist.log_prob(model_action)
        total_log_prob = server_log_prob + model_log_prob
        
        action = torch.stack([server_action, model_action], dim=-1)
        
        return action, total_log_prob, value, uncertainty, adaptation_factor
    
    def evaluate_action(self, state, action):
        """
        Evaluate given action with value estimation
        
        Args:
            state: State vector
            action: Action to evaluate
            
        Returns:
            log_prob: Log probability of action
            value: Value estimate
            entropy: Policy entropy
            uncertainty: Value uncertainty
        """
        server_logits, model_logits, value, uncertainty, _ = self.forward(state)
        
        # Create distributions
        server_dist = torch.distributions.Categorical(logits=server_logits)
        model_dist = torch.distributions.Categorical(logits=model_logits)
        
        # Extract actions
        server_action = action[:, 0] if action.dim() > 1 else action[0]
        model_action = action[:, 1] if action.dim() > 1 else action[1]
        
        # Calculate log probabilities and entropy
        server_log_prob = server_dist.log_prob(server_action)
        model_log_prob = model_dist.log_prob(model_action)
        total_log_prob = server_log_prob + model_log_prob
        
        entropy = server_dist.entropy() + model_dist.entropy()
        
        return total_log_prob, value.squeeze(), entropy, uncertainty.squeeze()

class FeedForwardNN(nn.Module):
    """
    Enhanced Feed Forward Neural Network for value adaptive learning
    """
    
    def __init__(self, in_dim, out_dim):
        """
        Initialize feed forward network
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
        """
        super(FeedForwardNN, self).__init__()
        
        # Enhanced architecture with batch normalization
        self.layer1 = nn.Linear(in_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, out_dim)
        
        # Activation functions
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, obs):
        """
        Forward propagation
        
        Args:
            obs: Input observations
            
        Returns:
            output: Network output
        """
        x = self.layer1(obs)
        
        # Apply batch normalization if batch size > 1
        if x.size(0) > 1:
            x = self.bn1(x)
        
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        
        # Apply batch normalization if batch size > 1
        if x.size(0) > 1:
            x = self.bn2(x)
        
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        
        return x 

# ============================================================================
# VAMAPPO Multi-Agent Extensions
# ============================================================================

class VAMAPPOAgentNetwork(nn.Module):
    """
    VAMAPPO Agent Network for Multi-Agent PPO with Value-Adaptive Learning
    Each agent has its own network with local observation processing and coordination
    """
    
    def __init__(self, local_state_dim, n_local_servers, max_models_per_server, 
                 agent_id, n_agents, coordination_dim=16):
        """
        Initialize VAMAPPO Agent Network
        
        Args:
            local_state_dim: Local state dimension for the agent
            n_local_servers: Number of servers managed by this agent
            max_models_per_server: Maximum models per server
            agent_id: Agent identifier
            n_agents: Total number of agents
            coordination_dim: Coordination information dimension
        """
        super(VAMAPPOAgentNetwork, self).__init__()
        
        self.local_state_dim = local_state_dim
        self.n_local_servers = n_local_servers
        self.max_models_per_server = max_models_per_server
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.coordination_dim = coordination_dim
        
        # Local state encoder
        self.local_encoder = nn.Sequential(
            nn.Linear(local_state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Coordination module for multi-agent communication
        self.coordination_module = CoordinationModule(64, coordination_dim, n_agents)
        
        # Enhanced local actor with coordination
        self.local_actor = nn.Sequential(
            nn.Linear(64 + coordination_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Server selection head (local servers only)
        self.server_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_local_servers)
        )
        
        # Model selection head
        self.model_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, max_models_per_server)
        )
        
        # Local value critic with uncertainty estimation
        self.local_critic = nn.Sequential(
            nn.Linear(64 + coordination_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Value and uncertainty heads
        self.value_head = nn.Linear(32, 1)
        self.uncertainty_head = nn.Linear(32, 1)
        
        # Adaptive learning rate factor
        self.adaptation_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, local_state, coordination_info=None):
        """
        Forward pass for VAMAPPO agent
        
        Args:
            local_state: Local state observation
            coordination_info: Coordination information from other agents
            
        Returns:
            server_logits: Local server selection logits
            model_logits: Model selection logits
            value: Value estimate
            uncertainty: Value uncertainty
            adaptation_factor: Learning rate adaptation factor
            coordination_signal: Signal to send to other agents
        """
        # Process input
        if isinstance(local_state, np.ndarray):
            local_state = torch.tensor(local_state, dtype=torch.float32)
        
        local_state = local_state.to(next(self.parameters()).device)
        
        # Local state encoding
        local_features = self.local_encoder(local_state)
        
        # Ensure local_features has batch dimension
        if local_features.dim() == 1:
            local_features = local_features.unsqueeze(0)
        
        # Coordination processing
        coordination_features, coordination_signal = self.coordination_module(
            local_features, coordination_info
        )
        
        # Ensure coordination_features matches local_features dimensions
        if coordination_features.dim() == 1 and local_features.dim() == 2:
            coordination_features = coordination_features.unsqueeze(0)
        elif coordination_features.dim() == 2 and local_features.dim() == 1:
            local_features = local_features.unsqueeze(0)
        
        # Combined features for action selection
        combined_features = torch.cat([local_features, coordination_features], dim=-1)
        actor_features = self.local_actor(combined_features)
        
        # Action logits
        server_logits = self.server_head(actor_features)
        model_logits = self.model_head(actor_features)
        
        # Value estimation
        critic_features = self.local_critic(combined_features)
        value = self.value_head(critic_features)
        uncertainty = torch.abs(self.uncertainty_head(critic_features)) + 1e-8
        adaptation_factor = self.adaptation_head(critic_features)
        
        return server_logits, model_logits, value, uncertainty, adaptation_factor, coordination_signal
    
    def get_action_and_value(self, local_state, coordination_info=None, action_mask=None):
        """
        Get action and value estimate for the agent
        
        Args:
            local_state: Local state observation
            coordination_info: Coordination information from other agents
            action_mask: Action mask (optional)
            
        Returns:
            action: Selected action [local_server_idx, model_idx]
            log_prob: Log probability of action
            value: Value estimate
            uncertainty: Value uncertainty
            adaptation_factor: Learning rate adaptation factor
            coordination_signal: Signal to send to other agents
        """
        server_logits, model_logits, value, uncertainty, adaptation_factor, coordination_signal = \
            self.forward(local_state, coordination_info)
        
        # Apply action mask if provided
        if action_mask is not None:
            server_mask, model_mask = action_mask
            server_logits = server_logits.masked_fill(~server_mask, -float('inf'))
            model_logits = model_logits.masked_fill(~model_mask, -float('inf'))
        
        # Create distributions
        server_dist = torch.distributions.Categorical(logits=server_logits)
        model_dist = torch.distributions.Categorical(logits=model_logits)
        
        # Sample actions
        server_action = server_dist.sample()
        model_action = model_dist.sample()
        
        # Calculate log probabilities
        server_log_prob = server_dist.log_prob(server_action)
        model_log_prob = model_dist.log_prob(model_action)
        total_log_prob = server_log_prob + model_log_prob
        
        action = torch.stack([server_action, model_action], dim=-1)
        
        return action, total_log_prob, value.squeeze(), uncertainty.squeeze(), \
               adaptation_factor.squeeze(), coordination_signal
    
    def evaluate_action(self, local_state, action, coordination_info=None):
        """
        Evaluate given action
        
        Args:
            local_state: Local state observation
            action: Action to evaluate
            coordination_info: Coordination information from other agents
            
        Returns:
            log_prob: Log probability of action
            value: Value estimate
            entropy: Policy entropy
            uncertainty: Value uncertainty
        """
        server_logits, model_logits, value, uncertainty, _, _ = \
            self.forward(local_state, coordination_info)
        
        # Create distributions
        server_dist = torch.distributions.Categorical(logits=server_logits)
        model_dist = torch.distributions.Categorical(logits=model_logits)
        
        # Extract actions
        server_action = action[:, 0] if action.dim() > 1 else action[0]
        model_action = action[:, 1] if action.dim() > 1 else action[1]
        
        # Calculate log probabilities and entropy
        server_log_prob = server_dist.log_prob(server_action)
        model_log_prob = model_dist.log_prob(model_action)
        total_log_prob = server_log_prob + model_log_prob
        
        entropy = server_dist.entropy() + model_dist.entropy()
        
        return total_log_prob, value.squeeze(), entropy, uncertainty.squeeze()


class CoordinationModule(nn.Module):
    """
    Coordination Module for Multi-Agent Communication
    Processes coordination information and generates communication signals
    """
    
    def __init__(self, local_feature_dim, coordination_dim, n_agents):
        """
        Initialize Coordination Module
        
        Args:
            local_feature_dim: Local feature dimension
            coordination_dim: Coordination information dimension
            n_agents: Number of agents
        """
        super(CoordinationModule, self).__init__()
        
        self.local_feature_dim = local_feature_dim
        self.coordination_dim = coordination_dim
        self.n_agents = n_agents
        
        # Coordination information processor
        self.coord_processor = nn.Sequential(
            nn.Linear(coordination_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Communication signal generator
        self.signal_generator = nn.Sequential(
            nn.Linear(local_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, coordination_dim)
        )
        
        # Attention mechanism for coordination
        self.attention = nn.MultiheadAttention(
            embed_dim=16, num_heads=2, batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(16, coordination_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.MultiheadAttention):
            torch.nn.init.xavier_uniform_(module.in_proj_weight)
            torch.nn.init.constant_(module.in_proj_bias, 0)
    
    def forward(self, local_features, coordination_info=None):
        """
        Process coordination information and generate communication signals
        
        Args:
            local_features: Local feature representation
            coordination_info: Coordination information from other agents
            
        Returns:
            coordination_features: Processed coordination features
            coordination_signal: Signal to send to other agents
        """
        # Generate communication signal
        coordination_signal = self.signal_generator(local_features)
        
        # Process coordination information
        if coordination_info is not None:
            # Process coordination information
            coord_features = self.coord_processor(coordination_info)
            
            # Apply attention mechanism
            if coord_features.dim() == 1:
                coord_features = coord_features.unsqueeze(0).unsqueeze(0)
            elif coord_features.dim() == 2:
                coord_features = coord_features.unsqueeze(1)
            
            attended_features, _ = self.attention(coord_features, coord_features, coord_features)
            
            # Properly squeeze dimensions to match expected output
            if attended_features.dim() == 3:
                attended_features = attended_features.squeeze(1)
            
            # Output projection
            coordination_features = self.output_proj(attended_features)
            
            # Ensure coordination_features matches the batch dimension of local_features
            batch_size = local_features.shape[0] if local_features.dim() > 1 else 1
            
            if coordination_features.dim() == 1:
                # coordination_features is 1D, expand to match batch size
                coordination_features = coordination_features.unsqueeze(0).expand(batch_size, -1)
            elif coordination_features.dim() == 2:
                # coordination_features is 2D, check if batch dimension matches
                if coordination_features.shape[0] != batch_size:
                    if coordination_features.shape[0] == 1:
                        # Expand single batch to match
                        coordination_features = coordination_features.expand(batch_size, -1)
                    else:
                        # Take first element and expand
                        coordination_features = coordination_features[0:1].expand(batch_size, -1)
        else:
            # No coordination information available
            coordination_features = torch.zeros_like(coordination_signal)
            
            # Ensure dimensions match local_features batch size
            batch_size = local_features.shape[0] if local_features.dim() > 1 else 1
            
            if coordination_features.dim() == 1:
                # coordination_features is 1D, expand to match batch size
                coordination_features = coordination_features.unsqueeze(0).expand(batch_size, -1)
            elif coordination_features.dim() == 2 and coordination_features.shape[0] != batch_size:
                # coordination_features batch size doesn't match
                if coordination_features.shape[0] == 1:
                    coordination_features = coordination_features.expand(batch_size, -1)
                else:
                    coordination_features = coordination_features[0:1].expand(batch_size, -1)
        
        return coordination_features, coordination_signal


class VAMAPPOMultiAgentSystem(nn.Module):
    """
    VAMAPPO Multi-Agent System
    Manages multiple agents with coordination and communication
    """
    
    def __init__(self, agent_configs: List[Dict]):
        """
        Initialize VAMAPPO Multi-Agent System
        
        Args:
            agent_configs: List of agent configurations
                Each config should contain:
                - local_state_dim: Local state dimension
                - n_local_servers: Number of local servers
                - max_models_per_server: Maximum models per server
                - agent_id: Agent identifier
        """
        super(VAMAPPOMultiAgentSystem, self).__init__()
        
        self.n_agents = len(agent_configs)
        self.agent_configs = agent_configs
        
        # Create agent networks
        self.agents = nn.ModuleDict()
        for config in agent_configs:
            agent_id = config['agent_id']
            self.agents[agent_id] = VAMAPPOAgentNetwork(
                local_state_dim=config['local_state_dim'],
                n_local_servers=config['n_local_servers'],
                max_models_per_server=config['max_models_per_server'],
                agent_id=agent_id,
                n_agents=self.n_agents,
                coordination_dim=config.get('coordination_dim', 16)
            )
        
        # Global coordination tracker
        self.global_coordination_state = {}
        
    def forward(self, observations: Dict[str, torch.Tensor], 
                coordination_info: Optional[Dict[str, torch.Tensor]] = None):
        """
        Forward pass for all agents
        
        Args:
            observations: Dictionary of agent observations
            coordination_info: Dictionary of coordination information
            
        Returns:
            Dictionary of agent outputs
        """
        outputs = {}
        new_coordination_signals = {}
        
        for agent_id, agent_network in self.agents.items():
            if agent_id in observations:
                local_state = observations[agent_id]
                coord_info = coordination_info.get(agent_id) if coordination_info else None
                
                # Forward pass for agent
                server_logits, model_logits, value, uncertainty, adaptation_factor, coordination_signal = \
                    agent_network(local_state, coord_info)
                
                outputs[agent_id] = {
                    'server_logits': server_logits,
                    'model_logits': model_logits,
                    'value': value,
                    'uncertainty': uncertainty,
                    'adaptation_factor': adaptation_factor
                }
                
                new_coordination_signals[agent_id] = coordination_signal
        
        # Update global coordination state
        self.global_coordination_state = new_coordination_signals
        
        return outputs
    
    def get_joint_actions_and_values(self, observations: Dict[str, torch.Tensor],
                                     coordination_info: Optional[Dict[str, torch.Tensor]] = None,
                                     action_masks: Optional[Dict[str, Tuple]] = None):
        """
        Get joint actions and values for all agents
        
        Args:
            observations: Dictionary of agent observations
            coordination_info: Dictionary of coordination information
            action_masks: Dictionary of action masks
            
        Returns:
            Dictionary of joint actions, log probabilities, values, etc.
        """
        joint_actions = {}
        joint_log_probs = {}
        joint_values = {}
        joint_uncertainties = {}
        joint_adaptation_factors = {}
        new_coordination_signals = {}
        
        for agent_id, agent_network in self.agents.items():
            if agent_id in observations:
                local_state = observations[agent_id]
                coord_info = coordination_info.get(agent_id) if coordination_info else None
                action_mask = action_masks.get(agent_id) if action_masks else None
                
                # Get action and value for agent
                action, log_prob, value, uncertainty, adaptation_factor, coordination_signal = \
                    agent_network.get_action_and_value(local_state, coord_info, action_mask)
                
                joint_actions[agent_id] = action
                joint_log_probs[agent_id] = log_prob
                joint_values[agent_id] = value
                joint_uncertainties[agent_id] = uncertainty
                joint_adaptation_factors[agent_id] = adaptation_factor
                new_coordination_signals[agent_id] = coordination_signal
        
        # Update global coordination state
        self.global_coordination_state = new_coordination_signals
        
        return {
            'actions': joint_actions,
            'log_probs': joint_log_probs,
            'values': joint_values,
            'uncertainties': joint_uncertainties,
            'adaptation_factors': joint_adaptation_factors,
            'coordination_signals': new_coordination_signals
        }
    
    def evaluate_joint_actions(self, observations: Dict[str, torch.Tensor],
                               actions: Dict[str, torch.Tensor],
                               coordination_info: Optional[Dict[str, torch.Tensor]] = None):
        """
        Evaluate joint actions for all agents
        
        Args:
            observations: Dictionary of agent observations
            actions: Dictionary of agent actions
            coordination_info: Dictionary of coordination information
            
        Returns:
            Dictionary of evaluation results
        """
        joint_log_probs = {}
        joint_values = {}
        joint_entropies = {}
        joint_uncertainties = {}
        
        for agent_id, agent_network in self.agents.items():
            if agent_id in observations and agent_id in actions:
                local_state = observations[agent_id]
                action = actions[agent_id]
                coord_info = coordination_info.get(agent_id) if coordination_info else None
                
                # Evaluate action for agent
                log_prob, value, entropy, uncertainty = \
                    agent_network.evaluate_action(local_state, action, coord_info)
                
                joint_log_probs[agent_id] = log_prob
                joint_values[agent_id] = value
                joint_entropies[agent_id] = entropy
                joint_uncertainties[agent_id] = uncertainty
        
        return {
            'log_probs': joint_log_probs,
            'values': joint_values,
            'entropies': joint_entropies,
            'uncertainties': joint_uncertainties
        }
    
    def get_coordination_state(self):
        """Get current global coordination state"""
        return self.global_coordination_state.copy()
    
    def update_coordination_state(self, new_coordination_info: Dict[str, torch.Tensor]):
        """Update global coordination state"""
        self.global_coordination_state.update(new_coordination_info)