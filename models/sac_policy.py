#!/usr/bin/env python3
"""
SAC (Soft Actor-Critic) Policy Implementation
Simplified version for the manipulator environment

Author: RL Training Implementation
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """Experience replay buffer for SAC"""
    
    def __init__(self, capacity: int, device: str = 'cpu'):
        """
        Initialize replay buffer
        
        Args:
            capacity (int): Maximum buffer size
            device (str): Device for tensor storage
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state):
        """Add experience to buffer"""
        # Convert to tensors with consistent dtype (float32)
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        elif isinstance(state, torch.Tensor):
            state = state.to(dtype=torch.float32)
            
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)
        elif isinstance(action, torch.Tensor):
            action = action.to(dtype=torch.float32)
            
        if isinstance(reward, (int, float, np.number)):
            reward = torch.tensor([float(reward)], dtype=torch.float32)
        elif isinstance(reward, torch.Tensor):
            reward = reward.to(dtype=torch.float32)
            
        if isinstance(next_state, np.ndarray):
            next_state = torch.tensor(next_state, dtype=torch.float32)
        elif isinstance(next_state, torch.Tensor):
            next_state = next_state.to(dtype=torch.float32)
            
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(torch.stack, zip(*batch))
        return state.to(self.device), action.to(self.device), reward.to(self.device), next_state.to(self.device)
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """Actor network for SAC"""
    
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, action_size)
        self.log_std_head = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.xavier_uniform_(self.log_std_head.weight)
        
    def forward(self, state):
        """Forward pass"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample(self, state):
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class Critic(nn.Module):
    """Critic network for SAC"""
    
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
        super(Critic, self).__init__()
        
        # Q1 network
        self.q1_fc1 = nn.Linear(obs_size + action_size, hidden_size)
        self.q1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_fc3 = nn.Linear(hidden_size, 1)
        
        # Q2 network
        self.q2_fc1 = nn.Linear(obs_size + action_size, hidden_size)
        self.q2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q2_fc3 = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        for fc in [self.q1_fc1, self.q1_fc2, self.q1_fc3, self.q2_fc1, self.q2_fc2, self.q2_fc3]:
            nn.init.xavier_uniform_(fc.weight)
    
    def forward(self, state, action):
        """Forward pass for both Q networks"""
        x = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        
        # Q2
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        
        return q1, q2

class SACPolicy:
    """Soft Actor-Critic Policy"""
    
    def __init__(self, 
                 obs_size: int,
                 action_size: int,
                 device: str = 'cpu',
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 action_scale: float = 1.0):
        """
        Initialize SAC policy
        
        Args:
            obs_size (int): Observation dimension
            action_size (int): Action dimension
            device (str): Device for computation
            lr (float): Learning rate
            gamma (float): Discount factor
            tau (float): Soft update coefficient
            alpha (float): Entropy regularization coefficient
            action_scale (float): Action scaling factor
        """
        self.obs_size = obs_size
        self.action_size = action_size
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_scale = action_scale
        
        logger.info(f"🧠 Initializing SAC Policy")
        logger.info(f"   Obs size: {obs_size}, Action size: {action_size}")
        logger.info(f"   Device: {device}, LR: {lr}")
        
        # Networks
        self.actor = Actor(obs_size, action_size).to(device)
        self.critic = Critic(obs_size, action_size).to(device)
        self.critic_target = Critic(obs_size, action_size).to(device)
        
        # Copy weights to target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        self.target_entropy = -action_size
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        logger.info("✅ SAC Policy initialized successfully")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action from policy
        
        Args:
            state (np.ndarray): Current state
            deterministic (bool): Whether to use deterministic action selection
            
        Returns:
            np.ndarray: Selected action
        """
        # Ensure consistent dtype (float32)
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state)
        
        action = action.cpu().numpy().flatten()
        return action * self.action_scale
    
    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> Optional[Dict[str, float]]:
        """
        Update policy using SAC algorithm
        
        Args:
            replay_buffer (ReplayBuffer): Experience replay buffer
            batch_size (int): Batch size for training
            
        Returns:
            Optional[Dict[str, float]]: Training metrics
        """
        if len(replay_buffer) < batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states = replay_buffer.sample(batch_size)
        
        # Current alpha
        alpha = self.log_alpha.exp()
        
        # Compute target Q values
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - alpha * next_log_probs
            q_target = rewards + self.gamma * q_target
        
        # Critic loss
        q1_current, q2_current = self.critic(states, actions)
        critic_loss = F.mse_loss(q1_current, q_target) + F.mse_loss(q2_current, q_target)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_probs - q_new).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Alpha loss (entropy regularization)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item()
        }
    
    def save(self, filepath: str):
        """Save policy to file"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'config': {
                'obs_size': self.obs_size,
                'action_size': self.action_size,
                'lr': self.lr,
                'gamma': self.gamma,
                'tau': self.tau,
                'action_scale': self.action_scale
            }
        }, filepath)
        
        logger.info(f"💾 Policy saved to {filepath}")
    
    def load(self, filepath: str):
        """Load policy from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        
        logger.info(f"📂 Policy loaded from {filepath}")

def test_sac_policy():
    """Test SAC policy implementation"""
    logger.info("🧪 Testing SAC Policy implementation...")
    
    # Create policy
    policy = SACPolicy(obs_size=18, action_size=6, device='cpu')
    
    # Test action selection
    state = np.random.randn(18)
    action = policy.select_action(state)
    logger.info(f"✅ Action selection test: {action.shape}")
    
    # Test replay buffer
    replay_buffer = ReplayBuffer(capacity=1000, device='cpu')
    
    # Add some dummy experiences
    for _ in range(50):
        state = np.random.randn(18)
        action = np.random.randn(6)
        reward = np.random.randn()
        next_state = np.random.randn(18)
        replay_buffer.push(state, action, reward, next_state)
    
    # Test policy update
    metrics = policy.update(replay_buffer, batch_size=16)
    logger.info(f"✅ Policy update test: {metrics}")
    
    logger.info("🎉 SAC Policy test completed successfully!")

if __name__ == "__main__":
    test_sac_policy()