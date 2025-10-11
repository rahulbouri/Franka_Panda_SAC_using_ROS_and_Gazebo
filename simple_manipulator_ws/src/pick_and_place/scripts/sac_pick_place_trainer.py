#!/usr/bin/env python3

"""
High-Level SAC Strategic Controller for Pick-and-Place Tasks
Implements hierarchical control with SAC as high-level strategic decision maker

This module implements the high-level SAC controller that learns strategic
task sequencing and phase transitions, while low-level controllers handle
precise motion execution using physics-informed priors.

Author: Physics-Informed RL Implementation
Date: 2024
"""

import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque
from typing import Dict, List, Tuple, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HighLevelReplayBuffer:
    """Experience replay buffer for high-level SAC strategic decisions"""
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        # Statistics for monitoring
        self.total_added = 0
        self.total_sampled = 0
    
    def add(self, state, action, reward, next_state, done, info=None):
        """Add high-level strategic experience to buffer"""
        experience = (state, action, reward, next_state, done, info)
        self.buffer.append(experience)
        self.total_added += 1
    
    def sample(self, batch_size):
        """Sample batch of high-level strategic decisions"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, infos = zip(*batch)
        
        self.total_sampled += batch_size
        
        return {
            'states': np.array(states),
            'actions': np.array(actions), 
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones),
            'infos': list(infos)
        }
    
    def __len__(self):
        return len(self.buffer)
    
    def get_stats(self):
        """Get buffer statistics"""
        return {
            'buffer_size': len(self.buffer),
            'total_added': self.total_added,
            'total_sampled': self.total_sampled
        }


class HighLevelActor(nn.Module):
    """
    High-level strategic actor network for SAC
    
    Outputs strategic decisions for task phase transitions and object selection
    """
    
    def __init__(self, state_dim=42, action_dim=8, hidden_dim=256):
        """
        Initialize high-level strategic actor
        
        Args:
            state_dim: Input state dimension (joint states + vision + context)
            action_dim: Strategic action dimension (8: 5 phases + 3 object selection)
            hidden_dim: Hidden layer dimension
        """
        super(HighLevelActor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Strategic decision network
        self.strategic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log_std heads for strategic actions
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Activation functions
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"High-level strategic actor initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        """Forward pass for strategic decisions"""
        x = self.strategic_net(state)
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Numerical stability
        
        return mean, log_std
    
    def sample(self, state):
        """Sample strategic action from policy"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        # Apply tanh activation for bounded actions
        y_t = self.output_activation(x_t)
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob
    
    def decode_strategic_action(self, action: torch.Tensor) -> Dict[str, float]:
        """
        Decode strategic action into interpretable components
        
        Args:
            action: Strategic action vector (batch_size, 8)
            
        Returns:
            strategic_decisions: Dictionary with decoded strategic components
        """
        action_np = action.detach().cpu().numpy()
        
        strategic_decisions = {
            'phase_transition': action_np[0, :5],  # Phase transition probabilities
            'object_selection': action_np[0, 5:8],  # Object selection preferences
            'exploration_bonus': action_np[0, 7]  # Exploration vs exploitation
        }
        
        return strategic_decisions


class HighLevelCritic(nn.Module):
    """
    High-level strategic critic network for SAC
    
    Evaluates the value of strategic decisions for task completion
    """
    
    def __init__(self, state_dim=42, action_dim=8, hidden_dim=256):
        """
        Initialize high-level strategic critic
        
        Args:
            state_dim: Input state dimension
            action_dim: Strategic action dimension
            hidden_dim: Hidden layer dimension
        """
        super(HighLevelCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Strategic value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"High-level strategic critic initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state, action):
        """Forward pass to compute strategic action value"""
        x = torch.cat([state, action], dim=1)
        return self.value_net(x)


class HighLevelSACAgent:
    """
    High-level SAC agent for strategic pick-and-place task planning
    
    Learns strategic decisions for task phase transitions and object selection
    """
    
    def __init__(self, state_dim=42, action_dim=8, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        """
        Initialize high-level SAC agent
        
        Args:
            state_dim: State dimension (joint states + vision + context)
            action_dim: Strategic action dimension (8: 5 phases + 3 object selection)
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update rate
            alpha: Entropy coefficient
        """
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # High-level strategic networks
        self.actor = HighLevelActor(state_dim, action_dim)
        self.critic1 = HighLevelCritic(state_dim, action_dim)
        self.critic2 = HighLevelCritic(state_dim, action_dim)
        
        # Target networks for stable learning
        self.critic1_target = HighLevelCritic(state_dim, action_dim)
        self.critic2_target = HighLevelCritic(state_dim, action_dim)
        
        # Initialize target networks with same weights
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Entropy coefficient
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Training statistics
        self.training_step = 0
        self.total_updates = 0
        
        logger.info(f"High-level SAC agent initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def select_strategic_action(self, state, evaluate=False):
        """
        Select strategic action using current high-level policy
        
        Args:
            state: Current state (joint states + vision + context)
            evaluate: Whether to use deterministic policy
            
        Returns:
            strategic_action: Strategic decision for phase transitions and object selection
        """
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if evaluate:
            with torch.no_grad():
                action, _ = self.actor.sample(state_tensor)
        else:
            action, _ = self.actor.sample(state_tensor)
        
        strategic_action = action.detach().cpu().numpy().flatten()
        
        # Decode strategic action for logging
        decoded = self.actor.decode_strategic_action(action)
        
        return strategic_action, decoded
    
    def update_strategic_policy(self, replay_buffer, batch_size=256):
        """
        Update high-level strategic policy using SAC algorithm
        
        Args:
            replay_buffer: High-level experience replay buffer
            batch_size: Batch size for training
        """
        
        if len(replay_buffer) < batch_size:
            return
        
        # Sample batch of strategic experiences
        batch_data = replay_buffer.sample(batch_size)
        if batch_data is None:
            return
        
        states = torch.FloatTensor(batch_data['states'])
        actions = torch.FloatTensor(batch_data['actions'])
        rewards = torch.FloatTensor(batch_data['rewards']).unsqueeze(1)
        next_states = torch.FloatTensor(batch_data['next_states'])
        dones = torch.BoolTensor(batch_data['dones']).unsqueeze(1)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + self.gamma * (1 - dones.float()) * q_next
        
        # Current Q-values
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        # Critic losses
        critic1_loss = nn.MSELoss()(q1, q_target)
        critic2_loss = nn.MSELoss()(q2, q_target)
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update entropy coefficient
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        # Update statistics
        self.training_step += 1
        self.total_updates += 1
        
        if self.training_step % 100 == 0:
            logger.info(f"Strategic policy update {self.training_step}: "
                      f"Actor Loss={actor_loss.item():.4f}, "
                      f"Critic1 Loss={critic1_loss.item():.4f}, "
                      f"Alpha={self.alpha.item():.4f}")
    
    def _soft_update(self, local_model, target_model):
        """Soft update target network parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save_strategic_policy(self, filepath):
        """
        Save high-level strategic policy
        
        Args:
            filepath: Path to save the model
        """
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'training_step': self.training_step,
            'total_updates': self.total_updates,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'tau': self.tau,
                'alpha': self.alpha
            }
        }, filepath)
        
        logger.info(f"High-level strategic policy saved to {filepath}")
    
    def load_strategic_policy(self, filepath):
        """
        Load high-level strategic policy
        
        Args:
            filepath: Path to load the model from
        """
        
        checkpoint = torch.load(filepath)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.training_step = checkpoint.get('training_step', 0)
        self.total_updates = checkpoint.get('total_updates', 0)
        
        logger.info(f"High-level strategic policy loaded from {filepath}")
        logger.info(f"Training step: {self.training_step}, Total updates: {self.total_updates}")


class HierarchicalPickPlaceTrainer:
    """
    Main trainer class for hierarchical SAC pick-and-place learning
    
    Coordinates high-level strategic learning with low-level physics-informed control
    """
    
    def __init__(self, 
                 max_episodes=20000,
                 max_steps=500,
                 batch_size=256,
                 update_frequency=1,
                 save_frequency=100,
                 models_dir="models"):
        """
        Initialize hierarchical SAC trainer
        
        Args:
            max_episodes: Maximum number of training episodes
            max_steps: Maximum steps per episode
            batch_size: Batch size for training
            update_frequency: How often to update networks
            save_frequency: How often to save model
            models_dir: Directory to save models
        """
        
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency
        self.models_dir = models_dir
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize hierarchical environment
        from pick_place_sac_env import PickPlaceSACEnvironment
        self.env = PickPlaceSACEnvironment(max_episode_steps=max_steps)
        
        # Initialize high-level SAC agent for strategic decisions
        self.high_level_agent = HighLevelSACAgent(
            state_dim=42,  # 14 joint states + 20 object features + 8 context
            action_dim=8   # 8 strategic actions (5 phases + 3 object selection)
        )
        
        # Initialize high-level replay buffer
        self.high_level_replay_buffer = HighLevelReplayBuffer(capacity=100000)
        
        # Initialize residual controller for low-level physics-informed control
        from residual_controller import ResidualController
        self.residual_controller = ResidualController()
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.strategic_success_rate = []
        self.training_step = 0
        self.episode_count = 0
        
        # Performance tracking
        self.best_success_rate = 0.0
        self.convergence_episodes = []
        
        logger.info("Hierarchical Pick-and-Place Trainer initialized!")
        logger.info(f"High-level agent: state_dim=42, action_dim=8")
        logger.info(f"Training for {max_episodes} episodes with {max_steps} max steps")
    
    def train_hierarchical_policy(self):
        """Main hierarchical training loop"""
        logger.info("Starting hierarchical SAC training...")
        
        for episode in range(self.max_episodes):
            episode_reward = 0
            episode_length = 0
            strategic_decisions = []
            
            # Reset environment
            state = self.env.reset()
            
            for step in range(self.max_steps):
                # High-level strategic decision making
                strategic_action, decoded_decision = self.high_level_agent.select_strategic_action(state)
                strategic_decisions.append(decoded_decision)
                
                # Convert strategic action to low-level commands
                low_level_commands = self._strategic_to_low_level(strategic_action, state)
                
                # Execute low-level commands using physics-informed control
                next_state, reward, done, info = self.env.step(low_level_commands)
                
                # Store high-level strategic experience
                strategic_info = {
                    'phase_transition': decoded_decision['phase_transition'],
                    'object_selection': decoded_decision['object_selection'],
                    'exploration_bonus': decoded_decision['exploration_bonus'],
                    'low_level_success': info.get('low_level_success', False),
                    'task_progress': info.get('task_progress', 0.0)
                }
                
                self.high_level_replay_buffer.add(state, strategic_action, reward, next_state, done, strategic_info)
                
                # Update high-level strategic policy
                if len(self.high_level_replay_buffer) > self.batch_size and step % self.update_frequency == 0:
                    self.high_level_agent.update_strategic_policy(self.high_level_replay_buffer, self.batch_size)
                    self.training_step += 1
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Log progress
                if step % 100 == 0:
                    logger.info(f"Episode {episode}, Step {step}, Reward: {reward:.3f}, "
                              f"Phase: {info.get('state_machine_state', 0)}, "
                              f"Strategic Decision: {decoded_decision}")
                
                if done:
                    break
            
            # Calculate episode statistics
            success = info.get('task_completed', False)
            strategic_success = len([d for d in strategic_decisions if d['exploration_bonus'] > 0.5]) / len(strategic_decisions)
            
            # Log episode results
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.strategic_success_rate.append(strategic_success)
            self.episode_count += 1
            
            logger.info(f"Episode {episode} completed: Reward={episode_reward:.3f}, "
                       f"Length={episode_length}, Success={success}, "
                       f"Strategic Success Rate={strategic_success:.3f}")
            
            # Save models
            if episode % self.save_frequency == 0:
                self._save_models(episode)
                
                # Track best performance
                if success and episode_reward > self.best_success_rate:
                    self.best_success_rate = episode_reward
                    self._save_models(episode, suffix="_best")
                    logger.info(f"New best model saved at episode {episode}")
            
            # Check for convergence
            if self._check_convergence(episode):
                logger.info(f"Training converged at episode {episode}")
                break
        
        logger.info("Hierarchical training completed!")
        self._save_models(self.episode_count, suffix="_final")
    
    def _strategic_to_low_level(self, strategic_action: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Convert high-level strategic action to low-level joint commands
        
        Args:
            strategic_action: High-level strategic decision (8,)
            state: Current state (42,)
            
        Returns:
            low_level_commands: Joint position targets (7,)
        """
        
        # Extract current joint positions and velocities
        joint_positions = state[:7]
        joint_velocities = state[7:14]
        
        # Decode strategic action
        phase_transition = strategic_action[:5]
        object_selection = strategic_action[5:8]
        
        # Determine target joint positions based on strategic decision
        # This is a simplified mapping - in practice, this would use inverse kinematics
        
        # Neutral pose as base
        neutral_pose = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        # Phase-based target modifications
        phase_idx = np.argmax(phase_transition)
        
        if phase_idx == 0:  # Home phase
            target_pose = neutral_pose
        elif phase_idx == 1:  # Approach phase
            target_pose = neutral_pose + np.array([0.1, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0])
        elif phase_idx == 2:  # Grasp phase
            target_pose = neutral_pose + np.array([0.2, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1])
        elif phase_idx == 3:  # Transport phase
            target_pose = neutral_pose + np.array([-0.1, 0.1, 0.2, -0.1, 0.2, -0.1, 0.2])
        else:  # Place phase
            target_pose = neutral_pose + np.array([-0.2, 0.0, 0.3, -0.2, 0.3, -0.2, 0.3])
        
        # Add object selection influence
        object_influence = np.sum(object_selection) * 0.1
        target_pose += object_influence * np.random.normal(0, 0.1, 7)
        
        # Smooth transition from current position
        alpha = 0.3  # Smoothing factor
        low_level_commands = alpha * target_pose + (1 - alpha) * joint_positions
        
        return low_level_commands
    
    def _save_models(self, episode: int, suffix: str = ""):
        """Save hierarchical models"""
        
        # Save high-level strategic policy
        high_level_path = os.path.join(self.models_dir, f"high_level_sac_episode_{episode}{suffix}.pth")
        self.high_level_agent.save_strategic_policy(high_level_path)
        
        # Save residual controller
        residual_path = os.path.join(self.models_dir, f"residual_controller_episode_{episode}{suffix}.pth")
        self.residual_controller.save_model(residual_path)
        
        logger.info(f"Models saved at episode {episode}")
    
    def _check_convergence(self, episode: int) -> bool:
        """Check if training has converged"""
        
        if len(self.episode_rewards) < 100:
            return False
        
        # Check if average reward over last 100 episodes is stable
        recent_rewards = self.episode_rewards[-100:]
        avg_reward = np.mean(recent_rewards)
        reward_std = np.std(recent_rewards)
        
        # Check if success rate is consistently high
        recent_success = self.strategic_success_rate[-100:]
        avg_success = np.mean(recent_success)
        
        # Convergence criteria
        reward_converged = reward_std < 50.0 and avg_reward > 100.0
        success_converged = avg_success > 0.8
        
        if reward_converged and success_converged:
            self.convergence_episodes.append(episode)
            return True
        
        return False
    
    def evaluate_hierarchical_policy(self, num_episodes=10):
        """
        Evaluate trained hierarchical policy
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            evaluation_results: Dictionary with evaluation metrics
        """
        
        logger.info(f"Evaluating hierarchical policy over {num_episodes} episodes...")
        
        eval_rewards = []
        eval_lengths = []
        success_count = 0
        strategic_success_count = 0
        
        for episode in range(num_episodes):
            episode_reward = 0
            episode_length = 0
            strategic_decisions = []
            
            state = self.env.reset()
            
            for step in range(self.max_steps):
                # High-level strategic decision (evaluation mode)
                strategic_action, decoded_decision = self.high_level_agent.select_strategic_action(
                    state, evaluate=True
                )
                strategic_decisions.append(decoded_decision)
                
                # Convert to low-level commands
                low_level_commands = self._strategic_to_low_level(strategic_action, state)
                
                # Execute
                next_state, reward, done, info = self.env.step(low_level_commands)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    if info.get('task_completed', False):
                        success_count += 1
                    break
            
            # Calculate strategic success
            strategic_success = len([d for d in strategic_decisions if d['exploration_bonus'] > 0.5]) / len(strategic_decisions)
            if strategic_success > 0.7:
                strategic_success_count += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            logger.info(f"Eval Episode {episode}: Reward={episode_reward:.3f}, Length={episode_length}, "
                       f"Success={info.get('task_completed', False)}, Strategic Success={strategic_success:.3f}")
        
        # Calculate evaluation metrics
        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        task_success_rate = success_count / num_episodes
        strategic_success_rate = strategic_success_count / num_episodes
        
        evaluation_results = {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'task_success_rate': task_success_rate,
            'strategic_success_rate': strategic_success_rate,
            'reward_std': np.std(eval_rewards),
            'length_std': np.std(eval_lengths)
        }
        
        logger.info(f"Hierarchical Policy Evaluation Results:")
        logger.info(f"  Average Reward: {avg_reward:.3f} ± {np.std(eval_rewards):.3f}")
        logger.info(f"  Average Length: {avg_length:.1f} ± {np.std(eval_lengths):.1f}")
        logger.info(f"  Task Success Rate: {task_success_rate:.3f}")
        logger.info(f"  Strategic Success Rate: {strategic_success_rate:.3f}")
        
        return evaluation_results


def main():
    """Main function for hierarchical SAC training"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Hierarchical SAC Pick-and-Place Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'test'],
                       help='Mode: train, eval, or test')
    parser.add_argument('--episodes', type=int, default=20000,
                       help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--model_path', type=str, default='models/high_level_sac_episode_best.pth',
                       help='Path to trained model for evaluation')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    # Initialize ROS node
    rospy.init_node('hierarchical_sac_trainer', anonymous=True)
    
    try:
        # Initialize hierarchical trainer
        trainer = HierarchicalPickPlaceTrainer(
            max_episodes=args.episodes,
            max_steps=args.steps,
            batch_size=256,
            update_frequency=1,
            save_frequency=100
        )
        
        if args.mode == 'train':
            # Start hierarchical training
            trainer.train_hierarchical_policy()
            
            # Evaluate trained policy
            trainer.evaluate_hierarchical_policy(num_episodes=args.eval_episodes)
            
        elif args.mode == 'eval':
            # Load trained model and evaluate
            if os.path.exists(args.model_path):
                trainer.high_level_agent.load_strategic_policy(args.model_path)
                trainer.evaluate_hierarchical_policy(num_episodes=args.eval_episodes)
            else:
                logger.error(f"Model file not found: {args.model_path}")
                
        elif args.mode == 'test':
            # Test mode - run a few episodes with detailed logging
            logger.info("Running test episodes...")
            trainer.evaluate_hierarchical_policy(num_episodes=3)
        
    except rospy.ROSInterruptException:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        logger.info("Hierarchical SAC training session ended")


if __name__ == "__main__":
    main()
