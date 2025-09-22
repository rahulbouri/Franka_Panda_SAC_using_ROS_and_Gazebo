#!/usr/bin/env python3

"""
SAC Training Script for Pick-and-Place Tasks
Integrates with state machine and online replay buffer

Author: RL Training Implementation
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
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience replay buffer for SAC training"""
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample batch from buffer"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for SAC"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
        
    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = self.output_activation(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob


class Critic(nn.Module):
    """Critic network for SAC"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.ReLU()
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return self.fc4(x)


class SACAgent:
    """Soft Actor-Critic agent for pick-and-place tasks"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Networks
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        
        # Target networks
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        
        # Copy weights to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Target entropy
        self.target_entropy = -action_dim
        
        # Log alpha
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, state, evaluate=False):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        if evaluate:
            with torch.no_grad():
                action, _ = self.actor.sample(state)
        else:
            action, _ = self.actor.sample(state)
        
        return action.detach().cpu().numpy().flatten()
    
    def update(self, replay_buffer, batch_size=256):
        """Update networks using SAC algorithm"""
        if len(replay_buffer) < batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + self.gamma * (1 - dones.float()) * q_next
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        critic1_loss = nn.MSELoss()(q1, q_target)
        critic2_loss = nn.MSELoss()(q2, q_target)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
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
        
        # Update alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # Update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
    
    def _soft_update(self, local_model, target_model):
        """Soft update target network"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'log_alpha': self.log_alpha
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.log_alpha = checkpoint['log_alpha']


class PickPlaceSACTrainer:
    """Main trainer class for SAC pick-and-place learning"""
    
    def __init__(self, 
                 max_episodes=1000,
                 max_steps=1000,
                 batch_size=256,
                 update_frequency=1,
                 save_frequency=100):
        """
        Initialize SAC trainer
        
        Args:
            max_episodes: Maximum number of training episodes
            max_steps: Maximum steps per episode
            batch_size: Batch size for training
            update_frequency: How often to update networks
            save_frequency: How often to save model
        """
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency
        
        # Initialize environment
        from pick_place_sac_env import PickPlaceSACEnvironment
        self.env = PickPlaceSACEnvironment(max_episode_steps=max_steps)
        
        # Initialize agent
        self.agent = SACAgent(
            state_dim=42,  # 14 joint states + 20 object features + 8 context
            action_dim=7   # 7 joint torques for Franka Panda
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_step = 0
        
        logger.info("SAC Pick-and-Place Trainer initialized!")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting SAC training...")
        
        for episode in range(self.max_episodes):
            episode_reward = 0
            episode_length = 0
            
            # Reset environment
            state = self.env.reset()
            
            for step in range(self.max_steps):
                # Select action
                action = self.agent.select_action(state)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Update agent
                if len(self.replay_buffer) > self.batch_size and step % self.update_frequency == 0:
                    self.agent.update(self.replay_buffer, self.batch_size)
                    self.training_step += 1
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Log progress
                if step % 100 == 0:
                    logger.info(f"Episode {episode}, Step {step}, Reward: {reward:.3f}, "
                              f"State: {info['state_machine_state']}, Progress: {info['completion']:.3f}")
                
                if done:
                    break
            
            # Log episode results
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            logger.info(f"Episode {episode} completed: Reward={episode_reward:.3f}, "
                       f"Length={episode_length}, Avg Reward={np.mean(self.episode_rewards[-10:]):.3f}")
            
            # Save model
            if episode % self.save_frequency == 0:
                self.agent.save(f"pick_place_sac_episode_{episode}.pth")
                logger.info(f"Model saved at episode {episode}")
            
            # Early stopping if performance is good
            if len(self.episode_rewards) >= 100:
                avg_reward = np.mean(self.episode_rewards[-100:])
                if avg_reward > 1000:  # Adjust threshold as needed
                    logger.info(f"Training converged at episode {episode} with avg reward {avg_reward:.3f}")
                    break
        
        logger.info("Training completed!")
        self.agent.save("pick_place_sac_final.pth")
    
    def evaluate(self, num_episodes=10):
        """Evaluate trained agent"""
        logger.info("Evaluating trained agent...")
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            episode_reward = 0
            episode_length = 0
            
            state = self.env.reset()
            
            for step in range(self.max_steps):
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            logger.info(f"Eval Episode {episode}: Reward={episode_reward:.3f}, Length={episode_length}")
        
        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        
        logger.info(f"Evaluation completed: Avg Reward={avg_reward:.3f}, Avg Length={avg_length:.3f}")
        
        return avg_reward, avg_length


def main():
    """Main function"""
    try:
        # Initialize trainer
        trainer = PickPlaceSACTrainer(
            max_episodes=1000,
            max_steps=1000,
            batch_size=256,
            update_frequency=1,
            save_frequency=100
        )
        
        # Train agent
        trainer.train()
        
        # Evaluate agent
        trainer.evaluate(num_episodes=10)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
