#!/usr/bin/env python3
"""
Prioritized Experience Replay Buffer for Online RL Training
Implements efficient experience storage and sampling for manipulator training

Author: RL Training Implementation
Date: 2024
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer for manipulator RL training
    
    This implementation provides:
    - Efficient experience storage and retrieval
    - Prioritized sampling based on TD error
    - Multi-objective reward tracking
    - Comprehensive logging and debugging
    """
    
    def __init__(self, 
                 capacity: int = 100000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 device: str = 'cpu'):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity (int): Maximum number of experiences to store
            alpha (float): Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta (float): Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment (float): Beta increment per sampling
            device (str): Device for tensor operations
        """
        logger.info(f"🚀 Initializing Prioritized Replay Buffer")
        logger.info(f"📊 Capacity: {capacity}, Alpha: {alpha}, Beta: {beta}")
        
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device
        
        # Storage for experiences
        self.states = np.zeros((capacity, 18), dtype=np.float32)  # 18D observation
        self.actions = np.zeros((capacity, 6), dtype=np.float32)  # 6D action
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, 18), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        
        # Priority storage
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
        
        # Buffer state
        self.position = 0
        self.size = 0
        self.total_added = 0
        
        # Statistics
        self.stats = {
            'total_added': 0,
            'total_sampled': 0,
            'high_priority_samples': 0,
            'avg_reward': 0.0,
            'avg_priority': 0.0
        }
        
        logger.info("✅ Prioritized Replay Buffer initialized successfully")
    
    def add(self, 
            state: np.ndarray, 
            action: np.ndarray, 
            reward: float, 
            next_state: np.ndarray, 
            done: bool,
            priority: Optional[float] = None):
        """
        Add experience to the replay buffer
        
        Args:
            state (np.ndarray): Current state observation
            action (np.ndarray): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state observation
            done (bool): Whether episode is done
            priority (float, optional): Priority for this experience
        """
        logger.debug(f"➕ Adding experience to buffer (position {self.position})")
        
        # Validate inputs
        if not self._validate_experience(state, action, reward, next_state, done):
            logger.warning("⚠️ Invalid experience, skipping")
            return
        
        # Store experience
        self.states[self.position] = state.astype(np.float32)
        self.actions[self.position] = action.astype(np.float32)
        self.rewards[self.position] = float(reward)
        self.next_states[self.position] = next_state.astype(np.float32)
        self.dones[self.position] = bool(done)
        
        # Set priority
        if priority is None:
            priority = self.max_priority
        self.priorities[self.position] = priority
        
        # Update buffer state
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        
        self.total_added += 1
        self.stats['total_added'] += 1
        
        # Update statistics
        self._update_stats(reward, priority)
        
        logger.debug(f"✅ Experience added. Buffer size: {self.size}/{self.capacity}")
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences with prioritized sampling
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            Tuple[Dict, np.ndarray, np.ndarray]: (batch_data, indices, weights)
        """
        logger.debug(f"🎲 Sampling batch of size {batch_size} from buffer of size {self.size}")
        
        if self.size < batch_size:
            logger.warning(f"⚠️ Buffer size ({self.size}) < batch size ({batch_size})")
            batch_size = self.size
        
        if self.size == 0:
            logger.error("❌ Cannot sample from empty buffer")
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(
            self.size, 
            size=batch_size, 
            replace=False, 
            p=probabilities
        )
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Prepare batch data
        batch_data = {
            'states': torch.tensor(self.states[indices], dtype=torch.float32, device=self.device),
            'actions': torch.tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            'rewards': torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            'next_states': torch.tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            'dones': torch.tensor(self.dones[indices], dtype=torch.bool, device=self.device)
        }
        
        # Update statistics
        self.stats['total_sampled'] += batch_size
        high_priority_count = np.sum(priorities[indices] > np.percentile(priorities, 75))
        self.stats['high_priority_samples'] += high_priority_count
        
        logger.debug(f"✅ Batch sampled. High priority samples: {high_priority_count}/{batch_size}")
        
        return batch_data, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled experiences
        
        Args:
            indices (np.ndarray): Indices of experiences to update
            priorities (np.ndarray): New priorities
        """
        logger.debug(f"🔄 Updating priorities for {len(indices)} experiences")
        
        # Clip priorities to avoid zero
        priorities = np.clip(priorities, 1e-6, None)
        
        # Update priorities
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
        
        logger.debug(f"✅ Priorities updated. Max priority: {self.max_priority:.4f}")
    
    def _validate_experience(self, 
                           state: np.ndarray, 
                           action: np.ndarray, 
                           reward: float, 
                           next_state: np.ndarray, 
                           done: bool) -> bool:
        """Validate experience before adding to buffer"""
        
        # Check shapes
        if state.shape != (18,):
            logger.warning(f"⚠️ Invalid state shape: {state.shape} != (18,)")
            return False
        
        if action.shape != (6,):
            logger.warning(f"⚠️ Invalid action shape: {action.shape} != (6,)")
            return False
        
        if next_state.shape != (18,):
            logger.warning(f"⚠️ Invalid next_state shape: {next_state.shape} != (18,)")
            return False
        
        # Check for NaN or Inf values
        for name, data in [("state", state), ("action", action), ("next_state", next_state)]:
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                logger.warning(f"⚠️ {name} contains NaN or Inf values")
                return False
        
        if np.isnan(reward) or np.isinf(reward):
            logger.warning(f"⚠️ Reward is NaN or Inf: {reward}")
            return False
        
        return True
    
    def _update_stats(self, reward: float, priority: float):
        """Update buffer statistics"""
        # Update average reward (exponential moving average)
        if self.stats['total_added'] == 1:
            self.stats['avg_reward'] = reward
        else:
            alpha = 0.01  # Learning rate for moving average
            self.stats['avg_reward'] = (1 - alpha) * self.stats['avg_reward'] + alpha * reward
        
        # Update average priority
        if self.stats['total_added'] == 1:
            self.stats['avg_priority'] = priority
        else:
            alpha = 0.01
            self.stats['avg_priority'] = (1 - alpha) * self.stats['avg_priority'] + alpha * priority
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            'size': self.size,
            'capacity': self.capacity,
            'utilization': self.size / self.capacity,
            'total_added': self.stats['total_added'],
            'total_sampled': self.stats['total_sampled'],
            'avg_reward': self.stats['avg_reward'],
            'avg_priority': self.stats['avg_priority'],
            'max_priority': self.max_priority,
            'beta': self.beta
        }
    
    def save(self, filepath: str):
        """Save buffer to file"""
        logger.info(f"💾 Saving replay buffer to {filepath}")
        
        data = {
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_states': self.next_states[:self.size],
            'dones': self.dones[:self.size],
            'priorities': self.priorities[:self.size],
            'position': self.position,
            'size': self.size,
            'total_added': self.total_added,
            'stats': self.stats,
            'max_priority': self.max_priority
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info("✅ Replay buffer saved successfully")
    
    def load(self, filepath: str):
        """Load buffer from file"""
        logger.info(f"📂 Loading replay buffer from {filepath}")
        
        if not os.path.exists(filepath):
            logger.error(f"❌ File not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Restore buffer state
            self.states[:data['size']] = data['states']
            self.actions[:data['size']] = data['actions']
            self.rewards[:data['size']] = data['rewards']
            self.next_states[:data['size']] = data['next_states']
            self.dones[:data['size']] = data['dones']
            self.priorities[:data['size']] = data['priorities']
            
            self.position = data['position']
            self.size = data['size']
            self.total_added = data['total_added']
            self.stats = data['stats']
            self.max_priority = data['max_priority']
            
            logger.info(f"✅ Replay buffer loaded successfully. Size: {self.size}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load replay buffer: {e}")
            return False
    
    def clear(self):
        """Clear the replay buffer"""
        logger.info("🗑️ Clearing replay buffer")
        
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(False)
        self.priorities.fill(0)
        
        self.position = 0
        self.size = 0
        self.total_added = 0
        self.max_priority = 1.0
        
        self.stats = {
            'total_added': 0,
            'total_sampled': 0,
            'high_priority_samples': 0,
            'avg_reward': 0.0,
            'avg_priority': 0.0
        }
        
        logger.info("✅ Replay buffer cleared")

# Test function for debugging
def test_replay_buffer():
    """Test function to verify replay buffer functionality"""
    logger.info("🧪 Testing replay buffer")
    
    try:
        # Create buffer
        buffer = PrioritizedReplayBuffer(capacity=1000)
        
        # Add some experiences
        logger.info("➕ Adding test experiences...")
        for i in range(100):
            state = np.random.randn(18)
            action = np.random.randn(6)
            reward = np.random.randn()
            next_state = np.random.randn(18)
            done = i % 10 == 9
            
            buffer.add(state, action, reward, next_state, done)
        
        # Test sampling
        logger.info("🎲 Testing sampling...")
        batch_data, indices, weights = buffer.sample(32)
        
        if batch_data is not None:
            logger.info(f"✅ Sampling successful. Batch size: {len(batch_data['states'])}")
            logger.info(f"📊 Weights shape: {weights.shape}, Indices shape: {indices.shape}")
        else:
            logger.error("❌ Sampling failed")
        
        # Test statistics
        stats = buffer.get_stats()
        logger.info(f"📊 Buffer stats: {stats}")
        
        logger.info("✅ Replay buffer test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Replay buffer test failed: {e}")
        raise

if __name__ == "__main__":
    test_replay_buffer()
