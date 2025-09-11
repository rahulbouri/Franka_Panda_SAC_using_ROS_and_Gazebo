#!/usr/bin/env python3
"""
Online RL Training Script for 6-DOF Manipulator
Implements Phase 1: Environment Integration & Data Collection

Author: RL Training Implementation
Date: 2024
"""

import os
import sys
import numpy as np
import torch
import rospy
import logging
import time
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import threading
from collections import deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from env.manipulator_env import ManipulatorEnvironment, EnvironmentConfig
from models.replay_buffer import PrioritizedReplayBuffer

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('online_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OnlineTrainer:
    """
    Online RL Trainer for 6-DOF Manipulator
    
    This class implements Phase 1 of the training plan:
    - Environment Integration & Data Collection
    - Replay Buffer Management
    - Comprehensive Logging and Debugging
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the online trainer
        
        Args:
            config (Dict): Training configuration
        """
        logger.info("🚀 Initializing Online RL Trainer")
        logger.info(f"📊 Configuration: {config}")
        
        # Store configuration
        self.config = config
        
        # Initialize ROS if not already done
        if not rospy.get_node_uri():
            rospy.init_node('online_trainer', anonymous=True)
            logger.info("✅ ROS node initialized: online_trainer")
        
        # Initialize environment
        self._initialize_environment()
        
        # Initialize replay buffer
        self._initialize_replay_buffer()
        
        # Initialize training state
        self._initialize_training_state()
        
        # Initialize logging and monitoring
        self._initialize_logging()
        
        logger.info("✅ Online RL Trainer initialized successfully")
    
    def _initialize_environment(self):
        """Initialize the manipulator environment"""
        logger.info("🌍 Initializing manipulator environment")
        
        # Create environment configuration
        env_config = EnvironmentConfig(
            max_episode_steps=self.config.get('max_episode_steps', 50),
            ros_rate=self.config.get('ros_rate', 50),
            target_tolerance=self.config.get('target_tolerance', 0.05),
            accuracy_weight=self.config.get('accuracy_weight', 15.0),
            speed_weight=self.config.get('speed_weight', 5.0),
            energy_weight=self.config.get('energy_weight', 0.01)
        )
        
        # Create environment
        self.env = ManipulatorEnvironment(env_config)
        logger.info("✅ Environment initialized successfully")
    
    def _initialize_replay_buffer(self):
        """Initialize the replay buffer"""
        logger.info("📚 Initializing replay buffer")
        
        buffer_config = self.config.get('replay_buffer', {})
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_config.get('capacity', 100000),
            alpha=buffer_config.get('alpha', 0.6),
            beta=buffer_config.get('beta', 0.4),
            beta_increment=buffer_config.get('beta_increment', 0.001),
            device=self.config.get('device', 'cpu')
        )
        
        logger.info("✅ Replay buffer initialized successfully")
    
    def _initialize_training_state(self):
        """Initialize training state variables"""
        logger.info("📊 Initializing training state")
        
        # Training counters
        self.episode_count = 0
        self.total_steps = 0
        self.total_episodes = self.config.get('total_episodes', 1000)
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'avg_rewards': [],
            'avg_lengths': [],
            'energy_consumption': [],
            'accuracy_rewards': [],
            'speed_rewards': [],
            'energy_rewards': []
        }
        
        # Best performance tracking
        self.best_episode_reward = float('-inf')
        self.best_success_rate = 0.0
        
        logger.info("✅ Training state initialized successfully")
    
    def _initialize_logging(self):
        """Initialize logging and monitoring systems"""
        logger.info("📝 Initializing logging systems")
        
        # Create logs directory
        self.logs_dir = self.config.get('logs_dir', './logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Create models directory
        self.models_dir = self.config.get('models_dir', './models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize episode log
        self.episode_log_file = os.path.join(self.logs_dir, 'episodes.jsonl')
        
        # Initialize metrics file
        self.metrics_file = os.path.join(self.logs_dir, 'training_metrics.json')
        
        logger.info(f"✅ Logging initialized. Logs dir: {self.logs_dir}, Models dir: {self.models_dir}")
    
    def train(self):
        """Main training loop"""
        logger.info("🎓 Starting online training")
        logger.info(f"📊 Training for {self.total_episodes} episodes")
        
        try:
            # Initial data collection phase
            if self.config.get('initial_data_collection', True):
                self._collect_initial_data()
            
            # Main training loop
            for episode in range(self.total_episodes):
                logger.info(f"🎬 Starting episode {episode + 1}/{self.total_episodes}")
                
                # Run episode
                episode_metrics = self._run_episode(episode)
                
                # Update training metrics
                self._update_training_metrics(episode_metrics)
                
                # Log episode results
                self._log_episode(episode, episode_metrics)
                
                # Save checkpoints
                if (episode + 1) % self.config.get('save_frequency', 100) == 0:
                    self._save_checkpoint(episode + 1)
                
                # Print progress
                if (episode + 1) % self.config.get('print_frequency', 10) == 0:
                    self._print_progress(episode + 1)
                
                # Update replay buffer beta
                self._update_beta()
                
                self.episode_count += 1
            
            logger.info("🎉 Training completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("⏹️ Training interrupted by user")
        except Exception as e:
            logger.error(f"❌ Training failed with error: {e}")
            raise
        finally:
            # Final cleanup
            self._finalize_training()
    
    def _collect_initial_data(self):
        """Collect initial random data for replay buffer"""
        logger.info("🎲 Collecting initial random data")
        
        initial_episodes = self.config.get('initial_episodes', 50)
        logger.info(f"📊 Collecting {initial_episodes} random episodes")
        
        for episode in range(initial_episodes):
            logger.debug(f"🎬 Random episode {episode + 1}/{initial_episodes}")
            
            # Reset environment with random target
            obs = self.env.reset()
            
            for step in range(self.config.get('max_episode_steps', 50)):
                # Random action
                action = self._get_random_action()
                
                # Step environment
                next_obs, reward, done, info = self.env.step(action)
                
                # Store experience
                self.replay_buffer.add(obs, action, reward, next_obs, done)
                
                obs = next_obs
                if done:
                    break
            
            # Log progress
            if (episode + 1) % 10 == 0:
                logger.info(f"📊 Initial data collection: {episode + 1}/{initial_episodes} episodes")
        
        logger.info(f"✅ Initial data collection complete. Buffer size: {self.replay_buffer.size}")
    
    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single training episode"""
        logger.debug(f"🎬 Running episode {episode + 1}")
        
        # Reset environment
        obs = self.env.reset()
        
        # Episode metrics
        episode_reward = 0.0
        episode_length = 0
        episode_energy = 0.0
        task_completed = False
        
        # Reward components
        accuracy_rewards = []
        speed_rewards = []
        energy_rewards = []
        
        # Run episode
        for step in range(self.config.get('max_episode_steps', 50)):
            logger.debug(f"🎮 Episode {episode + 1}, Step {step + 1}")
            
            # Get action (random for now, will be replaced with policy)
            action = self._get_action(obs, episode)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Store experience in replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            
            # Update episode metrics
            episode_reward += reward
            episode_length += 1
            episode_energy += info.get('energy_consumed', 0.0)
            
            # Track reward components
            if 'reward_info' in info:
                reward_info = info['reward_info']
                accuracy_rewards.append(reward_info.get('accuracy_reward', 0.0))
                speed_rewards.append(reward_info.get('speed_reward', 0.0))
                energy_rewards.append(reward_info.get('energy_reward', 0.0))
            
            # Check task completion
            if info.get('task_status') == 'success':
                task_completed = True
                logger.info(f"🎉 Task completed in episode {episode + 1}, step {step + 1}")
            
            # Update state
            obs = next_obs
            self.total_steps += 1
            
            if done:
                break
        
        # Prepare episode metrics
        episode_metrics = {
            'episode': episode + 1,
            'reward': episode_reward,
            'length': episode_length,
            'energy': episode_energy,
            'task_completed': task_completed,
            'avg_accuracy_reward': np.mean(accuracy_rewards) if accuracy_rewards else 0.0,
            'avg_speed_reward': np.mean(speed_rewards) if speed_rewards else 0.0,
            'avg_energy_reward': np.mean(energy_rewards) if energy_rewards else 0.0,
            'success_rate': 1.0 if task_completed else 0.0
        }
        
        logger.info(f"📊 Episode {episode + 1} complete: reward={episode_reward:.4f}, "
                   f"length={episode_length}, completed={task_completed}")
        
        return episode_metrics
    
    def _get_action(self, obs: np.ndarray, episode: int) -> np.ndarray:
        """Get action for current observation"""
        # For Phase 1, we use random actions
        # In later phases, this will be replaced with policy network
        
        if episode < self.config.get('random_episodes', 100):
            # Random exploration
            action = self._get_random_action()
            logger.debug(f"🎲 Random action: {action}")
        else:
            # TODO: Implement policy network action selection
            action = self._get_random_action()
            logger.debug(f"🤖 Policy action (placeholder): {action}")
        
        return action
    
    def _get_random_action(self) -> np.ndarray:
        """Generate random action within joint limits"""
        # Random torque commands (adjust limits based on your robot)
        max_torque = 20.0  # Nm
        action = np.random.uniform(-max_torque, max_torque, 6)
        return action
    
    def _update_training_metrics(self, episode_metrics: Dict[str, Any]):
        """Update training metrics with episode results"""
        logger.debug("📊 Updating training metrics")
        
        # Update rolling averages
        self.episode_rewards.append(episode_metrics['reward'])
        self.episode_lengths.append(episode_metrics['length'])
        self.success_rate.append(episode_metrics['success_rate'])
        
        # Update training metrics
        self.training_metrics['episode_rewards'].append(episode_metrics['reward'])
        self.training_metrics['episode_lengths'].append(episode_metrics['length'])
        self.training_metrics['success_rates'].append(episode_metrics['success_rate'])
        self.training_metrics['energy_consumption'].append(episode_metrics['energy'])
        self.training_metrics['accuracy_rewards'].append(episode_metrics['avg_accuracy_reward'])
        self.training_metrics['speed_rewards'].append(episode_metrics['avg_speed_reward'])
        self.training_metrics['energy_rewards'].append(episode_metrics['avg_energy_reward'])
        
        # Calculate rolling averages
        if len(self.episode_rewards) > 0:
            self.training_metrics['avg_rewards'].append(np.mean(self.episode_rewards))
            self.training_metrics['avg_lengths'].append(np.mean(self.episode_lengths))
        
        # Update best performance
        if episode_metrics['reward'] > self.best_episode_reward:
            self.best_episode_reward = episode_metrics['reward']
            logger.info(f"🏆 New best episode reward: {self.best_episode_reward:.4f}")
        
        if episode_metrics['success_rate'] > self.best_success_rate:
            self.best_success_rate = episode_metrics['success_rate']
            logger.info(f"🏆 New best success rate: {self.best_success_rate:.4f}")
    
    def _log_episode(self, episode: int, episode_metrics: Dict[str, Any]):
        """Log episode results to file"""
        logger.debug(f"📝 Logging episode {episode + 1}")
        
        # Add timestamp
        episode_metrics['timestamp'] = datetime.now().isoformat()
        
        # Log to JSONL file
        with open(self.episode_log_file, 'a') as f:
            f.write(json.dumps(episode_metrics) + '\n')
        
        # Save metrics periodically
        if (episode + 1) % self.config.get('metrics_save_frequency', 10) == 0:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
    
    def _print_progress(self, episode: int):
        """Print training progress"""
        logger.info("=" * 60)
        logger.info(f"📊 Training Progress - Episode {episode}")
        logger.info("=" * 60)
        
        # Current episode metrics
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards)
            avg_length = np.mean(self.episode_lengths)
            current_success_rate = np.mean(self.success_rate)
            
            logger.info(f"📈 Recent Performance (last 100 episodes):")
            logger.info(f"   Average Reward: {avg_reward:.4f}")
            logger.info(f"   Average Length: {avg_length:.2f}")
            logger.info(f"   Success Rate: {current_success_rate:.2%}")
        
        # Buffer statistics
        buffer_stats = self.replay_buffer.get_stats()
        logger.info(f"📚 Replay Buffer:")
        logger.info(f"   Size: {buffer_stats['size']}/{buffer_stats['capacity']}")
        logger.info(f"   Utilization: {buffer_stats['utilization']:.2%}")
        logger.info(f"   Total Added: {buffer_stats['total_added']}")
        logger.info(f"   Total Sampled: {buffer_stats['total_sampled']}")
        logger.info(f"   Average Reward: {buffer_stats['avg_reward']:.4f}")
        
        # Best performance
        logger.info(f"🏆 Best Performance:")
        logger.info(f"   Best Episode Reward: {self.best_episode_reward:.4f}")
        logger.info(f"   Best Success Rate: {self.best_success_rate:.2%}")
        
        logger.info("=" * 60)
    
    def _update_beta(self):
        """Update replay buffer beta parameter"""
        if hasattr(self.replay_buffer, 'beta'):
            self.replay_buffer.beta = min(1.0, self.replay_buffer.beta + self.replay_buffer.beta_increment)
            logger.debug(f"🔄 Updated beta to {self.replay_buffer.beta:.4f}")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        logger.info(f"💾 Saving checkpoint for episode {episode}")
        
        checkpoint = {
            'episode': episode,
            'total_steps': self.total_steps,
            'training_metrics': self.training_metrics,
            'best_episode_reward': self.best_episode_reward,
            'best_success_rate': self.best_success_rate,
            'config': self.config
        }
        
        checkpoint_file = os.path.join(self.models_dir, f'checkpoint_episode_{episode}.json')
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Save replay buffer
        buffer_file = os.path.join(self.models_dir, f'replay_buffer_episode_{episode}.pkl')
        self.replay_buffer.save(buffer_file)
        
        logger.info(f"✅ Checkpoint saved: {checkpoint_file}")
    
    def _finalize_training(self):
        """Finalize training and cleanup"""
        logger.info("🏁 Finalizing training")
        
        # Save final metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        # Save final checkpoint
        self._save_checkpoint(self.episode_count)
        
        # Close environment
        self.env.close()
        
        # Print final statistics
        self._print_final_statistics()
        
        logger.info("✅ Training finalized successfully")
    
    def _print_final_statistics(self):
        """Print final training statistics"""
        logger.info("=" * 60)
        logger.info("🏁 FINAL TRAINING STATISTICS")
        logger.info("=" * 60)
        
        if len(self.episode_rewards) > 0:
            logger.info(f"📊 Overall Performance:")
            logger.info(f"   Total Episodes: {self.episode_count}")
            logger.info(f"   Total Steps: {self.total_steps}")
            logger.info(f"   Average Reward: {np.mean(self.episode_rewards):.4f}")
            logger.info(f"   Average Length: {np.mean(self.episode_lengths):.2f}")
            logger.info(f"   Final Success Rate: {np.mean(self.success_rate):.2%}")
            logger.info(f"   Best Episode Reward: {self.best_episode_reward:.4f}")
            logger.info(f"   Best Success Rate: {self.best_success_rate:.2%}")
        
        # Buffer statistics
        buffer_stats = self.replay_buffer.get_stats()
        logger.info(f"📚 Replay Buffer Final Stats:")
        logger.info(f"   Total Experiences: {buffer_stats['total_added']}")
        logger.info(f"   Total Samples: {buffer_stats['total_sampled']}")
        logger.info(f"   Average Reward: {buffer_stats['avg_reward']:.4f}")
        
        logger.info("=" * 60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Online RL Training for 6-DOF Manipulator')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Total number of episodes')
    parser.add_argument('--max_steps', type=int, default=50, help='Max steps per episode')
    parser.add_argument('--ros_rate', type=int, default=50, help='ROS control rate')
    
    # Reward weights
    parser.add_argument('--accuracy_weight', type=float, default=15.0, help='Accuracy reward weight')
    parser.add_argument('--speed_weight', type=float, default=5.0, help='Speed reward weight')
    parser.add_argument('--energy_weight', type=float, default=0.01, help='Energy reward weight')
    
    # Replay buffer parameters
    parser.add_argument('--buffer_capacity', type=int, default=100000, help='Replay buffer capacity')
    parser.add_argument('--initial_episodes', type=int, default=50, help='Initial random episodes')
    
    # Logging parameters
    parser.add_argument('--logs_dir', type=str, default='./logs', help='Logs directory')
    parser.add_argument('--models_dir', type=str, default='./models', help='Models directory')
    parser.add_argument('--save_frequency', type=int, default=100, help='Save frequency')
    parser.add_argument('--print_frequency', type=int, default=10, help='Print frequency')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'total_episodes': args.episodes,
        'max_episode_steps': args.max_steps,
        'ros_rate': args.ros_rate,
        'accuracy_weight': args.accuracy_weight,
        'speed_weight': args.speed_weight,
        'energy_weight': args.energy_weight,
        'replay_buffer': {
            'capacity': args.buffer_capacity,
            'alpha': 0.6,
            'beta': 0.4,
            'beta_increment': 0.001
        },
        'initial_episodes': args.initial_episodes,
        'logs_dir': args.logs_dir,
        'models_dir': args.models_dir,
        'save_frequency': args.save_frequency,
        'print_frequency': args.print_frequency,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create and run trainer
    trainer = OnlineTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
