#!/usr/bin/env python3
"""
Comprehensive RL Training Framework for 6-DOF Manipulator
Based on ROS Noetic and Gazebo simulation

This framework implements:
1. Model-Based RL with Lagrangian Neural Networks
2. Constraint-aware control with joint limits
3. Real-time ROS communication
4. Multi-objective reward function
5. Episode management and data collection
"""

import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym
from gym import spaces
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, Point
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
import time
import json
import os
from collections import deque
import matplotlib.pyplot as plt

class ConstraintAwareController:
    """Constraint-aware controller with joint limits and barrier functions"""
    
    def __init__(self, joint_limits, barrier_strength=1.0):
        self.joint_limits = joint_limits
        self.barrier_strength = barrier_strength
        self.joint_names = list(joint_limits.keys())
        
    def apply_barrier_function(self, joint_positions):
        """Apply barrier function to enforce joint limits"""
        barrier_penalty = 0.0
        barrier_gradients = np.zeros(len(self.joint_names))
        
        for i, joint_name in enumerate(self.joint_names):
            pos = joint_positions[i]
            lower = self.joint_limits[joint_name]['lower']
            upper = self.joint_limits[joint_name]['upper']
            
            # Barrier function: -log(upper - pos) - log(pos - lower)
            if pos > lower and pos < upper:
                barrier_penalty += -np.log(upper - pos) - np.log(pos - lower)
                barrier_gradients[i] = 1.0/(upper - pos) - 1.0/(pos - lower)
            else:
                # Large penalty for constraint violation
                barrier_penalty += 1000.0
                barrier_gradients[i] = 1000.0 if pos >= upper else -1000.0
                
        return barrier_penalty, barrier_gradients
    
    def clamp_efforts(self, efforts):
        """Clamp efforts to joint limits"""
        clamped_efforts = np.zeros_like(efforts)
        
        for i, joint_name in enumerate(self.joint_names):
            effort = efforts[i]
            max_effort = self.joint_limits[joint_name]['effort']
            clamped_efforts[i] = np.clip(effort, -max_effort, max_effort)
            
        return clamped_efforts

class LagrangianNeuralNetwork(nn.Module):
    """Lagrangian Neural Network for dynamics learning with constraints"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, constraint_dim=6):
        super(LagrangianNeuralNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_dim = constraint_dim
        
        # Main dynamics network
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Constraint network for joint limits
        self.constraint_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, constraint_dim),
            nn.Sigmoid()
        )
        
        # Value function for policy
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and std
        )
        
    def forward(self, state, action=None):
        """Forward pass through the network"""
        if action is not None:
            # Dynamics prediction
            x = torch.cat([state, action], dim=-1)
            next_state = self.dynamics_net(x)
            
            # Constraint prediction
            constraints = self.constraint_net(state)
            
            return next_state, constraints
        else:
            # Policy and value prediction
            value = self.value_net(state)
            policy_params = self.policy_net(state)
            
            mean = policy_params[..., :self.action_dim]
            std = torch.softplus(policy_params[..., self.action_dim:]) + 1e-5
            
            return value, mean, std

class RLEnvironment(gym.Env):
    """RL Environment for 6-DOF Manipulator Training"""
    
    def __init__(self):
        super(RLEnvironment, self).__init__()
        
        # Initialize ROS
        rospy.init_node('rl_manipulator_env', anonymous=True)
        
        # Joint configuration
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Joint limits (from URDF)
        self.joint_limits = {
            'shoulder_pan_joint': {'lower': -6.28, 'upper': 6.28, 'effort': 150.0},
            'shoulder_lift_joint': {'lower': -6.28, 'upper': 6.28, 'effort': 150.0},
            'elbow_joint': {'lower': -3.14, 'upper': 3.14, 'effort': 150.0},
            'wrist_1_joint': {'lower': -6.28, 'upper': 6.28, 'effort': 28.0},
            'wrist_2_joint': {'lower': -6.28, 'upper': 6.28, 'effort': 28.0},
            'wrist_3_joint': {'lower': -6.28, 'upper': 6.28, 'effort': 28.0}
        }
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.joint_names),), dtype=np.float32
        )
        
        # State: joint positions + velocities + end-effector pose + target pose
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6*2 + 3 + 3,), dtype=np.float32
        )
        
        # ROS publishers and subscribers
        self.joint_publishers = {}
        for joint_name in self.joint_names:
            topic = f'/manipulator/{joint_name}_effort/command'
            self.joint_publishers[joint_name] = rospy.Publisher(
                topic, Float64, queue_size=1
            )
        
        self.joint_state_sub = rospy.Subscriber(
            '/manipulator/joint_states', JointState, self.joint_state_callback
        )
        
        # Gazebo services
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # State variables
        self.current_joint_states = None
        self.target_pose = None
        self.episode_step = 0
        self.max_episode_steps = 200
        
        # Constraint-aware controller
        self.controller = ConstraintAwareController(self.joint_limits)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=100000)
        
        # Wait for ROS services
        rospy.wait_for_message('/manipulator/joint_states', JointState, timeout=10)
        
    def joint_state_callback(self, msg):
        """Callback for joint state updates"""
        self.current_joint_states = msg
        
    def get_observation(self):
        """Get current observation"""
        if self.current_joint_states is None:
            return np.zeros(self.observation_space.shape[0])
        
        # Extract joint positions and velocities
        joint_positions = np.zeros(len(self.joint_names))
        joint_velocities = np.zeros(len(self.joint_names))
        
        for i, joint_name in enumerate(self.joint_names):
            try:
                idx = self.current_joint_states.name.index(joint_name)
                joint_positions[i] = self.current_joint_states.position[idx]
                if len(self.current_joint_states.velocity) > idx:
                    joint_velocities[i] = self.current_joint_states.velocity[idx]
            except ValueError:
                pass
        
        # Get end-effector pose (simplified - using forward kinematics approximation)
        ee_position = self.forward_kinematics(joint_positions)
        
        # Combine observation
        observation = np.concatenate([
            joint_positions,
            joint_velocities, 
            ee_position,
            self.target_pose[:3] if self.target_pose is not None else np.zeros(3)
        ])
        
        return observation.astype(np.float32)
    
    def forward_kinematics(self, joint_positions):
        """Simplified forward kinematics for end-effector position"""
        # This is a simplified approximation - in practice, you'd use proper FK
        # For now, return a rough estimate based on joint positions
        x = 0.5 + 0.3 * np.cos(joint_positions[0]) * np.cos(joint_positions[1])
        y = 0.3 * np.sin(joint_positions[0]) * np.cos(joint_positions[1])
        z = 0.5 + 0.3 * np.sin(joint_positions[1]) + 0.2 * np.sin(joint_positions[2])
        
        return np.array([x, y, z])
    
    def reset(self):
        """Reset environment for new episode"""
        self.episode_step = 0
        
        # Randomize target pose
        self.target_pose = np.array([
            np.random.uniform(0.3, 0.9),  # x
            np.random.uniform(-0.3, 0.3), # y  
            np.random.uniform(0.6, 0.8)   # z
        ])
        
        # Randomize initial joint positions
        initial_joints = np.zeros(len(self.joint_names))
        for i, joint_name in enumerate(self.joint_names):
            lower = self.joint_limits[joint_name]['lower']
            upper = self.joint_limits[joint_name]['upper']
            initial_joints[i] = np.random.uniform(lower, upper)
        
        # Set initial joint positions
        self.set_joint_positions(initial_joints)
        
        return self.get_observation()
    
    def set_joint_positions(self, positions):
        """Set joint positions using Gazebo service"""
        # This would require implementing a service call to set joint positions
        # For now, we'll use effort control to move to positions
        pass
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        self.episode_step += 1
        
        # Convert action to efforts (scale by joint effort limits)
        efforts = np.zeros(len(self.joint_names))
        for i, joint_name in enumerate(self.joint_names):
            max_effort = self.joint_limits[joint_name]['effort']
            efforts[i] = action[i] * max_effort
        
        # Apply constraint-aware control
        efforts = self.controller.clamp_efforts(efforts)
        
        # Send effort commands
        for i, joint_name in enumerate(self.joint_names):
            self.joint_publishers[joint_name].publish(Float64(efforts[i]))
        
        # Wait for state update
        rospy.sleep(0.1)
        
        # Get new observation
        observation = self.get_observation()
        
        # Calculate reward
        reward = self.calculate_reward(observation, action)
        
        # Check if episode is done
        done = self.episode_step >= self.max_episode_steps
        
        # Store experience in replay buffer
        if hasattr(self, 'last_observation') and hasattr(self, 'last_action'):
            self.replay_buffer.append((
                self.last_observation,
                self.last_action,
                reward,
                observation,
                done
            ))
        
        self.last_observation = observation
        self.last_action = action
        
        info = {
            'episode_step': self.episode_step,
            'target_pose': self.target_pose,
            'constraint_violations': self.check_constraint_violations(observation)
        }
        
        return observation, reward, done, info
    
    def calculate_reward(self, observation, action):
        """Calculate multi-objective reward"""
        # Extract components
        joint_positions = observation[:6]
        joint_velocities = observation[6:12]
        ee_position = observation[12:15]
        target_position = observation[15:18]
        
        # Accuracy reward (distance to target)
        distance_to_target = np.linalg.norm(ee_position - target_position)
        accuracy_reward = -distance_to_target * 10.0
        
        # Speed reward (encourage faster movement)
        speed_reward = -np.linalg.norm(joint_velocities) * 0.1
        
        # Energy efficiency reward (penalize high efforts)
        effort_penalty = -np.linalg.norm(action) * 0.01
        
        # Constraint violation penalty
        constraint_penalty = 0.0
        for i, joint_name in enumerate(self.joint_names):
            pos = joint_positions[i]
            lower = self.joint_limits[joint_name]['lower']
            upper = self.joint_limits[joint_name]['upper']
            if pos < lower or pos > upper:
                constraint_penalty -= 100.0
        
        # Success bonus
        success_bonus = 100.0 if distance_to_target < 0.05 else 0.0
        
        total_reward = accuracy_reward + speed_reward + effort_penalty + constraint_penalty + success_bonus
        
        return total_reward
    
    def check_constraint_violations(self, observation):
        """Check for joint limit violations"""
        joint_positions = observation[:6]
        violations = []
        
        for i, joint_name in enumerate(self.joint_names):
            pos = joint_positions[i]
            lower = self.joint_limits[joint_name]['lower']
            upper = self.joint_limits[joint_name]['upper']
            if pos < lower or pos > upper:
                violations.append(joint_name)
        
        return violations

class ModelBasedRLTrainer:
    """Model-Based RL Trainer with Lagrangian Neural Networks"""
    
    def __init__(self, env, lr=3e-4, batch_size=256):
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.model = LagrangianNeuralNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.batch_size = batch_size
        self.training_data = []
        
    def collect_data(self, num_episodes=100):
        """Collect training data from environment"""
        print(f"Collecting data for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            observation = self.env.reset()
            episode_data = []
            
            for step in range(self.env.max_episode_steps):
                # Random action for data collection
                action = self.env.action_space.sample()
                next_observation, reward, done, info = self.env.step(action)
                
                episode_data.append((observation, action, reward, next_observation, done))
                observation = next_observation
                
                if done:
                    break
            
            self.training_data.extend(episode_data)
            print(f"Episode {episode+1}/{num_episodes} completed")
        
        print(f"Collected {len(self.training_data)} transitions")
    
    def train_model(self, epochs=100):
        """Train the Lagrangian Neural Network"""
        if len(self.training_data) < self.batch_size:
            print("Not enough data for training")
            return
        
        print(f"Training model for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Sample batch
            batch_indices = np.random.choice(len(self.training_data), self.batch_size, replace=False)
            batch = [self.training_data[i] for i in batch_indices]
            
            # Prepare data
            states = torch.FloatTensor([t[0] for t in batch]).to(self.device)
            actions = torch.FloatTensor([t[1] for t in batch]).to(self.device)
            rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
            next_states = torch.FloatTensor([t[3] for t in batch]).to(self.device)
            
            # Forward pass
            predicted_next_states, constraints = self.model(states, actions)
            
            # Dynamics loss
            dynamics_loss = nn.MSELoss()(predicted_next_states, next_states)
            
            # Constraint loss (encourage constraint satisfaction)
            constraint_loss = torch.mean(constraints)
            
            # Total loss
            total_loss = dynamics_loss + 0.1 * constraint_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")
    
    def train_policy(self, epochs=50):
        """Train the policy using the learned model"""
        print("Training policy...")
        
        for epoch in range(epochs):
            # Sample batch from replay buffer
            if len(self.env.replay_buffer) < self.batch_size:
                continue
                
            batch = np.random.choice(len(self.env.replay_buffer), self.batch_size, replace=False)
            batch_data = [self.env.replay_buffer[i] for i in batch]
            
            states = torch.FloatTensor([t[0] for t in batch_data]).to(self.device)
            actions = torch.FloatTensor([t[1] for t in batch_data]).to(self.device)
            rewards = torch.FloatTensor([t[2] for t in batch_data]).to(self.device)
            next_states = torch.FloatTensor([t[3] for t in batch_data]).to(self.device)
            
            # Get policy and value predictions
            values, means, stds = self.model(states)
            
            # Policy loss (REINFORCE)
            dist = Normal(means, stds)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            policy_loss = -(log_probs * rewards).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), rewards)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Policy Epoch {epoch}: Loss = {total_loss.item():.4f}")
    
    def evaluate(self, num_episodes=10):
        """Evaluate the trained policy"""
        print(f"Evaluating policy for {num_episodes} episodes...")
        
        total_rewards = []
        
        for episode in range(num_episodes):
            observation = self.env.reset()
            total_reward = 0
            
            for step in range(self.env.max_episode_steps):
                # Get action from policy
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                    _, mean, std = self.model(state_tensor)
                    dist = Normal(mean, std)
                    action = dist.sample().cpu().numpy()[0]
                
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            total_rewards.append(total_reward)
            print(f"Episode {episode+1}: Reward = {total_reward:.2f}")
        
        avg_reward = np.mean(total_rewards)
        print(f"Average reward: {avg_reward:.2f}")
        return avg_reward

def main():
    """Main training loop"""
    print("🤖 Starting RL Training Framework for 6-DOF Manipulator")
    print("=" * 60)
    
    # Create environment
    env = RLEnvironment()
    
    # Create trainer
    trainer = ModelBasedRLTrainer(env)
    
    # Training loop
    print("\n📊 Phase 1: Data Collection")
    trainer.collect_data(num_episodes=50)
    
    print("\n🧠 Phase 2: Model Training")
    trainer.train_model(epochs=100)
    
    print("\n🎯 Phase 3: Policy Training")
    trainer.train_policy(epochs=50)
    
    print("\n📈 Phase 4: Evaluation")
    avg_reward = trainer.evaluate(num_episodes=10)
    
    print(f"\n✅ Training completed! Average reward: {avg_reward:.2f}")
    
    # Save model
    torch.save(trainer.model.state_dict(), 'lagrangian_manipulator_model.pth')
    print("Model saved as 'lagrangian_manipulator_model.pth'")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
