#!/usr/bin/env python3
"""
Integrated LNN-RL Framework for 6-DOF Manipulator Control
Combines enhanced Lagrangian Neural Network with comprehensive reward function.
"""

import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs

from enhanced_lnn_manipulator import EnhancedLNNManipulator, ConstraintAwareRewardFunction

class IntegratedLNNRLFramework:
    """
    Integrated framework combining LNN dynamics modeling with RL training.
    Implements Model-Based RL with physics-informed neural networks.
    """
    
    def __init__(self, target_position=[0.6, 0.0, 0.8], max_episode_steps=1000):
        rospy.init_node('integrated_lnn_rl_framework', anonymous=True)
        
        # Joint configuration
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Joint limits
        self.joint_limits = {
            'shoulder_pan_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 150.0},
            'shoulder_lift_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 150.0},
            'elbow_joint': {'lower': -3.141592653589793, 'upper': 3.141592653589793, 'effort': 150.0},
            'wrist_1_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 28.0},
            'wrist_2_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 28.0},
            'wrist_3_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 28.0}
        }
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
        
        # Enhanced LNN for dynamics modeling
        self.lnn = EnhancedLNNManipulator(
            n_joints=len(self.joint_names),
            obs_size=2 * len(self.joint_names),  # positions + velocities
            action_size=len(self.joint_names),
            dt=0.01,
            device=self.device
        ).to(self.device)
        
        # Reward function
        self.reward_function = ConstraintAwareRewardFunction(
            joint_names=self.joint_names,
            target_position=target_position,
            max_episode_steps=max_episode_steps
        )
        
        # RL components
        self.obs_size = 2 * len(self.joint_names)
        self.action_size = len(self.joint_names)
        
        # Actor-Critic networks
        self.actor = self._create_actor_network().to(self.device)
        self.critic = self._create_critic_network().to(self.device)
        self.critic_target = self._create_critic_network().to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=3e-4)
        self.lnn_optimizer = optim.AdamW(self.lnn.parameters(), lr=3e-4)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000, device=self.device)
        
        # ROS setup
        self._setup_ros()
        
        # Training parameters
        self.max_episode_steps = max_episode_steps
        self.episode_count = 0
        self.step_count = 0
        
        # State tracking
        self.current_joint_states = None
        self.joint_states_received = False
        self.prev_torques = None
        
        rospy.loginfo("🚀 Integrated LNN-RL Framework initialized")
        
        # Wait for joint states
        self._wait_for_joint_states()
    
    def _create_actor_network(self):
        """Create actor network for policy."""
        return nn.Sequential(
            nn.Linear(self.obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size),
            nn.Tanh()  # Actions in [-1, 1]
        )
    
    def _create_critic_network(self):
        """Create critic network for value function."""
        return nn.Sequential(
            nn.Linear(self.obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def _setup_ros(self):
        """Setup ROS publishers and subscribers."""
        # Effort publishers
        self.effort_publishers = {}
        for joint in self.joint_names:
            topic = f'/manipulator/{joint}_effort/command'
            self.effort_publishers[joint] = rospy.Publisher(topic, Float64, queue_size=1)
        
        # Joint states subscriber
        self.joint_states_sub = rospy.Subscriber(
            '/manipulator/joint_states', 
            JointState, 
            self._joint_states_callback
        )
        
        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
    
    def _joint_states_callback(self, msg):
        """Callback for joint state messages."""
        self.current_joint_states = msg
        self.joint_states_received = True
    
    def _wait_for_joint_states(self, timeout=10.0):
        """Wait for joint states to be received."""
        rospy.loginfo("Waiting for joint states...")
        start_time = time.time()
        while not self.joint_states_received and (time.time() - start_time) < timeout:
            rospy.sleep(0.1)
        
        if self.joint_states_received:
            rospy.loginfo("✓ Joint states received!")
        else:
            rospy.logerr("✗ Timeout waiting for joint states!")
            return False
        return True
    
    def get_observation(self):
        """Get current observation (joint positions and velocities)."""
        if self.current_joint_states is None:
            return None
        
        positions = []
        velocities = []
        
        for joint in self.joint_names:
            try:
                idx = self.current_joint_states.name.index(joint)
                positions.append(self.current_joint_states.position[idx])
                velocities.append(self.current_joint_states.velocity[idx])
            except ValueError:
                positions.append(0.0)
                velocities.append(0.0)
        
        return np.array(positions), np.array(velocities)
    
    def get_end_effector_pose(self):
        """Get end-effector pose using TF."""
        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rospy.Time())
            pose = PoseStamped()
            pose.header = transform.header
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            return pose
        except Exception as e:
            rospy.logwarn(f"Could not get end-effector pose: {e}")
            return None
    
    def apply_action(self, action):
        """Apply action (normalized torques) to joints."""
        # Scale action to torque limits
        torques = action * np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
        
        # Apply torques
        for i, joint in enumerate(self.joint_names):
            msg = Float64()
            msg.data = float(torques[i])
            self.effort_publishers[joint].publish(msg)
        
        return torques
    
    def reset_episode(self):
        """Reset episode with random valid joint configuration."""
        rospy.loginfo(f"🔄 Resetting episode {self.episode_count}")
        
        # Sample random valid joint positions
        positions = []
        for joint in self.joint_names:
            limits = self.joint_limits[joint]
            # Sample within safe bounds
            lower = limits['lower'] + 0.5
            upper = limits['upper'] - 0.5
            pos = np.random.uniform(lower, upper)
            positions.append(pos)
        
        positions = np.array(positions)
        velocities = np.zeros_like(positions)
        
        # Set joint configuration
        self._set_joint_configuration(positions)
        
        self.episode_count += 1
        self.step_count = 0
        self.prev_torques = None
        
        return positions, velocities
    
    def _set_joint_configuration(self, positions):
        """Set joint configuration using Gazebo service."""
        try:
            from gazebo_msgs.srv import SetModelConfiguration
            rospy.wait_for_service('/gazebo/set_model_configuration')
            set_config = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
            
            response = set_config(
                model_name='manipulator',
                urdf_param_name='robot_description',
                joint_names=self.joint_names,
                joint_positions=positions.tolist()
            )
            
            if response.success:
                rospy.loginfo("✓ Joint configuration set successfully")
            else:
                rospy.logwarn(f"⚠️  Failed to set joint configuration: {response.status_message}")
                
        except Exception as e:
            rospy.logwarn(f"⚠️  Could not set joint configuration: {e}")
    
    def step(self, action):
        """Execute one RL step."""
        self.step_count += 1
        
        # Get current observation
        positions, velocities = self.get_observation()
        if positions is None:
            return None, 0.0, True, {}
        
        # Apply action
        torques = self.apply_action(action)
        
        # Wait for dynamics to update
        rospy.sleep(0.1)
        
        # Get new observation
        new_positions, new_velocities = self.get_observation()
        if new_positions is None:
            return None, 0.0, True, {}
        
        # Compute reward
        reward, reward_info = self.reward_function.compute_reward(
            new_positions, new_velocities, torques, 
            self.step_count, self.joint_limits, self.prev_torques
        )
        
        # Update previous torques
        self.prev_torques = torques.copy()
        
        # Check for termination
        constraint_violations = self.reward_function.compute_constraint_penalty(
            new_positions, self.joint_limits
        )[1]
        done = len(constraint_violations) > 0 or self.step_count >= self.max_episode_steps
        
        # Prepare observation
        observation = np.concatenate([new_positions, new_velocities])
        
        info = {
            'constraint_violations': constraint_violations,
            'torques_applied': torques,
            'reward_info': reward_info
        }
        
        return observation, reward, done, info
    
    def train_lnn(self, batch_size=64, num_epochs=10):
        """Train the LNN on collected data."""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        obs, actions, rewards, next_obs = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        
        # Train LNN
        self.lnn_optimizer.zero_grad()
        
        # Predict next state
        pred_next_obs = self.lnn(obs, actions)
        
        # Compute loss
        lnn_loss = nn.MSELoss()(pred_next_obs, next_obs)
        
        # Backward pass
        lnn_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lnn.parameters(), 1.0)
        self.lnn_optimizer.step()
        
        return lnn_loss.item()
    
    def train_actor_critic(self, batch_size=64, num_epochs=10):
        """Train actor-critic networks."""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        obs, actions, rewards, next_obs = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        
        # Train critic
        self.critic_optimizer.zero_grad()
        values = self.critic(obs).squeeze()
        target_values = rewards + 0.99 * self.critic_target(next_obs).squeeze()
        critic_loss = nn.MSELoss()(values, target_values.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Train actor
        self.actor_optimizer.zero_grad()
        pred_actions = self.actor(obs)
        actor_loss = -self.critic(obs).mean()  # Simple policy gradient
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update target critic
        self._soft_update_target_critic()
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update_target_critic(self, tau=0.005):
        """Soft update target critic network."""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def run_episode(self, render=False):
        """Run a complete episode."""
        rospy.loginfo(f"🎬 Starting episode {self.episode_count}")
        
        # Reset episode
        positions, velocities = self.reset_episode()
        observation = np.concatenate([positions, velocities])
        
        total_reward = 0.0
        episode_data = []
        
        for step in range(self.max_episode_steps):
            # Get action from actor
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                action = self.actor(obs_tensor).cpu().numpy()[0]
            
            # Execute step
            next_observation, reward, done, info = self.step(action)
            
            if next_observation is None:
                rospy.logwarn("⚠️  Episode terminated due to observation failure")
                break
            
            # Store experience
            self.replay_buffer.push(observation, action, reward, next_observation)
            episode_data.append((observation, action, reward, next_observation))
            
            total_reward += reward
            observation = next_observation
            
            # Log progress
            if step % 50 == 0:
                rospy.loginfo(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
                if info['constraint_violations']:
                    rospy.logwarn(f"    Constraint violations: {info['constraint_violations']}")
            
            if done:
                rospy.loginfo(f"🏁 Episode {self.episode_count} completed in {step} steps")
                break
        
        # Train networks
        if len(self.replay_buffer) > 100:
            lnn_loss = self.train_lnn()
            critic_loss, actor_loss = self.train_actor_critic()
            
            rospy.loginfo(f"📊 Training losses - LNN: {lnn_loss:.4f}, Critic: {critic_loss:.4f}, Actor: {actor_loss:.4f}")
        
        # Episode summary
        summary = self.reward_function.get_episode_summary()
        rospy.loginfo(f"📊 Episode {self.episode_count} summary:")
        rospy.loginfo(f"  Total reward: {total_reward:.3f}")
        rospy.loginfo(f"  Steps: {step}")
        rospy.loginfo(f"  Average reward: {total_reward/step:.3f}")
        if summary:
            rospy.loginfo(f"  Energy efficiency: {summary.get('energy_efficiency', 0):.3f}")
        
        return total_reward, step, summary

class ReplayBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, obs, action, reward, next_obs):
        """Add experience to buffer."""
        self.buffer.append((obs, action, reward, next_obs))
    
    def sample(self, batch_size):
        """Sample batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs = zip(*batch)
        return np.array(obs), np.array(actions), np.array(rewards), np.array(next_obs)
    
    def __len__(self):
        return len(self.buffer)

def main():
    """Main function to run the integrated LNN-RL framework."""
    try:
        # Create framework
        framework = IntegratedLNNRLFramework(
            target_position=[0.6, 0.0, 0.8],  # Target position for coke can
            max_episode_steps=1000
        )
        
        # Run training episodes
        rospy.loginfo("🚀 Starting Integrated LNN-RL Training")
        
        for episode in range(10):  # Train for 10 episodes
            total_reward, steps, summary = framework.run_episode()
            rospy.sleep(1.0)  # Pause between episodes
        
        rospy.loginfo("🎉 Integrated LNN-RL training completed!")
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Framework interrupted by user")
    except Exception as e:
        rospy.logerr(f"Framework failed: {e}")

if __name__ == "__main__":
    main()
