#!/usr/bin/env python3
"""
Single Episode Model-Based RL Experiment for 6-DOF Manipulator
Tests if the policy can learn correct joint controller with 100x scaling.
"""

import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('single_episode_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleActorCritic(nn.Module):
    """Simple Actor-Critic network for policy learning"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(SimpleActorCritic, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        shared = self.shared_layers(state)
        
        # Actor outputs
        mean = self.actor_mean(shared)
        log_std = self.actor_log_std(shared)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        # Critic output
        value = self.critic(shared)
        
        return mean, std, value
    
    def select_action(self, state, deterministic=False):
        """Select action from policy"""
        mean, std, _ = self.forward(state)
        
        if deterministic:
            return torch.tanh(mean)
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            return torch.tanh(action)

class SimpleDynamicsModel(nn.Module):
    """Simple dynamics model for state prediction"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(SimpleDynamicsModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        """Predict next state"""
        input_tensor = torch.cat([state, action], dim=-1)
        return self.network(input_tensor)

class SingleEpisodeMBRLExperiment:
    """
    Single episode experiment to test if MBRL can learn correct joint controller.
    Includes 100x scaling as requested and comprehensive logging.
    """
    
    def __init__(self):
        logger.info("🚀 Initializing Single Episode MBRL Experiment")
        
        # Initialize ROS
        rospy.init_node('single_episode_mbrl_experiment', anonymous=True)
        
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
            'shoulder_pan_joint': {'lower': -6.28, 'upper': 6.28, 'effort': 150.0},
            'shoulder_lift_joint': {'lower': -6.28, 'upper': 6.28, 'effort': 150.0},
            'elbow_joint': {'lower': -3.14, 'upper': 3.14, 'effort': 150.0},
            'wrist_1_joint': {'lower': -6.28, 'upper': 6.28, 'effort': 28.0},
            'wrist_2_joint': {'lower': -6.28, 'upper': 6.28, 'effort': 28.0},
            'wrist_3_joint': {'lower': -6.28, 'upper': 6.28, 'effort': 28.0}
        }
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # State and action dimensions
        self.state_dim = 2 * len(self.joint_names)  # positions + velocities
        self.action_dim = len(self.joint_names)
        
        # Initialize models
        self._initialize_models()
        
        # Initialize ROS components
        self._initialize_ros()
        
        # Experiment parameters
        self.max_episode_steps = 1000
        self.target_position = np.array([0.6, 0.0, 0.8])  # Target for coke can
        self.position_tolerance = 0.05  # 5cm tolerance
        
        # Action scaling (100x as requested)
        self.action_scale = 100.0
        logger.info(f"Action scaling set to {self.action_scale}x")
        
        # Experiment state
        self.episode_step = 0
        self.total_reward = 0.0
        self.success = False
        
        # Data collection
        self.replay_buffer = deque(maxlen=10000)
        self.episode_data = []
        
        logger.info("✅ Single Episode MBRL Experiment initialized")
    
    def _initialize_models(self):
        """Initialize neural network models"""
        logger.info("🧠 Initializing neural network models")
        
        # Actor-Critic model
        self.actor_critic = SimpleActorCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=256
        ).to(self.device)
        
        # Dynamics model
        self.dynamics_model = SimpleDynamicsModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=256
        ).to(self.device)
        
        # Optimizers
        self.actor_critic_optimizer = optim.AdamW(
            self.actor_critic.parameters(), lr=3e-4
        )
        self.dynamics_optimizer = optim.AdamW(
            self.dynamics_model.parameters(), lr=3e-4
        )
        
        logger.info("✅ Models initialized")
    
    def _initialize_ros(self):
        """Initialize ROS publishers and subscribers"""
        logger.info("🔗 Initializing ROS components")
        
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
        
        # State tracking
        self.current_joint_states = None
        self.joint_states_received = False
        
        # Wait for joint states
        self._wait_for_joint_states()
        
        logger.info("✅ ROS components initialized")
    
    def _joint_states_callback(self, msg):
        """Callback for joint state messages"""
        self.current_joint_states = msg
        self.joint_states_received = True
    
    def _wait_for_joint_states(self, timeout=10.0):
        """Wait for joint states to be received"""
        logger.info("⏳ Waiting for joint states...")
        start_time = time.time()
        while not self.joint_states_received and (time.time() - start_time) < timeout:
            rospy.sleep(0.1)
        
        if self.joint_states_received:
            logger.info("✅ Joint states received!")
        else:
            logger.error("❌ Timeout waiting for joint states!")
            raise RuntimeError("Failed to receive joint states")
    
    def get_observation(self) -> Optional[np.ndarray]:
        """Get current observation (joint positions and velocities)"""
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
        
        return np.concatenate([positions, velocities])
    
    def get_end_effector_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get end-effector position and orientation"""
        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rospy.Time())
            
            position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            orientation = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
            
            return position, orientation
            
        except Exception as e:
            logger.warning(f"Could not get end-effector pose: {e}")
            return None
    
    def apply_action(self, action: np.ndarray) -> np.ndarray:
        """Apply action (normalized torques) to joints with 100x scaling"""
        # Scale action by 100x as requested
        scaled_action = action * self.action_scale
        
        # Scale to torque limits
        torque_limits = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
        torques = scaled_action * torque_limits
        
        # Apply torques
        for i, joint in enumerate(self.joint_names):
            msg = Float64()
            msg.data = float(torques[i])
            self.effort_publishers[joint].publish(msg)
        
        logger.debug(f"Applied torques: {torques}")
        return torques
    
    def compute_reward(self, state: np.ndarray, action: np.ndarray, 
                      next_state: np.ndarray, step: int) -> Tuple[float, Dict]:
        """Compute comprehensive reward function"""
        
        # Get end-effector pose
        ee_pose = self.get_end_effector_pose()
        if ee_pose is None:
            return 0.0, {'error': 'No end-effector pose'}
        
        ee_position, _ = ee_pose
        
        # Position accuracy reward
        position_error = np.linalg.norm(ee_position - self.target_position)
        if position_error < self.position_tolerance:
            accuracy_reward = 100.0  # Success bonus
            self.success = True
        else:
            accuracy_reward = np.exp(-position_error / 0.1)
        
        # Energy efficiency reward
        torques = action * self.action_scale * np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
        energy_penalty = np.sum(np.abs(torques)) * 0.001
        
        # Time efficiency reward
        time_reward = (self.max_episode_steps - step) / self.max_episode_steps
        
        # Constraint penalty
        constraint_penalty = 0.0
        for i, joint in enumerate(self.joint_names):
            pos = state[i]
            limits = self.joint_limits[joint]
            if pos < limits['lower'] or pos > limits['upper']:
                constraint_penalty += 10.0
        
        # Total reward
        total_reward = accuracy_reward - energy_penalty + time_reward - constraint_penalty
        
        reward_info = {
            'accuracy_reward': accuracy_reward,
            'energy_penalty': energy_penalty,
            'time_reward': time_reward,
            'constraint_penalty': constraint_penalty,
            'position_error': position_error,
            'total_reward': total_reward,
            'success': self.success
        }
        
        return total_reward, reward_info
    
    def train_dynamics_model(self, batch_size: int = 64) -> float:
        """Train dynamics model on collected data"""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        # Train dynamics model
        self.dynamics_optimizer.zero_grad()
        pred_next_states = self.dynamics_model(states, actions)
        dynamics_loss = nn.MSELoss()(pred_next_states, next_states)
        dynamics_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), 1.0)
        self.dynamics_optimizer.step()
        
        return dynamics_loss.item()
    
    def train_actor_critic(self, batch_size: int = 64) -> Tuple[float, float]:
        """Train actor-critic model"""
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
        
        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        # Train actor-critic
        self.actor_critic_optimizer.zero_grad()
        
        # Get current values
        mean, std, values = self.actor_critic(states)
        _, _, next_values = self.actor_critic(next_states)
        
        # Compute advantages
        advantages = rewards + 0.99 * next_values.squeeze() - values.squeeze()
        
        # Critic loss
        critic_loss = nn.MSELoss()(values.squeeze(), rewards + 0.99 * next_values.squeeze())
        
        # Actor loss (policy gradient)
        normal = torch.distributions.Normal(mean, std)
        log_probs = normal.log_prob(actions).sum(dim=-1)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Total loss
        total_loss = critic_loss + actor_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
        self.actor_critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def run_episode(self) -> Dict:
        """Run single episode experiment"""
        logger.info("🎬 Starting single episode experiment")
        
        # Reset episode state
        self.episode_step = 0
        self.total_reward = 0.0
        self.success = False
        self.episode_data = []
        
        # Get initial observation
        state = self.get_observation()
        if state is None:
            logger.error("❌ Failed to get initial observation")
            return {'success': False, 'error': 'No initial observation'}
        
        logger.info(f"📊 Initial state: {state[:6]} (positions), {state[6:]} (velocities)")
        
        # Run episode
        for step in range(self.max_episode_steps):
            self.episode_step = step
            
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.actor_critic.select_action(state_tensor, deterministic=False)
                action = action.cpu().numpy()[0]
            
            logger.debug(f"Step {step}: Action = {action}")
            
            # Apply action
            torques = self.apply_action(action)
            
            # Wait for dynamics to update
            rospy.sleep(0.1)
            
            # Get next observation
            next_state = self.get_observation()
            if next_state is None:
                logger.warning(f"⚠️  Step {step}: Failed to get next observation")
                break
            
            # Compute reward
            reward, reward_info = self.compute_reward(state, action, next_state, step)
            self.total_reward += reward
            
            # Store transition
            self.replay_buffer.append((state, action, reward, next_state))
            self.episode_data.append({
                'step': step,
                'state': state.copy(),
                'action': action.copy(),
                'reward': reward,
                'next_state': next_state.copy(),
                'reward_info': reward_info
            })
            
            # Train models every 10 steps
            if step % 10 == 0 and len(self.replay_buffer) > 32:
                dynamics_loss = self.train_dynamics_model()
                actor_loss, critic_loss = self.train_actor_critic()
                
                logger.debug(f"Step {step}: Dynamics loss = {dynamics_loss:.4f}, "
                           f"Actor loss = {actor_loss:.4f}, Critic loss = {critic_loss:.4f}")
            
            # Log progress
            if step % 50 == 0:
                logger.info(f"Step {step}: Reward = {reward:.3f}, Total = {self.total_reward:.3f}, "
                           f"Position error = {reward_info['position_error']:.3f}")
            
            # Check for success
            if self.success:
                logger.info(f"🎉 SUCCESS at step {step}! Position error = {reward_info['position_error']:.3f}")
                break
            
            # Update state
            state = next_state
        
        # Episode summary
        episode_summary = {
            'success': self.success,
            'total_reward': self.total_reward,
            'episode_length': self.episode_step + 1,
            'final_position_error': self.episode_data[-1]['reward_info']['position_error'] if self.episode_data else float('inf'),
            'total_energy': sum([np.sum(np.abs(step['action'] * self.action_scale * np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0]))) for step in self.episode_data]),
            'episode_data': self.episode_data
        }
        
        logger.info(f"📊 Episode Summary:")
        logger.info(f"  Success: {self.success}")
        logger.info(f"  Total Reward: {self.total_reward:.3f}")
        logger.info(f"  Episode Length: {self.episode_step + 1}")
        logger.info(f"  Final Position Error: {episode_summary['final_position_error']:.3f}")
        logger.info(f"  Total Energy: {episode_summary['total_energy']:.3f}")
        
        return episode_summary

def main():
    """Main function to run the experiment"""
    try:
        # Create experiment
        experiment = SingleEpisodeMBRLExperiment()
        
        # Run single episode
        results = experiment.run_episode()
        
        # Print final results
        print("\n" + "="*60)
        print("🎯 SINGLE EPISODE MBRL EXPERIMENT RESULTS")
        print("="*60)
        print(f"Success: {results['success']}")
        print(f"Total Reward: {results['total_reward']:.3f}")
        print(f"Episode Length: {results['episode_length']}")
        print(f"Final Position Error: {results['final_position_error']:.3f}m")
        print(f"Total Energy: {results['total_energy']:.3f}")
        print("="*60)
        
        if results['success']:
            print("🎉 EXPERIMENT SUCCESSFUL! Policy learned to reach target!")
        else:
            print("❌ Experiment failed. Policy needs more training.")
        
    except rospy.ROSInterruptException:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()
