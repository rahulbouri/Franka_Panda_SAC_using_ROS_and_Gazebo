#!/usr/bin/env python3
"""
Simplified Manipulator Environment Wrapper for Online RL Training
This version works without PyKDL dependency for basic functionality

Author: RL Training Implementation
Date: 2024
"""

import os
import sys
import numpy as np
import rospy
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# ROS message imports
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import Float64, Float64MultiArray, Bool
from gazebo_msgs.srv import GetModelState, SetModelState, SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState

# Try to import tf2_geometry_msgs, but don't fail if not available
try:
    from tf2_ros import TransformListener, Buffer
    import tf2_geometry_msgs
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False
    print("⚠️ tf2_geometry_msgs not available, some features may be limited")

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('manipulator_env.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task completion status"""
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class EnvironmentConfig:
    """Configuration for the manipulator environment"""
    # Episode parameters
    max_episode_steps: int = 50  # As requested
    ros_rate: int = 50
    target_tolerance: float = 0.05  # 5cm tolerance
    
    # Reward weights (accuracy highest, then speed, then energy)
    accuracy_weight: float = 15.0    # Highest priority
    speed_weight: float = 5.0        # Second priority  
    energy_weight: float = 0.01      # Third priority
    
    # Joint limits (from URDF analysis)
    joint_limits: Dict[str, Tuple[float, float]] = None
    
    # Workspace bounds
    workspace_bounds: Dict[str, Tuple[float, float]] = None
    
    # Safety parameters
    max_joint_velocity: float = 2.0
    collision_threshold: float = 0.02
    
    def __post_init__(self):
        """Set default values after initialization"""
        if self.joint_limits is None:
            self.joint_limits = {
                'shoulder_pan_joint': (-np.pi, np.pi),
                'shoulder_lift_joint': (-np.pi/2, np.pi/2),
                'elbow_joint': (-np.pi, np.pi),
                'wrist_1_joint': (-np.pi, np.pi),
                'wrist_2_joint': (-np.pi/2, np.pi/2),
                'wrist_3_joint': (-np.pi, np.pi)
            }
        
        if self.workspace_bounds is None:
            self.workspace_bounds = {
                'x': (0.2, 1.0),
                'y': (-0.5, 0.5),
                'z': (0.1, 0.8)
            }

class ManipulatorEnvironmentSimple:
    """
    Simplified 6-DOF Manipulator Environment for Online RL Training
    
    This version works without PyKDL dependency and provides basic functionality
    for testing and development.
    """
    
    def __init__(self, config: EnvironmentConfig = None):
        """
        Initialize the manipulator environment
        
        Args:
            config (EnvironmentConfig): Environment configuration
        """
        logger.info("🚀 Initializing Simplified Manipulator Environment")
        
        # Store configuration
        self.config = config or EnvironmentConfig()
        
        # Initialize ROS node if not already done
        if not rospy.get_node_uri():
            rospy.init_node('manipulator_env_simple', anonymous=True)
            logger.info("✅ ROS node initialized: manipulator_env_simple")
        
        # Environment state
        self.current_step = 0
        self.episode_count = 0
        self.task_status = TaskStatus.IN_PROGRESS
        self.episode_start_time = None
        self.total_energy_consumed = 0.0
        
        # State variables
        self.joint_states = JointState()
        self.end_effector_pose = Pose()
        self.target_pose = Pose()
        self.previous_joint_positions = None
        self.previous_joint_velocities = None
        
        # Threading and synchronization
        self.lock = threading.Lock()
        self.running = False
        self.control_thread = None
        
        # ROS communication setup
        self._setup_ros_communication()
        
        # Wait for services to be available
        self._wait_for_services()
        
        # Initialize episode
        self._reset_episode_state()
        
        logger.info("✅ Simplified Manipulator Environment initialized successfully")
        logger.info(f"📊 Configuration: max_steps={self.config.max_episode_steps}, "
                   f"accuracy_weight={self.config.accuracy_weight}, "
                   f"speed_weight={self.config.speed_weight}, "
                   f"energy_weight={self.config.energy_weight}")
    
    def _setup_ros_communication(self):
        """Setup ROS publishers, subscribers, and services"""
        logger.info("🔧 Setting up ROS communication")
        
        # Publishers for control commands
        self.joint_torque_pubs = {}
        joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        for joint_name in joint_names:
            topic = f'/manipulator/{joint_name}_controller/command'
            self.joint_torque_pubs[joint_name] = rospy.Publisher(
                topic, Float64, queue_size=1
            )
            logger.debug(f"📡 Created torque publisher: {topic}")
        
        # Subscribers for state observation
        self.joint_states_sub = rospy.Subscriber(
            '/manipulator/joint_states', JointState, 
            self._joint_states_callback, queue_size=1
        )
        logger.debug("📡 Created joint states subscriber")
        
        # Service proxies
        self.get_model_state_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.spawn_model_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        
        # TF buffer and listener (if available)
        if TF2_AVAILABLE:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer)
            logger.debug("📡 TF2 listener initialized")
        else:
            logger.warning("⚠️ TF2 not available, end-effector position will be estimated")
        
        logger.info("✅ ROS communication setup complete")
    
    def _wait_for_services(self):
        """Wait for required ROS services to be available"""
        logger.info("⏳ Waiting for ROS services...")
        
        services = [
            '/gazebo/get_model_state',
            '/gazebo/set_model_state', 
            '/gazebo/spawn_sdf_model',
            '/gazebo/delete_model'
        ]
        
        for service in services:
            try:
                rospy.wait_for_service(service, timeout=10.0)
                logger.debug(f"✅ Service available: {service}")
            except rospy.ROSException as e:
                logger.error(f"❌ Service not available: {service} - {e}")
                raise RuntimeError(f"Required service {service} not available")
        
        logger.info("✅ All required services available")
    
    def _joint_states_callback(self, msg: JointState):
        """Callback for joint states"""
        with self.lock:
            self.joint_states = msg
            logger.debug(f"📊 Joint states updated: {len(msg.position)} positions, "
                        f"{len(msg.velocity)} velocities")
    
    def _reset_episode_state(self):
        """Reset episode-specific state variables"""
        logger.debug("🔄 Resetting episode state")
        
        self.current_step = 0
        self.task_status = TaskStatus.IN_PROGRESS
        self.episode_start_time = time.time()
        self.total_energy_consumed = 0.0
        self.previous_joint_positions = None
        self.previous_joint_velocities = None
        
        logger.debug("✅ Episode state reset complete")
    
    def reset(self, target_position: Optional[List[float]] = None) -> np.ndarray:
        """
        Reset the environment for a new episode
        
        Args:
            target_position (List[float], optional): Target position [x, y, z]
            
        Returns:
            np.ndarray: Initial observation state
        """
        logger.info(f"🎬 Resetting environment for episode {self.episode_count + 1}")
        
        # Reset episode state
        self._reset_episode_state()
        
        # Randomize target position if not provided
        if target_position is None:
            target_position = self._generate_random_target()
        
        logger.info(f"🎯 Target position: {target_position}")
        
        # Set target pose
        self.target_pose.position.x = target_position[0]
        self.target_pose.position.y = target_position[1] 
        self.target_pose.position.z = target_position[2]
        
        # Randomize initial joint positions
        initial_joints = self._generate_random_joint_configuration()
        logger.info(f"🔧 Initial joint positions: {initial_joints}")
        
        # Set initial joint configuration
        self._set_joint_configuration(initial_joints)
        
        # Wait for state to stabilize
        rospy.sleep(0.5)
        
        # Get initial observation
        observation = self._get_observation()
        
        logger.info(f"✅ Environment reset complete. Observation shape: {observation.shape}")
        logger.debug(f"📊 Initial observation: {observation}")
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action (np.ndarray): Joint torque commands [6D]
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict]: (observation, reward, done, info)
        """
        logger.debug(f"🎮 Executing step {self.current_step + 1} with action: {action}")
        
        # Validate action
        if not self._validate_action(action):
            logger.warning(f"⚠️ Invalid action detected: {action}")
            action = self._clip_action(action)
        
        # Execute action
        self._execute_action(action)
        
        # Wait for action to take effect
        rospy.sleep(1.0 / self.config.ros_rate)
        
        # Get new observation
        next_observation = self._get_observation()
        
        # Calculate reward
        reward, reward_info = self._calculate_reward(action)
        
        # Check if episode is done
        done = self._check_done()
        
        # Update step counter
        self.current_step += 1
        
        # Log step information
        logger.debug(f"📊 Step {self.current_step}: reward={reward:.4f}, "
                    f"done={done}, status={self.task_status}")
        
        # Prepare info dictionary
        info = {
            'step': self.current_step,
            'task_status': self.task_status.value,
            'reward_info': reward_info,
            'energy_consumed': self.total_energy_consumed,
            'episode_time': time.time() - self.episode_start_time
        }
        
        return next_observation, reward, done, info
    
    def _generate_random_target(self) -> List[float]:
        """Generate random target position within workspace bounds"""
        logger.debug("🎲 Generating random target position")
        
        x = np.random.uniform(
            self.config.workspace_bounds['x'][0],
            self.config.workspace_bounds['x'][1]
        )
        y = np.random.uniform(
            self.config.workspace_bounds['y'][0], 
            self.config.workspace_bounds['y'][1]
        )
        z = np.random.uniform(
            self.config.workspace_bounds['z'][0],
            self.config.workspace_bounds['z'][1]
        )
        
        target = [x, y, z]
        logger.debug(f"🎯 Random target generated: {target}")
        return target
    
    def _generate_random_joint_configuration(self) -> List[float]:
        """Generate random joint configuration within limits"""
        logger.debug("🎲 Generating random joint configuration")
        
        joint_config = []
        joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint', 
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        for joint_name in joint_names:
            min_val, max_val = self.config.joint_limits[joint_name]
            joint_config.append(np.random.uniform(min_val, max_val))
        
        logger.debug(f"🔧 Random joint configuration: {joint_config}")
        return joint_config
    
    def _set_joint_configuration(self, joint_positions: List[float]):
        """Set joint configuration using Gazebo service"""
        logger.debug(f"🔧 Setting joint configuration: {joint_positions}")
        
        try:
            # This would use the set_model_configuration service
            # Implementation depends on your specific Gazebo setup
            logger.debug("✅ Joint configuration set (placeholder)")
        except Exception as e:
            logger.error(f"❌ Failed to set joint configuration: {e}")
    
    def _validate_action(self, action: np.ndarray) -> bool:
        """Validate action before execution"""
        if not isinstance(action, np.ndarray):
            logger.warning("⚠️ Action is not numpy array")
            return False
        
        if action.shape != (6,):
            logger.warning(f"⚠️ Action shape {action.shape} != (6,)")
            return False
        
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            logger.warning("⚠️ Action contains NaN or Inf values")
            return False
        
        return True
    
    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip action to valid range"""
        logger.debug(f"✂️ Clipping action: {action}")
        
        # Clip to reasonable torque limits (adjust based on your robot)
        max_torque = 50.0  # Nm
        clipped_action = np.clip(action, -max_torque, max_torque)
        
        logger.debug(f"✂️ Clipped action: {clipped_action}")
        return clipped_action
    
    def _execute_action(self, action: np.ndarray):
        """Execute joint torque commands"""
        logger.debug(f"🎮 Executing action: {action}")
        
        joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint', 
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        for i, joint_name in enumerate(joint_names):
            if joint_name in self.joint_torque_pubs:
                torque_msg = Float64()
                torque_msg.data = float(action[i])
                self.joint_torque_pubs[joint_name].publish(torque_msg)
                logger.debug(f"📡 Published torque for {joint_name}: {action[i]:.4f}")
        
        logger.debug("✅ Action execution complete")
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        logger.debug("👁️ Getting observation")
        
        with self.lock:
            # Joint positions (6D)
            joint_positions = np.array(self.joint_states.position[:6])
            
            # Joint velocities (6D) 
            joint_velocities = np.array(self.joint_states.velocity[:6])
            
            # End-effector position (3D) - simplified estimation
            # In a real implementation, this would come from TF or forward kinematics
            end_effector_pos = np.array([0.0, 0.0, 0.0])  # Placeholder
            
            # Target position (3D)
            target_pos = np.array([
                self.target_pose.position.x,
                self.target_pose.position.y,
                self.target_pose.position.z
            ])
            
            # Combine into observation vector
            observation = np.concatenate([
                joint_positions,      # 6D
                joint_velocities,     # 6D  
                end_effector_pos,     # 3D
                target_pos            # 3D
            ])
        
        logger.debug(f"👁️ Observation shape: {observation.shape}")
        return observation
    
    def _calculate_reward(self, action: np.ndarray) -> Tuple[float, Dict]:
        """Calculate reward based on accuracy, speed, and energy efficiency"""
        logger.debug("💰 Calculating reward")
        
        # Get current state
        observation = self._get_observation()
        joint_positions = observation[:6]
        joint_velocities = observation[6:12]
        end_effector_pos = observation[12:15]
        target_pos = observation[15:18]
        
        # Calculate accuracy reward (highest priority)
        distance_to_target = np.linalg.norm(end_effector_pos - target_pos)
        accuracy_reward = -distance_to_target * self.config.accuracy_weight
        logger.debug(f"🎯 Accuracy reward: {accuracy_reward:.4f} (distance: {distance_to_target:.4f})")
        
        # Calculate speed reward (second priority)
        if self.current_step < self.config.max_episode_steps:
            remaining_steps = self.config.max_episode_steps - self.current_step
            speed_reward = remaining_steps * self.config.speed_weight
        else:
            speed_reward = 0.0
        logger.debug(f"⚡ Speed reward: {speed_reward:.4f}")
        
        # Calculate energy efficiency reward (third priority)
        if self.previous_joint_positions is not None:
            # Energy is proportional to torque squared
            energy_consumed = np.sum(action ** 2)
            self.total_energy_consumed += energy_consumed
            energy_reward = -energy_consumed * self.config.energy_weight
        else:
            energy_reward = 0.0
        logger.debug(f"🔋 Energy reward: {energy_reward:.4f}")
        
        # Check for task completion
        if distance_to_target < self.config.target_tolerance:
            self.task_status = TaskStatus.SUCCESS
            completion_bonus = 100.0  # Large bonus for task completion
            logger.info(f"🎉 Task completed! Distance: {distance_to_target:.4f}")
        else:
            completion_bonus = 0.0
        
        # Total reward
        total_reward = accuracy_reward + speed_reward + energy_reward + completion_bonus
        
        # Store previous state for next iteration
        self.previous_joint_positions = joint_positions.copy()
        self.previous_joint_velocities = joint_velocities.copy()
        
        # Prepare reward info
        reward_info = {
            'accuracy_reward': accuracy_reward,
            'speed_reward': speed_reward, 
            'energy_reward': energy_reward,
            'completion_bonus': completion_bonus,
            'total_reward': total_reward,
            'distance_to_target': distance_to_target,
            'task_status': self.task_status.value
        }
        
        logger.debug(f"💰 Total reward: {total_reward:.4f}")
        return total_reward, reward_info
    
    def _check_done(self) -> bool:
        """Check if episode is done"""
        logger.debug(f"🔍 Checking if done: step {self.current_step}/{self.config.max_episode_steps}")
        
        # Check maximum steps
        if self.current_step >= self.config.max_episode_steps:
            if self.task_status == TaskStatus.IN_PROGRESS:
                self.task_status = TaskStatus.TIMEOUT
            logger.info(f"⏰ Episode done: reached max steps ({self.config.max_episode_steps})")
            return True
        
        # Check task completion
        if self.task_status == TaskStatus.SUCCESS:
            logger.info("🎉 Episode done: task completed successfully")
            return True
        
        return False
    
    def close(self):
        """Close the environment and cleanup resources"""
        logger.info("🔒 Closing manipulator environment")
        
        # Stop control thread if running
        if self.control_thread and self.control_thread.is_alive():
            self.running = False
            self.control_thread.join()
        
        # Close ROS subscribers
        if hasattr(self, 'joint_states_sub'):
            self.joint_states_sub.unregister()
        
        logger.info("✅ Environment closed successfully")

# Test function for debugging
def test_environment():
    """Test function to verify environment setup"""
    logger.info("🧪 Testing simplified manipulator environment")
    
    try:
        # Create environment
        config = EnvironmentConfig()
        env = ManipulatorEnvironmentSimple(config)
        
        # Test reset
        logger.info("🔄 Testing reset...")
        obs = env.reset()
        logger.info(f"✅ Reset successful. Observation shape: {obs.shape}")
        
        # Test step
        logger.info("🎮 Testing step...")
        action = np.random.uniform(-10, 10, 6)
        obs, reward, done, info = env.step(action)
        logger.info(f"✅ Step successful. Reward: {reward:.4f}, Done: {done}")
        
        # Close environment
        env.close()
        logger.info("✅ Environment test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        raise

if __name__ == "__main__":
    test_environment()
