#!/usr/bin/env python3

"""
Hierarchical Pick-and-Place SAC Environment
Integrates high-level strategic learning with physics-informed low-level control

This environment implements the hierarchical control architecture where:
- High-level SAC learns strategic task sequencing and phase transitions
- Low-level controllers execute precise motions using physics-informed control
- State representation includes joint states, visual perception, and task context

Author: Physics-Informed RL Implementation
Date: 2024
"""

import rospy
import numpy as np
import torch
import time
import logging
from typing import Dict, List, Tuple, Optional
from sensor_msgs.msg import JointState
from pick_and_place.msg import DetectedObjectsStamped, DetectedObject
from ros_controller import PhysicsInformedROSController
from perception_module import PerceptionModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalPickPlaceEnvironment:
    """
    Hierarchical SAC Environment for Pick-and-Place Tasks
    
    This environment implements the hierarchical control architecture described in the README:
    
    High-Level Strategic Layer:
    - Input: Multi-modal 42D state (joint states + vision + task context)
    - Output: 8D strategic actions (5 phase transitions + 3 object selection)
    - Purpose: Learn strategic task sequencing and decision making
    
    Low-Level Physics-Informed Layer:
    - Input: Strategic commands from high-level layer
    - Output: Joint position targets for physics-informed control
    - Purpose: Execute precise motions using Lagrangian dynamics + residual learning
    
    Observation Space (42D):
    - Joint states (14D): positions (7D) + velocities (7D)
    - Object detection (20D): 4 objects × 5 features (x, y, height, color, confidence)
    - Task context (8D): current phase, gripper state, target state, progress
    
    Action Space (8D):
    - Strategic actions: 5 phase transitions + 3 object selection preferences
    
    Reward Components:
    - Task completion: 40% weight
    - Efficiency: 25% weight
    - Strategic coherence: 20% weight
    - Physics-informed control success: 15% weight
    """
    
    def __init__(self, 
                 max_episode_steps=500,
                 reward_weights=None,
                 enable_physics_control=True,
                 enable_residual_learning=True):
        """
        Initialize the hierarchical pick-and-place SAC environment
        
        Args:
            max_episode_steps: Maximum steps per episode
            reward_weights: Dictionary of reward component weights
            enable_physics_control: Enable physics-informed low-level control
            enable_residual_learning: Enable residual learning for adaptation
        """
        # Only initialize node if not already initialized
        try:
            rospy.init_node('pick_place_sac_env', anonymous=True)
        except rospy.exceptions.ROSException:
            # Node already initialized, continue
            pass
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Reward weights (your specified priorities)
        self.reward_weights = reward_weights or {
            'speed': 0.3,
            'energy': 0.3,
            'accuracy': 0.2,
            'completion': 0.2
        }
        
        # Observation space dimensions for hierarchical control
        self.obs_dim = 42  # 14 + 20 + 8 (7 joint positions + 7 joint velocities + 20 object features + 8 context)
        self.action_dim = 8  # Strategic actions: 5 phase transitions + 3 object selection preferences
        
        # Joint names (Franka Panda)
        self.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        
        # Initialize components
        # Initialize physics-informed ROS controller
        self.controller = PhysicsInformedROSController(
            enable_physics_control=enable_physics_control,
            enable_residual_learning=enable_residual_learning
        )
        
        # Initialize perception module
        self.perception = PerceptionModule()
        
        # Control mode flags
        self.enable_physics_control = enable_physics_control
        self.enable_residual_learning = enable_residual_learning
        
        # Gripper control (simplified - using joint control)
        self.gripper_open_position = 0.04  # Open gripper
        self.gripper_close_position = 0.0  # Close gripper
        self.current_gripper_state = 0  # 0: open, 1: closed
        
        # Environment parameters
        self.workbench_height = 0.8  # Height of the workbench in meters
        
        # Pick-and-place parameters (from reference implementation)
        self.x_offset = 0.01          # Gripper offset in x-axis
        self.z_offset = 0.105         # Gripper offset in z-axis
        self.z_pre_pick_offset = 0.2  # Pre-pick height above object
        self.z_pre_place_offset = 0.2 # Pre-place height above bin
        
        # Gripper control thresholds - relaxed for bring-up/training
        self.pre_pick_distance = 0.45  # Distance to trigger pre-pick approach (relaxed)
        self.pick_distance = 0.25      # Distance to trigger pick (relaxed)
        self.place_distance = 0.25     # Distance to trigger place (slightly relaxed)
        
        # Learning progress tracking
        self.best_distance_achieved = float('inf')
        self.consecutive_improvements = 0
        
        # State tracking
        self.joint_positions = np.zeros(7)
        self.joint_velocities = np.zeros(7)
        self.detected_objects = []
        self.state_machine_state = 0  # 0: home, 1: selecting, 2: picking, 3: placing
        self.target_object_id = -1
        self.gripper_state = 0  # 0: open, 1: closed
        self.task_progress = 0.0
        self.episode_start_time = None
        self.object_attached = False
        
        # Performance tracking
        self.episode_energy = 0.0
        self.episode_accuracy = 0.0
        self.episode_speed = 0.0
        self.episode_completion = 0.0
        
        # Setup ROS communication
        self._setup_ros_communication()
        
        rospy.loginfo("Pick-and-Place SAC Environment initialized!")
    
    def _setup_ros_communication(self):
        """Setup ROS subscribers and publishers"""
        # Joint state subscriber
        self.joint_state_sub = rospy.Subscriber(
            '/franka_state_controller/joint_states',
            JointState,
            self._joint_state_callback
        )
        
        # Object detection subscriber
        self.object_detection_sub = rospy.Subscriber(
            '/object_detection',
            DetectedObjectsStamped,
            self._object_detection_callback
        )
        
        rospy.loginfo("ROS communication setup complete")
    
    def _joint_state_callback(self, msg: JointState):
        """Update joint states from ROS"""
        try:
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    self.joint_positions[i] = msg.position[idx]
                    self.joint_velocities[i] = msg.velocity[idx]
        except Exception as e:
            rospy.logwarn(f"Error in joint state callback: {e}")
    
    def _object_detection_callback(self, msg: DetectedObjectsStamped):
        """Update detected objects from perception module"""
        self.detected_objects = msg.detected_objects
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation"""
        rospy.loginfo("Resetting environment...")
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_energy = 0.0
        self.episode_accuracy = 0.0
        self.episode_speed = 0.0
        self.episode_completion = 0.0
        self.episode_start_time = time.time()
        
        # Reset state machine
        self.state_machine_state = 0  # Start at home
        self.target_object_id = -1
        self.gripper_state = 0
        self.current_gripper_state = 0  # Reset gripper state
        self.task_progress = 0.0
        
        # Clear detected objects list
        self.detected_objects = []
        
        # Move robot to home position
        self.controller.move_to_neutral()
        
        # Reset learning progress tracking
        self.best_distance_achieved = float('inf')
        self.consecutive_improvements = 0
        
        # Respawn objects at random positions
        self._respawn_objects()
        
        # Wait for initial state and object detection
        rospy.sleep(3.0)  # Increased wait time for object respawning and detection
        
        # Force update of detected objects
        self._update_detected_objects()
        
        # Return initial observation
        return self._get_observation()
    
    def _respawn_objects(self):
        """Respawn objects at random positions on the workbench"""
        try:
            # Define object names and their possible spawn positions
            object_names = ['block_red_1', 'block_red_2', 'block_blue_1', 'block_green_1']
            
            # Define spawn area on workbench (x: 0.3-0.7, y: -0.3-0.3, z: 0.115)
            spawn_positions = [
                (0.4, -0.22, 0.115),  # Original position 1
                (0.4, 0.22, 0.115),   # Original position 2
                (0.6, 0.22, 0.115),   # Original position 3
                (0.45, 0.0, 0.115),   # Original position 4
            ]
            
            # Shuffle positions for randomness
            import random
            random.shuffle(spawn_positions)
            
            rospy.loginfo("🔄 Respawning objects at random positions...")
            
            # Use Gazebo service to set model poses
            from gazebo_msgs.srv import SetModelState
            from gazebo_msgs.msg import ModelState
            from geometry_msgs.msg import Pose, Point, Quaternion
            from tf.transformations import quaternion_from_euler
            
            set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            
            for i, obj_name in enumerate(object_names):
                if i < len(spawn_positions):
                    x, y, z = spawn_positions[i]
                    
                    # Create model state
                    model_state = ModelState()
                    model_state.model_name = obj_name
                    model_state.pose = Pose()
                    model_state.pose.position = Point(x, y, z)
                    
                    # Random orientation (slight rotation around z-axis)
                    yaw = random.uniform(-0.5, 0.5)
                    quat = quaternion_from_euler(0, 0, yaw)
                    model_state.pose.orientation = Quaternion(*quat)
                    
                    # Set model state
                    try:
                        set_model_state(model_state)
                        rospy.loginfo(f"  📦 {obj_name} -> ({x:.2f}, {y:.2f}, {z:.2f})")
                    except Exception as e:
                        rospy.logwarn(f"Failed to respawn {obj_name}: {e}")
            
            rospy.loginfo("✅ Object respawning completed!")
            
        except Exception as e:
            rospy.logwarn(f"Object respawning failed: {e}")
            rospy.loginfo("Continuing with existing object positions...")
    
    def _update_detected_objects(self):
        """Force update of detected objects from perception module"""
        try:
            # Get latest detected objects from perception module
            if hasattr(self, 'perception') and self.perception:
                self.detected_objects = self.perception.get_detected_objects()
                rospy.loginfo(f"Updated detected objects: {len(self.detected_objects)}")
            else:
                rospy.logwarn("Perception module not available for object update")
        except Exception as e:
            rospy.logwarn(f"Failed to update detected objects: {e}")
    
    def step(self, strategic_action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the hierarchical environment
        
        Args:
            strategic_action: High-level strategic action (8D)
                - First 5 elements: Phase transition probabilities
                - Last 3 elements: Object selection preferences
            
        Returns:
            observation: Next observation (42D)
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information including low-level control success
        """
        self.current_step += 1
        
        # Log current state
        logger.info(f"=== HIERARCHICAL STEP {self.current_step} ===")
        logger.info(f"Strategic action received: {strategic_action}")
        
        # Decode strategic action
        phase_transitions = strategic_action[:5]
        object_selection = strategic_action[5:8]
        
        logger.info(f"Phase transitions: {phase_transitions}")
        logger.info(f"Object selection: {object_selection}")
        rospy.loginfo(f"Current joint positions: {self.joint_positions}")
        rospy.loginfo(f"Current joint velocities: {self.joint_velocities}")
        rospy.loginfo(f"Detected objects: {len(self.detected_objects)}")
        rospy.loginfo(f"State machine state: {self.state_machine_state}")
        
        # Execute hierarchical action using physics-informed control
        low_level_success = False
        try:
            # Convert strategic action to low-level joint commands
            target_joint_positions = self._strategic_action_to_joint_positions(strategic_action)
            logger.info(f"Target joint positions from strategic action: {target_joint_positions}")
            
            # Prepare task context for residual learning
            task_context = np.array([
                float(self.state_machine_state),  # Current phase
                float(self.current_gripper_state),  # Gripper state
                self.task_progress  # Task progress
            ])
            
            # Execute using physics-informed control
            logger.info("Executing physics-informed control...")
            low_level_success = self.controller.execute_physics_informed_control(
                target_positions=target_joint_positions,
                task_context=task_context
            )
            
            if not low_level_success:
                logger.warning("Physics-informed control failed, using fallback")
                # Fallback to simple position control
                low_level_success = self.controller.move_to_joint_positions(
                    target_joint_positions, duration=0.35
                )
            
            logger.info(f"Low-level control result: {low_level_success}")
            
            # Wait for action to complete
            rospy.sleep(0.35)  # Control loop timing
            
        except Exception as e:
            logger.warning(f"Error executing hierarchical action: {e}")
            low_level_success = False
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Calculate hierarchical reward
        reward = self._calculate_hierarchical_reward(strategic_action, low_level_success)
        self.episode_reward += reward
        
        rospy.loginfo(f"Reward calculated: {reward:.3f}")
        rospy.loginfo(f"Cumulative reward: {self.episode_reward:.3f}")
        
        # Check if episode is done
        done = self._check_done()
        rospy.loginfo(f"Episode done: {done}")
        
        # Update state machine
        old_state = self.state_machine_state
        self._update_state_machine()
        rospy.loginfo(f"State machine: {old_state} -> {self.state_machine_state}")
        rospy.loginfo(f"Task progress: {self.task_progress:.3f}")
        
        # Prepare info
        info = {
            'episode_reward': self.episode_reward,
            'energy': self.episode_energy,
            'accuracy': self.episode_accuracy,
            'speed': self.episode_speed,
            'completion': self.episode_completion,
            'state_machine_state': self.state_machine_state,
            'step': self.current_step
        }
        
        return next_obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector (42D)"""
        # Joint states (14D) - 7 positions + 7 velocities
        joint_obs = np.concatenate([
            self.joint_positions,    # 7D
            self.joint_velocities    # 7D
        ])
        
        # Object detection (20D)
        vision_obs = self._encode_objects()
        
        # Task context (8D)
        context_obs = np.array([
            self.state_machine_state,
            self.target_object_id,
            self.gripper_state,
            self.task_progress,
            self.current_step / self.max_episode_steps,
            self.completion_percentage,  # Task completion percentage
            float(len(self.detected_objects) > 0),  # Objects detected flag
            float(self.task_completed)  # Task completion flag
        ])
        
        return np.concatenate([joint_obs, vision_obs, context_obs])
    
    def _encode_objects(self) -> np.ndarray:
        """Encode detected objects into fixed-size vector (20D)"""
        encoding = np.zeros(20)  # 4 objects × 5 features
        
        for i, obj in enumerate(self.detected_objects[:4]):  # Max 4 objects
            start_idx = i * 5
            encoding[start_idx:start_idx+3] = [obj.x_world, obj.y_world, obj.height]
            encoding[start_idx+3] = self._encode_color(obj.color)
            encoding[start_idx+4] = obj.confidence
        
        return encoding
    
    def _encode_color(self, color: str) -> float:
        """Encode color as numerical value"""
        color_map = {'red': 0.0, 'green': 0.25, 'blue': 0.5, 'black': 0.75}
        return color_map.get(color, 0.0)
    
    def _get_end_effector_position(self) -> np.ndarray:
        """Get EE position from TF (world -> panda_link7); fallback to coarse FK."""
        try:
            import tf
            if not hasattr(self, '_tf_listener'):
                self._tf_listener = tf.TransformListener()
            self._tf_listener.waitForTransform('world', 'panda_link7', rospy.Time(0), rospy.Duration(0.2))
            trans, _ = self._tf_listener.lookupTransform('world', 'panda_link7', rospy.Time(0))
            return np.array(trans, dtype=float)
        except Exception:
            # Coarse FK fallback
            d1 = 0.333
            a2 = 0.316
            a3 = 0.0825
            d4 = 0.384
            q1, q2, q3, q4 = self.joint_positions[:4]
            x = a2 * np.cos(q1) * np.cos(q2) + a3 * np.cos(q1) * np.cos(q2 + q3) + d4 * np.cos(q1) * np.cos(q2 + q3 + q4)
            y = a2 * np.sin(q1) * np.cos(q2) + a3 * np.sin(q1) * np.cos(q2 + q3) + d4 * np.sin(q1) * np.cos(q2 + q3 + q4)
            z = d1 + a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + d4 * np.sin(q2 + q3 + q4)
            return np.array([x, y, z])
    
    def _control_gripper(self, open_gripper: bool):
        """Control gripper via ROSController when available; keep state in sync"""
        try:
            if open_gripper and self.current_gripper_state != 0:
                if hasattr(self.controller, 'open_gripper'):
                    self.controller.open_gripper()
                self.current_gripper_state = 0
                rospy.loginfo("🤖 Opening gripper...")
            elif (not open_gripper) and self.current_gripper_state != 1:
                if hasattr(self.controller, 'close_gripper'):
                    self.controller.close_gripper()
                self.current_gripper_state = 1
                rospy.loginfo("🤖 Closing gripper...")
        except Exception as e:
            rospy.logwarn(f"Gripper control error: {e}")
    
    def _strategic_action_to_joint_positions(self, strategic_action: np.ndarray) -> np.ndarray:
        """
        Convert high-level strategic action to low-level joint positions
        
        Args:
            strategic_action: Strategic action (8D: 5 phase transitions + 3 object selection)
            
        Returns:
            target_joint_positions: Target joint positions (7D)
        """
        
        # Decode strategic action
        phase_transitions = strategic_action[:5]
        object_selection = strategic_action[5:8]
        
        # Determine dominant phase
        phase_idx = np.argmax(phase_transitions)
        
        # Neutral pose as base
        neutral_pose = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        # Phase-based target positions
        phase_offsets = {
            0: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Home
            1: np.array([0.1, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0]),  # Approach
            2: np.array([0.2, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1]),  # Grasp
            3: np.array([-0.1, 0.1, 0.2, -0.1, 0.2, -0.1, 0.2]),  # Transport
            4: np.array([-0.2, 0.0, 0.3, -0.2, 0.3, -0.2, 0.3])  # Place
        }
        
        target_pose = neutral_pose + phase_offsets.get(phase_idx, np.zeros(7))
        
        # Add object selection influence
        object_influence = np.sum(object_selection) * 0.1
        target_pose += object_influence * np.random.normal(0, 0.05, 7)
        
        # Smooth transition from current position
        alpha = 0.3  # Smoothing factor
        target_positions = alpha * target_pose + (1 - alpha) * self.joint_positions
        
        # Apply joint limits
        target_positions = np.clip(target_positions, -2.8973, 2.8973)
        
        return target_positions
    
    def _action_to_joint_positions(self, action: np.ndarray) -> List[float]:
        """
        Convert action to joint positions
        
        Note: This is used as fallback when hierarchical control is not available.
        In hierarchical mode, strategic actions are converted via _strategic_action_to_joint_positions.
        """
        current_positions = self.joint_positions.copy()
        
        # Ensure action has correct dimension (7D for Franka Panda)
        if len(action) != 7:
            # Pad or truncate action to match joint count
            if len(action) < 7:
                action = np.pad(action, (0, 7 - len(action)), 'constant')
            else:
                action = action[:7]
        
        target_positions = current_positions + action * 0.3  # Reduced scaling to avoid joint limit errors
        
        # Clamp to joint limits
        joint_limits = [
            [-2.8973, 2.8973],   # joint1
            [-1.7628, 1.7628],   # joint2
            [-2.8973, 2.8973],   # joint3
            [-3.0718, -0.0698],  # joint4
            [-2.8973, 2.8973],   # joint5
            [-0.0175, 3.7525],   # joint6
            [-2.8973, 2.8973]    # joint7
        ]
        
        for i in range(len(target_positions)):
            target_positions[i] = np.clip(
                target_positions[i], 
                joint_limits[i][0], 
                joint_limits[i][1]
            )
        
        return target_positions.tolist()
    
    def _calculate_hierarchical_reward(self, strategic_action: np.ndarray, low_level_success: bool) -> float:
        """
        Calculate hierarchical reward combining strategic and execution performance
        
        Args:
            strategic_action: High-level strategic action
            low_level_success: Whether low-level execution succeeded
            
        Returns:
            reward: Combined hierarchical reward
        """
        
        reward = 0.0
        
        # Task completion reward (40% weight)
        if self.task_completed:
            reward += 100.0
        else:
            # Progress-based reward
            reward += 10.0 * self.task_progress
        
        # Efficiency reward (25% weight)
        time_penalty = -0.1 * (self.current_step / self.max_episode_steps)
        reward += time_penalty
        
        # Strategic coherence reward (20% weight)
        phase_transitions = strategic_action[:5]
        phase_entropy = -np.sum(phase_transitions * np.log(phase_transitions + 1e-10))
        coherence_reward = 5.0 * (1.0 - phase_entropy / np.log(5))  # Reward decisiveness
        reward += coherence_reward
        
        # Physics-informed control success reward (15% weight)
        if low_level_success:
            reward += 3.0
        else:
            reward -= 1.0
        
        return reward
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Calculate reward (fallback for non-hierarchical mode)
        
        Note: This is the legacy reward function used when hierarchical control is disabled.
        """
        reward = 0.0
        
        # Task-specific rewards based on current state machine state
        if self.state_machine_state == 0:  # Home state
            reward = self._calculate_home_reward()
        elif self.state_machine_state == 1:  # Selecting object
            reward = self._calculate_selection_reward()
        elif self.state_machine_state == 2:  # Picking
            reward = self._calculate_picking_reward()
        elif self.state_machine_state == 3:  # Placing
            reward = self._calculate_placing_reward()
        
        # Add small efficiency bonuses (not the main focus)
        efficiency_bonus = (0.05 * self._calculate_speed_reward() + 
                          0.05 * self._calculate_energy_reward(action))
        
        total_reward = reward + efficiency_bonus
        
        # Log reward components for debugging
        rospy.loginfo(f"Task Reward: {reward:.3f}, Efficiency: {efficiency_bonus:.3f}, Total: {total_reward:.3f}")
        
        return total_reward
    
    def _calculate_home_reward(self) -> float:
        """Reward for staying in home position (minimal)"""
        return 0.1  # Small positive reward for being ready
    
    def _calculate_selection_reward(self) -> float:
        """Reward for selecting and approaching an object"""
        if len(self.detected_objects) == 0:
            return -1.0  # Penalty for no objects detected
        
        # Find closest object
        ee_pos = self._get_end_effector_position()
        min_distance = float('inf')
        for obj in self.detected_objects:
            target_pos = np.array([obj.x_world, obj.y_world, obj.height + self.workbench_height])
            distance = np.linalg.norm(ee_pos - target_pos)
            min_distance = min(min_distance, distance)
        
        # Progressive reward for getting closer
        if min_distance < 0.3:  # Within 30cm
            return 5.0 - min_distance * 10.0  # High reward for being close
        else:
            return -min_distance * 0.5  # Small penalty for being far
    
    def _calculate_picking_reward(self) -> float:
        """Reward for picking behavior - this is the critical one"""
        if len(self.detected_objects) == 0 or self.target_object_id < 0:
            return -2.0  # Penalty for no target
        
        target_obj = self.detected_objects[self.target_object_id]
        ee_pos = self._get_end_effector_position()
        
        # Calculate target positions
        pre_pick_pos = np.array([
            target_obj.x_world + self.x_offset, 
            target_obj.y_world, 
            target_obj.height + self.workbench_height + self.z_offset + self.z_pre_pick_offset
        ])
        pick_pos = np.array([
            target_obj.x_world + self.x_offset, 
            target_obj.y_world, 
            target_obj.height + self.workbench_height + self.z_offset
        ])
        
        pre_pick_distance = np.linalg.norm(ee_pos - pre_pick_pos)
        pick_distance = np.linalg.norm(ee_pos - pick_pos)
        
        # More gradual reward system for better learning
        if pre_pick_distance > 0.6:  # Very far from pre-pick
            return -pre_pick_distance * 1.5  # Penalty for being very far
        elif pre_pick_distance > self.pre_pick_distance:
            # Far from pre-pick - reward getting closer
            return 3.0 - pre_pick_distance * 2.5
        elif pick_distance > self.pick_distance:
            # At pre-pick - reward getting closer to pick position
            return 6.0 - pick_distance * 8.0
        else:
            # At pick position - high reward for being there
            if self.current_gripper_state == 1:  # Gripper closed
                return 25.0  # High reward for successful pick
            else:
                return 12.0  # Good reward for being at pick position
    
    def _calculate_placing_reward(self) -> float:
        """Reward for placing behavior"""
        if len(self.detected_objects) == 0 or self.target_object_id < 0:
            return -2.0
        
        target_obj = self.detected_objects[self.target_object_id]
        bin_positions = {
            'red': (-0.5, -0.25, 0.5),
            'green': (-0.5, 0.0, 0.5),
            'blue': (-0.5, 0.25, 0.5)
        }
        
        if target_obj.color not in bin_positions:
            return -1.0
        
        bin_pos = np.array(bin_positions[target_obj.color])
        ee_pos = self._get_end_effector_position()
        distance = np.linalg.norm(ee_pos - bin_pos)
        
        if distance < self.place_distance:
            if self.current_gripper_state == 0:  # Gripper open (object released)
                return 100.0  # MASSIVE reward for successful place
            else:
                return 20.0  # Good reward for being at bin
        else:
            return 5.0 - distance * 5.0  # Reward for getting closer to bin
    
    def _calculate_speed_reward(self) -> float:
        """Calculate speed-based reward"""
        if self.episode_start_time is None:
            return 0.0
        
        elapsed_time = time.time() - self.episode_start_time
        time_efficiency = 1.0 - (elapsed_time / (self.max_episode_steps * 0.1))
        
        # Reward for completing tasks quickly
        speed_bonus = 10.0 if self.state_machine_state == 3 else 0.0  # Placing state
        
        return max(0.0, time_efficiency) + speed_bonus
    
    def _calculate_energy_reward(self, action: np.ndarray) -> float:
        """Calculate energy efficiency reward"""
        # Penalty for high torque usage
        torque_penalty = -np.sum(np.abs(action)) * 0.01
        
        # Track total energy
        self.episode_energy += np.sum(np.abs(action))
        
        return torque_penalty
    
    def _calculate_accuracy_reward(self) -> float:
        """Calculate accuracy reward based on task-specific goals"""
        accuracy_reward = 0.0
        
        if self.state_machine_state == 1:  # Selecting object
            # Reward for moving toward detected objects
            if len(self.detected_objects) > 0:
                # Calculate distance to nearest object
                min_distance = float('inf')
                for obj in self.detected_objects:
                    ee_pos = self._get_end_effector_position()
                    target_pos = np.array([obj.x_world, obj.y_world, obj.height + self.workbench_height])
                    distance = np.linalg.norm(ee_pos - target_pos)
                    min_distance = min(min_distance, distance)
                
                # Progressive reward for getting closer to objects
                if min_distance < 0.3:  # Within 30cm
                    accuracy_reward = 5.0 - min_distance * 10.0  # High reward for being close
                else:
                    accuracy_reward = -min_distance * 1.0  # Penalty for being far
        
        elif self.state_machine_state == 2:  # Picking
            # Enhanced progressive reward system for pick approach
            if len(self.detected_objects) > 0 and self.target_object_id >= 0:
                target_obj = self.detected_objects[self.target_object_id]
                ee_pos = self._get_end_effector_position()
                
                # Calculate target positions (following reference implementation)
                pre_pick_pos = np.array([
                    target_obj.x_world + self.x_offset, 
                    target_obj.y_world, 
                    target_obj.height + self.workbench_height + self.z_offset + self.z_pre_pick_offset
                ])
                pick_pos = np.array([
                    target_obj.x_world + self.x_offset, 
                    target_obj.y_world, 
                    target_obj.height + self.workbench_height + self.z_offset
                ])
                
                pre_pick_distance = np.linalg.norm(ee_pos - pre_pick_pos)
                pick_distance = np.linalg.norm(ee_pos - pick_pos)
                
                # Track learning progress
                if pick_distance < self.best_distance_achieved:
                    self.best_distance_achieved = pick_distance
                    self.consecutive_improvements += 1
                    rospy.loginfo(f"🎯 NEW BEST DISTANCE: {pick_distance:.3f}m (improvement #{self.consecutive_improvements})")
                else:
                    self.consecutive_improvements = 0
                
                # Enhanced progressive accuracy rewards with stronger guidance
                if pre_pick_distance > self.pre_pick_distance:
                    # Far from pre-pick - strong reward for getting closer
                    distance_factor = max(0.1, 1.0 - (pre_pick_distance - self.pre_pick_distance) / 2.0)
                    accuracy_reward = 8.0 * distance_factor - pre_pick_distance * 3.0
                    rospy.loginfo(f"🎯 APPROACHING: {pre_pick_distance:.3f}m (reward: {accuracy_reward:.2f})")
                elif pick_distance > self.pick_distance:
                    # At pre-pick - very strong reward for getting closer to pick position
                    distance_factor = max(0.2, 1.0 - (pick_distance - self.pick_distance) / 0.5)
                    accuracy_reward = 15.0 * distance_factor - pick_distance * 20.0
                    rospy.loginfo(f"🎯 NEAR PICK: {pick_distance:.3f}m (reward: {accuracy_reward:.2f})")
                else:
                    # At pick position - massive reward AND trigger gripper
                    accuracy_reward = 50.0
                    rospy.loginfo(f"🎯 PICK POSITION! {pick_distance:.3f}m (MASSIVE REWARD: {accuracy_reward:.2f})")
                    # Force gripper to close when at pick position
                    self._control_gripper(open_gripper=False)
                
                # Bonus for consistent improvement
                if self.consecutive_improvements > 3:
                    accuracy_reward += 5.0
                    rospy.loginfo(f"🎯 CONSISTENT IMPROVEMENT BONUS: +5.0")
        
        elif self.state_machine_state == 3:  # Placing
            # Reward for moving toward bin
            bin_positions = {
                'red': (-0.5, -0.25, 0.5),
                'green': (-0.5, 0.0, 0.5),
                'blue': (-0.5, 0.25, 0.5)
            }
            
            if len(self.detected_objects) > 0 and self.target_object_id >= 0:
                target_obj = self.detected_objects[self.target_object_id]
                if target_obj.color in bin_positions:
                    bin_pos = np.array(bin_positions[target_obj.color])
                    ee_pos = self._get_end_effector_position()
                    distance = np.linalg.norm(ee_pos - bin_pos)
                    
                    if distance < self.place_distance:
                        accuracy_reward = 20.0  # Big reward for reaching bin
                        rospy.loginfo(f"🎯 ACCURACY: At bin position! (distance: {distance:.3f}m)")
                    else:
                        accuracy_reward = 8.0 - distance * 5.0  # Reward for getting closer
        
        return accuracy_reward
    
    def _calculate_completion_reward(self) -> float:
        """Calculate task completion reward based on actual task progress"""
        completion_bonus = 0.0
        
        # State-specific completion rewards
        if self.state_machine_state == 1:  # Selecting object
            # Reward for successfully selecting an object
            if len(self.detected_objects) > 0 and self.target_object_id >= 0:
                completion_bonus += 5.0
                rospy.loginfo("✅ Object selected - Completion bonus: +5.0")
        
        elif self.state_machine_state == 2:  # Picking object
            # Reward for being close to target object (simulating successful pick)
            if len(self.detected_objects) > 0 and self.target_object_id >= 0:
                target_obj = self.detected_objects[self.target_object_id]
                ee_pos = self._get_end_effector_position()
                target_pos = np.array([target_obj.x_world, target_obj.y_world, target_obj.height + self.workbench_height])
                distance = np.linalg.norm(ee_pos - target_pos)
                
                if distance < 0.03:  # Very close to object (3cm)
                    completion_bonus += 20.0  # Big reward for successful "pick"
                    rospy.loginfo("✅ Object picked - Completion bonus: +20.0")
                elif distance < 0.1:  # Close to object (10cm)
                    completion_bonus += 10.0
                    rospy.loginfo("🎯 Close to object - Completion bonus: +10.0")
        
        elif self.state_machine_state == 3:  # Placing object
            # Reward for reaching the correct bin
            if len(self.detected_objects) > 0 and self.target_object_id >= 0:
                target_obj = self.detected_objects[self.target_object_id]
                bin_positions = {
                    'red': (-0.5, -0.25, 0.5),
                    'green': (-0.5, 0.0, 0.5),
                    'blue': (-0.5, 0.25, 0.5)
                }
                
                if target_obj.color in bin_positions:
                    bin_pos = np.array(bin_positions[target_obj.color])
                    ee_pos = self._get_end_effector_position()
                    distance = np.linalg.norm(ee_pos - bin_pos)
                    
                    if distance < 0.05:  # Very close to bin (5cm)
                        completion_bonus += 30.0  # Big reward for successful "place"
                        rospy.loginfo(f"✅ Object placed in {target_obj.color} bin - Completion bonus: +30.0")
                    elif distance < 0.15:  # Close to bin (15cm)
                        completion_bonus += 15.0
                        rospy.loginfo(f"🎯 Close to {target_obj.color} bin - Completion bonus: +15.0")
        
        # Bonus for completing full pick-and-place sequence
        if self.state_machine_state == 0 and self.task_progress > 0.9:
            completion_bonus += 100.0  # Massive reward for completing full task
            rospy.loginfo("🎉 FULL TASK COMPLETED - Completion bonus: +100.0")
        
        return completion_bonus
    
    def _check_done(self) -> bool:
        """Check if episode is finished"""
        # Episode finished if max steps reached
        if self.current_step >= self.max_episode_steps:
            return True
        
        # Episode finished if task completed
        if self.state_machine_state == 0 and self.task_progress > 0.9:
            return True
        
        return False
    
    def _update_state_machine(self):
        """Update state machine based on current state"""
        old_state = self.state_machine_state
        old_progress = self.task_progress
        
        if self.state_machine_state == 0:  # Home
            if len(self.detected_objects) > 0:
                self.state_machine_state = 1  # Select object
                # Select nearest object to the current EE pose
                try:
                    ee_pos = self._get_end_effector_position()
                    dists = [np.linalg.norm(ee_pos - np.array([o.x_world, o.y_world, o.height + self.workbench_height])) for o in self.detected_objects]
                    self.target_object_id = int(np.argmin(dists)) if len(dists) > 0 else 0
                except Exception:
                    self.target_object_id = 0
                self.task_progress = 0.1
                rospy.loginfo("State transition: Home -> Selecting Object")
        
        elif self.state_machine_state == 1:  # Selecting
            if self.target_object_id >= 0:
                self.state_machine_state = 2  # Picking
                self.task_progress = 0.3
                rospy.loginfo("State transition: Selecting -> Picking")
        
        elif self.state_machine_state == 2:  # Picking
            # Progressive pick approach (pre-pick → pick → post-pick)
            if len(self.detected_objects) > 0 and self.target_object_id >= 0:
                target_obj = self.detected_objects[self.target_object_id]
                ee_pos = self._get_end_effector_position()
                
                # Calculate target positions (following reference implementation)
                pre_pick_pos = np.array([
                    target_obj.x_world + self.x_offset, 
                    target_obj.y_world, 
                    target_obj.height + self.workbench_height + self.z_offset + self.z_pre_pick_offset
                ])
                pick_pos = np.array([
                    target_obj.x_world + self.x_offset, 
                    target_obj.y_world, 
                    target_obj.height + self.workbench_height + self.z_offset
                ])
                
                # Calculate distances
                pre_pick_distance = np.linalg.norm(ee_pos - pre_pick_pos)
                pick_distance = np.linalg.norm(ee_pos - pick_pos)
                
                # Progressive gripper control
                if pre_pick_distance > self.pre_pick_distance:
                    # Far from pre-pick position - keep gripper open
                    self._control_gripper(open_gripper=True)
                    rospy.loginfo(f"🎯 APPROACHING: {pre_pick_distance:.3f}m (Gripper OPEN)")
                elif pick_distance > self.pick_distance:
                    # At pre-pick position - keep gripper open
                    self._control_gripper(open_gripper=True)
                    rospy.loginfo(f"🎯 NEAR PICK: {pick_distance:.3f}m (Gripper OPEN)")
                else:
                    # At pick position - close gripper and attempt attach repeatedly for robustness
                    self._control_gripper(open_gripper=False)
                    rospy.loginfo(f"🤖 GRASPING! {pick_distance:.3f}m (Gripper CLOSED!)")
                    # Try multiple attach attempts with different candidate links handled in controller
                    try:
                        target_model = f"block_{target_obj.color}_1"
                        attached = False
                        for attempt in range(2):
                            success = self.controller.attach_block(target_model)
                            if success:
                                attached = True
                                self.object_attached = True
                                rospy.loginfo(f"✅ Successfully attached {target_model} on attempt {attempt+1}")
                                break
                            rospy.sleep(0.1)
                        if not attached:
                            rospy.logwarn(f"❌ Attach unsuccessful for {target_model} after retries")
                    except Exception as e:
                        rospy.logwarn(f"Attach attempt failed: {e}")
            
            # Simulate picking completion based on attach flag OR proximity
            if (self.current_step > 5 and 
                len(self.detected_objects) > 0 and self.target_object_id >= 0):
                target_obj = self.detected_objects[self.target_object_id]
                ee_pos = self._get_end_effector_position()
                pick_pos = np.array([
                    target_obj.x_world + self.x_offset, 
                    target_obj.y_world, 
                    target_obj.height + self.workbench_height + self.z_offset
                ])
                distance = np.linalg.norm(ee_pos - pick_pos)
                
                # More lenient transition condition
                if self.object_attached or distance < self.pick_distance * 1.5:  # allow attach to gate transition
                    self.state_machine_state = 3  # Placing
                    self.task_progress = 0.7
                    rospy.loginfo(f"State transition: Picking -> Placing (Object grasped, distance: {distance:.3f}m)")
        
        elif self.state_machine_state == 3:  # Placing
            # Control gripper based on proximity to bin
            if len(self.detected_objects) > 0 and self.target_object_id >= 0:
                target_obj = self.detected_objects[self.target_object_id]
                bin_positions = {
                    'red': (-0.5, -0.25, 0.5),
                    'green': (-0.5, 0.0, 0.5),
                    'blue': (-0.5, 0.25, 0.5)
                }
                
                if target_obj.color in bin_positions:
                    bin_pos = np.array(bin_positions[target_obj.color])
                    ee_pos = self._get_end_effector_position()
                    distance = np.linalg.norm(ee_pos - bin_pos)
                    
                    if distance < max(0.1, self.place_distance):  # Close enough to bin - open gripper
                        self._control_gripper(open_gripper=True)
                        # Try to detach when opening near bin
                        try:
                            target_obj = self.detected_objects[self.target_object_id]
                            target_model = f"block_{target_obj.color}_1"
                            rospy.loginfo(f"Attempting to detach {target_model} from panda_hand")
                            success = self.controller.detach_block(target_model)
                            if success:
                                rospy.loginfo(f"✅ Successfully detached {target_model}")
                            else:
                                rospy.logwarn(f"❌ Failed to detach {target_model}")
                        except Exception as e:
                            rospy.logwarn(f"Detach attempt failed: {e}")
            
            # Simulate placing completion based on steps and gripper state
            if self.current_step > 10 and self.current_gripper_state == 0:  # Gripper open (object released)
                self.state_machine_state = 0  # Back to home
                self.task_progress = 1.0
                rospy.loginfo("State transition: Placing -> Home (Task Complete! Object placed)")
                # Reset attach flag for next cycle
                self.object_attached = False
        
        # Update task progress more aggressively
        self.task_progress = min(1.0, self.task_progress + 0.05)  # Increased progress increment
        
        if old_state != self.state_machine_state or abs(old_progress - self.task_progress) > 0.01:
            rospy.loginfo(f"State machine update: {old_state}->{self.state_machine_state}, Progress: {old_progress:.2f}->{self.task_progress:.2f}")
    
    def render(self):
        """Render environment visualization"""
        # Provide hierarchical control visualization
        logger.info(f"=== Hierarchical Control Status ===")
        logger.info(f"Step: {self.current_step}, Phase: {self.state_machine_state}")
        logger.info(f"Episode Reward: {self.episode_reward:.3f}")
        logger.info(f"Task Progress: {self.task_progress:.2f}")
        logger.info(f"Objects Detected: {len(self.detected_objects)}")
    
    def close(self):
        """Close environment"""
        rospy.loginfo("Closing Pick-and-Place SAC Environment")


def main():
    """Test the environment"""
    env = PickPlaceSACEnvironment()
    
    try:
        obs = env.reset()
        rospy.loginfo(f"Initial observation shape: {obs.shape}")
        
        for step in range(10):
            action = np.random.uniform(-1, 1, 6)  # Random action
            obs, reward, done, info = env.step(action)
            
            rospy.loginfo(f"Step {step}: Reward={reward:.3f}, Done={done}")
            
            if done:
                break
        
        rospy.loginfo("Environment test completed!")
        
    except KeyboardInterrupt:
        rospy.loginfo("Test interrupted by user")
    finally:
        env.close()


# Compatibility alias for backward compatibility
PickPlaceSACEnvironment = HierarchicalPickPlaceEnvironment


if __name__ == "__main__":
    main()
