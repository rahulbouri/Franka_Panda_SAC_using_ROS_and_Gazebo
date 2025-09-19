#!/usr/bin/env python3
"""
Enhanced Gazebo Environment with Proper Collision Detection
Based on gym-gazebo patterns and PINN_RL reward system

Author: RL Training Implementation
Date: 2024
"""

import numpy as np
import rospy
import logging
import time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelStates, ContactsState
from gazebo_msgs.srv import GetModelState, SetModelState, SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GazeboEnvEnhanced:
    """
    Enhanced Gazebo Environment with proper collision detection and reward system
    Based on gym-gazebo patterns and PINN_RL reward system
    """
    
    def __init__(self, config=None):
        """Initialize the environment with enhanced safety and reward system"""
        self.config = config or {}
        
        # Environment state
        self.joint_positions = np.zeros(6)
        self.joint_velocities = np.zeros(6)
        self.target_position = np.zeros(3)
        self.end_effector_position = np.zeros(3)
        self.coke_can_position = np.zeros(3)
        self.coke_can_orientation = np.zeros(4)
        
        # Safety and collision detection
        self.collision_detected = False
        self.contact_points = []
        self.safety_distance = 0.03  # 3cm safety margin (collision detection)
        self.target_tolerance = 0.08  # 8cm success tolerance (reach coke can)
        
        # Joint limits (UR5e specifications)
        self.joint_limits = {
            'min': np.array([-6.28, -6.28, -3.14, -6.28, -6.28, -6.28]),
            'max': np.array([6.28, 6.28, 3.14, 6.28, 6.28, 6.28])
        }
        self.max_joint_velocity = 2.0  # rad/s
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = self.config.get('max_episode_steps', 1000)
        self.episode_reward = 0.0
        self.total_torque = 0.0
        self.target_reached = False
        self.coke_can_tipped = False
        self.episode_start_time = time.time()
        
        # Reward system parameters (based on PINN_RL)
        self.accuracy_weight = 10.0
        self.energy_weight = 0.01
        self.task_completion_weight = 100.0
        self.safety_weight = 1000.0
        self.smoothness_weight = 0.1
        self.time_weight = 1.0
        
        # Setup publishers and subscribers
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_services()
        
        # Wait for services to be available
        self._wait_for_services()
        
        logger.info("✅ Enhanced Gazebo Environment initialized with proper collision detection")
    
    def _setup_publishers(self):
        """Setup joint control publishers"""
        self.joint_pubs = {}
        joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        for joint_name in joint_names:
            topic = f'/manipulator/{joint_name}_effort/command'
            self.joint_pubs[joint_name] = rospy.Publisher(topic, Float64, queue_size=1)
    
    def _setup_subscribers(self):
        """Setup state and collision subscribers"""
        try:
            self.joint_sub = rospy.Subscriber('/manipulator/joint_states', JointState, self._joint_states_callback)
            self.model_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self._model_states_callback)
            # Subscribe to contact sensors for collision detection
            self.contact_sub = rospy.Subscriber('/gazebo/contacts', ContactsState, self._contact_callback)
        except Exception as e:
            logger.warning(f"⚠️ Subscriber setup error: {e}")
    
    def _setup_services(self):
        """Setup Gazebo services"""
        try:
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        except Exception as e:
            logger.warning(f"⚠️ Service setup error: {e}")
    
    def _wait_for_services(self):
        """Wait for required services to be available"""
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=5.0)
            rospy.wait_for_service('/gazebo/set_model_state', timeout=5.0)
            logger.info("✅ Gazebo services available")
        except rospy.ROSException:
            logger.warning("⚠️ Gazebo services not available, continuing...")
    
    def _joint_states_callback(self, msg):
        """Callback for joint states"""
        try:
            if len(msg.position) >= 6:
                self.joint_positions = np.array(msg.position[:6])
            if len(msg.velocity) >= 6:
                self.joint_velocities = np.array(msg.velocity[:6])
        except Exception as e:
            logger.warning(f"⚠️ Joint states callback error: {e}")
    
    def _model_states_callback(self, msg):
        """Callback for model states"""
        try:
            for i, name in enumerate(msg.name):
                if name == 'coke_can':
                    pose = msg.pose[i]
                    self.coke_can_position = np.array([
                        pose.position.x, pose.position.y, pose.position.z
                    ])
                    self.coke_can_orientation = np.array([
                        pose.orientation.x, pose.orientation.y, 
                        pose.orientation.z, pose.orientation.w
                    ])
        except Exception as e:
            logger.warning(f"⚠️ Model states callback error: {e}")
    
    def _contact_callback(self, msg):
        """Callback for contact detection"""
        try:
            self.contact_points = []
            for contact in msg.states:
                # Check if contact involves manipulator and coke_can
                if ('manipulator' in contact.collision1_name and 'coke_can' in contact.collision2_name) or \
                   ('coke_can' in contact.collision1_name and 'manipulator' in contact.collision2_name):
                    self.contact_points.append(contact)
                    self.collision_detected = True
                    logger.warning("⚠️ COLLISION DETECTED! Manipulator contacted coke can!")
                    logger.warning(f"   Contact details: {contact.collision1_name} <-> {contact.collision2_name}")
        except Exception as e:
            logger.warning(f"⚠️ Contact callback error: {e}")
    
    def _check_collision_distance(self):
        """Check collision using distance-based detection as backup"""
        # Calculate distance between end-effector and coke can
        distance = np.linalg.norm(self.end_effector_position - self.coke_can_position)
        
        # If distance is very small, consider it a collision
        if distance < self.safety_distance:
            if not self.collision_detected:  # Only log once
                logger.warning(f"⚠️ DISTANCE COLLISION DETECTED! Distance: {distance:.4f}m")
                self.collision_detected = True
            return True
        
        return False
    
    def reset(self):
        """Reset the environment with proper randomization"""
        self.step_count = 0
        self.episode_reward = 0.0
        self.total_torque = 0.0
        self.collision_detected = False
        self.target_reached = False
        self.coke_can_tipped = False
        self.contact_points = []
        self.episode_start_time = time.time()
        
        # Randomize target position (on table surface)
        table_bounds = {
            'x': [0.2, 1.0],
            'y': [-0.4, 0.4],
            'z': 0.75  # Table height
        }
        
        self.target_position = np.array([
            np.random.uniform(table_bounds['x'][0], table_bounds['x'][1]),
            np.random.uniform(table_bounds['y'][0], table_bounds['y'][1]),
            table_bounds['z']
        ])
        
        # Randomize coke can position (on table surface, different from target)
        coke_can_position = np.array([
            np.random.uniform(table_bounds['x'][0], table_bounds['x'][1]),
            np.random.uniform(table_bounds['y'][0], table_bounds['y'][1]),
            table_bounds['z']
        ])
        
        # Ensure coke can is not too close to target
        while np.linalg.norm(coke_can_position - self.target_position) < 0.1:
            coke_can_position = np.array([
                np.random.uniform(table_bounds['x'][0], table_bounds['x'][1]),
                np.random.uniform(table_bounds['y'][0], table_bounds['y'][1]),
                table_bounds['z']
            ])
        
        # Set coke can position in Gazebo
        self._set_coke_can_position(coke_can_position)
        
        # Randomize joint positions (within safe limits)
        joint_positions = np.random.uniform(
            self.joint_limits['min'] * 0.8, 
            self.joint_limits['max'] * 0.8, 
            6
        )
        
        # Set initial joint positions
        self._set_joint_positions(joint_positions)
        
        # Wait for joints to move
        time.sleep(0.5)
        
        # Update end-effector position
        self.end_effector_position = self._estimate_end_effector_position(self.joint_positions)
        
        # Get initial coke can position
        self._update_coke_can_state()
        
        state = self._get_state()
        
        logger.info(f"🔄 Environment reset - Target: {self.target_position}")
        logger.info(f"🎯 Coke can position: {self.coke_can_position}")
        logger.info(f"🤖 Initial joints: {self.joint_positions[:3]}...")
        return state
    
    def step(self, action):
        """Execute one step with enhanced safety and reward system"""
        self.step_count += 1
        
        # Apply safety constraints to action
        safe_action = self._apply_safety_constraints(action)
        
        # Apply action to joints
        self._apply_action(safe_action)
        
        # Wait for physics to update
        time.sleep(0.1)
        
        # Update state
        self.end_effector_position = self._estimate_end_effector_position(self.joint_positions)
        self._update_coke_can_state()
        
        # Check for safety violations
        self._check_safety_violations()
        
        # Get current state
        state = self._get_state()
        
        # Calculate comprehensive reward (based on PINN_RL) - AFTER safety checks
        reward = self._calculate_enhanced_reward(safe_action)
        self.episode_reward += reward
        
        # Log reward breakdown for debugging
        if self.collision_detected or self.coke_can_tipped:
            logger.warning(f"⚠️ SAFETY VIOLATION REWARD: {reward:.2f}")
            logger.warning(f"   Collision: {self.collision_detected}, Tipped: {self.coke_can_tipped}")
            logger.warning(f"   Episode reward so far: {self.episode_reward:.2f}")
        
        # Check if done
        done = self._is_done()
        
        # Info dictionary
        distance_to_coke = np.linalg.norm(self.end_effector_position - self.coke_can_position)
        distance_to_target = np.linalg.norm(self.end_effector_position - self.target_position)
        
        info = {
            'step': self.step_count,
            'target_position': self.target_position.copy(),
            'end_effector_position': self.end_effector_position.copy(),
            'coke_can_position': self.coke_can_position.copy(),
            'distance': distance_to_coke,  # Primary distance (to coke can)
            'distance_to_target': distance_to_target,  # Reference distance
            'collision_detected': self.collision_detected,
            'target_reached': self.target_reached,
            'coke_can_tipped': self.coke_can_tipped,
            'episode_reward': self.episode_reward,
            'total_torque': self.total_torque,
            'contact_points': len(self.contact_points),
            'success': self.target_reached and not self.collision_detected
        }
        
        return state, reward, done, info
    
    def _apply_safety_constraints(self, action):
        """Apply safety constraints to action"""
        # Clamp action to reasonable range
        action = np.clip(action, -1.0, 1.0)
        
        # Scale action to joint torque range
        max_torque = 50.0  # Nm
        scaled_action = action * max_torque
        
        # Check joint limits
        for i in range(6):
            if self.joint_positions[i] + scaled_action[i] * 0.01 > self.joint_limits['max'][i]:
                scaled_action[i] = (self.joint_limits['max'][i] - self.joint_positions[i]) * 100
            elif self.joint_positions[i] + scaled_action[i] * 0.01 < self.joint_limits['min'][i]:
                scaled_action[i] = (self.joint_positions[i] - self.joint_limits['min'][i]) * 100
        
        # Check velocity limits
        for i in range(6):
            if abs(self.joint_velocities[i]) > self.max_joint_velocity:
                scaled_action[i] *= 0.1  # Reduce torque if velocity is too high
        
        return scaled_action
    
    def _check_safety_violations(self):
        """Check for safety violations"""
        # Check collision using both contact sensors and distance
        if not self.collision_detected:
            self._check_collision_distance()
        
        # Check if coke can is reached (success criteria)
        distance_to_coke = np.linalg.norm(self.end_effector_position - self.coke_can_position)
        if distance_to_coke < self.target_tolerance and not self.collision_detected:
            self.target_reached = True
            logger.info(f"✅ COKE CAN REACHED! Distance: {distance_to_coke:.3f}m")
        
        # Also check distance to original target for reference
        distance_to_target = np.linalg.norm(self.end_effector_position - self.target_position)
        
        # Check if coke can is tipped over or fell off
        if self._check_coke_can_tipped():
            self.coke_can_tipped = True
            logger.warning("⚠️ COKE CAN TIPPED/FELL! Episode terminated!")
    
    def _check_coke_can_tipped(self):
        """Check if coke can is tipped over or fell off platform"""
        # Check if coke can fell off the platform (z < 0.5)
        if self.coke_can_position[2] < 0.5:
            logger.warning(f"⚠️ Coke can fell off platform! Z position: {self.coke_can_position[2]}")
            return True
        
        # Check if coke can orientation indicates tipping
        if len(self.coke_can_orientation) >= 4:
            z_orientation = abs(self.coke_can_orientation[3])  # w component
            if z_orientation < 0.7:  # Threshold for tipping
                logger.warning(f"⚠️ Coke can tipped over! Orientation: {z_orientation}")
                return True
        
        # Check if coke can moved too far from table (x or y out of bounds)
        table_bounds = {'x': [0.0, 1.2], 'y': [-0.6, 0.6]}
        if (self.coke_can_position[0] < table_bounds['x'][0] or 
            self.coke_can_position[0] > table_bounds['x'][1] or
            self.coke_can_position[1] < table_bounds['y'][0] or 
            self.coke_can_position[1] > table_bounds['y'][1]):
            logger.warning(f"⚠️ Coke can moved off table! Position: {self.coke_can_position}")
            return True
        
        return False
    
    def _set_joint_positions(self, positions):
        """Set joint positions"""
        logger.info(f"🤖 Setting joint positions: {positions[:3]}...")
        for i, joint_name in enumerate(self.joint_pubs.keys()):
            try:
                # Publish position command (scaled for position control)
                self.joint_pubs[joint_name].publish(Float64(positions[i] * 0.1))
            except Exception as e:
                logger.warning(f"⚠️ Failed to publish to {joint_name}: {e}")
        
        # Update internal joint positions
        self.joint_positions = positions.copy()
    
    def _apply_action(self, action):
        """Apply action to joints"""
        for i, joint_name in enumerate(self.joint_pubs.keys()):
            try:
                self.joint_pubs[joint_name].publish(Float64(action[i]))
                # Track total torque for energy efficiency
                self.total_torque += abs(action[i])
            except Exception as e:
                logger.warning(f"⚠️ Failed to publish action to {joint_name}: {e}")
    
    def _set_coke_can_position(self, position):
        """Set coke can position in Gazebo"""
        try:
            if hasattr(self, 'set_model_state'):
                # Create pose message
                pose = Pose()
                pose.position.x = position[0]
                pose.position.y = position[1]
                pose.position.z = position[2]
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0
                pose.orientation.w = 1.0
                
                # Set model state
                from gazebo_msgs.msg import ModelState
                model_state = ModelState()
                model_state.model_name = 'coke_can'
                model_state.pose = pose
                model_state.reference_frame = 'world'
                
                response = self.set_model_state(model_state)
                if response.success:
                    logger.info(f"✅ Coke can moved to position: {position}")
                else:
                    logger.warning(f"⚠️ Failed to move coke can: {response.status_message}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to set coke can position: {e}")
    
    def _update_coke_can_state(self):
        """Update coke can state from Gazebo"""
        try:
            if hasattr(self, 'get_model_state'):
                response = self.get_model_state('coke_can', 'world')
                if response.success:
                    pose = response.pose
                    self.coke_can_position = np.array([
                        pose.position.x, pose.position.y, pose.position.z
                    ])
                    self.coke_can_orientation = np.array([
                        pose.orientation.x, pose.orientation.y,
                        pose.orientation.z, pose.orientation.w
                    ])
        except Exception as e:
            logger.warning(f"⚠️ Failed to update coke can state: {e}")
    
    def _get_state(self):
        """Get current state with enhanced information"""
        # Create state vector with enhanced information
        state = np.concatenate([
            self.joint_positions,      # 6 joint positions
            self.joint_velocities,     # 6 joint velocities  
            self.target_position,      # 3 target position
            self.end_effector_position, # 3 end-effector position
            self.coke_can_position,    # 3 coke can position
            [float(self.collision_detected)],  # 1 collision flag
            [float(self.target_reached)],      # 1 target reached flag
            [float(self.coke_can_tipped)]      # 1 coke can tipped flag
        ])
        
        return state
    
    def _calculate_enhanced_reward(self, action):
        """Calculate enhanced reward based on PINN_RL system"""
        # Use distance to coke can as the primary success metric
        distance = np.linalg.norm(self.end_effector_position - self.coke_can_position)
        
        # 1. Accuracy Reward (highest priority)
        if distance < self.target_tolerance:
            accuracy_reward = 100.0  # Large bonus for reaching target
        else:
            accuracy_reward = -distance * self.accuracy_weight
        
        # 2. Energy Efficiency Penalty
        energy_penalty = -np.sum(np.abs(action)) * self.energy_weight
        
        # 3. Task Completion Bonus
        task_reward = 0.0
        if self.target_reached and not self.collision_detected and not self.coke_can_tipped:
            task_reward = self.task_completion_weight
        elif self.target_reached and (self.collision_detected or self.coke_can_tipped):
            task_reward = 0.0  # No bonus if safety violation occurred
        
        # 4. Safety Penalty (CRITICAL - must be applied)
        safety_penalty = 0.0
        if self.collision_detected:
            safety_penalty = -self.safety_weight
            logger.warning(f"⚠️ COLLISION PENALTY: {safety_penalty}")
        
        if self.coke_can_tipped:
            safety_penalty += -self.safety_weight * 2  # Double penalty for tipping
            logger.warning(f"⚠️ COKE CAN TIPPED PENALTY: {safety_penalty}")
        
        # 5. Smoothness Penalty
        smoothness_penalty = -np.sum(np.abs(np.diff(action))) * self.smoothness_weight
        
        # 6. Time Efficiency
        time_penalty = -self.step_count * self.time_weight
        
        # Calculate total reward
        total_reward = (accuracy_reward + energy_penalty + task_reward + 
                       safety_penalty + smoothness_penalty + time_penalty)
        
        # Debug logging for safety violations
        if self.collision_detected or self.coke_can_tipped:
            logger.warning(f"🔍 REWARD BREAKDOWN:")
            logger.warning(f"   Accuracy: {accuracy_reward:.2f}")
            logger.warning(f"   Energy: {energy_penalty:.2f}")
            logger.warning(f"   Task: {task_reward:.2f}")
            logger.warning(f"   Safety: {safety_penalty:.2f}")
            logger.warning(f"   Smoothness: {smoothness_penalty:.2f}")
            logger.warning(f"   Time: {time_penalty:.2f}")
            logger.warning(f"   TOTAL: {total_reward:.2f}")
        
        return total_reward
    
    def _is_done(self):
        """Check if episode is done"""
        # Episode ends if:
        # 1. Maximum steps reached
        # 2. Target reached successfully
        # 3. Collision detected
        # 4. Coke can tipped over
        return (self.step_count >= self.max_steps or 
                self.target_reached or 
                self.collision_detected or 
                self.coke_can_tipped)
    
    def _estimate_end_effector_position(self, joint_positions):
        """Estimate end-effector position using improved forward kinematics"""
        # Improved forward kinematics for UR5e (simplified but more accurate)
        # Base to shoulder: 0.1625m
        # Shoulder to elbow: 0.425m  
        # Elbow to wrist: 0.3922m
        # Wrist to end-effector: 0.09475m
        
        q1, q2, q3, q4, q5, q6 = joint_positions
        
        # Simplified forward kinematics
        # This is a rough approximation - in practice you'd use proper DH parameters
        L1 = 0.1625  # Base to shoulder
        L2 = 0.425   # Shoulder to elbow
        L3 = 0.3922  # Elbow to wrist
        L4 = 0.09475 # Wrist to end-effector
        
        # Calculate end-effector position (simplified)
        x = L1 + L2 * np.cos(q2) * np.cos(q1) + L3 * np.cos(q2 + q3) * np.cos(q1) + L4 * np.cos(q2 + q3) * np.cos(q1)
        y = L1 + L2 * np.cos(q2) * np.sin(q1) + L3 * np.cos(q2 + q3) * np.sin(q1) + L4 * np.cos(q2 + q3) * np.sin(q1)
        z = L2 * np.sin(q2) + L3 * np.sin(q2 + q3) + L4 * np.sin(q2 + q3)
        
        return np.array([x, y, z])
    
    def close(self):
        """Close the environment"""
        logger.info("🔒 Enhanced Environment closed")
