#!/usr/bin/env python3
"""
Comprehensive Reward Function for Manipulator Reach/Pregrasp Tasks
Implements accuracy, efficiency, and energy optimization as requested.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import rospy
from geometry_msgs.msg import PoseStamped
import tf2_ros

class ComprehensiveManipulatorReward:
    """
    Comprehensive reward function for manipulator reach/pregrasp tasks.
    
    Implements three main objectives:
    1. Accuracy: End-effector position error to target
    2. Efficiency: Time-based rewards for quick completion
    3. Energy: Torque minimization for mechanical efficiency
    
    Based on research from:
    - IvLabs/Manipulator-Control-using-RL
    - Feedforward Control for Manipulator with Flexure Joints Using LNN
    - A Reinforcement Learning Neural Network for Robotic Manipulator Control
    """
    
    def __init__(self, 
                 target_position: np.ndarray,
                 joint_names: List[str],
                 joint_limits: Dict,
                 max_episode_steps: int = 1000,
                 position_tolerance: float = 0.05,
                 orientation_tolerance: float = 0.1):
        """
        Initialize comprehensive reward function.
        
        Args:
            target_position: [x, y, z] target position for end-effector
            joint_names: List of joint names
            joint_limits: Dictionary with joint limits
            max_episode_steps: Maximum steps per episode
            position_tolerance: Position accuracy tolerance (meters)
            orientation_tolerance: Orientation accuracy tolerance (radians)
        """
        self.target_position = np.array(target_position)
        self.joint_names = joint_names
        self.joint_limits = joint_limits
        self.max_episode_steps = max_episode_steps
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        
        # Reward weights (hyperparameters - tunable)
        self.weights = {
            'accuracy': 10.0,      # Position accuracy reward
            'orientation': 5.0,    # Orientation accuracy reward
            'efficiency': 1.0,     # Time efficiency reward
            'energy': 0.01,        # Energy efficiency reward
            'smoothness': 0.1,     # Motion smoothness reward
            'constraint': 5.0,     # Constraint violation penalty
            'collision': 10.0,     # Collision penalty
            'success': 100.0       # Success bonus
        }
        
        # Energy and torque tracking
        self.energy_history = []
        self.torque_history = []
        self.power_history = []
        
        # Success tracking
        self.success_count = 0
        self.episode_rewards = []
        
        # TF listener for end-effector pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        rospy.loginfo("🎯 Comprehensive Reward Function initialized")
        rospy.loginfo(f"Target position: {self.target_position}")
        rospy.loginfo(f"Position tolerance: {self.position_tolerance}m")
    
    def get_end_effector_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get end-effector position and orientation using TF.
        
        Returns:
            Tuple of (position, orientation) or None if failed
        """
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
            rospy.logwarn(f"Could not get end-effector pose: {e}")
            return None
    
    def compute_accuracy_reward(self, 
                               joint_positions: np.ndarray, 
                               joint_velocities: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute accuracy reward based on end-effector position and orientation error.
        
        Args:
            joint_positions: Current joint positions
            joint_velocities: Current joint velocities
            
        Returns:
            Tuple of (accuracy_reward, accuracy_info)
        """
        # Get end-effector pose
        ee_pose = self.get_end_effector_pose()
        if ee_pose is None:
            return 0.0, {'position_error': float('inf'), 'orientation_error': float('inf')}
        
        ee_position, ee_orientation = ee_pose
        
        # Position error
        position_error = np.linalg.norm(ee_position - self.target_position)
        
        # Orientation error (simplified - in practice, use proper orientation distance)
        # For now, we'll focus on position accuracy
        orientation_error = 0.0  # Placeholder for orientation error
        
        # Position accuracy reward
        if position_error < self.position_tolerance:
            # Success bonus - high reward when very close to target
            position_reward = self.weights['success'] * np.exp(-position_error / 0.01)
        else:
            # Distance-based reward with exponential decay
            position_reward = np.exp(-position_error / 0.1)
        
        # Orientation accuracy reward (placeholder)
        orientation_reward = 1.0  # Placeholder - implement proper orientation reward
        
        # Total accuracy reward
        accuracy_reward = (
            self.weights['accuracy'] * position_reward +
            self.weights['orientation'] * orientation_reward
        )
        
        accuracy_info = {
            'position_error': position_error,
            'orientation_error': orientation_error,
            'position_reward': position_reward,
            'orientation_reward': orientation_reward,
            'accuracy_reward': accuracy_reward,
            'ee_position': ee_position.tolist(),
            'target_position': self.target_position.tolist()
        }
        
        return accuracy_reward, accuracy_info
    
    def compute_efficiency_reward(self, step_count: int) -> Tuple[float, Dict]:
        """
        Compute time efficiency reward.
        Encourages reaching target quickly.
        
        Args:
            step_count: Current step in episode
            
        Returns:
            Tuple of (efficiency_reward, efficiency_info)
        """
        # Linear decay reward - higher reward for fewer steps
        efficiency_reward = (self.max_episode_steps - step_count) / self.max_episode_steps
        
        # Bonus for early completion
        if step_count < self.max_episode_steps * 0.5:  # Complete in first half
            efficiency_reward *= 2.0
        
        efficiency_info = {
            'step_count': step_count,
            'max_steps': self.max_episode_steps,
            'efficiency_ratio': step_count / self.max_episode_steps,
            'efficiency_reward': efficiency_reward
        }
        
        return efficiency_reward, efficiency_info
    
    def compute_energy_reward(self, 
                             torques: np.ndarray, 
                             velocities: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute energy efficiency reward.
        Penalizes high torques and encourages smooth motion.
        
        Args:
            torques: Applied joint torques
            velocities: Joint velocities
            
        Returns:
            Tuple of (energy_reward, energy_info)
        """
        # Power consumption: P = |tau * qdot|
        power = np.abs(torques * velocities)
        total_power = np.sum(power)
        
        # Store energy history
        self.energy_history.append(total_power)
        self.power_history.append(power)
        
        # Energy efficiency reward (inverse of power)
        energy_reward = 1.0 / (1.0 + total_power)
        
        # Penalty for excessive torque usage
        max_torque = np.max(np.abs(torques))
        torque_penalty = np.exp(-max_torque / 50.0)  # Penalize high torques
        
        # Combined energy reward
        total_energy_reward = energy_reward * torque_penalty
        
        energy_info = {
            'total_power': total_power,
            'max_power': np.max(power),
            'max_torque': max_torque,
            'energy_reward': energy_reward,
            'torque_penalty': torque_penalty,
            'total_energy_reward': total_energy_reward
        }
        
        return total_energy_reward, energy_info
    
    def compute_smoothness_reward(self, 
                                 torques: np.ndarray, 
                                 prev_torques: Optional[np.ndarray]) -> Tuple[float, Dict]:
        """
        Compute motion smoothness reward.
        Penalizes abrupt changes in torque.
        
        Args:
            torques: Current joint torques
            prev_torques: Previous joint torques
            
        Returns:
            Tuple of (smoothness_reward, smoothness_info)
        """
        if prev_torques is None:
            return 0.0, {'torque_change': 0.0, 'smoothness_reward': 0.0}
        
        # Torque change penalty
        torque_change = np.linalg.norm(torques - prev_torques)
        
        # Smoothness reward (exponential decay with torque change)
        smoothness_reward = np.exp(-torque_change / 10.0)
        
        # Store torque history
        self.torque_history.append(torques.copy())
        
        smoothness_info = {
            'torque_change': torque_change,
            'smoothness_reward': smoothness_reward
        }
        
        return smoothness_reward, smoothness_info
    
    def compute_constraint_penalty(self, 
                                  joint_positions: np.ndarray) -> Tuple[float, List[str]]:
        """
        Compute constraint violation penalty.
        
        Args:
            joint_positions: Current joint positions
            
        Returns:
            Tuple of (constraint_penalty, violations)
        """
        penalty = 0.0
        violations = []
        
        for i, joint in enumerate(self.joint_names):
            pos = joint_positions[i]
            limits = self.joint_limits[joint]
            
            # Position limit violation
            if pos < limits['lower'] or pos > limits['upper']:
                violation = min(abs(pos - limits['lower']), abs(pos - limits['upper']))
                penalty += violation * 10.0
                violations.append(f"{joint}: {pos:.3f} outside [{limits['lower']:.3f}, {limits['upper']:.3f}]")
            
            # Velocity limit violation (if available)
            if 'velocity' in limits:
                # This would need joint velocities - placeholder for now
                pass
        
        return penalty, violations
    
    def compute_collision_penalty(self, 
                                 joint_positions: np.ndarray) -> Tuple[float, Optional[str]]:
        """
        Compute collision penalty.
        Simplified collision detection - in practice, use proper collision checking.
        
        Args:
            joint_positions: Current joint positions
            
        Returns:
            Tuple of (collision_penalty, collision_info)
        """
        # Get end-effector pose
        ee_pose = self.get_end_effector_pose()
        if ee_pose is None:
            return 0.0, None
        
        ee_position, _ = ee_pose
        
        # Check if end-effector is below table surface (collision)
        if ee_position[2] < 0.75:  # Table height
            return 50.0, "Collision with table"
        
        # Check if end-effector is too far from workspace
        workspace_radius = 1.0
        if np.linalg.norm(ee_position[:2]) > workspace_radius:
            return 20.0, "Outside workspace"
        
        # Check for self-collision (simplified)
        # In practice, use proper collision checking
        return 0.0, None
    
    def compute_reward(self, 
                      joint_positions: np.ndarray,
                      joint_velocities: np.ndarray,
                      torques: np.ndarray,
                      step_count: int,
                      prev_torques: Optional[np.ndarray] = None) -> Tuple[float, Dict]:
        """
        Compute comprehensive reward function.
        
        Args:
            joint_positions: Current joint positions
            joint_velocities: Current joint velocities
            torques: Applied joint torques
            step_count: Current step in episode
            prev_torques: Previous joint torques (for smoothness)
            
        Returns:
            Tuple of (total_reward, reward_info)
        """
        # Accuracy reward
        accuracy_reward, accuracy_info = self.compute_accuracy_reward(
            joint_positions, joint_velocities
        )
        
        # Efficiency reward
        efficiency_reward, efficiency_info = self.compute_efficiency_reward(step_count)
        
        # Energy reward
        energy_reward, energy_info = self.compute_energy_reward(torques, joint_velocities)
        
        # Smoothness reward
        smoothness_reward, smoothness_info = self.compute_smoothness_reward(
            torques, prev_torques
        )
        
        # Constraint penalty
        constraint_penalty, violations = self.compute_constraint_penalty(joint_positions)
        
        # Collision penalty
        collision_penalty, collision_info = self.compute_collision_penalty(joint_positions)
        
        # Total reward
        total_reward = (
            accuracy_reward +
            self.weights['efficiency'] * efficiency_reward +
            self.weights['energy'] * energy_reward +
            self.weights['smoothness'] * smoothness_reward -
            self.weights['constraint'] * constraint_penalty -
            self.weights['collision'] * collision_penalty
        )
        
        # Check for success
        if accuracy_info['position_error'] < self.position_tolerance:
            self.success_count += 1
        
        # Store episode reward
        self.episode_rewards.append(total_reward)
        
        # Comprehensive reward info
        reward_info = {
            'accuracy': accuracy_info,
            'efficiency': efficiency_info,
            'energy': energy_info,
            'smoothness': smoothness_info,
            'constraint_penalty': constraint_penalty,
            'collision_penalty': collision_penalty,
            'violations': violations,
            'collision_info': collision_info,
            'total_reward': total_reward,
            'success': accuracy_info['position_error'] < self.position_tolerance
        }
        
        return total_reward, reward_info
    
    def get_episode_summary(self) -> Dict:
        """
        Get summary statistics for the episode.
        
        Returns:
            Dictionary with episode statistics
        """
        if not self.energy_history:
            return {}
        
        return {
            'total_energy': np.sum(self.energy_history),
            'average_power': np.mean(self.energy_history),
            'max_power': np.max(self.energy_history),
            'energy_efficiency': 1.0 / (1.0 + np.sum(self.energy_history)),
            'torque_variance': np.var(self.torque_history, axis=0).tolist() if self.torque_history else [],
            'success_count': self.success_count,
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'max_reward': np.max(self.episode_rewards) if self.episode_rewards else 0.0,
            'min_reward': np.min(self.episode_rewards) if self.episode_rewards else 0.0
        }
    
    def reset_episode(self):
        """Reset episode statistics."""
        self.energy_history = []
        self.torque_history = []
        self.power_history = []
        self.episode_rewards = []
    
    def update_weights(self, new_weights: Dict):
        """
        Update reward weights.
        
        Args:
            new_weights: Dictionary with new weight values
        """
        self.weights.update(new_weights)
        rospy.loginfo(f"Updated reward weights: {self.weights}")

def test_reward_function():
    """Test the comprehensive reward function."""
    print("🧪 Testing Comprehensive Reward Function")
    
    # Mock data
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    joint_limits = {
        'joint1': {'lower': -6.28, 'upper': 6.28, 'effort': 150.0},
        'joint2': {'lower': -6.28, 'upper': 6.28, 'effort': 150.0},
        'joint3': {'lower': -3.14, 'upper': 3.14, 'effort': 150.0},
        'joint4': {'lower': -6.28, 'upper': 6.28, 'effort': 28.0},
        'joint5': {'lower': -6.28, 'upper': 6.28, 'effort': 28.0},
        'joint6': {'lower': -6.28, 'upper': 6.28, 'effort': 28.0}
    }
    
    target_position = np.array([0.6, 0.0, 0.8])
    
    # Create reward function
    reward_func = ComprehensiveManipulatorReward(
        target_position=target_position,
        joint_names=joint_names,
        joint_limits=joint_limits,
        max_episode_steps=1000
    )
    
    # Test data
    joint_positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    joint_velocities = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    torques = np.array([10.0, 15.0, 20.0, 5.0, 8.0, 12.0])
    step_count = 50
    
    # Compute reward
    reward, info = reward_func.compute_reward(
        joint_positions, joint_velocities, torques, step_count
    )
    
    print(f"✓ Reward computation successful: {reward:.3f}")
    print(f"✓ Reward info keys: {list(info.keys())}")
    
    # Test episode summary
    summary = reward_func.get_episode_summary()
    print(f"✓ Episode summary: {summary}")
    
    print("🎉 Comprehensive Reward Function test completed!")

if __name__ == "__main__":
    test_reward_function()
