#!/usr/bin/env python3
"""
RL Environment with Constraint-Aware Joint Control
Integrates joint limits, barrier functions, and Lagrangian Neural Networks for RL training.
"""

import rospy
import numpy as np
import torch
import torch.nn as nn
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs
from lagrangian_constraint_integration import ConstraintAwareLagrangianNN, ConstraintAwareRLEnvironment

class RLConstraintEnvironment:
    """
    RL Environment with constraint-aware joint control for manipulator training.
    Implements all strategies from web resources for joint constraint handling.
    """
    
    def __init__(self):
        rospy.init_node('rl_constraint_environment', anonymous=True)
        
        # Joint configuration
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Joint limits from URDF
        self.joint_limits = {
            'shoulder_pan_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 150.0},
            'shoulder_lift_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 150.0},
            'elbow_joint': {'lower': -3.141592653589793, 'upper': 3.141592653589793, 'effort': 150.0},
            'wrist_1_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 28.0},
            'wrist_2_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 28.0},
            'wrist_3_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 28.0}
        }
        
        # Constraint parameters
        self.safety_margin = 0.1
        self.constraint_penalty = 10.0
        self.barrier_gain = 5.0
        
        # ROS publishers and subscribers
        self.effort_publishers = {}
        for joint in self.joint_names:
            topic = f'/manipulator/{joint}_effort/command'
            self.effort_publishers[joint] = rospy.Publisher(topic, Float64, queue_size=1)
        
        self.joint_states_sub = rospy.Subscriber('/manipulator/joint_states', JointState, self.joint_states_callback)
        self.current_joint_states = None
        self.joint_states_received = False
        
        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Lagrangian Neural Network
        self.lnn = ConstraintAwareLagrangianNN(n_joints=len(self.joint_names))
        
        # RL environment state
        self.episode_count = 0
        self.step_count = 0
        self.max_steps = 1000
        
        rospy.loginfo("RL Constraint Environment initialized")
        rospy.loginfo("Joint limits loaded and constraint handling enabled")
        
        # Wait for joint states
        self.wait_for_joint_states()
    
    def joint_states_callback(self, msg):
        """Callback for joint state messages"""
        self.current_joint_states = msg
        self.joint_states_received = True
    
    def wait_for_joint_states(self, timeout=10.0):
        """Wait for joint states to be received"""
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
        
        return np.array(positions), np.array(velocities)
    
    def get_end_effector_pose(self):
        """Get end-effector pose"""
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
    
    def check_constraints(self, positions, velocities=None):
        """Check joint constraint violations"""
        violations = []
        
        for i, joint in enumerate(self.joint_names):
            pos = positions[i]
            limits = self.joint_limits[joint]
            
            # Check position limits
            if pos < limits['lower'] or pos > limits['upper']:
                violations.append(f"{joint}: position {pos:.3f} outside [{limits['lower']:.3f}, {limits['upper']:.3f}]")
            
            # Check velocity limits if provided
            if velocities is not None:
                vel = velocities[i]
                if abs(vel) > limits.get('velocity', 3.14):
                    violations.append(f"{joint}: velocity {vel:.3f} exceeds limit")
        
        return violations
    
    def compute_barrier_function(self, positions):
        """Compute barrier function for constraint enforcement"""
        barrier_values = []
        
        for i, joint in enumerate(self.joint_names):
            pos = positions[i]
            limits = self.joint_limits[joint]
            
            # Safety bounds
            lower = limits['lower'] + self.safety_margin
            upper = limits['upper'] - self.safety_margin
            
            # Barrier function: B(q) = -log((q - q_min)(q_max - q))
            if pos <= lower or pos >= upper:
                barrier_values.append(float('inf'))
            else:
                barrier_value = -np.log((pos - lower) * (upper - pos))
                barrier_values.append(barrier_value)
        
        return np.array(barrier_values)
    
    def compute_barrier_gradient(self, positions):
        """Compute gradient of barrier function"""
        gradients = []
        
        for i, joint in enumerate(self.joint_names):
            pos = positions[i]
            limits = self.joint_limits[joint]
            
            # Safety bounds
            lower = limits['lower'] + self.safety_margin
            upper = limits['upper'] - self.safety_margin
            
            # Gradient: dB/dq = 1/(q - q_min) - 1/(q_max - q)
            if pos <= lower or pos >= upper:
                gradients.append(float('inf'))
            else:
                gradient = 1.0 / (pos - lower) - 1.0 / (upper - pos)
                gradients.append(gradient)
        
        return np.array(gradients)
    
    def compute_constraint_aware_effort(self, desired_effort, positions, velocities):
        """Compute constraint-aware effort using barrier functions"""
        # Compute barrier function gradient
        barrier_grad = self.compute_barrier_gradient(positions)
        
        # Apply barrier function correction
        constraint_effort = np.zeros_like(desired_effort)
        
        for i, joint in enumerate(self.joint_names):
            if np.isfinite(barrier_grad[i]):
                # Apply barrier correction
                constraint_effort[i] = desired_effort[i] - self.barrier_gain * barrier_grad[i]
            else:
                # At limits, apply maximum constraint force
                limits = self.joint_limits[joint]
                if positions[i] <= limits['lower'] + self.safety_margin:
                    constraint_effort[i] = limits['effort']  # Push away from lower limit
                elif positions[i] >= limits['upper'] - self.safety_margin:
                    constraint_effort[i] = -limits['effort']  # Push away from upper limit
                else:
                    constraint_effort[i] = desired_effort[i]
            
            # Clamp to effort limits
            max_effort = self.joint_limits[joint]['effort']
            constraint_effort[i] = np.clip(constraint_effort[i], -max_effort, max_effort)
        
        return constraint_effort
    
    def compute_reward(self, positions, velocities, target_positions, target_velocities, effort):
        """Compute reward function with constraint penalties"""
        # Position error
        pos_error = np.linalg.norm(positions - target_positions)
        
        # Velocity error
        vel_error = np.linalg.norm(velocities - target_velocities)
        
        # Effort penalty
        effort_penalty = np.linalg.norm(effort)
        
        # Constraint violation penalty
        barrier_values = self.compute_barrier_function(positions)
        constraint_penalty = np.sum(barrier_values[np.isfinite(barrier_values)])
        
        # Task-specific reward (reach target)
        target_reward = 1.0 / (1.0 + pos_error)  # Higher when closer to target
        
        # Total reward
        reward = target_reward - 0.1 * vel_error - 0.01 * effort_penalty - constraint_penalty
        
        return reward, {
            'position_error': pos_error,
            'velocity_error': vel_error,
            'effort_penalty': effort_penalty,
            'constraint_penalty': constraint_penalty,
            'target_reward': target_reward,
            'total_reward': reward
        }
    
    def apply_effort(self, effort):
        """Apply effort commands to joints"""
        for i, joint in enumerate(self.joint_names):
            msg = Float64()
            msg.data = effort[i]
            self.effort_publishers[joint].publish(msg)
    
    def reset_episode(self):
        """Reset episode with random valid joint configuration"""
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
        
        # Set joint configuration using Gazebo service
        self.set_joint_configuration(positions)
        
        self.episode_count += 1
        self.step_count = 0
        
        return positions, velocities
    
    def set_joint_configuration(self, positions):
        """Set joint configuration using Gazebo service"""
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
        """Execute one RL step"""
        self.step_count += 1
        
        # Get current observation
        positions, velocities = self.get_observation()
        if positions is None:
            return None, 0.0, True, {}
        
        # Convert action to effort (action is normalized [-1, 1])
        desired_effort = action * np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
        
        # Compute constraint-aware effort
        constraint_effort = self.compute_constraint_aware_effort(desired_effort, positions, velocities)
        
        # Apply effort
        self.apply_effort(constraint_effort)
        
        # Wait for dynamics to update
        rospy.sleep(0.1)
        
        # Get new observation
        new_positions, new_velocities = self.get_observation()
        if new_positions is None:
            return None, 0.0, True, {}
        
        # Compute reward (simplified target: center of workspace)
        target_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        target_velocities = np.zeros_like(target_positions)
        
        reward, reward_info = self.compute_reward(
            new_positions, new_velocities, target_positions, target_velocities, constraint_effort
        )
        
        # Check for termination
        constraint_violations = self.check_constraints(new_positions, new_velocities)
        done = len(constraint_violations) > 0 or self.step_count >= self.max_steps
        
        # Prepare observation
        observation = np.concatenate([new_positions, new_velocities])
        
        info = {
            'constraint_violations': constraint_violations,
            'effort_applied': constraint_effort,
            'reward_info': reward_info
        }
        
        return observation, reward, done, info
    
    def run_episode(self, max_steps=100):
        """Run a complete episode"""
        rospy.loginfo(f"🎬 Starting episode {self.episode_count}")
        
        # Reset episode
        positions, velocities = self.reset_episode()
        observation = np.concatenate([positions, velocities])
        
        total_reward = 0.0
        step_count = 0
        
        for step in range(max_steps):
            # Random action for demonstration
            action = np.random.uniform(-1.0, 1.0, size=len(self.joint_names))
            
            # Execute step
            observation, reward, done, info = self.step(action)
            
            if observation is None:
                rospy.logwarn("⚠️  Episode terminated due to observation failure")
                break
            
            total_reward += reward
            step_count += 1
            
            # Log progress
            if step % 10 == 0:
                rospy.loginfo(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
                if info['constraint_violations']:
                    rospy.logwarn(f"    Constraint violations: {info['constraint_violations']}")
            
            if done:
                rospy.loginfo(f"🏁 Episode {self.episode_count} completed in {step_count} steps")
                break
        
        rospy.loginfo(f"📊 Episode {self.episode_count} summary:")
        rospy.loginfo(f"  Total reward: {total_reward:.3f}")
        rospy.loginfo(f"  Steps: {step_count}")
        rospy.loginfo(f"  Average reward: {total_reward/step_count:.3f}")
        
        return total_reward, step_count

def main():
    """Main function to demonstrate RL constraint environment"""
    try:
        # Create RL environment
        env = RLConstraintEnvironment()
        
        # Run demonstration episodes
        rospy.loginfo("🚀 Starting RL Constraint Environment Demonstration")
        
        for episode in range(3):
            total_reward, steps = env.run_episode(max_steps=50)
            rospy.sleep(2.0)  # Pause between episodes
        
        rospy.loginfo("🎉 RL Constraint Environment demonstration completed!")
        
    except rospy.ROSInterruptException:
        rospy.loginfo("RL Environment interrupted by user")
    except Exception as e:
        rospy.logerr(f"RL Environment failed: {e}")

if __name__ == '__main__':
    main()
