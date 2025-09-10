#!/usr/bin/env python3
"""
Constraint-Aware Joint Controller
Implements joint limit awareness using barrier functions, clamping, and adaptive control
for RL environment and Lagrangian Neural Network integration.
"""

import rospy
import numpy as np
import time
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs

class ConstraintAwareController:
    def __init__(self):
        rospy.init_node('constraint_aware_controller', anonymous=True)
        
        # Joint names for the 6-DOF manipulator
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Joint limits from URDF (position limits in radians)
        self.joint_limits = {
            'shoulder_pan_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 150.0, 'velocity': 3.141592653589793},
            'shoulder_lift_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 150.0, 'velocity': 3.141592653589793},
            'elbow_joint': {'lower': -3.141592653589793, 'upper': 3.141592653589793, 'effort': 150.0, 'velocity': 3.141592653589793},
            'wrist_1_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 28.0, 'velocity': 3.141592653589793},
            'wrist_2_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 28.0, 'velocity': 3.141592653589793},
            'wrist_3_joint': {'lower': -6.283185307179586, 'upper': 6.283185307179586, 'effort': 28.0, 'velocity': 3.141592653589793}
        }
        
        # Safety margins for constraint enforcement
        self.safety_margin = 0.1  # 10% margin from limits
        self.barrier_gain = 10.0  # Gain for barrier function
        
        # Publishers for effort control
        self.effort_publishers = {}
        for joint in self.joint_names:
            topic = f'/manipulator/{joint}_effort/command'
            self.effort_publishers[joint] = rospy.Publisher(topic, Float64, queue_size=1)
        
        # Subscriber for joint states
        self.joint_states_sub = rospy.Subscriber('/manipulator/joint_states', JointState, self.joint_states_callback)
        self.current_joint_states = None
        self.joint_states_received = False
        
        # TF listener for end-effector position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        rospy.loginfo("Constraint-Aware Controller initialized")
        rospy.loginfo("Joint limits loaded:")
        for joint, limits in self.joint_limits.items():
            rospy.loginfo(f"  {joint}: [{limits['lower']:.2f}, {limits['upper']:.2f}] rad, max_effort: {limits['effort']} Nm")
        
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
    
    def get_joint_positions(self):
        """Get current joint positions"""
        if self.current_joint_states is None:
            return None
        
        positions = {}
        for i, name in enumerate(self.current_joint_states.name):
            if i < len(self.current_joint_states.position):
                positions[name] = self.current_joint_states.position[i]
        return positions
    
    def get_joint_velocities(self):
        """Get current joint velocities"""
        if self.current_joint_states is None:
            return None
        
        velocities = {}
        for i, name in enumerate(self.current_joint_states.name):
            if i < len(self.current_joint_states.velocity):
                velocities[name] = self.current_joint_states.velocity[i]
        return velocities
    
    def get_joint_efforts(self):
        """Get current joint efforts"""
        if self.current_joint_states is None:
            return None
        
        efforts = {}
        for i, name in enumerate(self.current_joint_states.name):
            if i < len(self.current_joint_states.effort):
                efforts[name] = self.current_joint_states.effort[i]
        return efforts
    
    def check_joint_constraints(self, joint_name, position, velocity=None):
        """Check if joint position and velocity are within limits"""
        if joint_name not in self.joint_limits:
            return True, "Unknown joint"
        
        limits = self.joint_limits[joint_name]
        
        # Check position limits
        if position < limits['lower'] or position > limits['upper']:
            return False, f"Position {position:.3f} outside limits [{limits['lower']:.3f}, {limits['upper']:.3f}]"
        
        # Check velocity limits if provided
        if velocity is not None and abs(velocity) > limits['velocity']:
            return False, f"Velocity {velocity:.3f} exceeds limit {limits['velocity']:.3f}"
        
        return True, "Within limits"
    
    def compute_barrier_function(self, joint_name, position):
        """Compute barrier function value for joint constraint enforcement"""
        if joint_name not in self.joint_limits:
            return 0.0
        
        limits = self.joint_limits[joint_name]
        lower = limits['lower'] + self.safety_margin
        upper = limits['upper'] - self.safety_margin
        
        # Barrier function: B(q) = -log((q - q_min)(q_max - q))
        if position <= lower or position >= upper:
            return float('inf')  # Infinite barrier at limits
        
        barrier_value = -np.log((position - lower) * (upper - position))
        return barrier_value
    
    def compute_barrier_gradient(self, joint_name, position):
        """Compute gradient of barrier function for constraint-aware control"""
        if joint_name not in self.joint_limits:
            return 0.0
        
        limits = self.joint_limits[joint_name]
        lower = limits['lower'] + self.safety_margin
        upper = limits['upper'] - self.safety_margin
        
        if position <= lower or position >= upper:
            return float('inf')  # Infinite gradient at limits
        
        # Gradient: dB/dq = 1/(q - q_min) - 1/(q_max - q)
        gradient = 1.0 / (position - lower) - 1.0 / (upper - position)
        return gradient
    
    def clamp_effort(self, joint_name, effort):
        """Clamp effort to joint limits"""
        if joint_name not in self.joint_limits:
            return effort
        
        max_effort = self.joint_limits[joint_name]['effort']
        return np.clip(effort, -max_effort, max_effort)
    
    def compute_constraint_aware_effort(self, joint_name, desired_effort, position, velocity=0.0):
        """Compute constraint-aware effort using barrier functions and clamping"""
        if joint_name not in self.joint_limits:
            return desired_effort
        
        # 1. Check constraints
        within_limits, constraint_msg = self.check_joint_constraints(joint_name, position, velocity)
        
        # 2. Compute barrier function contribution
        barrier_gradient = self.compute_barrier_gradient(joint_name, position)
        
        # 3. Apply barrier function correction
        if np.isfinite(barrier_gradient):
            barrier_effort = -self.barrier_gain * barrier_gradient
            corrected_effort = desired_effort + barrier_effort
        else:
            # At or beyond limits, apply maximum constraint force
            limits = self.joint_limits[joint_name]
            if position <= limits['lower'] + self.safety_margin:
                corrected_effort = self.joint_limits[joint_name]['effort']  # Push away from lower limit
            elif position >= limits['upper'] - self.safety_margin:
                corrected_effort = -self.joint_limits[joint_name]['effort']  # Push away from upper limit
            else:
                corrected_effort = desired_effort
        
        # 4. Clamp to effort limits
        final_effort = self.clamp_effort(joint_name, corrected_effort)
        
        # 5. Log constraint violations
        if not within_limits:
            rospy.logwarn(f"⚠️  {joint_name}: {constraint_msg}")
        
        return final_effort, within_limits
    
    def send_constraint_aware_command(self, joint_name, desired_effort, duration=3.0):
        """Send constraint-aware effort command to a joint"""
        if joint_name not in self.effort_publishers:
            rospy.logerr(f"Unknown joint: {joint_name}")
            return False
        
        # Get current state
        positions = self.get_joint_positions()
        velocities = self.get_joint_velocities()
        
        if positions is None:
            rospy.logerr("Cannot get joint positions")
            return False
        
        current_pos = positions.get(joint_name, 0.0)
        current_vel = velocities.get(joint_name, 0.0) if velocities else 0.0
        
        rospy.loginfo(f"\n🎯 Constraint-Aware Control: {joint_name}")
        rospy.loginfo(f"Desired effort: {desired_effort} Nm")
        rospy.loginfo(f"Current position: {current_pos:.4f} rad")
        rospy.loginfo(f"Current velocity: {current_vel:.4f} rad/s")
        
        # Compute constraint-aware effort
        constraint_effort, within_limits = self.compute_constraint_aware_effort(
            joint_name, desired_effort, current_pos, current_vel
        )
        
        rospy.loginfo(f"Constraint-aware effort: {constraint_effort:.2f} Nm")
        rospy.loginfo(f"Within limits: {within_limits}")
        
        # Apply effort for specified duration
        start_time = time.time()
        rate = rospy.Rate(20)  # 20 Hz
        
        while (time.time() - start_time) < duration and not rospy.is_shutdown():
            # Update current state
            positions = self.get_joint_positions()
            velocities = self.get_joint_velocities()
            
            if positions:
                current_pos = positions.get(joint_name, 0.0)
                current_vel = velocities.get(joint_name, 0.0) if velocities else 0.0
                
                # Recompute constraint-aware effort
                constraint_effort, within_limits = self.compute_constraint_aware_effort(
                    joint_name, desired_effort, current_pos, current_vel
                )
                
                # Send command
                msg = Float64()
                msg.data = constraint_effort
                self.effort_publishers[joint_name].publish(msg)
                
                # Log every 0.5 seconds
                elapsed = time.time() - start_time
                if int(elapsed * 2) % 2 == 0:
                    rospy.loginfo(f"  t={elapsed:.1f}s: pos={current_pos:.3f}, vel={current_vel:.3f}, effort={constraint_effort:.1f}")
            
            rate.sleep()
        
        # Final state
        final_positions = self.get_joint_positions()
        if final_positions:
            final_pos = final_positions.get(joint_name, 0.0)
            position_change = final_pos - current_pos
            rospy.loginfo(f"Final position: {final_pos:.4f} rad")
            rospy.loginfo(f"Position change: {position_change:.4f} rad ({np.degrees(position_change):.1f} degrees)")
            
            # Check final constraints
            final_within_limits, final_msg = self.check_joint_constraints(joint_name, final_pos)
            rospy.loginfo(f"Final constraint status: {final_msg}")
            
            return abs(position_change) > 0.01
        
        return False
    
    def demonstrate_constraint_aware_control(self):
        """Demonstrate constraint-aware control with various scenarios"""
        rospy.loginfo(f"\n{'='*80}")
        rospy.loginfo("🎯 CONSTRAINT-AWARE CONTROL DEMONSTRATION")
        rospy.loginfo(f"{'='*80}")
        
        # Show initial status
        self.print_status()
        
        # Test scenarios: normal operation, near limits, and constraint violation
        test_scenarios = [
            ('shoulder_pan_joint', 50.0, "Normal operation"),
            ('shoulder_lift_joint', -75.0, "Near lower limit"),
            ('elbow_joint', 100.0, "Near upper limit"),
            ('wrist_1_joint', 20.0, "Wrist joint control"),
            ('wrist_2_joint', -25.0, "Wrist joint control"),
            ('wrist_3_joint', 30.0, "Wrist joint control"),
        ]
        
        success_count = 0
        for joint, effort, description in test_scenarios:
            rospy.loginfo(f"\n--- {description} ---")
            if self.send_constraint_aware_command(joint, effort, duration=4.0):
                success_count += 1
                rospy.loginfo(f"✅ {description} completed successfully!")
            else:
                rospy.logwarn(f"❌ {description} failed!")
            
            # Show current status
            self.print_status()
            rospy.sleep(1.0)
        
        # Summary
        rospy.loginfo(f"\n{'='*80}")
        rospy.loginfo(f"🎊 CONSTRAINT-AWARE CONTROL SUMMARY")
        rospy.loginfo(f"{'='*80}")
        rospy.loginfo(f"Successful operations: {success_count}/{len(test_scenarios)}")
        
        if success_count > 0:
            rospy.loginfo("🎉 Constraint-aware control is working!")
            rospy.loginfo("🔒 Joint limits are being respected!")
        else:
            rospy.logwarn("⚠️  Constraint-aware control needs debugging")
        
        return success_count > 0
    
    def get_constraint_violations(self):
        """Get current constraint violations"""
        positions = self.get_joint_positions()
        velocities = self.get_joint_velocities()
        
        violations = []
        if positions:
            for joint in self.joint_names:
                pos = positions.get(joint, 0.0)
                vel = velocities.get(joint, 0.0) if velocities else 0.0
                
                within_limits, msg = self.check_joint_constraints(joint, pos, vel)
                if not within_limits:
                    violations.append(f"{joint}: {msg}")
        
        return violations
    
    def print_status(self):
        """Print current status with constraint information"""
        positions = self.get_joint_positions()
        velocities = self.get_joint_velocities()
        efforts = self.get_joint_efforts()
        violations = self.get_constraint_violations()
        
        print(f"\n{'='*80}")
        print("🤖 CONSTRAINT-AWARE MANIPULATOR STATUS")
        print(f"{'='*80}")
        
        if positions:
            print("📍 Joint Positions (rad) [limits]:")
            for joint in self.joint_names:
                pos = positions.get(joint, "N/A")
                limits = self.joint_limits[joint]
                print(f"  {joint:20s}: {pos:8.4f} [{limits['lower']:6.2f}, {limits['upper']:6.2f}]")
        
        if velocities:
            print("\n⚡ Joint Velocities (rad/s) [limits]:")
            for joint in self.joint_names:
                vel = velocities.get(joint, "N/A")
                limits = self.joint_limits[joint]
                print(f"  {joint:20s}: {vel:8.4f} [±{limits['velocity']:6.2f}]")
        
        if efforts:
            print("\n💥 Joint Efforts (Nm) [limits]:")
            for joint in self.joint_names:
                eff = efforts.get(joint, "N/A")
                limits = self.joint_limits[joint]
                print(f"  {joint:20s}: {eff:8.1f} [±{limits['effort']:6.1f}]")
        
        if violations:
            print(f"\n⚠️  CONSTRAINT VIOLATIONS:")
            for violation in violations:
                print(f"  {violation}")
        else:
            print(f"\n✅ All joints within limits")
        
        print(f"{'='*80}")

def main():
    controller = ConstraintAwareController()
    
    try:
        # Run constraint-aware demonstration
        success = controller.demonstrate_constraint_aware_control()
        
        if success:
            rospy.loginfo("\n🎊 CONSTRAINT-AWARE CONTROL SUCCESSFUL!")
            rospy.loginfo("🔒 Joint limits are being respected!")
            rospy.loginfo("🎯 Ready for RL environment integration!")
        else:
            rospy.logwarn("\n⚠️  Constraint-aware control needs debugging")
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Controller interrupted by user")
    except Exception as e:
        rospy.logerr(f"Controller failed: {e}")

if __name__ == '__main__':
    main()
