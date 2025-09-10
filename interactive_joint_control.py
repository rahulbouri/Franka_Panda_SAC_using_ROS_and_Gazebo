#!/usr/bin/env python3
"""
Interactive Joint Control with Constraint Awareness
Allows interactive control of manipulator joints with real-time joint state monitoring.
"""

import rospy
import numpy as np
import time
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs

class InteractiveJointController:
    def __init__(self):
        rospy.init_node('interactive_joint_controller', anonymous=True)
        
        # Joint names for the 6-DOF manipulator
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
        self.barrier_gain = 5.0
        
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
        
        # Control state
        self.is_controlling = False
        self.control_rate = 20  # Hz
        
        rospy.loginfo("Interactive Joint Controller initialized")
        rospy.loginfo("Joint limits loaded with constraint awareness")
        
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
    
    def compute_barrier_gradient(self, positions):
        """Compute gradient of barrier function for constraint enforcement"""
        gradients = []
        
        for i, (joint, limits) in enumerate(self.joint_limits.items()):
            pos = positions[i]
            lower = limits['lower'] + self.safety_margin
            upper = limits['upper'] - self.safety_margin
            
            # Gradient: dB/dq = 1/(q - q_min) - 1/(q_max - q)
            if pos <= lower or pos >= upper:
                gradients.append(float('inf'))
            else:
                gradient = 1.0 / (pos - lower) - 1.0 / (upper - pos)
                gradients.append(gradient)
        
        return np.array(gradients)
    
    def compute_constraint_aware_effort(self, desired_effort, positions):
        """Compute constraint-aware effort using barrier functions"""
        # Compute barrier function gradient
        barrier_grad = self.compute_barrier_gradient(positions)
        
        # Apply barrier function correction
        constraint_effort = np.zeros_like(desired_effort)
        
        for i, (joint, limits) in enumerate(self.joint_limits.items()):
            if np.isfinite(barrier_grad[i]):
                # Apply barrier correction
                constraint_effort[i] = desired_effort[i] - self.barrier_gain * barrier_grad[i]
            else:
                # At limits, apply maximum constraint force
                if positions[i] <= limits['lower'] + self.safety_margin:
                    constraint_effort[i] = limits['effort']  # Push away from lower limit
                elif positions[i] >= limits['upper'] - self.safety_margin:
                    constraint_effort[i] = -limits['effort']  # Push away from upper limit
                else:
                    constraint_effort[i] = desired_effort[i]
            
            # Clamp to effort limits
            constraint_effort[i] = np.clip(constraint_effort[i], -limits['effort'], limits['effort'])
        
        return constraint_effort
    
    def check_constraints(self, positions):
        """Check joint constraint violations"""
        violations = []
        
        for i, (joint, limits) in enumerate(self.joint_limits.items()):
            pos = positions[i]
            if pos < limits['lower'] or pos > limits['upper']:
                violations.append(f"{joint}: {pos:.3f} outside [{limits['lower']:.3f}, {limits['upper']:.3f}]")
        
        return violations
    
    def apply_effort(self, effort):
        """Apply effort commands to joints"""
        for i, joint in enumerate(self.joint_names):
            msg = Float64()
            msg.data = effort[i]
            self.effort_publishers[joint].publish(msg)
    
    def print_status(self):
        """Print current status with constraint information"""
        positions = self.get_joint_positions()
        velocities = self.get_joint_velocities()
        efforts = self.get_joint_efforts()
        ee_pose = self.get_end_effector_pose()
        violations = self.check_constraints(list(positions.values()) if positions else [])
        
        print(f"\n{'='*80}")
        print("🤖 MANIPULATOR STATUS - INTERACTIVE CONTROL")
        print(f"{'='*80}")
        
        if positions:
            print("📍 Joint Positions (rad) [limits]:")
            for i, joint in enumerate(self.joint_names):
                pos = positions.get(joint, "N/A")
                limits = self.joint_limits[joint]
                print(f"  {joint:20s}: {pos:8.4f} [{limits['lower']:6.2f}, {limits['upper']:6.2f}]")
        
        if velocities:
            print("\n⚡ Joint Velocities (rad/s):")
            for joint in self.joint_names:
                vel = velocities.get(joint, "N/A")
                print(f"  {joint:20s}: {vel:8.4f}")
        
        if efforts:
            print("\n💥 Joint Efforts (Nm):")
            for joint in self.joint_names:
                eff = efforts.get(joint, "N/A")
                print(f"  {joint:20s}: {eff:8.1f}")
        
        if ee_pose:
            print(f"\n🎯 End-Effector Position (m):")
            print(f"  X: {ee_pose.pose.position.x:.4f}")
            print(f"  Y: {ee_pose.pose.position.y:.4f}")
            print(f"  Z: {ee_pose.pose.position.z:.4f}")
        
        if violations:
            print(f"\n⚠️  CONSTRAINT VIOLATIONS:")
            for violation in violations:
                print(f"  {violation}")
        else:
            print(f"\n✅ All joints within limits")
        
        print(f"{'='*80}")
    
    def move_joint(self, joint_name, effort, duration=3.0):
        """Move a specific joint with constraint awareness"""
        if joint_name not in self.joint_names:
            rospy.logerr(f"Unknown joint: {joint_name}")
            return False
        
        rospy.loginfo(f"\n🎯 Moving {joint_name} with effort {effort} Nm for {duration}s")
        
        # Get current positions
        positions = self.get_joint_positions()
        if positions is None:
            rospy.logerr("Cannot get joint positions")
            return False
        
        # Convert to array for constraint computation
        pos_array = np.array([positions.get(joint, 0.0) for joint in self.joint_names])
        desired_effort = np.zeros(len(self.joint_names))
        
        # Set effort for target joint
        joint_idx = self.joint_names.index(joint_name)
        desired_effort[joint_idx] = effort
        
        # Apply constraint-aware control
        start_time = time.time()
        rate = rospy.Rate(self.control_rate)
        
        while (time.time() - start_time) < duration and not rospy.is_shutdown():
            # Update current positions
            positions = self.get_joint_positions()
            if positions:
                pos_array = np.array([positions.get(joint, 0.0) for joint in self.joint_names])
                
                # Compute constraint-aware effort
                constraint_effort = self.compute_constraint_aware_effort(desired_effort, pos_array)
                
                # Apply effort
                self.apply_effort(constraint_effort)
                
                # Log every 0.5 seconds
                elapsed = time.time() - start_time
                if int(elapsed * 2) % 2 == 0:
                    current_pos = positions.get(joint_name, 0.0)
                    rospy.loginfo(f"  t={elapsed:.1f}s: {joint_name} = {current_pos:.4f} rad")
            
            rate.sleep()
        
        # Stop all efforts
        self.apply_effort(np.zeros(len(self.joint_names)))
        
        rospy.loginfo(f"✅ {joint_name} movement completed")
        return True
    
    def move_all_joints(self, efforts, duration=3.0):
        """Move all joints with constraint awareness"""
        rospy.loginfo(f"\n🎯 Moving all joints with efforts: {efforts}")
        
        # Apply constraint-aware control
        start_time = time.time()
        rate = rospy.Rate(self.control_rate)
        
        while (time.time() - start_time) < duration and not rospy.is_shutdown():
            # Update current positions
            positions = self.get_joint_positions()
            if positions:
                pos_array = np.array([positions.get(joint, 0.0) for joint in self.joint_names])
                
                # Compute constraint-aware effort
                constraint_effort = self.compute_constraint_aware_effort(efforts, pos_array)
                
                # Apply effort
                self.apply_effort(constraint_effort)
                
                # Log every 0.5 seconds
                elapsed = time.time() - start_time
                if int(elapsed * 2) % 2 == 0:
                    rospy.loginfo(f"  t={elapsed:.1f}s: Applying constraint-aware efforts")
            
            rate.sleep()
        
        # Stop all efforts
        self.apply_effort(np.zeros(len(self.joint_names)))
        
        rospy.loginfo(f"✅ All joints movement completed")
        return True
    
    def interactive_control(self):
        """Interactive control interface"""
        rospy.loginfo(f"\n{'='*80}")
        rospy.loginfo("🎮 INTERACTIVE JOINT CONTROL")
        rospy.loginfo(f"{'='*80}")
        rospy.loginfo("Available commands:")
        rospy.loginfo("  's' - Show current status")
        rospy.loginfo("  'm <joint_name> <effort> <duration>' - Move single joint")
        rospy.loginfo("  'a <effort1> <effort2> <effort3> <effort4> <effort5> <effort6> <duration>' - Move all joints")
        rospy.loginfo("  'd' - Demo sequence")
        rospy.loginfo("  'q' - Quit")
        rospy.loginfo("\nAvailable joints:")
        for i, joint in enumerate(self.joint_names):
            rospy.loginfo(f"  {i+1}. {joint}")
        
        while not rospy.is_shutdown():
            try:
                command = input("\n🎮 Enter command: ").strip().split()
                if not command:
                    continue
                
                if command[0] == 'q':
                    rospy.loginfo("👋 Exiting interactive control")
                    break
                elif command[0] == 's':
                    self.print_status()
                elif command[0] == 'm' and len(command) >= 3:
                    joint_name = command[1]
                    effort = float(command[2])
                    duration = float(command[3]) if len(command) > 3 else 3.0
                    self.move_joint(joint_name, effort, duration)
                elif command[0] == 'a' and len(command) >= 8:
                    efforts = [float(x) for x in command[1:7]]
                    duration = float(command[7]) if len(command) > 7 else 3.0
                    self.move_all_joints(efforts, duration)
                elif command[0] == 'd':
                    self.demo_sequence()
                else:
                    rospy.logwarn("❌ Invalid command. Type 'q' to quit or 's' for status.")
                    
            except (ValueError, IndexError) as e:
                rospy.logwarn(f"❌ Invalid input: {e}")
            except KeyboardInterrupt:
                rospy.loginfo("👋 Interrupted by user")
                break
    
    def demo_sequence(self):
        """Run a demonstration sequence"""
        rospy.loginfo("\n🎬 Running demonstration sequence...")
        
        # Demo movements
        demo_movements = [
            ('shoulder_pan_joint', 50.0, 3.0),
            ('shoulder_lift_joint', -30.0, 3.0),
            ('elbow_joint', -40.0, 3.0),
            ('wrist_1_joint', 20.0, 2.0),
            ('wrist_2_joint', -20.0, 2.0),
            ('wrist_3_joint', 15.0, 2.0),
        ]
        
        for joint, effort, duration in demo_movements:
            rospy.loginfo(f"\n--- Moving {joint} ---")
            self.move_joint(joint, effort, duration)
            self.print_status()
            rospy.sleep(1.0)
        
        rospy.loginfo("🎉 Demonstration sequence completed!")

def main():
    """Main function"""
    try:
        # Create interactive controller
        controller = InteractiveJointController()
        
        # Show initial status
        controller.print_status()
        
        # Start interactive control
        controller.interactive_control()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Controller interrupted by user")
    except Exception as e:
        rospy.logerr(f"Controller failed: {e}")

if __name__ == '__main__':
    main()
