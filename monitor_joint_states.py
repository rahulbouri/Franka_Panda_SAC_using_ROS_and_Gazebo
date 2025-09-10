#!/usr/bin/env python3
"""
Joint State Monitor
Continuously monitors and displays joint states in real-time.
"""

import rospy
import numpy as np
import time
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs

class JointStateMonitor:
    def __init__(self):
        rospy.init_node('joint_state_monitor', anonymous=True)
        
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
        
        # Subscriber for joint states
        self.joint_states_sub = rospy.Subscriber('/manipulator/joint_states', JointState, self.joint_states_callback)
        self.current_joint_states = None
        self.joint_states_received = False
        
        # TF listener for end-effector position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Monitoring state
        self.monitoring = False
        self.update_rate = 10  # Hz
        
        rospy.loginfo("Joint State Monitor initialized")
        
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
            return None
    
    def check_constraints(self, positions):
        """Check joint constraint violations"""
        violations = []
        
        for i, (joint, limits) in enumerate(self.joint_limits.items()):
            pos = positions[i]
            if pos < limits['lower'] or pos > limits['upper']:
                violations.append(f"{joint}: {pos:.3f} outside [{limits['lower']:.3f}, {limits['upper']:.3f}]")
        
        return violations
    
    def print_status(self):
        """Print current status"""
        positions = self.get_joint_positions()
        velocities = self.get_joint_velocities()
        efforts = self.get_joint_efforts()
        ee_pose = self.get_end_effector_pose()
        
        if positions is None:
            print("❌ No joint state data available")
            return
        
        # Convert to arrays for constraint checking
        pos_array = np.array([positions.get(joint, 0.0) for joint in self.joint_names])
        violations = self.check_constraints(pos_array)
        
        # Clear screen and print header
        print("\033[2J\033[H")  # Clear screen
        print(f"{'='*80}")
        print("📊 REAL-TIME JOINT STATE MONITOR")
        print(f"{'='*80}")
        print(f"Time: {rospy.get_time():.2f}s")
        
        # Joint positions
        print("\n📍 Joint Positions (rad) [limits]:")
        for i, joint in enumerate(self.joint_names):
            pos = positions.get(joint, "N/A")
            limits = self.joint_limits[joint]
            status = "⚠️ " if pos < limits['lower'] or pos > limits['upper'] else "✅"
            print(f"  {status} {joint:20s}: {pos:8.4f} [{limits['lower']:6.2f}, {limits['upper']:6.2f}]")
        
        # Joint velocities
        if velocities:
            print("\n⚡ Joint Velocities (rad/s):")
            for joint in self.joint_names:
                vel = velocities.get(joint, "N/A")
                print(f"  {joint:20s}: {vel:8.4f}")
        
        # Joint efforts
        if efforts:
            print("\n💥 Joint Efforts (Nm):")
            for joint in self.joint_names:
                eff = efforts.get(joint, "N/A")
                print(f"  {joint:20s}: {eff:8.1f}")
        
        # End-effector pose
        if ee_pose:
            print(f"\n🎯 End-Effector Position (m):")
            print(f"  X: {ee_pose.pose.position.x:.4f}")
            print(f"  Y: {ee_pose.pose.position.y:.4f}")
            print(f"  Z: {ee_pose.pose.position.z:.4f}")
        
        # Constraint violations
        if violations:
            print(f"\n⚠️  CONSTRAINT VIOLATIONS:")
            for violation in violations:
                print(f"  {violation}")
        else:
            print(f"\n✅ All joints within limits")
        
        print(f"{'='*80}")
        print("Press Ctrl+C to stop monitoring")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        rospy.loginfo("🔍 Starting joint state monitoring...")
        rospy.loginfo("Press Ctrl+C to stop")
        
        self.monitoring = True
        rate = rospy.Rate(self.update_rate)
        
        try:
            while not rospy.is_shutdown() and self.monitoring:
                self.print_status()
                rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("\n👋 Monitoring stopped by user")
        except Exception as e:
            rospy.logerr(f"Monitoring failed: {e}")
        finally:
            self.monitoring = False
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False

def main():
    """Main function"""
    try:
        # Create monitor
        monitor = JointStateMonitor()
        
        # Start monitoring
        monitor.start_monitoring()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Monitor interrupted by user")
    except Exception as e:
        rospy.logerr(f"Monitor failed: {e}")

if __name__ == '__main__':
    main()
