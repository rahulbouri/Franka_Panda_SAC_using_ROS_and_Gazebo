#!/usr/bin/env python3
"""
Environment Debugging Script
Systematically checks and fixes common issues with Gazebo + ROS control setup.
"""

import rospy
import subprocess
import time
import os
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

class EnvironmentDebugger:
    def __init__(self):
        rospy.init_node('environment_debugger', anonymous=True)
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
    def run_command(self, command, timeout=10):
        """Run a shell command and return output"""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def check_ros_master(self):
        """Check if ROS master is running"""
        print("🔍 Checking ROS Master...")
        code, stdout, stderr = self.run_command("rostopic list")
        if code == 0 and "gazebo" in stdout:
            print("✅ ROS Master is running")
            return True
        else:
            print("❌ ROS Master is not running")
            print(f"Error: {stderr}")
            return False
    
    def check_gazebo_services(self):
        """Check if Gazebo services are available"""
        print("\n🔍 Checking Gazebo Services...")
        services = [
            "/gazebo/get_physics_properties",
            "/gazebo/set_physics_properties", 
            "/gazebo/pause_physics",
            "/gazebo/unpause_physics"
        ]
        
        available_services = []
        for service in services:
            code, stdout, stderr = self.run_command(f"rosservice list | grep {service}")
            if code == 0 and service in stdout:
                available_services.append(service)
                print(f"✅ {service}")
            else:
                print(f"❌ {service}")
        
        return len(available_services) > 0
    
    def check_physics_state(self):
        """Check if physics is running"""
        print("\n🔍 Checking Physics State...")
        try:
            from gazebo_msgs.srv import GetPhysicsProperties
            rospy.wait_for_service('/gazebo/get_physics_properties', timeout=5)
            get_physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
            response = get_physics()
            
            if response.pause:
                print("⚠️  Physics is PAUSED")
                return False
            else:
                print("✅ Physics is RUNNING")
                return True
        except Exception as e:
            print(f"❌ Could not check physics state: {e}")
            return False
    
    def unpause_physics(self):
        """Unpause physics"""
        print("\n🔧 Unpausing Physics...")
        try:
            from gazebo_msgs.srv import Empty
            rospy.wait_for_service('/gazebo/unpause_physics', timeout=5)
            unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
            response = unpause()
            print("✅ Physics unpaused")
            return True
        except Exception as e:
            print(f"❌ Could not unpause physics: {e}")
            return False
    
    def check_controllers(self):
        """Check if controllers are loaded and running"""
        print("\n🔍 Checking Controllers...")
        try:
            from controller_manager_msgs.srv import ListControllers
            rospy.wait_for_service('/manipulator/controller_manager/list_controllers', timeout=5)
            list_controllers = rospy.ServiceProxy('/manipulator/controller_manager/list_controllers', ListControllers)
            response = list_controllers()
            
            if response.controllers:
                print("✅ Controllers found:")
                for controller in response.controllers:
                    status = "RUNNING" if controller.state == "running" else controller.state
                    print(f"  - {controller.name}: {status}")
                return True
            else:
                print("❌ No controllers found")
                return False
        except Exception as e:
            print(f"❌ Could not check controllers: {e}")
            return False
    
    def check_joint_states(self):
        """Check if joint states are being published"""
        print("\n🔍 Checking Joint States...")
        try:
            # Wait for joint states
            joint_states = rospy.wait_for_message('/manipulator/joint_states', JointState, timeout=5)
            
            if joint_states.name:
                print("✅ Joint states are being published")
                print(f"  Found {len(joint_states.name)} joints:")
                for i, name in enumerate(joint_states.name):
                    if i < len(joint_states.position):
                        pos = joint_states.position[i]
                        print(f"    {name}: {pos:.4f} rad")
                return True
            else:
                print("❌ Joint states are empty")
                return False
        except Exception as e:
            print(f"❌ Could not get joint states: {e}")
            return False
    
    def check_effort_topics(self):
        """Check if effort command topics exist"""
        print("\n🔍 Checking Effort Command Topics...")
        code, stdout, stderr = self.run_command("rostopic list | grep effort")
        
        if code == 0 and "effort" in stdout:
            print("✅ Effort command topics found:")
            for line in stdout.split('\n'):
                if 'effort' in line and 'command' in line:
                    print(f"  {line}")
            return True
        else:
            print("❌ No effort command topics found")
            return False
    
    def test_joint_control(self):
        """Test joint control by sending a small effort command"""
        print("\n🔍 Testing Joint Control...")
        try:
            # Test shoulder_pan_joint
            topic = '/manipulator/shoulder_pan_joint_effort/command'
            pub = rospy.Publisher(topic, Float64, queue_size=1)
            
            # Send small effort command
            msg = Float64()
            msg.data = 1.0
            pub.publish(msg)
            
            print(f"✅ Sent effort command to {topic}")
            print("  Check Gazebo simulation for movement")
            return True
        except Exception as e:
            print(f"❌ Could not test joint control: {e}")
            return False
    
    def fix_common_issues(self):
        """Attempt to fix common issues"""
        print("\n🔧 Attempting to Fix Common Issues...")
        
        # Unpause physics
        if not self.check_physics_state():
            self.unpause_physics()
        
        # Check if controllers need to be restarted
        if not self.check_controllers():
            print("⚠️  Controllers may need to be restarted")
            print("  Try: roslaunch simple_manipulator training_env.launch")
        
        return True
    
    def run_full_diagnostic(self):
        """Run complete diagnostic"""
        print("🚀 Starting Environment Diagnostic")
        print("="*50)
        
        # Check ROS master
        if not self.check_ros_master():
            print("\n❌ ROS Master is not running. Please start it first:")
            print("  roscore &")
            return False
        
        # Check Gazebo services
        if not self.check_gazebo_services():
            print("\n❌ Gazebo services not available. Please start Gazebo:")
            print("  roslaunch simple_manipulator training_env.launch")
            return False
        
        # Check physics
        self.check_physics_state()
        
        # Check controllers
        controllers_ok = self.check_controllers()
        
        # Check joint states
        joint_states_ok = self.check_joint_states()
        
        # Check effort topics
        effort_topics_ok = self.check_effort_topics()
        
        # Test joint control
        if controllers_ok and joint_states_ok and effort_topics_ok:
            self.test_joint_control()
        
        # Summary
        print("\n" + "="*50)
        print("📊 DIAGNOSTIC SUMMARY")
        print("="*50)
        
        if controllers_ok and joint_states_ok and effort_topics_ok:
            print("✅ Environment appears to be working correctly!")
            print("🎯 You should be able to control joints now")
        else:
            print("❌ Issues detected. Common fixes:")
            print("  1. Restart Gazebo: roslaunch simple_manipulator training_env.launch")
            print("  2. Unpause physics: rosservice call /gazebo/unpause_physics")
            print("  3. Check controller status: rosservice call /manipulator/controller_manager/list_controllers")
            print("  4. Check joint states: rostopic echo /manipulator/joint_states")
        
        return controllers_ok and joint_states_ok and effort_topics_ok

def main():
    """Main function"""
    try:
        debugger = EnvironmentDebugger()
        debugger.run_full_diagnostic()
    except rospy.ROSInterruptException:
        print("\n👋 Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")

if __name__ == '__main__':
    main()
