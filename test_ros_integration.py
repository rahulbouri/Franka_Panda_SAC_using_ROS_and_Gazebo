#!/usr/bin/env python3
"""
Test ROS integration with manipulator environment
"""
import rospy
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ros_connection():
    """Test basic ROS connection"""
    print("🧪 Testing ROS connection...")
    try:
        rospy.init_node('test_manipulator_env', anonymous=True)
        print("✅ ROS node initialized")
        return True
    except Exception as e:
        print(f"❌ ROS connection failed: {e}")
        return False

def test_environment_creation():
    """Test environment creation"""
    print("🧪 Testing environment creation...")
    try:
        from env.manipulator_env_simple import ManipulatorEnvironmentSimple
        env = ManipulatorEnvironmentSimple()
        print("✅ Environment created successfully")
        return True
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False

def test_environment_reset():
    """Test environment reset"""
    print("🧪 Testing environment reset...")
    try:
        from env.manipulator_env_simple import ManipulatorEnvironmentSimple
        env = ManipulatorEnvironmentSimple()
        state = env.reset()
        print(f"✅ Environment reset successful. State shape: {state.shape}")
        return True
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        return False

def test_environment_step():
    """Test environment step"""
    print("🧪 Testing environment step...")
    try:
        from env.manipulator_env_simple import ManipulatorEnvironmentSimple
        env = ManipulatorEnvironmentSimple()
        state = env.reset()
        action = np.random.randn(6)  # Random 6-DOF action
        next_state, reward, done, info = env.step(action)
        print(f"✅ Environment step successful. Reward: {reward:.3f}, Done: {done}")
        return True
    except Exception as e:
        print(f"❌ Environment step failed: {e}")
        return False

def test_ros_topics():
    """Test ROS topics are available"""
    print("🧪 Testing ROS topics...")
    try:
        import rospy
        from sensor_msgs.msg import JointState
        from geometry_msgs.msg import PoseStamped
        
        # Wait for topics to be available
        rospy.sleep(2)
        
        # Check joint states topic
        try:
            joint_states = rospy.wait_for_message('/manipulator/joint_states', JointState, timeout=5)
            print(f"✅ Joint states topic working. {len(joint_states.position)} joints")
        except rospy.ROSException:
            print("❌ Joint states topic not available")
            return False
        
        # Check if we can get model states
        try:
            from gazebo_msgs.msg import ModelStates
            model_states = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
            print(f"✅ Gazebo model states available. {len(model_states.name)} models")
        except rospy.ROSException:
            print("❌ Gazebo model states not available")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ROS topics test failed: {e}")
        return False

def test_gazebo_services():
    """Test Gazebo services are available"""
    print("🧪 Testing Gazebo services...")
    try:
        import rospy
        from gazebo_msgs.srv import GetModelState, SetModelState
        
        # Wait for services to be available
        rospy.sleep(2)
        
        # Check get model state service
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=5)
            print("✅ Get model state service available")
        except rospy.ROSException:
            print("❌ Get model state service not available")
            return False
        
        # Check set model state service
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=5)
            print("✅ Set model state service available")
        except rospy.ROSException:
            print("❌ Set model state service not available")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Gazebo services test failed: {e}")
        return False

def main():
    """Run all ROS integration tests"""
    print("🚀 Starting ROS Integration Tests")
    print("=" * 50)
    
    tests = [
        ("ROS Connection", test_ros_connection),
        ("ROS Topics", test_ros_topics),
        ("Gazebo Services", test_gazebo_services),
        ("Environment Creation", test_environment_creation),
        ("Environment Reset", test_environment_reset),
        ("Environment Step", test_environment_step)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test passed")
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All ROS integration tests passed!")
        print("\n📋 Next steps:")
        print("1. Run: python3 collect_training_data.py")
        print("2. Run: python3 analyze_data.py")
        print("3. Proceed to Phase 3: Policy Architecture")
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure ROS is running: roscore &")
        print("2. Make sure Gazebo is running: roslaunch simple_manipulator training_env.launch &")
        print("3. Check ROS topics: rostopic list")
        print("4. Check Gazebo services: rosservice list | grep gazebo")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
