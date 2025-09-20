#!/usr/bin/env python3
"""
Test Reward System with Safety Violations
Verify that proper penalties are applied for collisions and violations

Author: RL Training Implementation
Date: 2024
"""

import os
import sys
import numpy as np
import rospy
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test reward system with safety violations"""
    print("🧪 Testing Reward System with Safety Violations")
    print("=" * 60)
    
    # Initialize ROS node
    print("🔧 Initializing ROS node...")
    try:
        rospy.init_node('test_reward_system', anonymous=True, disable_signals=True)
        print("✅ ROS node initialized successfully")
    except rospy.ROSException as e:
        print(f"❌ Failed to initialize ROS node: {e}")
        return
    
    # Initialize enhanced environment
    print("\n🔧 Initializing Enhanced Gazebo environment...")
    from env.gazebo_env_enhanced import GazeboEnvEnhanced
    
    config = {'max_episode_steps': 20}
    env = GazeboEnvEnhanced(config)
    print("✅ Enhanced environment initialized")
    
    print("\n🎓 Starting reward system test...")
    print("🛡️ Testing reward penalties for safety violations")
    print("🎮 Watch the Gazebo window and console logs!")
    print("=" * 60)
    
    try:
        # Test episode with aggressive actions to trigger violations
        print("\n🎬 Test Episode - Aggressive Actions to Trigger Violations")
        print("-" * 50)
        
        # Reset environment
        print("🔄 Resetting environment...")
        state = env.reset()
        print(f"✅ Reset complete - Target: {env.target_position}")
        print(f"🎯 Coke can: {env.coke_can_position}")
        print(f"🤖 Initial joints: {env.joint_positions[:3]}...")
        
        episode_reward = 0
        step_count = 0
        
        # Run episode with very aggressive actions to force collision
        for step in range(20):
            print(f"\n  Step {step + 1}/20:")
            
            # Very aggressive random action to force collision
            action = np.random.uniform(-1.5, 1.5, 6)  # More aggressive than normal
            
            print(f"    Action: {action[:3]}...")
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Log detailed information
            print(f"    Reward: {reward:.2f}")
            print(f"    Episode Reward: {episode_reward:.2f}")
            print(f"    Distance to target: {info['distance']:.4f}")
            print(f"    Distance to coke can: {np.linalg.norm(env.end_effector_position - env.coke_can_position):.4f}")
            print(f"    End-effector: {env.end_effector_position}")
            print(f"    Coke can: {env.coke_can_position}")
            print(f"    Contact points: {info['contact_points']}")
            
            # Check for violations
            if info['collision_detected']:
                print(f"    ⚠️ COLLISION DETECTED! Reward should be heavily negative!")
            
            if info['coke_can_tipped']:
                print(f"    ⚠️ COKE CAN TIPPED! Reward should be heavily negative!")
            
            if info['target_reached']:
                print(f"    ✅ TARGET REACHED!")
            
            if done:
                print(f"    🏁 Episode done at step {step + 1}")
                print(f"    Reason: {'Target reached' if info['target_reached'] else 'Collision' if info['collision_detected'] else 'Coke tipped' if info['coke_can_tipped'] else 'Max steps'}")
                break
            
            # Small delay
            time.sleep(0.3)
        
        print(f"\n📊 Episode Summary:")
        print(f"    Total Steps: {step_count}")
        print(f"    Final Episode Reward: {episode_reward:.2f}")
        print(f"    Collision: {info['collision_detected']}")
        print(f"    Coke Tipped: {info['coke_can_tipped']}")
        print(f"    Target Reached: {info['target_reached']}")
        print(f"    Contact Points: {info['contact_points']}")
        
        # Analyze reward
        if episode_reward < -500:
            print("✅ GOOD: Heavy negative reward applied for safety violations!")
        elif episode_reward < 0:
            print("⚠️ PARTIAL: Some penalty applied, but may need adjustment")
        else:
            print("❌ BAD: No significant penalty applied for violations!")
        
        print("\n🎉 Reward System Test completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n🔒 Environment closed")

if __name__ == "__main__":
    main()

