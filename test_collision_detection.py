#!/usr/bin/env python3
"""
Test Collision Detection and Safety Violations
Quick test to verify the enhanced environment works properly

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
    """Test collision detection and safety violations"""
    print("🧪 Testing Collision Detection and Safety Violations")
    print("=" * 60)
    
    # Initialize ROS node
    print("🔧 Initializing ROS node...")
    try:
        rospy.init_node('test_collision_detection', anonymous=True, disable_signals=True)
        print("✅ ROS node initialized successfully")
    except rospy.ROSException as e:
        print(f"❌ Failed to initialize ROS node: {e}")
        return
    
    # Initialize enhanced environment
    print("\n🔧 Initializing Enhanced Gazebo environment...")
    from env.gazebo_env_enhanced import GazeboEnvEnhanced
    
    config = {'max_episode_steps': 100}
    env = GazeboEnvEnhanced(config)
    print("✅ Enhanced environment initialized")
    
    print("\n🎓 Starting collision detection test...")
    print("🛡️ Testing features:")
    print("   - Coke can position monitoring")
    print("   - Collision detection")
    print("   - Episode termination on violations")
    print("🎮 Watch the Gazebo window!")
    print("=" * 60)
    
    try:
        # Test multiple episodes
        for episode in range(3):
            print(f"\n🎬 Test Episode {episode + 1}/3")
            print("-" * 40)
            
            # Reset environment
            print("🔄 Resetting environment...")
            state = env.reset()
            print(f"✅ Reset complete - Target: {env.target_position}")
            print(f"🎯 Coke can: {env.coke_can_position}")
            print(f"🤖 Initial joints: {env.joint_positions[:3]}...")
            
            # Run episode with random actions
            for step in range(50):
                print(f"\n  Step {step + 1}/50:")
                
                # Random action (more aggressive to test collision)
                action = np.random.uniform(-0.8, 0.8, 6)
                
                print(f"    Action: {action[:3]}...")
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Log progress
                print(f"    Reward: {reward:.2f}")
                print(f"    Distance to target: {info['distance']:.4f}")
                print(f"    Distance to coke can: {np.linalg.norm(env.end_effector_position - env.coke_can_position):.4f}")
                print(f"    End-effector: {env.end_effector_position}")
                print(f"    Coke can: {env.coke_can_position}")
                print(f"    Contact points: {info['contact_points']}")
                
                # Check for violations
                if info['collision_detected']:
                    print(f"    ⚠️ COLLISION DETECTED!")
                
                if info['coke_can_tipped']:
                    print(f"    ⚠️ COKE CAN TIPPED/FELL!")
                
                if info['target_reached']:
                    print(f"    ✅ TARGET REACHED!")
                
                if done:
                    print(f"    🏁 Episode done at step {step + 1}")
                    print(f"    Reason: {'Target reached' if info['target_reached'] else 'Collision' if info['collision_detected'] else 'Coke tipped' if info['coke_can_tipped'] else 'Max steps'}")
                    break
                
                # Small delay
                time.sleep(0.2)
            
            print(f"\n📊 Episode {episode + 1} Summary:")
            print(f"    Final Distance: {info['distance']:.4f}")
            print(f"    Collision: {info['collision_detected']}")
            print(f"    Coke Tipped: {info['coke_can_tipped']}")
            print(f"    Target Reached: {info['target_reached']}")
            print(f"    Contact Points: {info['contact_points']}")
        
        print("\n🎉 Collision Detection Test completed!")
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
