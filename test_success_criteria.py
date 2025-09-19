#!/usr/bin/env python3
"""
Test Success Criteria and Distance Measurements
Verify that success is properly defined and measured

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
    """Test success criteria and distance measurements"""
    print("🧪 Testing Success Criteria and Distance Measurements")
    print("=" * 60)
    
    # Initialize ROS node
    print("🔧 Initializing ROS node...")
    try:
        rospy.init_node('test_success_criteria', anonymous=True, disable_signals=True)
        print("✅ ROS node initialized successfully")
    except rospy.ROSException as e:
        print(f"❌ Failed to initialize ROS node: {e}")
        return
    
    # Initialize enhanced environment
    print("\n🔧 Initializing Enhanced Gazebo environment...")
    from env.gazebo_env_enhanced import GazeboEnvEnhanced
    
    config = {'max_episode_steps': 50}
    env = GazeboEnvEnhanced(config)
    print("✅ Enhanced environment initialized")
    
    print("\n🎓 Starting success criteria test...")
    print("🎯 Testing success definition: Reach coke can within 5cm without collision")
    print("🎮 Watch the Gazebo window and console logs!")
    print("=" * 60)
    
    try:
        # Test episode with careful actions
        print("\n🎬 Test Episode - Careful Actions to Test Success")
        print("-" * 50)
        
        # Reset environment
        print("🔄 Resetting environment...")
        state = env.reset()
        print(f"✅ Reset complete - Target: {env.target_position}")
        print(f"🎯 Coke can: {env.coke_can_position}")
        print(f"🤖 Initial joints: {env.joint_positions[:3]}...")
        
        episode_reward = 0
        step_count = 0
        success_achieved = False
        
        # Run episode with careful actions
        for step in range(50):
            print(f"\n  Step {step + 1}/50:")
            
            # Careful random action to avoid collision
            action = np.random.uniform(-0.2, 0.2, 6)  # Small, careful actions
            
            print(f"    Action: {action[:3]}...")
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Log detailed distance information
            distance_to_coke = info['distance']
            distance_to_target = info['distance_to_target']
            
            print(f"    Reward: {reward:.2f}")
            print(f"    Distance to coke can: {distance_to_coke:.4f}m")
            print(f"    Distance to target: {distance_to_target:.4f}m")
            print(f"    End-effector: {env.end_effector_position}")
            print(f"    Coke can: {env.coke_can_position}")
            
            # Check success criteria
            if distance_to_coke < 0.05:  # 5cm tolerance
                if not env.collision_detected:
                    success_achieved = True
                    print(f"    ✅ SUCCESS! Reached coke can within 5cm without collision!")
                else:
                    print(f"    ❌ FAILED! Reached coke can but collision detected!")
            
            # Check for violations
            if info['collision_detected']:
                print(f"    ⚠️ COLLISION DETECTED!")
            
            if info['coke_can_tipped']:
                print(f"    ⚠️ COKE CAN TIPPED!")
            
            if info['target_reached']:
                print(f"    ✅ TARGET REACHED!")
            
            if done:
                print(f"    🏁 Episode done at step {step + 1}")
                print(f"    Reason: {'Target reached' if info['target_reached'] else 'Collision' if info['collision_detected'] else 'Coke tipped' if info['coke_can_tipped'] else 'Max steps'}")
                break
            
            # Small delay
            time.sleep(0.2)
        
        print(f"\n📊 Episode Summary:")
        print(f"    Total Steps: {step_count}")
        print(f"    Final Episode Reward: {episode_reward:.2f}")
        print(f"    Success Achieved: {success_achieved}")
        print(f"    Collision: {info['collision_detected']}")
        print(f"    Coke Tipped: {info['coke_can_tipped']}")
        print(f"    Target Reached: {info['target_reached']}")
        print(f"    Final Distance to Coke: {info['distance']:.4f}m")
        print(f"    Final Distance to Target: {info['distance_to_target']:.4f}m")
        
        # Analyze success
        if success_achieved:
            print("✅ SUCCESS: Manipulator reached coke can without collision!")
        elif info['collision_detected']:
            print("❌ FAILED: Collision detected before reaching coke can")
        elif info['coke_can_tipped']:
            print("❌ FAILED: Coke can tipped before reaching it")
        else:
            print("⚠️ PARTIAL: Did not reach coke can within tolerance")
        
        print("\n🎉 Success Criteria Test completed!")
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
