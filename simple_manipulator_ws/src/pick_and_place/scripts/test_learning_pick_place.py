#!/usr/bin/env python3

"""
Learning Pick-and-Place Test
Demonstrates the robot learning to pick and place objects through RL

Author: RL Training Implementation
Date: 2024
"""

import rospy
import numpy as np
import time
import sys
import os

# Add the scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pick_place_sac_env import PickPlaceSACEnvironment
from sac_pick_place_trainer import SACAgent, ReplayBuffer


def test_learning_pick_place():
    """Test the robot learning to pick and place objects"""
    print("=" * 80)
    print("ROBOT LEARNING PICK-AND-PLACE TEST")
    print("=" * 80)
    print("This test demonstrates:")
    print("1. Robot moves toward detected objects")
    print("2. Robot learns to pick objects (close gripper when near)")
    print("3. Robot learns to place objects in correct bins")
    print("4. Reward function guides learning toward task completion")
    print("=" * 80)
    
    try:
        # Initialize environment and agent
        print("\n🔧 Initializing Learning System...")
        env = PickPlaceSACEnvironment(max_episode_steps=2000)  # Longer episodes
        agent = SACAgent(state_dim=42, action_dim=7)
        replay_buffer = ReplayBuffer(capacity=10000)  # Create separate replay buffer
        print("✅ System initialized!")
        
        # Run learning episodes
        print("\n🎓 Starting EXTENDED Learning Episodes...")
        print("Watch the robot in Gazebo - it should learn to pick and place!")
        print("Enhanced reward tracking enabled for learning analysis")
        print("-" * 70)
        
        episode_rewards = []
        episode_distances = []  # Track closest distances achieved
        episode_successes = []
        
        for episode in range(5):  # Run 5 learning episodes
            print(f"\n🎬 EPISODE {episode + 1}/5 (EXTENDED)")
            print("=" * 50)
            
            print(f"\n🔄 RESETTING EPISODE {episode + 1}...")
            print("  - Moving robot to home position")
            print("  - Respawning objects at random positions")
            print("  - Clearing previous episode data")
            
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            closest_distance = float('inf')
            
            # Reset learning progress tracking
            env.best_distance_achieved = float('inf')
            env.consecutive_improvements = 0
            
            print(f"✅ Episode {episode + 1} Reset Complete!")
            print(f"  Initial state: {env.state_machine_state}")
            print(f"  Detected objects: {len(env.detected_objects)}")
            print(f"  Target object: {env.target_object_id}")
            print(f"  Robot at home position: ✅")
            
            for step in range(200):  # Much longer episodes - 200 steps
                # Get action from SAC agent
                action = agent.select_action(obs)
                
                # Execute action
                next_obs, reward, done, info = env.step(action)
                
                # Store experience for learning
                replay_buffer.add(obs, action, reward, next_obs, done)
                
                # Update agent more frequently for better learning
                if len(replay_buffer) > 50:  # Start learning earlier
                    agent.update(replay_buffer, batch_size=32)  # More updates per step
                
                # Track closest distance to any object
                if len(env.detected_objects) > 0:
                    ee_pos = env._get_end_effector_position()
                    for obj in env.detected_objects:
                        target_pos = np.array([obj.x_world, obj.y_world, obj.height + env.workbench_height])
                        distance = np.linalg.norm(ee_pos - target_pos)
                        closest_distance = min(closest_distance, distance)
                
                # Update for next step
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                
                # Detailed logging every 25 steps
                if step % 25 == 0:
                    print(f"  Step {step:3d}: Reward={reward:7.2f}, Total={episode_reward:9.2f}, "
                          f"State={info['state_machine_state']}, Closest={closest_distance:.3f}m")
                    
                    # Show learning progress
                    if env.best_distance_achieved < float('inf'):
                        print(f"    🎯 Best Distance: {env.best_distance_achieved:.3f}m "
                              f"(Improvements: {env.consecutive_improvements})")
                
                # Check for learning milestones with new reward structure
                if reward > 50.0:
                    print(f"🎯 MASSIVE REWARD! {reward:.2f} - Robot successfully picked object!")
                elif reward > 20.0:
                    print(f"🎯 HIGH REWARD! {reward:.2f} - Robot is learning!")
                elif reward > 100.0:
                    print(f"🏆 EPIC REWARD! {reward:.2f} - Robot successfully placed object!")
                
                if info['state_machine_state'] == 2 and env.current_gripper_state == 1:
                    print("🤖 GRASPING! Robot learned to pick object!")
                if info['state_machine_state'] == 3 and env.current_gripper_state == 0:
                    print("📦 PLACING! Robot learned to place object!")
                
                # Show gripper state and distance info
                if step % 10 == 0:  # Every 10 steps
                    print(f"    Gripper: {'OPEN' if env.current_gripper_state == 0 else 'CLOSED'}, "
                          f"Closest: {closest_distance:.3f}m, "
                          f"Pick Threshold: {env.pick_distance:.3f}m")
                
                if done:
                    episode_success = info.get('task_completed', False)
                    print(f"\n✅ Episode {episode + 1} completed in {episode_length} steps!")
                    print(f"   Final Reward: {reward:.2f}")
                    print(f"   Task Success: {'✅ YES' if episode_success else '❌ NO'}")
                    break
                
                # Small delay for visualization
                time.sleep(0.1)  # Reduced delay for faster testing
            
            episode_rewards.append(episode_reward)
            episode_distances.append(closest_distance)
            episode_successes.append(episode_success if 'episode_success' in locals() else False)
            
            print(f"\n📊 Episode {episode + 1} Summary:")
            print(f"   Total Reward: {episode_reward:9.2f}")
            print(f"   Closest Distance: {closest_distance:.3f}m")
            print(f"   Best Distance Achieved: {env.best_distance_achieved:.3f}m")
            print(f"   Final State: {info['state_machine_state']}")
            print(f"   Task Progress: {info['completion']:.3f}")
            print(f"   Steps: {episode_length}")
            print(f"   Success: {'✅ YES' if episode_successes[-1] else '❌ NO'}")
        
        # Enhanced learning analysis
        print(f"\n📈 EXTENDED LEARNING ANALYSIS")
        print("=" * 50)
        
        if len(episode_rewards) >= 2:
            reward_improvement = episode_rewards[-1] - episode_rewards[0]
            distance_improvement = episode_distances[0] - episode_distances[-1]  # Lower is better
            
            print(f"Episode 1: Reward={episode_rewards[0]:8.2f}, Distance={episode_distances[0]:.3f}m")
            print(f"Episode 5: Reward={episode_rewards[-1]:8.2f}, Distance={episode_distances[-1]:.3f}m")
            print(f"Reward Improvement: {reward_improvement:8.2f}")
            print(f"Distance Improvement: {distance_improvement:+.3f}m")
            
            if reward_improvement > 0 and distance_improvement > 0:
                print("🎉 EXCELLENT LEARNING! Robot is getting closer AND earning more rewards!")
            elif reward_improvement > 0:
                print("📈 REWARD LEARNING: Robot is earning more rewards (may need distance guidance)")
            elif distance_improvement > 0:
                print("🎯 DISTANCE LEARNING: Robot is getting closer (may need reward tuning)")
            else:
                print("⚠️  Limited learning detected - may need longer training or parameter tuning")
        
        # Episode-by-episode analysis
        print(f"\n📊 Episode-by-Episode Analysis:")
        for i, (reward, success, distance) in enumerate(zip(episode_rewards, episode_successes, episode_distances)):
            print(f"   Episode {i+1}: Reward={reward:8.2f}, Success={'✅' if success else '❌'}, "
                  f"Closest={distance:.3f}m")
        
        # Check if robot achieved pick-and-place
        successful_episodes = sum(episode_successes)
        high_reward_episodes = sum(1 for reward in episode_rewards if reward > 100.0)
        close_distance_episodes = sum(1 for distance in episode_distances if distance < 0.1)
        
        print(f"\n🎯 Success Metrics:")
        print(f"   Task Success: {successful_episodes}/{len(episode_successes)} episodes")
        print(f"   High Reward (>100): {high_reward_episodes}/{len(episode_rewards)} episodes")
        print(f"   Close Distance (<0.1m): {close_distance_episodes}/{len(episode_distances)} episodes")
        
        if successful_episodes > 0:
            print("🎉 SUCCESS! Robot learned to pick and place objects!")
        elif high_reward_episodes > 0 or close_distance_episodes > 0:
            print("📈 PROGRESS! Robot is learning but needs more training")
        else:
            print("🔄 Robot needs more training to master pick-and-place")
        
        return True
        
    except Exception as e:
        print(f"❌ Learning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            env.close()
        except:
            pass


def main():
    """Main test function"""
    print("ROBOT LEARNING PICK-AND-PLACE TEST")
    print("=" * 80)
    print("Make sure Gazebo is running with the pick-and-place world!")
    print("The robot should learn to pick and place objects through RL.")
    print("=" * 80)
    
    # Initialize ROS
    try:
        rospy.init_node('test_learning_pick_place', anonymous=True)
        print("✅ ROS node initialized")
    except:
        print("⚠️  ROS node already initialized")
    
    # Run test
    print(f"\n🔍 Running Learning Test...")
    result = test_learning_pick_place()
    
    if result:
        print("\n✅ Learning test completed!")
        print("🚀 Robot is ready for full training!")
    else:
        print("\n❌ Learning test failed!")
        print("🔧 Check the issues above and try again.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
