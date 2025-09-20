#!/usr/bin/env python3
"""
Enhanced SAC Training with Proper Collision Detection and Reward System
Based on gym-gazebo patterns and PINN_RL reward system

Author: RL Training Implementation
Date: 2024
"""

import os
import sys
import numpy as np
import torch
import logging
import time
import rospy
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Enhanced SAC training with proper collision detection and reward system"""
    print("🛡️ Starting Enhanced SAC Training with Proper Collision Detection")
    print("=" * 70)
    
    # Initialize ROS node
    print("🔧 Initializing ROS node...")
    try:
        rospy.init_node('enhanced_sac_training', anonymous=True, disable_signals=True)
        print("✅ ROS node initialized successfully")
    except rospy.ROSException as e:
        print(f"❌ Failed to initialize ROS node: {e}")
        return
    
    # Initialize enhanced environment
    print("\n🔧 Initializing Enhanced Gazebo environment...")
    from env.gazebo_env_enhanced import GazeboEnvEnhanced
    
    config = {'max_episode_steps': 1000}
    env = GazeboEnvEnhanced(config)
    print("✅ Enhanced environment initialized")
    
    # Initialize SAC policy
    print("\n🧠 Initializing SAC policy...")
    from models.sac_policy import SACPolicy, ReplayBuffer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Enhanced state size (6 + 6 + 3 + 3 + 3 + 1 + 1 + 1 = 24)
    policy = SACPolicy(
        obs_size=24,  # Enhanced state with safety information
        action_size=6,  # 6 joint torques
        device=device,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        action_scale=50.0  # Scaled for torque control
    )
    
    replay_buffer = ReplayBuffer(capacity=50000, device=device)
    print("✅ SAC policy initialized")
    
    # Training metrics
    episode_rewards = []
    success_rates = []
    collision_rates = []
    energy_efficiency = []
    safety_violations = []
    episode_times = []
    
    print("\n🎓 Starting Enhanced SAC training...")
    print("🛡️ Enhanced features:")
    print("   - Proper collision detection using Gazebo contact sensors")
    print("   - 0.2m success tolerance for target reaching")
    print("   - Immediate episode termination on collision")
    print("   - PINN_RL-based reward system")
    print("   - Energy efficiency optimization")
    print("🎮 Watch the Gazebo window to see safe manipulator learning!")
    print("=" * 70)
    
    try:
        # Training loop for 5 episodes
        for episode in range(5):
            print(f"\n🎬 Episode {episode + 1}/5")
            print("-" * 50)
            
            episode_start_time = time.time()
            
            # Reset environment
            print("🔄 Resetting environment...")
            state = env.reset()
            print(f"✅ Reset complete - Target: {env.target_position}")
            print(f"🎯 Coke can: {env.coke_can_position}")
            print(f"🤖 Initial joints: {env.joint_positions[:3]}...")
            
            episode_reward = 0
            episode_length = 0
            success = False
            collision = False
            coke_tipped = False
            episode_energy = 0
            
            # Run episode
            for step in range(1000):
                print(f"\n  Step {step + 1}/1000:")
                
                # Select action
                if episode < 2:
                    # Exploration phase - random actions with safety constraints
                    action = np.random.uniform(-0.3, 0.3, 6)  # Reduced range for safety
                    action_type = "RANDOM"
                else:
                    # Use SAC policy
                    action = policy.select_action(state, deterministic=False)
                    action_type = "SAC"
                
                print(f"    Action ({action_type}): {action[:3]}...")
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                action_tensor = torch.tensor(action, dtype=torch.float32, device=device)
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
                
                replay_buffer.push(state_tensor, action_tensor, reward_tensor, next_state_tensor)
                
                episode_reward += reward
                episode_length += 1
                episode_energy += info['total_torque']
                state = next_state
                
                # Log progress
                print(f"    Reward: {reward:.2f}")
                print(f"    Distance: {info['distance']:.4f}")
                print(f"    End-effector: {info['end_effector_position']}")
                print(f"    Coke can: {info['coke_can_position']}")
                print(f"    Contact points: {info['contact_points']}")
                
                # Check for safety violations
                if info['collision_detected']:
                    collision = True
                    print(f"    ⚠️ COLLISION DETECTED! Episode will terminate!")
                
                if info['coke_can_tipped']:
                    coke_tipped = True
                    print(f"    ⚠️ COKE CAN TIPPED! Episode will terminate!")
                
                if info['target_reached']:
                    success = True
                    print(f"    ✅ TARGET REACHED! Distance: {info['distance']:.3f}m")
                
                if done:
                    print(f"    🏁 Episode done at step {step + 1}")
                    break
                
                # Small delay to see movement
                time.sleep(0.1)
            
            # Episode summary
            episode_time = time.time() - episode_start_time
            episode_rewards.append(episode_reward)
            success_rates.append(1.0 if success else 0.0)
            collision_rates.append(1.0 if collision else 0.0)
            energy_efficiency.append(episode_energy / max(episode_length, 1))
            safety_violations.append(info['contact_points'])
            episode_times.append(episode_time)
            
            print(f"\n📊 Episode {episode + 1} Summary:")
            print(f"    Total Reward: {episode_reward:.2f}")
            print(f"    Steps: {episode_length}")
            print(f"    Time: {episode_time:.2f}s")
            print(f"    Success: {success}")
            print(f"    Collision: {collision}")
            print(f"    Coke Tipped: {coke_tipped}")
            print(f"    Final Distance: {info['distance']:.4f}")
            print(f"    Energy Efficiency: {episode_energy:.2f}")
            print(f"    Contact Points: {info['contact_points']}")
            print(f"    Buffer Size: {len(replay_buffer)}")
            
            # Update policy if we have enough experiences
            if len(replay_buffer) >= 64:
                print(f"\n🔄 Updating SAC policy with {len(replay_buffer)} experiences...")
                for update_step in range(10):  # More updates for better learning
                    policy.update(replay_buffer, 64)
                    print(f"    Policy update {update_step + 1}/10 completed")
            
            # Print progress
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards)
                current_success_rate = np.mean(success_rates)
                current_collision_rate = np.mean(collision_rates)
                avg_energy = np.mean(energy_efficiency)
                avg_violations = np.mean(safety_violations)
                avg_time = np.mean(episode_times)
                
                print(f"\n📈 Recent Performance:")
                print(f"    Average Reward: {avg_reward:.4f}")
                print(f"    Success Rate: {current_success_rate:.2%}")
                print(f"    Collision Rate: {current_collision_rate:.2%}")
                print(f"    Energy Efficiency: {avg_energy:.2f}")
                print(f"    Avg Contact Points: {avg_violations:.1f}")
                print(f"    Avg Episode Time: {avg_time:.2f}s")
        
        print("\n🎉 Enhanced SAC Training completed successfully!")
        print("=" * 70)
        print(f"📊 Final Statistics:")
        print(f"    Total Episodes: 5")
        print(f"    Average Reward: {np.mean(episode_rewards):.4f}")
        print(f"    Final Success Rate: {np.mean(success_rates):.2%}")
        print(f"    Final Collision Rate: {np.mean(collision_rates):.2%}")
        print(f"    Average Energy Efficiency: {np.mean(energy_efficiency):.2f}")
        print(f"    Average Contact Points: {np.mean(safety_violations):.1f}")
        print(f"    Average Episode Time: {np.mean(episode_times):.2f}s")
        print(f"    Replay Buffer Size: {len(replay_buffer)}")
        print("\n🎮 Check the Gazebo window for final manipulator position!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n🔒 Environment closed")

if __name__ == "__main__":
    main()

