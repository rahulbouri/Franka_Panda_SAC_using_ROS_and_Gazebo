#!/usr/bin/env python3
"""
Collect training data from Gazebo simulation
"""
import rospy
import numpy as np
import pickle
import os
import sys
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def collect_episodes(num_episodes=100, max_steps=50):
    """Collect random episodes for training"""
    print(f"🚀 Collecting {num_episodes} episodes...")
    print("=" * 50)
    
    try:
        from env.manipulator_env_simple import ManipulatorEnvironmentSimple
        from models.replay_buffer import PrioritizedReplayBuffer
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure you're in the correct directory and virtual environment is activated")
        return None
    
    # Initialize environment
    print("🔧 Initializing environment...")
    env = ManipulatorEnvironmentSimple()
    replay_buffer = PrioritizedReplayBuffer(capacity=10000)
    
    episodes_collected = 0
    total_steps = 0
    total_reward = 0
    success_count = 0
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    print(f"📊 Starting data collection...")
    print(f"   Episodes: {num_episodes}")
    print(f"   Max steps per episode: {max_steps}")
    print(f"   Buffer capacity: {replay_buffer.capacity}")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        print(f"\n🎬 Episode {episode + 1}/{num_episodes}")
        
        try:
            # Reset environment with random target
            state = env.reset()
            episode_steps = 0
            episode_reward = 0
            episode_success = False
            
            for step in range(max_steps):
                # Random action (normalized to [-1, 1])
                action = np.random.uniform(-1, 1, 6)
                
                # Step environment
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                replay_buffer.add(state, action, reward, next_state, done)
                
                # Update counters
                state = next_state
                episode_steps += 1
                episode_reward += reward
                
                # Check for success
                if info.get('success', False):
                    episode_success = True
                
                if done:
                    break
            
            # Update statistics
            episodes_collected += 1
            total_steps += episode_steps
            total_reward += episode_reward
            if episode_success:
                success_count += 1
            
            # Print episode summary
            success_rate = (success_count / episodes_collected) * 100
            avg_reward = total_reward / episodes_collected
            avg_steps = total_steps / episodes_collected
            
            print(f"   Steps: {episode_steps}/{max_steps}")
            print(f"   Reward: {episode_reward:.3f}")
            print(f"   Success: {'✅' if episode_success else '❌'}")
            print(f"   Overall Success Rate: {success_rate:.1f}%")
            print(f"   Avg Reward: {avg_reward:.3f}")
            print(f"   Avg Steps: {avg_steps:.1f}")
            
            # Save intermediate data every 25 episodes
            if (episode + 1) % 25 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'data/training_episodes_{episode + 1}_{timestamp}.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(replay_buffer, f)
                print(f"   💾 Intermediate data saved to {filename}")
        
        except Exception as e:
            print(f"   ❌ Episode {episode + 1} failed: {e}")
            continue
    
    # Calculate final statistics
    elapsed_time = time.time() - start_time
    final_success_rate = (success_count / episodes_collected) * 100 if episodes_collected > 0 else 0
    final_avg_reward = total_reward / episodes_collected if episodes_collected > 0 else 0
    final_avg_steps = total_steps / episodes_collected if episodes_collected > 0 else 0
    
    # Save final data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_filename = f'data/training_episodes_final_{timestamp}.pkl'
    with open(final_filename, 'wb') as f:
        pickle.dump(replay_buffer, f)
    
    # Print final summary
    print("\n" + "=" * 50)
    print("📊 DATA COLLECTION SUMMARY")
    print("=" * 50)
    print(f"✅ Episodes collected: {episodes_collected}/{num_episodes}")
    print(f"✅ Total steps: {total_steps}")
    print(f"✅ Success rate: {final_success_rate:.1f}%")
    print(f"✅ Average reward: {final_avg_reward:.3f}")
    print(f"✅ Average steps: {final_avg_steps:.1f}")
    print(f"✅ Collection time: {elapsed_time:.1f} seconds")
    print(f"✅ Data saved to: {final_filename}")
    print(f"✅ Buffer size: {len(replay_buffer)}")
    
    return replay_buffer

def analyze_collected_data(buffer):
    """Analyze the collected data"""
    print("\n🔍 ANALYZING COLLECTED DATA")
    print("=" * 30)
    
    if buffer is None or len(buffer) == 0:
        print("❌ No data to analyze")
        return
    
    # Sample some data for analysis
    batch_size = min(1000, len(buffer))
    batch_data, indices, weights = buffer.sample(batch_size)
    
    if batch_data is None:
        print("❌ Failed to sample data from buffer")
        return
    
    states = batch_data['states']
    actions = batch_data['actions']
    rewards = batch_data['rewards']
    dones = batch_data['dones']
    
    print(f"📊 Sample size: {len(states)}")
    print(f"📊 State shape: {states.shape}")
    print(f"📊 Action shape: {actions.shape}")
    print(f"📊 Reward shape: {rewards.shape}")
    
    print(f"\n📈 State statistics:")
    print(f"   Range: [{states.min():.3f}, {states.max():.3f}]")
    print(f"   Mean: {states.mean():.3f}")
    print(f"   Std: {states.std():.3f}")
    
    print(f"\n📈 Action statistics:")
    print(f"   Range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"   Mean: {actions.mean():.3f}")
    print(f"   Std: {actions.std():.3f}")
    
    print(f"\n📈 Reward statistics:")
    print(f"   Range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"   Mean: {rewards.mean():.3f}")
    print(f"   Std: {rewards.std():.3f}")
    
    print(f"\n📈 Episode termination:")
    print(f"   Done episodes: {dones.sum()}")
    print(f"   Done rate: {(dones.sum() / len(dones)) * 100:.1f}%")

def main():
    """Main data collection function"""
    print("🚀 Starting Training Data Collection")
    print("=" * 50)
    
    # Check if ROS is running
    try:
        import rospy
        if rospy.get_published_topics() == []:
            print("❌ ROS is not running. Please start ROS first:")
            print("   roscore &")
            print("   roslaunch simple_manipulator training_env.launch &")
            return 1
    except Exception as e:
        print(f"❌ ROS connection failed: {e}")
        print("Please start ROS first:")
        print("   roscore &")
        print("   roslaunch simple_manipulator training_env.launch &")
        return 1
    
    # Collect episodes
    buffer = collect_episodes(num_episodes=100, max_steps=50)
    
    if buffer is not None:
        # Analyze data
        analyze_collected_data(buffer)
        
        print("\n🎉 Data collection completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run: python3 analyze_data.py")
        print("2. Proceed to Phase 3: Policy Architecture")
        print("3. Start implementing SAC + LNN")
    else:
        print("❌ Data collection failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
