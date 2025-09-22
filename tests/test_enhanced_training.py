#!/usr/bin/env python3

import os
import sys
import rospy
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Add the scripts directory to Python path
sys.path.append('/home/bouri/roboset/simple_manipulator_ws/src/pick_and_place/scripts')

from pick_place_sac_env import PickPlaceSACEnvironment
from sac_pick_place_trainer import SACAgent, ReplayBuffer

class TrainingVisualizer:
    def __init__(self):
        self.episode_rewards = []
        self.episode_distances = []
        self.state_transitions = []
        self.gripper_states = []
        self.reward_components = {'task': [], 'efficiency': [], 'total': []}
        
    def update(self, reward, distance, state, gripper_state, reward_breakdown=None):
        self.episode_rewards.append(reward)
        self.episode_distances.append(distance)
        self.state_transitions.append(state)
        self.gripper_states.append(gripper_state)
        
        if reward_breakdown:
            self.reward_components['task'].append(reward_breakdown.get('task', 0))
            self.reward_components['efficiency'].append(reward_breakdown.get('efficiency', 0))
            self.reward_components['total'].append(reward_breakdown.get('total', 0))
    
    def plot_episode(self, episode_num):
        """Plot real-time training progress"""
        if len(self.episode_rewards) < 10:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Training Progress - Episode {episode_num}')
        
        # Plot 1: Episode rewards
        axes[0, 0].plot(self.episode_rewards[-100:])  # Last 100 steps
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Plot 2: Distance to target
        axes[0, 1].plot(self.episode_distances[-100:])
        axes[0, 1].set_title('Distance to Target')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Distance (m)')
        axes[0, 1].grid(True)
        
        # Plot 3: State machine progress
        axes[1, 0].plot(self.state_transitions[-100:])
        axes[1, 0].set_title('State Machine Progress')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('State (0=Home, 1=Select, 2=Pick, 3=Place)')
        axes[1, 0].set_ylim(-0.5, 3.5)
        axes[1, 0].grid(True)
        
        # Plot 4: Gripper state
        axes[1, 1].plot(self.gripper_states[-100:])
        axes[1, 1].set_title('Gripper State')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('State (0=Open, 1=Closed)')
        axes[1, 1].set_ylim(-0.5, 1.5)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'/home/bouri/roboset/tests/training_episode_{episode_num}.png')
        plt.close()

def test_enhanced_training():
    """Enhanced training test with comprehensive logging and visualization"""
    try:
        rospy.init_node('enhanced_training_test', anonymous=True)
    except rospy.exceptions.ROSException:
        pass

    print("\n🚀 ENHANCED SAC TRAINING TEST")
    print("=" * 50)
    
    # Initialize environment and agent
    env = PickPlaceSACEnvironment(max_episode_steps=500)
    agent = SACAgent(state_dim=42, action_dim=7)
    replay_buffer = ReplayBuffer(capacity=50000)
    visualizer = TrainingVisualizer()
    
    # Training parameters
    num_episodes = 10
    max_steps_per_episode = 200
    update_frequency = 10  # Update agent every 10 steps
    
    # Training statistics
    episode_rewards = []
    success_count = 0
    best_episode_reward = float('-inf')
    
    print(f"Starting training: {num_episodes} episodes, {max_steps_per_episode} steps each")
    print("Reward components will be logged for analysis")
    print("Visualization plots will be saved to /home/bouri/roboset/tests/")
    
    for episode in range(num_episodes):
        print(f"\n🔄 EPISODE {episode + 1}/{num_episodes}")
        print("-" * 30)
        
        # Reset environment
        obs = env.reset()
        episode_reward = 0.0
        episode_distance = float('inf')
        steps_in_state = {0: 0, 1: 0, 2: 0, 3: 0}  # Track time in each state
        
        # Wait for perception to initialize
        rospy.loginfo("Waiting for object detections...")
        start_wait = time.time()
        while len(env.detected_objects) == 0 and (time.time() - start_wait) < 5.0:
            rospy.sleep(0.1)
        
        print(f"Detected objects: {len(env.detected_objects)}")
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(obs)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            
            # Calculate distance to target for visualization
            if len(env.detected_objects) > 0 and env.target_object_id >= 0:
                target_obj = env.detected_objects[env.target_object_id]
                ee_pos = env._get_end_effector_position()
                target_pos = np.array([
                    target_obj.x_world, 
                    target_obj.y_world, 
                    target_obj.height + env.workbench_height + env.z_offset
                ])
                episode_distance = min(episode_distance, np.linalg.norm(ee_pos - target_pos))
            
            # Store experience
            replay_buffer.add(obs, action, reward, next_obs, done)
            
            # Update agent
            if len(replay_buffer) > 50 and step % update_frequency == 0:
                agent.update(replay_buffer, batch_size=32)
            
            # Update visualizer
            visualizer.update(
                reward, 
                episode_distance, 
                info['state_machine_state'], 
                env.current_gripper_state
            )
            
            # Track state transitions
            steps_in_state[info['state_machine_state']] += 1
            
            episode_reward += reward
            obs = next_obs
            
            # Log progress every 25 steps
            if step % 25 == 0:
                print(f"  Step {step:3d}: Reward={reward:6.2f}, State={info['state_machine_state']}, "
                      f"Gripper={env.current_gripper_state}, Distance={episode_distance:.3f}m")
            
            if done:
                break
        
        # Episode summary
        episode_rewards.append(episode_reward)
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
        
        # Check for success (reached placing state)
        if info['state_machine_state'] == 3:
            success_count += 1
            print(f"  ✅ SUCCESS: Reached placing state!")
        
        print(f"  📊 Episode Summary:")
        print(f"     Total Reward: {episode_reward:.2f}")
        print(f"     Best Distance: {episode_distance:.3f}m")
        print(f"     Final State: {info['state_machine_state']}")
        print(f"     Steps per state: {steps_in_state}")
        print(f"     Success: {'YES' if info['state_machine_state'] == 3 else 'NO'}")
        
        # Generate visualization
        visualizer.plot_episode(episode + 1)
        
        # Reset visualizer for next episode
        visualizer.episode_rewards.clear()
        visualizer.episode_distances.clear()
        visualizer.state_transitions.clear()
        visualizer.gripper_states.clear()
    
    # Final training summary
    print(f"\n🎯 TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Best Episode Reward: {best_episode_reward:.2f}")
    print(f"Final Episode Rewards: {episode_rewards[-3:]}")
    print(f"Visualization plots saved to: /home/bouri/roboset/tests/")
    
    # Plot overall training progress
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Training Progress - Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('/home/bouri/roboset/tests/training_progress.png')
    plt.close()
    
    print("✅ Enhanced training test completed!")

if __name__ == '__main__':
    test_enhanced_training()
