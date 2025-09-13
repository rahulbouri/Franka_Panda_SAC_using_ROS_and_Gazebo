#!/usr/bin/env python3
"""
Analyze collected training data
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

def find_latest_data_file():
    """Find the most recent data file"""
    data_files = glob.glob('data/training_episodes_*.pkl')
    if not data_files:
        print("❌ No data files found in data/ directory")
        return None
    
    # Sort by modification time and get the latest
    latest_file = max(data_files, key=os.path.getmtime)
    print(f"📁 Using data file: {latest_file}")
    return latest_file

def load_data(filename):
    """Load data from pickle file"""
    try:
        with open(filename, 'rb') as f:
            buffer = pickle.load(f)
        print(f"✅ Data loaded successfully")
        return buffer
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

def analyze_buffer_properties(buffer):
    """Analyze basic buffer properties"""
    print("\n📊 BUFFER PROPERTIES")
    print("=" * 30)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer capacity: {buffer.capacity}")
    print(f"Buffer alpha: {buffer.alpha}")
    print(f"Buffer beta: {buffer.beta}")
    
    if len(buffer) == 0:
        print("❌ Buffer is empty")
        return False
    
    return True

def analyze_sample_data(buffer, sample_size=1000):
    """Analyze sampled data from buffer"""
    print(f"\n📊 SAMPLE DATA ANALYSIS (n={sample_size})")
    print("=" * 40)
    
    # Sample data
    batch_data, indices, weights = buffer.sample(sample_size)
    
    if batch_data is None:
        print("❌ Failed to sample data from buffer")
        return False
    
    states = batch_data['states']
    actions = batch_data['actions']
    rewards = batch_data['rewards']
    next_states = batch_data['next_states']
    dones = batch_data['dones']
    
    print(f"Sample size: {len(states)}")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")
    print(f"Reward shape: {rewards.shape}")
    print(f"Next state shape: {next_states.shape}")
    print(f"Done shape: {dones.shape}")
    
    return True

def analyze_state_distribution(states):
    """Analyze state distribution"""
    print(f"\n📈 STATE DISTRIBUTION")
    print("=" * 25)
    
    print(f"State range: [{states.min():.3f}, {states.max():.3f}]")
    print(f"State mean: {states.mean():.3f}")
    print(f"State std: {states.std():.3f}")
    
    # Analyze each dimension
    print(f"\nState dimensions analysis:")
    for i in range(states.shape[1]):
        dim_data = states[:, i]
        print(f"  Dim {i:2d}: [{dim_data.min():.3f}, {dim_data.max():.3f}] "
              f"mean={dim_data.mean():.3f}, std={dim_data.std():.3f}")

def analyze_action_distribution(actions):
    """Analyze action distribution"""
    print(f"\n📈 ACTION DISTRIBUTION")
    print("=" * 25)
    
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"Action mean: {actions.mean():.3f}")
    print(f"Action std: {actions.std():.3f}")
    
    # Analyze each dimension
    print(f"\nAction dimensions analysis:")
    for i in range(actions.shape[1]):
        dim_data = actions[:, i]
        print(f"  Dim {i:2d}: [{dim_data.min():.3f}, {dim_data.max():.3f}] "
              f"mean={dim_data.mean():.3f}, std={dim_data.std():.3f}")

def analyze_reward_distribution(rewards):
    """Analyze reward distribution"""
    print(f"\n📈 REWARD DISTRIBUTION")
    print("=" * 25)
    
    print(f"Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"Reward mean: {rewards.mean():.3f}")
    print(f"Reward std: {rewards.std():.3f}")
    
    # Analyze reward components (if available)
    positive_rewards = rewards[rewards > 0]
    negative_rewards = rewards[rewards < 0]
    zero_rewards = rewards[rewards == 0]
    
    print(f"\nReward breakdown:")
    print(f"  Positive rewards: {len(positive_rewards)} ({len(positive_rewards)/len(rewards)*100:.1f}%)")
    print(f"  Negative rewards: {len(negative_rewards)} ({len(negative_rewards)/len(rewards)*100:.1f}%)")
    print(f"  Zero rewards: {len(zero_rewards)} ({len(zero_rewards)/len(rewards)*100:.1f}%)")
    
    if len(positive_rewards) > 0:
        print(f"  Positive reward mean: {positive_rewards.mean():.3f}")
    if len(negative_rewards) > 0:
        print(f"  Negative reward mean: {negative_rewards.mean():.3f}")

def analyze_episode_termination(dones):
    """Analyze episode termination patterns"""
    print(f"\n📈 EPISODE TERMINATION")
    print("=" * 25)
    
    total_episodes = len(dones)
    terminated_episodes = dones.sum()
    termination_rate = (terminated_episodes / total_episodes) * 100
    
    print(f"Total episodes: {total_episodes}")
    print(f"Terminated episodes: {terminated_episodes}")
    print(f"Termination rate: {termination_rate:.1f}%")
    
    # Find episode boundaries
    dones_array = np.array(dones) if hasattr(dones, 'astype') else dones
    episode_starts = np.where(np.diff(np.concatenate([[0], dones_array.astype(int)])) == 1)[0]
    episode_lengths = np.diff(np.concatenate([[0], episode_starts, [len(dones_array)]]))
    
    if len(episode_lengths) > 0:
        print(f"\nEpisode length statistics:")
        print(f"  Min length: {episode_lengths.min()}")
        print(f"  Max length: {episode_lengths.max()}")
        print(f"  Mean length: {episode_lengths.mean():.1f}")
        print(f"  Std length: {episode_lengths.std():.1f}")

def create_visualizations(states, actions, rewards, dones):
    """Create visualization plots"""
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("=" * 30)
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Data Analysis', fontsize=16)
        
        # Plot 1: State distribution
        axes[0, 0].hist(states.flatten(), bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('State Distribution')
        axes[0, 0].set_xlabel('State Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: Action distribution
        axes[0, 1].hist(actions.flatten(), bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('Action Distribution')
        axes[0, 1].set_xlabel('Action Value')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Reward distribution
        axes[1, 0].hist(rewards, bins=50, alpha=0.7, color='red')
        axes[1, 0].set_title('Reward Distribution')
        axes[1, 0].set_xlabel('Reward Value')
        axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Episode termination
        axes[1, 1].bar(['Not Done', 'Done'], [len(dones) - dones.sum(), dones.sum()], 
                       color=['lightblue', 'lightcoral'])
        axes[1, 1].set_title('Episode Termination')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'data/data_analysis_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {plot_filename}")
        
        # Show plot (if in interactive environment)
        try:
            plt.show()
        except:
            print("📊 Plot created but not displayed (non-interactive environment)")
        
    except Exception as e:
        print(f"❌ Failed to create visualizations: {e}")

def main():
    """Main analysis function"""
    print("🔍 Starting Training Data Analysis")
    print("=" * 50)
    
    # Find latest data file
    data_file = find_latest_data_file()
    if data_file is None:
        return 1
    
    # Load data
    buffer = load_data(data_file)
    if buffer is None:
        return 1
    
    # Analyze buffer properties
    if not analyze_buffer_properties(buffer):
        return 1
    
    # Sample data for analysis
    sample_size = min(1000, len(buffer))
    batch_data, indices, weights = buffer.sample(sample_size)
    
    if batch_data is None:
        print("❌ Failed to sample data from buffer")
        return 1
    
    states = batch_data['states']
    actions = batch_data['actions']
    rewards = batch_data['rewards']
    dones = batch_data['dones']
    
    # Perform analysis
    analyze_state_distribution(states)
    analyze_action_distribution(actions)
    analyze_reward_distribution(rewards)
    analyze_episode_termination(dones)
    
    # Create visualizations
    create_visualizations(states, actions, rewards, dones)
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"✅ Data file: {data_file}")
    print(f"✅ Buffer size: {len(buffer)}")
    print(f"✅ Sample analyzed: {len(states)}")
    print(f"✅ State dimensions: {states.shape[1]}")
    print(f"✅ Action dimensions: {actions.shape[1]}")
    print(f"✅ Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"✅ Termination rate: {(dones.sum() / len(dones)) * 100:.1f}%")
    
    print("\n🎉 Data analysis completed successfully!")
    print("\n📋 Next steps:")
    print("1. Review the analysis results")
    print("2. Check the visualization plots")
    print("3. Proceed to Phase 3: Policy Architecture")
    print("4. Start implementing SAC + LNN")
    
    return 0

if __name__ == "__main__":
    exit(main())
