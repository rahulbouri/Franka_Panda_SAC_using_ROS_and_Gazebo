#!/usr/bin/env python3
"""
Start RL Training for 6-DOF Manipulator
This script launches the environment and starts training
"""

import subprocess
import time
import os
import sys

def launch_environment():
    """Launch the Gazebo environment"""
    print("🚀 Launching Gazebo Environment...")
    
    # Launch the environment in background
    process = subprocess.Popen(
        ['./launch_fixed_environment.sh'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for environment to be ready
    print("⏳ Waiting for environment to initialize...")
    time.sleep(30)
    
    return process

def start_training():
    """Start the RL training"""
    print("🎯 Starting RL Training...")
    
    # Run the training framework
    subprocess.run([sys.executable, 'rl_training_framework.py'])

def main():
    """Main function"""
    print("🤖 RL Training Launcher for 6-DOF Manipulator")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('launch_fixed_environment.sh'):
        print("❌ Error: launch_fixed_environment.sh not found!")
        print("Please run this script from the /home/bouri/roboset directory")
        return
    
    # Launch environment
    env_process = launch_environment()
    
    try:
        # Start training
        start_training()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Error during training: {e}")
    finally:
        # Clean up
        print("🧹 Cleaning up...")
        env_process.terminate()
        env_process.wait()

if __name__ == '__main__':
    main()
