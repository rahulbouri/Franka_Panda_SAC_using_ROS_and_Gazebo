#!/bin/bash

# Phase 1 Training Launch Script
# Launches Gazebo simulation and starts online RL training

echo "🚀 Starting Phase 1: Environment Integration & Data Collection"
echo "=============================================================="

# Set up environment
export ROS_MASTER_URI=http://localhost:11311
export ROS_PACKAGE_PATH=/home/bouri/roboset/simple_manipulator_ws/src:$ROS_PACKAGE_PATH

# Function to cleanup background processes
cleanup() {
    echo "🧹 Cleaning up background processes..."
    pkill -f gazebo
    pkill -f roslaunch
    pkill -f roscore
    pkill -f python3
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if ROS is running
if ! pgrep -x "roscore" > /dev/null; then
    echo "🔄 Starting ROS master..."
    roscore &
    sleep 3
fi

# Check if Gazebo is running
if ! pgrep -x "gzserver" > /dev/null; then
    echo "🔄 Starting Gazebo simulation..."
    cd /home/bouri/roboset/simple_manipulator_ws
    source /opt/ros/noetic/setup.bash
    source devel/setup.bash
    roslaunch simple_manipulator training_env.launch &
    sleep 10
fi

# Wait for Gazebo to be ready
echo "⏳ Waiting for Gazebo to be ready..."
timeout 30 bash -c 'until rostopic list | grep -q "/gazebo/model_states"; do sleep 1; done'

if [ $? -eq 0 ]; then
    echo "✅ Gazebo is ready"
else
    echo "❌ Gazebo failed to start properly"
    exit 1
fi

# Unpause physics
echo "🔄 Unpausing physics..."
rosservice call /gazebo/unpause_physics

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source /home/bouri/roboset/rl_env/bin/activate

# Verify environment
echo "🔍 Verifying environment..."
python3 -c "
import sys
print(f'Python version: {sys.version}')

# Test PyTorch
try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')

# Test ROS
try:
    import rospy
    print('✅ ROS Python bindings working')
except ImportError as e:
    print(f'❌ ROS import failed: {e}')

# Test other dependencies
try:
    import numpy as np
    import yaml
    print('✅ Core dependencies working')
except ImportError as e:
    print(f'❌ Some dependencies failed: {e}')

print('🎉 Environment verification complete!')
"

# Start training
echo "🎓 Starting online RL training..."
cd /home/bouri/roboset

# Run training with comprehensive logging
python3 training/online_trainer.py \
    --episodes 1000 \
    --max_steps 50 \
    --accuracy_weight 15.0 \
    --speed_weight 5.0 \
    --energy_weight 0.01 \
    --buffer_capacity 100000 \
    --initial_episodes 50 \
    --logs_dir ./logs \
    --models_dir ./models \
    --save_frequency 100 \
    --print_frequency 10

# Wait for training to complete
wait

echo "✅ Phase 1 training completed!"
