#!/bin/bash
# Launch SAC Training with Proper ROS Environment Setup
# Based on gym-gazebo approach

echo "🚀 Starting SAC Training with Gazebo Environment"
echo "================================================"

# Source ROS environment
echo "🔧 Sourcing ROS environment..."
source /opt/ros/noetic/setup.bash
source /home/bouri/roboset/simple_manipulator_ws/devel/setup.bash

# Activate Python environment
echo "🐍 Activating Python environment..."
source /home/bouri/rl_env_py310/bin/activate

# Set ROS environment variables
export ROS_MASTER_URI=http://localhost:11311
export ROS_PACKAGE_PATH=/home/bouri/roboset/simple_manipulator_ws/src:$ROS_PACKAGE_PATH

# Verify ROS master is running
echo "🔍 Verifying ROS master..."
if ! rostopic list > /dev/null 2>&1; then
    echo "❌ ROS master is not running! Please start it first:"
    echo "   roscore &"
    echo "   sleep 3 && cd /home/bouri/roboset/simple_manipulator_ws && source /opt/ros/noetic/setup.bash && source devel/setup.bash && roslaunch simple_manipulator training_env.launch &"
    exit 1
fi

echo "✅ ROS master is running"

# Run training
echo "🎓 Starting SAC training..."
python3 train_sac_final.py --episodes 5 --max-steps 1000 --cuda

echo "🏁 Training completed!"


