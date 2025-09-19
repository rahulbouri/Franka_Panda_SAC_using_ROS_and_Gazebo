#!/bin/bash
# Launch Enhanced SAC Training with Proper Collision Detection
# Based on gym-gazebo patterns and PINN_RL reward system

echo "🛡️ Launching Enhanced SAC Training with Proper Collision Detection"
echo "=================================================================="

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f roscore
pkill -f gazebo
pkill -f roslaunch
sleep 2

# Start ROS master
echo "🔧 Starting ROS master..."
roscore &
ROS_PID=$!
sleep 3

# Start Gazebo simulation
echo "🎮 Starting Gazebo simulation..."
cd /home/bouri/roboset/simple_manipulator_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch simple_manipulator training_env.launch &
GAZEBO_PID=$!
sleep 5

# Verify ROS is running
echo "🔍 Verifying ROS master..."
if ! rostopic list > /dev/null 2>&1; then
    echo "❌ ROS master is not running!"
    exit 1
fi

echo "✅ ROS master is running"
echo "✅ Gazebo simulation is running"

# Go back to project directory
cd /home/bouri/roboset

# Activate Python environment
echo "🐍 Activating Python environment..."
source /home/bouri/roboset/venv/bin/activate

# Set ROS environment variables
export ROS_MASTER_URI=http://localhost:11311
export ROS_PACKAGE_PATH=/home/bouri/roboset/simple_manipulator_ws/src:$ROS_PACKAGE_PATH

# Run Enhanced SAC training
echo "🎓 Starting Enhanced SAC training..."
echo "🛡️ Enhanced features:"
echo "   - Proper collision detection using Gazebo contact sensors"
echo "   - 0.2m success tolerance for target reaching"
echo "   - Immediate episode termination on collision"
echo "   - PINN_RL-based reward system"
echo "   - Energy efficiency optimization"
echo ""

python3 train_sac_enhanced.py

echo ""
echo "🏁 Enhanced SAC Training completed!"
echo "🎮 Check the Gazebo window for final manipulator position!"

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    kill $GAZEBO_PID 2>/dev/null
    kill $ROS_PID 2>/dev/null
    pkill -f roscore
    pkill -f gazebo
    pkill -f roslaunch
    echo "✅ Cleanup complete"
}

# Set trap for cleanup on exit
trap cleanup EXIT

echo "Press Ctrl+C to exit and cleanup"
wait
