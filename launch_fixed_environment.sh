#!/bin/bash
# Fixed launch script for Gazebo environment with proper error handling

echo "🚀 Launching Fixed Gazebo Environment with Manipulator"
echo "======================================================"

# Source ROS environment
source /opt/ros/noetic/setup.bash
source /home/bouri/roboset/simple_manipulator_ws/devel/setup.bash

echo "✅ ROS environment sourced"

# Kill any existing Gazebo processes
echo "🧹 Cleaning up existing processes..."
pkill -f gazebo
pkill -f gzserver
pkill -f gzclient
sleep 2

# Start ROS master if not running
if ! pgrep -f "rosmaster" > /dev/null; then
    echo "🔄 Starting ROS master..."
    roscore &
    sleep 3
fi

# Launch Gazebo with training environment
echo "🎬 Starting Gazebo simulation..."
roslaunch simple_manipulator training_env.launch &
GAZEBO_PID=$!

# Wait for Gazebo to start
echo "⏳ Waiting for Gazebo to initialize..."
sleep 15

# Check if Gazebo is running
if pgrep -f "gzserver" > /dev/null; then
    echo "✅ Gazebo is running"
else
    echo "❌ Gazebo failed to start"
    exit 1
fi

# Wait for services to be available
echo "⏳ Waiting for Gazebo services..."
timeout 30 bash -c 'until rosservice list | grep -q gazebo; do sleep 1; done'

# Unpause physics
echo "🔧 Unpausing physics..."
rosservice call /gazebo/unpause_physics

# Wait for controllers to load
echo "⏳ Waiting for controllers to load..."
sleep 5

# Check controller status
echo "🔍 Checking controller status..."
if rosservice call /manipulator/controller_manager/list_controllers 2>/dev/null | grep -q "running"; then
    echo "✅ Controllers are running"
else
    echo "⚠️  Controllers may not be fully loaded yet"
fi

# Check joint states
echo "🔍 Checking joint states..."
if timeout 5 rostopic echo /manipulator/joint_states -n 1 >/dev/null 2>&1; then
    echo "✅ Joint states are being published"
else
    echo "⚠️  Joint states not available yet"
fi

echo ""
echo "🎉 Environment is ready!"
echo "========================="
echo "To monitor joint states, run:"
echo "  python3 /home/bouri/roboset/monitor_joint_states.py"
echo ""
echo "To control joints interactively, run:"
echo "  python3 /home/bouri/roboset/interactive_joint_control.py"
echo ""
echo "To debug issues, run:"
echo "  python3 /home/bouri/roboset/debug_environment.py"
echo ""
echo "To stop this environment, press Ctrl+C"
echo ""

# Keep script running and handle cleanup
trap 'echo "🛑 Shutting down..."; kill $GAZEBO_PID 2>/dev/null; exit 0' INT TERM
wait
