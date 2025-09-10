#!/bin/bash

# MBRL Single Episode Experiment Launch Script
# This script launches the Gazebo environment and runs the MBRL experiment

echo "🚀 Starting MBRL Single Episode Experiment"
echo "=========================================="

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

# Run debugger first
echo "🔍 Running system diagnostics..."
cd /home/bouri/roboset
source /home/bouri/roboset/noetic/bin/activate
python3 debug_mbrl_experiment.py

# Wait for user confirmation
echo "Press Enter to continue with the experiment..."
read

# Run the single episode experiment
echo "🎬 Starting single episode MBRL experiment..."
python3 single_episode_mbrl_experiment.py

# Wait for experiment to complete
wait

echo "✅ Experiment completed!"
