#!/bin/bash

# Setup script for RL training environment with PyKDL support
# This script creates a new virtual environment with all required dependencies

echo "🚀 Setting up RL Training Environment"
echo "====================================="

# Check Python version
echo "🐍 Checking Python version..."
python3 --version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📊 Detected Python version: $PYTHON_VERSION"

# Create new virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv rl_env

# Check if venv was created successfully
if [ ! -d "rl_env" ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source rl_env/bin/activate

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (as requested)
echo "🔥 Installing PyTorch with CUDA support..."
# Try CUDA 11.8 first (more compatible), then fallback to CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 || {
    echo "⚠️ CUDA 11.8 failed, trying CPU version..."
    pip install torch torchvision
}

# Install ROS dependencies first (to avoid conflicts)
echo "🤖 Installing ROS Python dependencies..."
sudo apt update
sudo apt install -y python3-rospy python3-rospkg python3-catkin-tools
sudo apt install -y python3-tf2-ros python3-tf2-geometry-msgs
sudo apt install -y python3-gazebo-ros python3-gazebo-ros-control

# Install PyKDL and related dependencies
echo "🔧 Installing PyKDL and geometry dependencies..."
sudo apt install -y liborocos-kdl-dev

# Try different PyKDL installation methods
echo "🔧 Installing PyKDL..."
pip install PyKDL || {
    echo "⚠️ PyKDL pip install failed, trying alternative method..."
    # Try installing from system packages
    sudo apt install -y python3-kdl || {
        echo "⚠️ System PyKDL not available, continuing without PyKDL..."
        echo "📝 Note: Some tf2_geometry_msgs features may not work"
    }
}

# Install other required packages
echo "📚 Installing additional dependencies..."
pip install numpy scipy sympy matplotlib seaborn
pip install tensorboard
pip install rospkg catkin_pkg
pip install gym
pip install opencv-python

# Install tf2_geometry_msgs (alternative approach)
echo "🌐 Installing tf2_geometry_msgs..."
pip install tf2_geometry_msgs || {
    echo "⚠️ tf2_geometry_msgs pip install failed, using system version..."
    # This should already be available from ROS installation
}

# Verify installation
echo "✅ Verifying installation..."
python3 -c "
import sys
print(f'Python version: {sys.version}')

# Test PyTorch
try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ CUDA device count: {torch.cuda.device_count()}')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')

# Test ROS
try:
    import rospy
    print('✅ ROS Python bindings working')
except ImportError as e:
    print(f'❌ ROS import failed: {e}')

# Test tf2_geometry_msgs
try:
    import tf2_geometry_msgs
    print('✅ tf2_geometry_msgs imported successfully')
except ImportError as e:
    print(f'⚠️ tf2_geometry_msgs import failed: {e}')

# Test PyKDL
try:
    import PyKDL
    print('✅ PyKDL imported successfully')
except ImportError as e:
    print(f'⚠️ PyKDL import failed: {e}')

# Test other dependencies
try:
    import numpy as np
    import scipy
    import sympy
    import matplotlib
    import seaborn
    import tensorboard
    import gym
    import cv2
    print('✅ All other dependencies working')
except ImportError as e:
    print(f'❌ Some dependencies failed: {e}')

print('🎉 Verification complete!')
"

# Add to PATH
echo "🔧 Adding scripts to PATH..."
echo 'export PATH="\$HOME/.local/bin:\$PATH"' >> ~/.bashrc

echo "🎉 Environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run: source rl_env/bin/activate"
echo "2. Run: source ~/.bashrc  # or restart terminal"
echo "3. Run: python3 test_phase1.py"
echo "4. Run: ./launch_phase1_training.sh"