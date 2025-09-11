#!/bin/bash

# Minimal setup script for RL training environment
# This version uses CPU-only PyTorch for fastest setup

echo "🚀 Setting up RL Training Environment (Minimal)"
echo "==============================================="

# Check Python version
echo "🐍 Checking Python version..."
python3 --version

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

# Install PyTorch CPU-only (much faster)
echo "🔥 Installing PyTorch (CPU-only)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install numpy scipy sympy matplotlib seaborn
pip install tensorboard
pip install gym
pip install opencv-python

# Install ROS Python packages (system-wide, already installed)
echo "🤖 ROS dependencies should already be installed system-wide"

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
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')

# Test core dependencies
try:
    import numpy as np
    import scipy
    import sympy
    import matplotlib
    import seaborn
    import tensorboard
    import gym
    import cv2
    print('✅ All core dependencies working')
except ImportError as e:
    print(f'❌ Some dependencies failed: {e}')

print('🎉 Minimal verification complete!')
"

echo "🎉 Minimal environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run: source rl_env/bin/activate"
echo "2. Run: python3 test_phase1.py"
echo "3. If ROS issues occur, run: source /opt/ros/noetic/setup.bash"
echo "4. Run: ./launch_phase1_training.sh"
