#!/usr/bin/env python3
"""
Simple Test Script for Phase 1 Implementation
Tests basic functionality without requiring ROS to be running

Author: RL Training Implementation
Date: 2024
"""

import os
import sys
import numpy as np
import logging
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        
        import numpy as np
        logger.info("✅ NumPy")
        
        import scipy
        logger.info("✅ SciPy")
        
        import sympy
        logger.info("✅ SymPy")
        
        import matplotlib
        logger.info("✅ Matplotlib")
        
        import seaborn
        logger.info("✅ Seaborn")
        
        import tensorboard
        logger.info("✅ TensorBoard")
        
        import gym
        logger.info("✅ Gym")
        
        import cv2
        logger.info("✅ OpenCV")
        
        import yaml
        logger.info("✅ PyYAML")
        
        # Test our modules
        from env.manipulator_env_simple import ManipulatorEnvironmentSimple, EnvironmentConfig
        logger.info("✅ ManipulatorEnvironmentSimple")
        
        from models.replay_buffer import PrioritizedReplayBuffer
        logger.info("✅ PrioritizedReplayBuffer")
        
        logger.info("✅ All imports successful!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_replay_buffer():
    """Test replay buffer functionality"""
    logger.info("🧪 Testing replay buffer...")
    
    try:
        from models.replay_buffer import PrioritizedReplayBuffer
        
        # Create buffer
        buffer = PrioritizedReplayBuffer(capacity=1000)
        
        # Add some experiences
        for i in range(100):
            state = np.random.randn(18)
            action = np.random.randn(6)
            reward = np.random.randn()
            next_state = np.random.randn(18)
            done = i % 10 == 9
            
            buffer.add(state, action, reward, next_state, done)
        
        # Test sampling
        batch_data, indices, weights = buffer.sample(32)
        
        if batch_data is not None:
            logger.info(f"✅ Replay buffer test passed. Batch size: {len(batch_data['states'])}")
            return True
        else:
            logger.error("❌ Replay buffer sampling failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Replay buffer test failed: {e}")
        return False

def test_environment_config():
    """Test environment configuration"""
    logger.info("🧪 Testing environment configuration...")
    
    try:
        from env.manipulator_env_simple import EnvironmentConfig
        
        # Create config
        config = EnvironmentConfig(
            max_episode_steps=50,
            accuracy_weight=15.0,
            speed_weight=5.0,
            energy_weight=0.01
        )
        
        logger.info(f"✅ Environment config created successfully")
        logger.info(f"   Max steps: {config.max_episode_steps}")
        logger.info(f"   Accuracy weight: {config.accuracy_weight}")
        logger.info(f"   Speed weight: {config.speed_weight}")
        logger.info(f"   Energy weight: {config.energy_weight}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment config test failed: {e}")
        return False

def test_training_config():
    """Test training configuration"""
    logger.info("🧪 Testing training configuration...")
    
    try:
        # Create training config
        config = {
            'total_episodes': 100,
            'max_episode_steps': 50,
            'accuracy_weight': 15.0,
            'speed_weight': 5.0,
            'energy_weight': 0.01,
            'replay_buffer': {
                'capacity': 10000,
                'alpha': 0.6,
                'beta': 0.4,
                'beta_increment': 0.001
            },
            'initial_episodes': 10,
            'logs_dir': './test_logs',
            'models_dir': './test_models',
            'device': 'cpu'
        }
        
        logger.info(f"✅ Training config created successfully")
        logger.info(f"   Total episodes: {config['total_episodes']}")
        logger.info(f"   Max steps per episode: {config['max_episode_steps']}")
        logger.info(f"   Replay buffer capacity: {config['replay_buffer']['capacity']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training config test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    logger.info("🧪 Testing file structure...")
    
    required_files = [
        'env/manipulator_env_simple.py',
        'models/replay_buffer.py',
        'training/online_trainer.py',
        'launch_phase1_training.sh',
        'setup_environment.sh'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Missing files: {missing_files}")
        return False
    else:
        logger.info("✅ All required files exist")
        return True

def main():
    """Run all tests"""
    logger.info("🚀 Starting Phase 1 Simple Tests")
    logger.info("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Environment Config", test_environment_config),
        ("Training Config", test_training_config),
        ("Replay Buffer", test_replay_buffer)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} test passed")
        else:
            logger.error(f"❌ {test_name} test failed")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Phase 1 implementation is ready.")
        logger.info("\n📋 Next steps:")
        logger.info("1. Start ROS: roscore &")
        logger.info("2. Start Gazebo: roslaunch simple_manipulator training_env.launch &")
        logger.info("3. Run: ./launch_phase1_training.sh")
        logger.info("4. Monitor logs in ./logs/ directory")
    else:
        logger.error("❌ Some tests failed. Please fix the issues before proceeding.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

