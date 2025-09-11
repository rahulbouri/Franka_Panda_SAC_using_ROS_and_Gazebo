# Phase 1: Environment Integration & Data Collection

This document describes the implementation of Phase 1 of the online RL training system for the 6-DOF manipulator.

## Overview

Phase 1 focuses on:
- **Environment Integration**: Unified interface between RL training and ROS/Gazebo simulation
- **Data Collection**: Efficient experience storage and retrieval using prioritized replay buffer
- **Comprehensive Logging**: Detailed debugging and monitoring capabilities
- **Multi-objective Rewards**: Accuracy (highest), speed, and energy efficiency optimization

## Key Findings from PINN_RL Analysis

### Policy Update Frequency
- **PINN_RL Updates**: Policy is updated every episode (not every step)
- **Imagination Horizon**: `T=16` parameter controls model-based rollouts
- **Evaluation Frequency**: Every 5 episodes (`eval_every=5`)
- **Our Implementation**: Will follow similar pattern for consistency

### Episode Configuration
- **Max Steps**: Set to 50 steps per episode (as requested)
- **Reward Weights**: Accuracy=15.0, Speed=5.0, Energy=0.01 (as requested)
- **Target Success Rate**: 100% (as requested)

## File Structure

```
roboset/
├── env/
│   └── manipulator_env.py          # Unified environment wrapper
├── models/
│   └── replay_buffer.py            # Prioritized experience replay
├── training/
│   └── online_trainer.py           # Main training script
├── scripts/
│   ├── setup_environment.sh        # Environment setup
│   └── launch_phase1_training.sh   # Training launcher
├── logs/                           # Training logs
├── models/                         # Saved models
├── test_phase1.py                  # Test script
└── PHASE1_README.md               # This file
```

## Implementation Details

### 1. Environment Integration (`env/manipulator_env.py`)

**Key Features:**
- **ROS Communication**: Handles all ROS topics and services
- **State Observation**: 18D observation vector (joints + velocities + end-effector + target)
- **Action Execution**: 6D torque commands for all joints
- **Reward Calculation**: Multi-objective reward system
- **Episode Management**: Random target and initial state generation

**State Space (18D):**
- Joint positions (6D): shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
- Joint velocities (6D): corresponding joint velocities
- End-effector position (3D): x, y, z coordinates
- Target position (3D): desired pick/place location

**Action Space (6D):**
- Joint torques (6D): torque commands for 6 main joints

**Reward Components:**
1. **Accuracy Reward** (Weight: 15.0): `-distance_to_target * accuracy_weight`
2. **Speed Reward** (Weight: 5.0): `remaining_steps * speed_weight`
3. **Energy Reward** (Weight: 0.01): `-torque_squared * energy_weight`
4. **Completion Bonus**: +100.0 for task completion

### 2. Replay Buffer (`models/replay_buffer.py`)

**Key Features:**
- **Prioritized Sampling**: Based on TD error importance
- **Experience Storage**: Efficient numpy-based storage
- **Statistics Tracking**: Comprehensive performance metrics
- **Checkpointing**: Save/load functionality

**Configuration:**
- **Capacity**: 100,000 experiences
- **Alpha**: 0.6 (prioritization exponent)
- **Beta**: 0.4 (importance sampling correction)
- **Beta Increment**: 0.001 per sampling

### 3. Online Trainer (`training/online_trainer.py`)

**Key Features:**
- **Initial Data Collection**: Random exploration phase
- **Online Learning**: Continuous data collection and storage
- **Comprehensive Logging**: Detailed metrics and debugging
- **Checkpointing**: Regular model and buffer saves

**Training Phases:**
1. **Initial Data Collection**: 50 random episodes
2. **Online Training**: 1000 episodes with policy updates
3. **Evaluation**: Performance monitoring and logging

## Usage Instructions

### 1. Environment Setup

```bash
# Run the environment setup script
./setup_environment.sh
```

This will:
- Create a new virtual environment (`rl_env/`)
- Install PyTorch with CUDA support
- Install ROS dependencies including PyKDL
- Install all required Python packages

### 2. Test Implementation

```bash
# Run the test script to verify everything works
python3 test_phase1.py
```

This will test:
- All imports and dependencies
- Replay buffer functionality
- Environment configuration
- File structure

### 3. Start Training

```bash
# Launch the complete training system
./launch_phase1_training.sh
```

This will:
- Start ROS master and Gazebo simulation
- Launch the manipulator environment
- Begin online RL training
- Save logs and checkpoints

### 4. Monitor Progress

```bash
# Monitor training logs
tail -f online_training.log

# Check episode logs
tail -f logs/episodes.jsonl

# Monitor ROS topics
rostopic list
rostopic echo /manipulator/joint_states
```

## Configuration Parameters

### Environment Configuration
```python
EnvironmentConfig(
    max_episode_steps=50,           # Max steps per episode
    ros_rate=50,                    # ROS control frequency
    target_tolerance=0.05,          # 5cm target tolerance
    accuracy_weight=15.0,           # Highest priority
    speed_weight=5.0,               # Second priority
    energy_weight=0.01              # Third priority
)
```

### Training Configuration
```python
{
    'total_episodes': 1000,         # Total training episodes
    'max_episode_steps': 50,        # Steps per episode
    'initial_episodes': 50,         # Random exploration episodes
    'accuracy_weight': 15.0,        # Accuracy reward weight
    'speed_weight': 5.0,            # Speed reward weight
    'energy_weight': 0.01,          # Energy reward weight
    'replay_buffer': {
        'capacity': 100000,         # Buffer capacity
        'alpha': 0.6,               # Prioritization exponent
        'beta': 0.4,                # Importance sampling
        'beta_increment': 0.001     # Beta increment
    }
}
```

## Logging and Debugging

### Log Files
- `online_training.log`: Main training log with detailed debugging
- `manipulator_env.log`: Environment-specific logs
- `logs/episodes.jsonl`: Episode-by-episode results
- `logs/training_metrics.json`: Aggregated training metrics

### Debug Information
The implementation includes comprehensive logging at multiple levels:
- **DEBUG**: Detailed step-by-step information
- **INFO**: Important events and progress updates
- **WARNING**: Potential issues and fallbacks
- **ERROR**: Critical errors and failures

### Monitoring Commands
```bash
# Check training progress
grep "Episode.*complete" online_training.log

# Monitor reward components
grep "reward" logs/episodes.jsonl | tail -10

# Check buffer statistics
grep "Buffer stats" online_training.log | tail -5

# Monitor ROS communication
rostopic hz /manipulator/joint_states
```

## Expected Performance

### Initial Phase (Random Actions)
- **Success Rate**: ~5-10% (random exploration)
- **Average Reward**: Low (due to random actions)
- **Episode Length**: 50 steps (max)

### After Training
- **Target Success Rate**: 100%
- **Average Reward**: High (optimized for accuracy, speed, energy)
- **Episode Length**: <50 steps (efficient completion)

## Troubleshooting

### Common Issues

1. **PyKDL Import Error**
   ```bash
   # Solution: Run setup script
   ./setup_environment.sh
   ```

2. **ROS Connection Issues**
   ```bash
   # Check ROS master
   roscore &
   # Check topics
   rostopic list
   ```

3. **Gazebo Not Starting**
   ```bash
   # Kill existing processes
   pkill -f gazebo
   # Restart simulation
   roslaunch simple_manipulator training_env.launch
   ```

4. **Memory Issues**
   ```bash
   # Reduce buffer capacity in config
   'capacity': 50000  # Instead of 100000
   ```

### Debug Commands
```bash
# Check system resources
htop
nvidia-smi  # If using GPU

# Check ROS nodes
rosnode list
rosnode info /manipulator_env

# Check file permissions
ls -la env/ models/ training/
```

## Next Steps (Phase 2)

After Phase 1 completion, the next steps will be:
1. **LNN Integration**: Physics-informed dynamics model
2. **Policy Network**: SAC agent implementation
3. **Model-Based Updates**: Imaginary rollouts and policy optimization
4. **Performance Tuning**: Hyperparameter optimization

## Questions and Support

If you encounter any issues or have questions:
1. Check the log files for detailed error messages
2. Run the test script to verify setup
3. Check ROS topics and services are available
4. Verify all dependencies are installed correctly

The implementation includes extensive debugging information to help identify and resolve issues quickly.
