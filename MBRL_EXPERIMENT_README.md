# Single Episode MBRL Experiment for 6-DOF Manipulator

## 🎯 Overview

This experiment tests whether a Model-Based Reinforcement Learning (MBRL) policy can learn the correct joint controller for a 6-DOF manipulator in a single episode. The experiment includes:

- **100x Action Scaling**: As requested, actions are scaled by 100x before being applied
- **Comprehensive Logging**: Detailed logs to understand what's happening
- **Real-time Monitoring**: System health checks and debugging tools
- **Physics Validation**: Proper integration with Gazebo simulation

## 📁 Files

- `single_episode_mbrl_experiment.py` - Main experiment script
- `debug_mbrl_experiment.py` - Debugging and monitoring tool
- `run_mbrl_experiment.sh` - Launch script for the complete experiment
- `MBRL_EXPERIMENT_README.md` - This documentation

## 🚀 Quick Start

### Option 1: Automated Launch (Recommended)

```bash
cd /home/bouri/roboset
./run_mbrl_experiment.sh
```

This script will:
1. Start ROS master if not running
2. Launch Gazebo simulation
3. Run system diagnostics
4. Execute the single episode experiment

### Option 2: Manual Launch

```bash
# Terminal 1: Start ROS and Gazebo
cd /home/bouri/roboset/simple_manipulator_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch simple_manipulator training_env.launch

# Terminal 2: Run diagnostics
cd /home/bouri/roboset
source noetic/bin/activate
python3 debug_mbrl_experiment.py

# Terminal 3: Run experiment
cd /home/bouri/roboset
source noetic/bin/activate
python3 single_episode_mbrl_experiment.py
```

## 🔍 System Requirements

### Prerequisites
- ROS Noetic
- Gazebo Classic
- Python 3.8+
- PyTorch
- NumPy

### Environment Setup
- Gazebo simulation running
- Manipulator spawned with controllers
- Coke can placed on table
- Physics unpaused

## 📊 Experiment Details

### Objective
Test if MBRL can learn to control a 6-DOF manipulator to reach a target position (coke can) in a single episode.

### Action Scaling
Actions are scaled by **100x** as requested:
```python
scaled_action = action * 100.0  # 100x scaling
torques = scaled_action * torque_limits
```

### Reward Function
- **Accuracy**: Position error to target (exponential reward)
- **Energy**: Torque minimization penalty
- **Time**: Efficiency reward for quick completion
- **Constraints**: Joint limit violation penalties
- **Success**: 100.0 bonus when within 5cm of target

### Models
- **Actor-Critic**: Policy and value function learning
- **Dynamics Model**: State transition prediction
- **Online Learning**: Models update every 10 steps

## 🔧 Debugging and Monitoring

### System Health Check
The debugger checks:
- ✅ ROS master status
- ✅ Gazebo services availability
- ✅ Controller status
- ✅ Joint states reception
- ✅ End-effector pose calculation
- ✅ Effort command topics
- ✅ Command execution

### Logging Levels
- **INFO**: General progress and status
- **DEBUG**: Detailed action/state information
- **WARNING**: Potential issues
- **ERROR**: Critical failures

### Log Files
- `single_episode_experiment.log` - Main experiment logs
- `debug_mbrl_experiment.log` - Debugger logs

## 📈 Expected Results

### Success Criteria
- **Position Error**: < 5cm from target
- **Episode Length**: < 1000 steps
- **Energy Efficiency**: Reasonable torque usage
- **Constraint Compliance**: No joint limit violations

### Typical Performance
- **Initial Position Error**: ~1.0m
- **Final Position Error**: < 0.05m (success)
- **Episode Length**: 200-800 steps
- **Total Reward**: 50-200 points

## 🐛 Troubleshooting

### Common Issues

#### 1. ROS Master Not Running
```bash
# Start ROS master
roscore &
```

#### 2. Gazebo Not Starting
```bash
# Check Gazebo installation
gazebo --version

# Start Gazebo manually
roslaunch simple_manipulator training_env.launch
```

#### 3. Joint States Not Received
```bash
# Check controller status
rosservice call /manipulator/controller_manager/list_controllers

# Check joint states topic
rostopic echo /manipulator/joint_states
```

#### 4. End-Effector Pose Issues
```bash
# Check TF tree
rosrun tf view_frames
evince frames.pdf

# Check transform
rosrun tf tf_echo base_link tool0
```

#### 5. Effort Commands Not Working
```bash
# Test effort commands manually
rostopic pub /manipulator/shoulder_pan_joint_effort/command std_msgs/Float64 "data: 10.0"
```

### Debug Commands

#### Check System Status
```bash
# Run comprehensive diagnostics
python3 debug_mbrl_experiment.py

# Check specific components
rostopic list
rosservice list
rosnode list
```

#### Monitor Experiment
```bash
# Monitor joint states
rostopic echo /manipulator/joint_states

# Monitor effort commands
rostopic echo /manipulator/shoulder_pan_joint_effort/command

# Check end-effector pose
rostopic echo /tf
```

## 📊 Performance Analysis

### Metrics Tracked
- **Position Error**: Distance to target over time
- **Energy Consumption**: Total torque applied
- **Episode Length**: Steps to completion
- **Success Rate**: Whether target was reached
- **Constraint Violations**: Joint limit violations

### Visualization
The experiment logs provide detailed information for analysis:
- Step-by-step action/state progression
- Reward component breakdown
- Model training losses
- System health status

## 🔬 Experiment Variations

### Different Target Positions
Modify the target position in the experiment:
```python
self.target_position = np.array([0.6, 0.0, 0.8])  # Default
self.target_position = np.array([0.5, 0.2, 0.6])  # Different target
```

### Different Action Scaling
Adjust the action scaling factor:
```python
self.action_scale = 100.0  # Default 100x
self.action_scale = 50.0   # 50x scaling
self.action_scale = 200.0  # 200x scaling
```

### Different Reward Weights
Modify reward function weights:
```python
# In compute_reward method
accuracy_reward = 100.0 * np.exp(-position_error / 0.1)
energy_penalty = 0.001 * np.sum(np.abs(torques))
time_reward = (self.max_episode_steps - step) / self.max_episode_steps
```

## 📝 Expected Log Output

### Successful Experiment
```
🎬 Starting single episode experiment
📊 Initial state: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] (positions), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (velocities)
Step 0: Reward = 0.123, Total = 0.123, Position error = 0.856
Step 50: Reward = 0.234, Total = 12.456, Position error = 0.234
...
Step 200: Reward = 15.678, Total = 156.789, Position error = 0.045
🎉 SUCCESS at step 200! Position error = 0.045
📊 Episode Summary:
  Success: True
  Total Reward: 156.789
  Episode Length: 201
  Final Position Error: 0.045
  Total Energy: 1234.567
```

### Failed Experiment
```
🎬 Starting single episode experiment
📊 Initial state: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] (positions), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (velocities)
Step 0: Reward = 0.123, Total = 0.123, Position error = 0.856
...
Step 999: Reward = 0.045, Total = 45.678, Position error = 0.234
📊 Episode Summary:
  Success: False
  Total Reward: 45.678
  Episode Length: 1000
  Final Position Error: 0.234
  Total Energy: 5678.901
```

## 🎯 Success Indicators

### Positive Signs
- ✅ Position error decreasing over time
- ✅ Reward increasing over time
- ✅ Joint movements visible in Gazebo
- ✅ No constraint violations
- ✅ Models learning (loss decreasing)

### Warning Signs
- ⚠️ Position error not decreasing
- ⚠️ Reward staying constant or decreasing
- ⚠️ No visible joint movement
- ⚠️ Constraint violations occurring
- ⚠️ Models not learning (loss increasing)

## 🔄 Next Steps

After running the experiment:

1. **Analyze Results**: Check if the policy learned to reach the target
2. **Debug Issues**: Use the debugger to identify problems
3. **Adjust Parameters**: Modify reward weights, action scaling, etc.
4. **Extend Training**: Run multiple episodes for better learning
5. **Validate Physics**: Ensure the LNN is modeling dynamics correctly

## 📚 References

- [PINN_RL Framework](https://github.com/your-repo/PINN_RL)
- [Gazebo Simulation](http://gazebosim.org/)
- [ROS Noetic](https://www.ros.org/)
- [PyTorch](https://pytorch.org/)

---

**Note**: This experiment is designed to test the basic functionality of MBRL with your manipulator. For production use, consider running multiple episodes and fine-tuning the hyperparameters.
