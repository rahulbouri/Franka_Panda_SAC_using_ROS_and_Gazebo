# Manipulator RL Environment

## Overview
This workspace contains a simplified manipulator environment for RL training with constraint-aware joint control.

## Essential Files

### 🚀 Launch Files
- **`launch_environment.sh`** - Main launch script for Gazebo environment
- **`simple_manipulator_ws/`** - Catkin workspace with manipulator package

### 🤖 Robot Description
- **`simple_manipulator_ws/src/simple_manipulator/urdf/manipulator.urdf`** - Robot URDF
- **`simple_manipulator_ws/src/simple_manipulator/world/training_world.world`** - Gazebo world
- **`simple_manipulator_ws/src/simple_manipulator/launch/training_env.launch`** - ROS launch file

### 🎮 Control Scripts
- **`interactive_joint_control.py`** - Interactive joint control with constraint awareness
- **`monitor_joint_states.py`** - Real-time joint state monitoring
- **`test_joint_control.py`** - Basic joint control testing

### 🧠 RL Environment
- **`rl_constraint_environment.py`** - RL environment with constraint handling
- **`constraint_aware_controller.py`** - Constraint-aware ROS controller
- **`lagrangian_constraint_integration.py`** - LNN with embedded constraints

## Quick Start

### 1. Launch Environment
```bash
# Make scripts executable
chmod +x *.py *.sh

# Launch Gazebo environment
./launch_environment.sh
```

### 2. Monitor Joint States
```bash
# In a new terminal
source /opt/ros/noetic/setup.bash
python3 monitor_joint_states.py
```

### 3. Control Joints Interactively
```bash
# In another terminal
source /opt/ros/noetic/setup.bash
python3 interactive_joint_control.py
```

## Interactive Control Commands

### Available Commands:
- **`s`** - Show current status
- **`m <joint_name> <effort> <duration>`** - Move single joint
- **`a <effort1> <effort2> <effort3> <effort4> <effort5> <effort6> <duration>`** - Move all joints
- **`d`** - Demo sequence
- **`q`** - Quit

### Available Joints:
1. `shoulder_pan_joint`
2. `shoulder_lift_joint`
3. `elbow_joint`
4. `wrist_1_joint`
5. `wrist_2_joint`
6. `wrist_3_joint`

### Example Commands:
```bash
# Move shoulder pan joint with 50 Nm effort for 3 seconds
m shoulder_pan_joint 50.0 3.0

# Move all joints with specified efforts
a 50.0 -30.0 20.0 10.0 -15.0 5.0 3.0

# Show current status
s

# Run demo sequence
d
```

## Constraint Awareness

The system includes joint limit awareness with:
- **Barrier functions** for smooth constraint enforcement
- **Effort clamping** to prevent physical limit violations
- **Real-time constraint monitoring** with violation warnings
- **Adaptive control** that respects joint limits

## Joint Limits

| Joint | Position Range (rad) | Max Effort (Nm) |
|-------|---------------------|-----------------|
| shoulder_pan_joint | [-6.28, 6.28] | 150.0 |
| shoulder_lift_joint | [-6.28, 6.28] | 150.0 |
| elbow_joint | [-3.14, 3.14] | 150.0 |
| wrist_1_joint | [-6.28, 6.28] | 28.0 |
| wrist_2_joint | [-6.28, 6.28] | 28.0 |
| wrist_3_joint | [-6.28, 6.28] | 28.0 |

## RL Training

For RL policy training, use:
```bash
python3 rl_constraint_environment.py
```

The RL environment includes:
- Constraint-aware reward function
- Joint limit enforcement
- Real-time constraint monitoring
- Integration with Lagrangian Neural Networks

## Troubleshooting

### Gazebo Not Starting
```bash
# Check if Gazebo is running
pgrep -f "gzserver"

# Unpause physics if needed
rosservice call /gazebo/unpause_physics
```

### Joint States Not Updating
```bash
# Check ROS topics
rostopic list | grep manipulator
rostopic echo /manipulator/joint_states
```

### Constraint Violations
The system will automatically:
- Apply barrier function corrections
- Clamp efforts to joint limits
- Display constraint violation warnings
- Prevent unsafe movements

## File Structure

```
/home/bouri/roboset/
├── launch_environment.sh          # Main launch script
├── interactive_joint_control.py   # Interactive control
├── monitor_joint_states.py        # Joint state monitoring
├── test_joint_control.py          # Basic testing
├── rl_constraint_environment.py   # RL environment
├── constraint_aware_controller.py # Constraint-aware controller
├── lagrangian_constraint_integration.py # LNN integration
├── simple_manipulator_ws/         # Catkin workspace
│   └── src/simple_manipulator/    # Manipulator package
│       ├── urdf/manipulator.urdf  # Robot description
│       ├── world/training_world.world # Gazebo world
│       └── launch/training_env.launch # ROS launch
└── README.md                      # This file
```

## Next Steps

1. **Test Environment**: Launch Gazebo and verify manipulator appears
2. **Test Control**: Use interactive control to move joints
3. **Verify Constraints**: Check that joint limits are respected
4. **RL Training**: Use RL environment for policy training
5. **Customize**: Modify reward functions and constraints as needed

## Support

For issues or questions:
1. Check ROS topics: `rostopic list`
2. Check joint states: `rostopic echo /manipulator/joint_states`
3. Check Gazebo: `gzclient` (if GUI available)
4. Review logs: `rosnode info <node_name>`
