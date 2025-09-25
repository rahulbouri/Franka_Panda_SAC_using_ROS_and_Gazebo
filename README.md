# Pick-and-Place Manipulation with Online SAC in Gazebo

## Overview
End-to-end system to train a Soft Actor-Critic (SAC) policy for robotic pick-and-place in the Gazebo simulator. The pipeline runs online in simulation: ROS nodes launch the Franka Panda, spawn objects/bins, publish detections, and the SAC trainer learns from live rollouts.

- Robot: Franka Emika Panda (sim)
- Simulator: Gazebo + ROS Noetic
- Task: Color-based pick-and-place into matching bins
- Learning: Online SAC with replay buffer

## Project Goal
Train a robust, online RL policy that perceives objects, controls the 7-DoF arm, and completes pick-and-place reliably in a realistic Gazebo world. This showcases full-stack robotics skills across simulation, perception, control, and reinforcement learning.

## Repository Structure
```
/home/bouri/roboset/
├── simple_manipulator_ws/ (catkin workspace)
│   └── src/pick_and_place/
│       ├── launch/
│       │   ├── panda_world.launch           # Launch Gazebo world + Panda
│       │   └── sac_training.launch          # World + perception + (optional) trainer
│       ├── worlds/pick_and_place.world      # World: table, bins, blocks, camera
│       ├── scripts/
│       │   ├── perception_module.py         # Publishes DetectedObjectsStamped
│       │   ├── ros_controller.py            # Joint-space control + gripper + attacher
│       │   ├── pick_place_sac_env.py        # RL environment wrapper
│       │   └── sac_pick_place_trainer.py    # Online SAC trainer
│       ├── msg/DetectedObject*.msg          # Perception messages
│       └── CMakeLists.txt, package.xml
└── README.md                                # This file
```

## Setup
- OS: Ubuntu 20.04 (ROS Noetic) or WSL2 Ubuntu 20.04 with GUI forwarding
- Dependencies:
  - ROS Noetic desktop-full
  - Gazebo (comes with ROS Noetic)
  - Python 3.8+, numpy, torch, opencv-python, cv-bridge, image-geometry
  - Franka description/gazebo packages: `franka_description`, `franka_gazebo`

### 1) Set up ROS and workspace
```bash
# ROS
sudo apt update
sudo apt install -y ros-noetic-desktop-full ros-noetic-cv-bridge ros-noetic-image-geometry \
  ros-noetic-gazebo-ros ros-noetic-gazebo-plugins ros-noetic-robot-state-publisher \
  ros-noetic-joint-state-publisher ros-noetic-control-msgs ros-noetic-trajectory-msgs \
  ros-noetic-controller-manager ros-noetic-tf ros-noetic-franka-description ros-noetic-franka-gazebo

# Python deps
python3 -m pip install --user numpy torch opencv-python

# Build catkin workspace
source /opt/ros/noetic/setup.bash
cd /home/bouri/roboset/simple_manipulator_ws
catkin_make
source devel/setup.bash
```

### 2) Verify environment variables (add to ~/.bashrc)
```bash
echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
echo 'source /home/bouri/roboset/simple_manipulator_ws/devel/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

## Launch Gazebo World
Launch the world with the Panda and scene objects.
```bash
roslaunch pick_and_place panda_world.launch
```
This brings up Gazebo with:
- Workbench and bins
- Colored blocks on the table
- Kinect camera (for perception)
- Panda robot with controllers

If you prefer to also spawn perception and see topics, use:
```bash
roslaunch pick_and_place sac_training.launch
```
Note: the trainer node is commented by default in `sac_training.launch`. You’ll run it manually below to control training runs and logging.

## Start Online Training
In a new terminal (after sourcing ROS and workspace):
```bash
rosrun pick_and_place sac_pick_place_trainer.py
```
Training flow:
- The `PickPlaceSACEnvironment` subscribes to joint states and detections, and commands joint targets through `ros_controller.py`.
- `sac_pick_place_trainer.py` collects online rollouts, learns with SAC, and periodically saves checkpoints.

Typical outputs:
- Logs: episodic reward, step reward components, state machine transitions
- Checkpoints: `pick_place_sac_episode_XXX.pth`, `pick_place_sac_final.pth`

## Monitoring & Debugging
- Topics: `rostopic list | grep -E "franka|object|joint|gazebo"`
- Joint states: `rostopic echo /franka_state_controller/joint_states`
- Perception stream: `rostopic echo /object_detection`
- Gazebo: ensure link attacher plugin is loaded by world (see `worlds/pick_and_place.world`).

## Results
- Task success rate (eval over multiple seeds): 30%
- Training duration: 20,000 episodes
- Max episode steps: 500
- Average reward (last 100 episodes): modest, plateauing with high variance
- Episode length to completion: typically near cap when failing; shorter on successful trials

Training reward curve (jagged, long-horizon training):
![Training Reward Curve](sac_timestep_jagged.png)

## Visual Demos
Gazebo environment overview:
![Gazebo Pick-and-Place Overview](pick_and_place_overview.png)


## Citation / Credits
- World and task design adapted from prior pick-and-place examples; Panda models from `franka_description` and `franka_gazebo`.
- RL implementation based on standard SAC, integrated for online ROS+Gazebo.
