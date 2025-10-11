# Pick-and-Place Manipulation with Hierarchical SAC and Physics-Informed Control (Franka Emika Panda)

## Overview
This project implements a **hierarchical robotic manipulation system** for a color-based pick-and-place task using the **Franka Emika Panda** robot in **Gazebo** and **ROS Noetic**.  
The design separates **strategic decision-making** (via Soft Actor-Critic, SAC) from **precise motion execution** (low-level controllers), and extends classical hierarchical control with **physics-informed priors** inspired by **Lagrangian dynamics** to improve sample efficiency.

- Robot: Franka Emika Panda (sim)
- Simulator: Gazebo + ROS Noetic
- Task: Color-based pick-and-place into matching bins
- Architecture: Hierarchical control with SAC high-level + physics-informed low-level controllers

## Motivation

Over the past few months I worked to bridge production ML with robot learning by building a **sample-efficient hierarchical SAC policy** for the Franka Panda manipulator  
(Repo: https://github.com/rahulbouri/Franka_Panda_SAC_using_ROS_and_Gazebo).

The goal was to test whether **injecting Lagrangian priors into a low-level controller**, guided by a **high-level goal-sequencing SAC policy**, could reduce the data required to train a pick-and-place task.  
While theoretically appealing, practical implementation revealed several core challenges in stability, simulation fidelity, and real-world scalability — all documented in this README.

## Project Goals

### Technical Objectives
- Implement hierarchical control architecture with SAC as the high-level strategic controller.
- Develop low-level controllers for precise joint-space trajectory execution.
- Integrate **Lagrangian priors** into low-level dynamics for improved sample efficiency.
- Design a unified ROS framework for online reinforcement learning and motion control.

### Research Objectives
- Demonstrate hierarchical RL for manipulation with physics priors.
- Evaluate the effect of Lagrangian structure on data efficiency and control stability.
- Document challenges in symbolic modeling, numerical stability, and simulation fidelity.

### Practical Impact
This project connects theoretical reinforcement learning, control systems, and robotics engineering.  
It highlights the challenges of **bridging physics priors with learning-based policies** and provides a foundation for future sim-to-real robotics research.

## Repository Structure
```
/home/bouri/roboset/
├── simple_manipulator_ws/
│   └── src/pick_and_place/
│       ├── launch/
│       │   ├── panda_world.launch
│       │   └── sac_training.launch
│       ├── worlds/pick_and_place.world
│       ├── scripts/
│       │   ├── perception_module.py
│       │   ├── ros_controller.py
│       │   ├── pick_place_sac_env.py
│       │   ├── sac_pick_place_trainer.py
│       │   ├── deep_lagrangian_experiments.py
│       │   ├── residual_controller.py
│       │   └── lagrangian_utils.py
│       ├── msg/DetectedObject*.msg
│       └── CMakeLists.txt, package.xml
└── README.md
```

## Dependencies

### Core Stack
- **OS:** Ubuntu 20.04 (ROS Noetic)
- **Simulator:** Gazebo 11
- **Robot Packages:** franka_description, franka_gazebo, franka_ros

### Python Packages
torch >= 1.9
opencv-python >= 4.5
numpy, scipy, matplotlib, tensorboard

### ROS Packages
ros-noetic-cv-bridge
ros-noetic-gazebo-ros
ros-noetic-tf
ros-noetic-franka-description
ros-noetic-gazebo-ros-link-attacher

## Setup

```bash
# Source ROS
source /opt/ros/noetic/setup.bash

# Build workspace
cd simple_manipulator_ws
catkin_make
source devel/setup.bash

# Verify
rosversion -d   # should output "noetic"
```

## Launch Simulation

```bash
roslaunch pick_and_place panda_world.launch
```

To start training (after Gazebo launches):

```bash
rosrun pick_and_place sac_pick_place_trainer.py
```

## Hierarchical Control Architecture

### 1. High-Level SAC Controller (`sac_pick_place_trainer.py`)
- Learns strategic sequencing of manipulation phases (Approach → Grasp → Transport → Place).
- Input: Multi-modal 42D state vector (joint states + vision + task context).
- Output: High-level task-phase command.
- Learns online using experience replay and reward shaping.

### 2. Low-Level Motion Controllers (`ros_controller.py`)
- Execute smooth joint-space trajectories and gripper commands.
- Use safety checks for collision avoidance and joint limits.
- Accept high-level commands from SAC.

### 3. Lagrangian Priors and Residual Strategy
To improve sample efficiency, low-level controllers were augmented with **Lagrangian structure**:

- Implemented **Deep Lagrangian Networks (DeLaN)**-style parameterization (Lutter et al., ICLR 2019).  
- Attempted **symbolic inverse dynamics** for the 7-DOF Panda using SymPy.
- Stabilization via Cholesky-based positive-definite mass matrices and diagonal damping.
- When instability persisted, transitioned to a **Residual RL** approach — combining classical inverse dynamics with a learned correction policy.

> This hybrid method proved empirically more stable than symbolic dynamics under Gazebo simulation.

## Implementation Details

### Learning Process
- SAC (γ=0.99, α=0.2, τ=0.005) with replay buffer = 1e5.
- Residual controller trained concurrently for low-level force adaptation.
- Real-time control loop: 0.35s.
- Training over 20,000 episodes (~8 hours simulated time).

### State Representation
Feature Group | Dimension | Description
--- | ---: | ---
Joint Positions | 7 | Proprioception
Joint Velocities | 7 | Dynamics context
Object Detections | 20 | (x, y, height, color, confidence) × 4 objects
Task Context | 8 | Current phase, gripper, target state

## Results — What Worked and What Didn't

### Quantitative Outcomes
Metric | Value
--- | ---
Training Episodes | 20,000
Success Rate | ~30%
Control Step | 0.35s
Reward Stability | Oscillatory early convergence; stable after 12k episodes

### Observations
- **Symbolic Lagrangian instability:** Full analytic inverse dynamics for 7-DOF caused frequent NaN gradients unless regularized.
- **Residual policy stabilized training:** Adding a residual learner on top of the nominal controller allowed smoother convergence.
- **Gazebo's limited contact fidelity** required extensive domain randomization to avoid overfitting to simulation physics.
- **Data efficiency gains were modest** due to low-fidelity contact modeling.

## Limitations & Lessons Learned

1. **Algebraic Intractability:**  
   Encoding full symbolic Lagrangian equations for a 7-DOF arm led to numerical instability unless strongly regularized — diminishing theoretical benefits.

2. **Simulator Fidelity:**  
   Gazebo's contact and friction models caused inconsistencies and required thousands of simulations to achieve robustness.  
   Higher-fidelity alternatives like MuJoCo or Drake are recommended.

3. **Sim-to-Real Gap:**  
   Contact-rich manipulation suffered due to actuator latency and simplified physics, echoing known limitations in literature.

4. **VLA (Vision-Language-Action) Constraints:**  
   Recent VLA models offer generalizable manipulation but demand large-scale data and GPUs, currently impractical for small-scale research setups.

## Reproducibility — Lagrangian Experiments

```bash
# Run isolated joint identification experiment
python3 scripts/deep_lagrangian_experiments.py --mode id_test --episodes 200

# Residual controller training
python3 scripts/residual_controller.py --train --episodes 1000
```

Key Hyperparameters:
Parameter | Value
--- | ---
Lagrangian Regularizer (λ_M) | 1e-3
Actor/Critic Learning Rate | 3e-4
Replay Buffer | 100,000
Residual Controller LR | 3e-4

Logs: logs/lagrangian/
Models: models/pick_place_sac_*

## Recommended Next Steps

- **Improve physics fidelity:** Port to MuJoCo or Drake for stable contact dynamics.  
- **Structured + residual hybrid:** Combine Lagrangian networks (for inertia) with residual policies for robustness.  
- **Safe RL:** Introduce safety constraints into SAC loss to prevent unstable transitions.  
- **Edge inference:** Explore lightweight controllers for embedded deployment.

Training reward curve (jagged, long-horizon training):
![Training Reward Curve](docs/media/sac_timestep_jagged.png)

## Visuals

Gazebo Simulation Overview:
![Gazebo Pick-and-Place Overview](docs/media/pick_and_place_overview.png)

## Educational & Research Value

This project showcases:
- Integration of **Reinforcement Learning** with **Classical Control Theory**.
- **Physics-informed learning** via Lagrangian priors.
- Real-world challenges of **sim-to-real transfer** and **data efficiency**.
- Hands-on experience with **ROS**, **Gazebo**, and **PyTorch** for robotic learning.

## References

1. M. Lutter et al., Deep Lagrangian Networks: Using Physics as Model Prior, ICLR 2019.
2. T. Johannink et al., Residual Reinforcement Learning for Robot Control, RSS 2019.
3. B. Acosta et al., Validating Robotics Simulators: Impacts & Contact Fidelity, IEEE RA-L, 2022.
4. P. Handa et al., Physics-Informed RL for Robotic Manipulation, 2023.
5. A. Driess et al., PaLM-E: An Embodied Multimodal Language Model, 2023 (VLA reference).
6. MuJoCo vs Gazebo contact modeling studies, 2021–2023.

## Author
Rahul Bouri
LinkedIn: https://www.linkedin.com/in/rahulbouri/
GitHub: https://github.com/rahulbouri
