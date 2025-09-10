#!/usr/bin/env python3
"""
Enhanced Lagrangian Neural Network for 6-DOF Manipulator Control
Implements proper physics-based dynamics modeling with constraint awareness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev
import numpy as np

class EnhancedLNNManipulator(nn.Module):
    """
    Enhanced Lagrangian Neural Network for 6-DOF manipulator dynamics.
    Implements proper physics-based modeling with joint constraints.
    """
    
    def __init__(self, n_joints=6, obs_size=12, action_size=6, dt=0.01, device='cpu'):
        super(EnhancedLNNManipulator, self).__init__()
        
        self.n_joints = n_joints
        self.obs_size = obs_size
        self.action_size = action_size
        self.dt = dt
        self.device = device
        
        # Input size for trigonometric transformation (6 joints -> 12 trig features)
        input_size = 2 * n_joints  # cos(q1), sin(q1), cos(q2), sin(q2), ...
        
        # Output size for Cholesky decomposition of mass matrix
        # For 6x6 matrix: 6*(6+1)/2 = 21 elements
        out_L = int(n_joints * (n_joints + 1) / 2)
        
        # Mass matrix (inertia) network
        self.fc1_L = nn.Linear(input_size, 128)
        self.fc2_L = nn.Linear(128, 128)
        self.fc3_L = nn.Linear(128, out_L)
        
        # Potential energy network (gravitational + elastic)
        self.fc1_V = nn.Linear(input_size, 128)
        self.fc2_V = nn.Linear(128, 128)
        self.fc3_V = nn.Linear(128, 1)
        
        # Joint limits and constraints
        self.joint_limits = {
            'lower': torch.tensor([-6.28, -6.28, -3.14, -6.28, -6.28, -6.28], device=device),
            'upper': torch.tensor([6.28, 6.28, 3.14, 6.28, 6.28, 6.28], device=device),
            'effort': torch.tensor([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], device=device)
        }
        
        # Safety margins for constraint handling
        self.safety_margin = 0.1
        
    def trig_transform_q(self, q):
        """
        Transform joint angles to trigonometric representation.
        For 6-DOF: [cos(q1), sin(q1), cos(q2), sin(q2), ..., cos(q6), sin(q6)]
        """
        batch_size = q.shape[0]
        trig_features = torch.zeros(batch_size, 2 * self.n_joints, device=q.device)
        
        for i in range(self.n_joints):
            trig_features[:, 2*i] = torch.cos(q[:, i])
            trig_features[:, 2*i + 1] = torch.sin(q[:, i])
            
        return trig_features
    
    def inverse_trig_transform(self, trig_q):
        """
        Convert trigonometric representation back to joint angles.
        """
        q = torch.zeros(trig_q.shape[0], self.n_joints, device=trig_q.device)
        
        for i in range(self.n_joints):
            q[:, i] = torch.atan2(trig_q[:, 2*i + 1], trig_q[:, 2*i])
            
        return q
    
    def compute_L(self, trig_q):
        """
        Compute Cholesky decomposition L of mass matrix M = L L^T.
        Ensures positive definiteness of mass matrix.
        """
        y1_L = F.softplus(self.fc1_L(trig_q))
        y2_L = F.softplus(self.fc2_L(y1_L))
        y_L = F.softplus(self.fc3_L(y2_L))  # Ensure positive values
        
        batch_size = y_L.shape[0]
        L = torch.zeros(batch_size, self.n_joints, self.n_joints, device=y_L.device)
        
        # Build lower triangular matrix L
        idx = 0
        for i in range(self.n_joints):
            for j in range(i + 1):
                L[:, i, j] = y_L[:, idx]
                idx += 1
                
        return L
    
    def get_mass_matrix(self, q):
        """
        Get mass matrix M = L L^T from Cholesky decomposition.
        """
        trig_q = self.trig_transform_q(q)
        L = self.compute_L(trig_q)
        M = torch.bmm(L, L.transpose(1, 2))
        return M
    
    def get_potential_energy(self, q):
        """
        Compute potential energy (gravitational + elastic).
        """
        trig_q = self.trig_transform_q(q)
        y1_V = F.softplus(self.fc1_V(trig_q))
        y2_V = F.softplus(self.fc2_V(y1_V))
        V = self.fc3_V(y2_V).squeeze()
        return V
    
    def get_kinetic_energy(self, q, qdot):
        """
        Compute kinetic energy T = 0.5 * qdot^T * M(q) * qdot.
        """
        M = self.get_mass_matrix(q)
        T = 0.5 * torch.bmm(qdot.unsqueeze(1), torch.bmm(M, qdot.unsqueeze(2))).squeeze()
        return T
    
    def get_coriolis_centrifugal(self, q, qdot):
        """
        Compute Coriolis and centrifugal forces using Christoffel symbols.
        C(q, qdot) * qdot = dM/dq * qdot - 0.5 * qdot^T * dM/dq * qdot
        """
        # Compute gradient of mass matrix
        dM_dq, M = jacrev(self.get_mass_matrix, has_aux=True)(q)
        
        # Reshape for Einstein summation
        dM_dq = dM_dq.view(q.shape[0], self.n_joints, self.n_joints, self.n_joints)
        
        # Coriolis matrix: C_ij = sum_k (dM_ik/dq_j - 0.5 * dM_kj/dq_i) * qdot_k
        C = torch.zeros_like(M)
        for i in range(self.n_joints):
            for j in range(self.n_joints):
                for k in range(self.n_joints):
                    C[:, i, j] += (dM_dq[:, i, k, j] - 0.5 * dM_dq[:, k, j, i]) * qdot[:, k]
        
        # Coriolis forces
        c = torch.bmm(C, qdot.unsqueeze(2)).squeeze(2)
        return c
    
    def get_gravity_forces(self, q):
        """
        Compute gravitational forces dV/dq.
        """
        dV_dq = jacrev(self.get_potential_energy)(q)
        return dV_dq
    
    def get_joint_constraints(self, q):
        """
        Compute joint constraint forces using barrier functions.
        """
        constraint_forces = torch.zeros_like(q)
        
        for i in range(self.n_joints):
            lower = self.joint_limits['lower'][i] + self.safety_margin
            upper = self.joint_limits['upper'][i] - self.safety_margin
            
            # Barrier function: B(q) = -log((q - q_min)(q_max - q))
            if torch.any(q[:, i] <= lower) or torch.any(q[:, i] >= upper):
                # At limits, apply maximum constraint force
                constraint_forces[:, i] = torch.where(
                    q[:, i] <= lower,
                    self.joint_limits['effort'][i],  # Push away from lower limit
                    torch.where(
                        q[:, i] >= upper,
                        -self.joint_limits['effort'][i],  # Push away from upper limit
                        torch.zeros_like(q[:, i])
                    )
                )
            else:
                # Barrier gradient: dB/dq = 1/(q - q_min) - 1/(q_max - q)
                barrier_grad = 1.0 / (q[:, i] - lower) - 1.0 / (upper - q[:, i])
                constraint_forces[:, i] = -5.0 * barrier_grad  # Barrier gain
        
        return constraint_forces
    
    def get_acceleration(self, q, qdot, tau):
        """
        Compute joint accelerations using Lagrangian dynamics:
        M(q) * qddot + C(q, qdot) * qdot + G(q) + F_constraint = tau
        
        Where:
        - M(q): mass matrix
        - C(q, qdot): Coriolis/centrifugal matrix
        - G(q): gravitational forces
        - F_constraint: joint constraint forces
        - tau: applied torques
        """
        # Get mass matrix
        M = self.get_mass_matrix(q)
        
        # Get Coriolis and centrifugal forces
        c = self.get_coriolis_centrifugal(q, qdot)
        
        # Get gravitational forces
        g = self.get_gravity_forces(q)
        
        # Get constraint forces
        f_constraint = self.get_joint_constraints(q)
        
        # Solve for acceleration: M * qddot = tau - c - g - f_constraint
        rhs = tau - c - g - f_constraint
        
        # Solve linear system M * qddot = rhs
        try:
            qddot = torch.linalg.solve(M, rhs.unsqueeze(2)).squeeze(2)
        except:
            # Fallback to pseudo-inverse if singular
            qddot = torch.bmm(torch.linalg.pinv(M), rhs.unsqueeze(2)).squeeze(2)
        
        return qddot
    
    def rk4_integration(self, q, qdot, tau):
        """
        Fourth-order Runge-Kutta integration for better accuracy.
        """
        # k1 = f(t, y)
        k1_q = qdot
        k1_qdot = self.get_acceleration(q, qdot, tau)
        
        # k2 = f(t + dt/2, y + dt*k1/2)
        k2_q = qdot + 0.5 * self.dt * k1_qdot
        k2_qdot = self.get_acceleration(q + 0.5 * self.dt * k1_q, k2_q, tau)
        
        # k3 = f(t + dt/2, y + dt*k2/2)
        k3_q = qdot + 0.5 * self.dt * k2_qdot
        k3_qdot = self.get_acceleration(q + 0.5 * self.dt * k2_q, k3_q, tau)
        
        # k4 = f(t + dt, y + dt*k3)
        k4_q = qdot + self.dt * k3_qdot
        k4_qdot = self.get_acceleration(q + self.dt * k3_q, k4_q, tau)
        
        # Update state
        q_new = q + (self.dt / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
        qdot_new = qdot + (self.dt / 6.0) * (k1_qdot + 2*k2_qdot + 2*k3_qdot + k4_qdot)
        
        return q_new, qdot_new
    
    def forward(self, obs, action):
        """
        Forward pass: predict next state given current state and action.
        obs: [q1, q2, q3, q4, q5, q6, qdot1, qdot2, qdot3, qdot4, qdot5, qdot6]
        action: [tau1, tau2, tau3, tau4, tau5, tau6]
        """
        # Extract joint positions and velocities
        q = obs[:, :self.n_joints]
        qdot = obs[:, self.n_joints:]
        
        # Scale action to torque limits
        tau = action * self.joint_limits['effort']
        
        # Integrate dynamics
        q_new, qdot_new = self.rk4_integration(q, qdot, tau)
        
        # Apply joint limits (clamping)
        q_new = torch.clamp(q_new, 
                           self.joint_limits['lower'], 
                           self.joint_limits['upper'])
        
        # Combine new state
        obs_new = torch.cat([q_new, qdot_new], dim=1)
        
        return obs_new
    
    def get_energy(self, q, qdot):
        """
        Get total energy (kinetic + potential).
        """
        T = self.get_kinetic_energy(q, qdot)
        V = self.get_potential_energy(q)
        return T + V
    
    def get_constraint_violations(self, q):
        """
        Check for joint constraint violations.
        """
        violations = []
        for i in range(self.n_joints):
            if torch.any(q[:, i] < self.joint_limits['lower'][i]) or \
               torch.any(q[:, i] > self.joint_limits['upper'][i]):
                violations.append(f"Joint {i} out of bounds")
        return violations

class ConstraintAwareRewardFunction:
    """
    Comprehensive reward function for manipulator reach/pregrasp tasks.
    Implements accuracy, efficiency, and energy optimization.
    """
    
    def __init__(self, joint_names, target_position, max_episode_steps=1000):
        self.joint_names = joint_names
        self.target_position = np.array(target_position)  # [x, y, z] target position
        self.max_episode_steps = max_episode_steps
        
        # Reward weights (tunable hyperparameters)
        self.weights = {
            'accuracy': 10.0,      # Position accuracy reward
            'efficiency': 1.0,     # Time efficiency reward
            'energy': 0.01,        # Energy efficiency reward
            'smoothness': 0.1,     # Motion smoothness reward
            'constraint': 5.0,     # Constraint violation penalty
            'collision': 10.0      # Collision penalty
        }
        
        # Target tolerance
        self.position_tolerance = 0.05  # 5cm tolerance
        self.orientation_tolerance = 0.1  # 0.1 rad tolerance
        
        # Energy tracking
        self.energy_history = []
        self.torque_history = []
        
    def get_end_effector_position(self, joint_positions):
        """
        Compute end-effector position from joint positions.
        This is a simplified forward kinematics - in practice, you'd use
        the actual manipulator's forward kinematics.
        """
        # Simplified forward kinematics for 6-DOF manipulator
        # In practice, use proper DH parameters or URDF-based FK
        
        # Approximate end-effector position based on joint angles
        # This is a placeholder - replace with actual FK
        x = 0.5 * np.cos(joint_positions[0]) * np.cos(joint_positions[1]) + \
            0.3 * np.cos(joint_positions[0]) * np.cos(joint_positions[1] + joint_positions[2])
        
        y = 0.5 * np.sin(joint_positions[0]) * np.cos(joint_positions[1]) + \
            0.3 * np.sin(joint_positions[0]) * np.cos(joint_positions[1] + joint_positions[2])
        
        z = 0.5 * np.sin(joint_positions[1]) + \
            0.3 * np.sin(joint_positions[1] + joint_positions[2]) + 0.75  # Base height
        
        return np.array([x, y, z])
    
    def compute_accuracy_reward(self, joint_positions):
        """
        Compute accuracy reward based on end-effector position error.
        """
        ee_position = self.get_end_effector_position(joint_positions)
        position_error = np.linalg.norm(ee_position - self.target_position)
        
        # Exponential reward: higher when closer to target
        if position_error < self.position_tolerance:
            # Success bonus
            accuracy_reward = 100.0
        else:
            # Distance-based reward
            accuracy_reward = np.exp(-position_error / 0.1)
        
        return accuracy_reward, position_error
    
    def compute_efficiency_reward(self, step_count):
        """
        Compute time efficiency reward.
        Encourages reaching target quickly.
        """
        # Linear decay reward
        efficiency_reward = (self.max_episode_steps - step_count) / self.max_episode_steps
        
        return efficiency_reward
    
    def compute_energy_reward(self, torques, velocities):
        """
        Compute energy efficiency reward.
        Penalizes high torques and encourages smooth motion.
        """
        # Power consumption: P = |tau * qdot|
        power = np.abs(torques * velocities)
        total_power = np.sum(power)
        
        # Energy efficiency reward (inverse of power)
        energy_reward = 1.0 / (1.0 + total_power)
        
        return energy_reward, total_power
    
    def compute_smoothness_reward(self, torques, prev_torques):
        """
        Compute motion smoothness reward.
        Penalizes abrupt changes in torque.
        """
        if prev_torques is None:
            return 0.0
        
        # Torque change penalty
        torque_change = np.linalg.norm(torques - prev_torques)
        smoothness_reward = np.exp(-torque_change / 10.0)
        
        return smoothness_reward
    
    def compute_constraint_penalty(self, joint_positions, joint_limits):
        """
        Compute constraint violation penalty.
        """
        penalty = 0.0
        violations = []
        
        for i, joint in enumerate(self.joint_names):
            pos = joint_positions[i]
            limits = joint_limits[joint]
            
            # Position limit violation
            if pos < limits['lower'] or pos > limits['upper']:
                violation = min(abs(pos - limits['lower']), abs(pos - limits['upper']))
                penalty += violation * 10.0
                violations.append(f"{joint}: {pos:.3f} outside [{limits['lower']:.3f}, {limits['upper']:.3f}]")
        
        return penalty, violations
    
    def compute_collision_penalty(self, joint_positions):
        """
        Compute collision penalty.
        Simplified collision detection - in practice, use proper collision checking.
        """
        # Placeholder collision detection
        # In practice, use collision checking with environment objects
        
        ee_position = self.get_end_effector_position(joint_positions)
        
        # Check if end-effector is below table surface (collision)
        if ee_position[2] < 0.75:  # Table height
            return 50.0, "Collision with table"
        
        # Check if end-effector is too far from workspace
        workspace_radius = 1.0
        if np.linalg.norm(ee_position[:2]) > workspace_radius:
            return 20.0, "Outside workspace"
        
        return 0.0, None
    
    def compute_reward(self, joint_positions, joint_velocities, torques, 
                      step_count, joint_limits, prev_torques=None):
        """
        Compute comprehensive reward function.
        """
        # Accuracy reward
        accuracy_reward, position_error = self.compute_accuracy_reward(joint_positions)
        
        # Efficiency reward
        efficiency_reward = self.compute_efficiency_reward(step_count)
        
        # Energy reward
        energy_reward, total_power = self.compute_energy_reward(torques, joint_velocities)
        
        # Smoothness reward
        smoothness_reward = self.compute_smoothness_reward(torques, prev_torques)
        
        # Constraint penalty
        constraint_penalty, violations = self.compute_constraint_penalty(joint_positions, joint_limits)
        
        # Collision penalty
        collision_penalty, collision_info = self.compute_collision_penalty(joint_positions)
        
        # Total reward
        total_reward = (
            self.weights['accuracy'] * accuracy_reward +
            self.weights['efficiency'] * efficiency_reward +
            self.weights['energy'] * energy_reward +
            self.weights['smoothness'] * smoothness_reward -
            self.weights['constraint'] * constraint_penalty -
            self.weights['collision'] * collision_penalty
        )
        
        # Store energy history
        self.energy_history.append(total_power)
        self.torque_history.append(torques.copy())
        
        # Reward breakdown
        reward_info = {
            'accuracy_reward': accuracy_reward,
            'efficiency_reward': efficiency_reward,
            'energy_reward': energy_reward,
            'smoothness_reward': smoothness_reward,
            'constraint_penalty': constraint_penalty,
            'collision_penalty': collision_penalty,
            'total_reward': total_reward,
            'position_error': position_error,
            'total_power': total_power,
            'violations': violations,
            'collision_info': collision_info
        }
        
        return total_reward, reward_info
    
    def get_episode_summary(self):
        """
        Get summary statistics for the episode.
        """
        if not self.energy_history:
            return {}
        
        return {
            'total_energy': np.sum(self.energy_history),
            'average_power': np.mean(self.energy_history),
            'max_power': np.max(self.energy_history),
            'energy_efficiency': 1.0 / (1.0 + np.sum(self.energy_history)),
            'torque_variance': np.var(self.torque_history, axis=0).tolist()
        }

def test_enhanced_lnn():
    """
    Test the enhanced LNN implementation.
    """
    print("🧪 Testing Enhanced LNN for 6-DOF Manipulator")
    
    # Create LNN
    lnn = EnhancedLNNManipulator(n_joints=6, obs_size=12, action_size=6, dt=0.01)
    
    # Test data
    batch_size = 4
    obs = torch.randn(batch_size, 12)  # [q1-q6, qdot1-qdot6]
    action = torch.randn(batch_size, 6)  # [tau1-tau6]
    
    # Forward pass
    obs_new = lnn(obs, action)
    print(f"✓ Forward pass successful: {obs.shape} -> {obs_new.shape}")
    
    # Test energy conservation
    q = obs[:, :6]
    qdot = obs[:, 6:]
    energy = lnn.get_energy(q, qdot)
    print(f"✓ Energy computation: {energy.shape}")
    
    # Test constraint violations
    violations = lnn.get_constraint_violations(q)
    print(f"✓ Constraint checking: {len(violations)} violations")
    
    print("🎉 Enhanced LNN test completed successfully!")

if __name__ == "__main__":
    test_enhanced_lnn()
