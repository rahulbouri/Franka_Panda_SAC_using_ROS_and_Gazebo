#!/usr/bin/env python3
"""
Lagrangian Neural Network with Joint Constraint Integration
Implements constraint-aware Lagrangian dynamics for RL environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstraintAwareLagrangianNN(nn.Module):
    """
    Lagrangian Neural Network with embedded joint constraints.
    Implements the strategies from web resources for constraint handling.
    """
    
    def __init__(self, n_joints=6, hidden_dim=128, constraint_penalty=1.0):
        super(ConstraintAwareLagrangianNN, self).__init__()
        
        self.n_joints = n_joints
        self.constraint_penalty = constraint_penalty
        
        # Joint limits (from URDF)
        self.joint_limits = torch.tensor([
            [-6.283185307179586, 6.283185307179586],  # shoulder_pan_joint
            [-6.283185307179586, 6.283185307179586],  # shoulder_lift_joint
            [-3.141592653589793, 3.141592653589793],  # elbow_joint
            [-6.283185307179586, 6.283185307179586],  # wrist_1_joint
            [-6.283185307179586, 6.283185307179586],  # wrist_2_joint
            [-6.283185307179586, 6.283185307179586]   # wrist_3_joint
        ], dtype=torch.float32)
        
        # Safety margins
        self.safety_margin = 0.1
        
        # Lagrangian network components
        self.lagrangian_net = nn.Sequential(
            nn.Linear(n_joints * 2, hidden_dim),  # q, q_dot
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Lagrangian L(q, q_dot)
        )
        
        # Constraint barrier network
        self.barrier_net = nn.Sequential(
            nn.Linear(n_joints, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, n_joints)  # Barrier function values
        )
        
        # Adaptive control network
        self.adaptive_net = nn.Sequential(
            nn.Linear(n_joints * 3, hidden_dim),  # q, q_dot, q_desired
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_joints)  # Adaptive control effort
        )
    
    def compute_barrier_function(self, q):
        """Compute barrier function for joint constraints"""
        # B(q) = -log((q - q_min)(q_max - q))
        q_min = self.joint_limits[:, 0] + self.safety_margin
        q_max = self.joint_limits[:, 1] - self.safety_margin
        
        # Check if within safe bounds
        safe_lower = q > q_min
        safe_upper = q < q_max
        safe_bounds = safe_lower & safe_upper
        
        # Compute barrier function
        barrier_values = torch.zeros_like(q)
        valid_mask = safe_bounds
        
        if valid_mask.any():
            q_safe = q[valid_mask]
            q_min_safe = q_min[valid_mask]
            q_max_safe = q_max[valid_mask]
            
            barrier_values[valid_mask] = -torch.log((q_safe - q_min_safe) * (q_max_safe - q_safe))
        
        # Set infinite barrier for unsafe positions
        barrier_values[~valid_mask] = float('inf')
        
        return barrier_values
    
    def compute_barrier_gradient(self, q):
        """Compute gradient of barrier function"""
        q_min = self.joint_limits[:, 0] + self.safety_margin
        q_max = self.joint_limits[:, 1] - self.safety_margin
        
        # Check if within safe bounds
        safe_lower = q > q_min
        safe_upper = q < q_max
        safe_bounds = safe_lower & safe_upper
        
        # Compute gradient: dB/dq = 1/(q - q_min) - 1/(q_max - q)
        gradient = torch.zeros_like(q)
        valid_mask = safe_bounds
        
        if valid_mask.any():
            q_safe = q[valid_mask]
            q_min_safe = q_min[valid_mask]
            q_max_safe = q_max[valid_mask]
            
            gradient[valid_mask] = 1.0 / (q_safe - q_min_safe) - 1.0 / (q_max_safe - q_safe)
        
        # Set infinite gradient for unsafe positions
        gradient[~valid_mask] = float('inf')
        
        return gradient
    
    def compute_lagrangian(self, q, q_dot):
        """Compute Lagrangian L(q, q_dot)"""
        # Concatenate position and velocity
        state = torch.cat([q, q_dot], dim=-1)
        
        # Compute Lagrangian
        L = self.lagrangian_net(state)
        
        return L
    
    def compute_lagrangian_equations(self, q, q_dot, q_ddot):
        """Compute Lagrangian equations of motion"""
        # Enable gradient computation
        q.requires_grad_(True)
        q_dot.requires_grad_(True)
        
        # Compute Lagrangian
        L = self.compute_lagrangian(q, q_dot)
        
        # Compute partial derivatives
        dL_dq = torch.autograd.grad(L, q, create_graph=True, retain_graph=True)[0]
        dL_dqdot = torch.autograd.grad(L, q_dot, create_graph=True, retain_graph=True)[0]
        
        # Compute time derivative of dL/dqdot
        dL_dqdot_dt = torch.autograd.grad(dL_dqdot, q, grad_outputs=q_dot, create_graph=True, retain_graph=True)[0]
        dL_dqdot_dt += torch.autograd.grad(dL_dqdot, q_dot, grad_outputs=q_ddot, create_graph=True, retain_graph=True)[0]
        
        # Lagrangian equations: d/dt(dL/dqdot) - dL/dq = tau
        tau = dL_dqdot_dt - dL_dq
        
        return tau, L
    
    def compute_constraint_aware_control(self, q, q_dot, q_desired, q_dot_desired):
        """Compute constraint-aware control using barrier functions"""
        # Compute barrier function gradient
        barrier_grad = self.compute_barrier_gradient(q)
        
        # Compute adaptive control
        control_input = torch.cat([q, q_dot, q_desired], dim=-1)
        adaptive_effort = self.adaptive_net(control_input)
        
        # Apply barrier function correction
        barrier_correction = -self.constraint_penalty * barrier_grad
        
        # Final control effort
        total_effort = adaptive_effort + barrier_correction
        
        # Clamp to joint effort limits
        effort_limits = torch.tensor([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=torch.float32)
        total_effort = torch.clamp(total_effort, -effort_limits, effort_limits)
        
        return total_effort, barrier_grad
    
    def compute_reward_function(self, q, q_dot, q_desired, q_dot_desired, effort):
        """Compute reward function with constraint penalties"""
        # Position error
        pos_error = torch.norm(q - q_desired)
        
        # Velocity error
        vel_error = torch.norm(q_dot - q_dot_desired)
        
        # Effort penalty
        effort_penalty = torch.norm(effort)
        
        # Constraint violation penalty
        barrier_values = self.compute_barrier_function(q)
        constraint_penalty = torch.sum(barrier_values[torch.isfinite(barrier_values)])
        
        # Total reward (higher is better)
        reward = -(pos_error + 0.1 * vel_error + 0.01 * effort_penalty + constraint_penalty)
        
        return reward, {
            'position_error': pos_error.item(),
            'velocity_error': vel_error.item(),
            'effort_penalty': effort_penalty.item(),
            'constraint_penalty': constraint_penalty.item(),
            'total_reward': reward.item()
        }
    
    def forward(self, q, q_dot, q_desired, q_dot_desired):
        """Forward pass for constraint-aware control"""
        # Compute constraint-aware control
        effort, barrier_grad = self.compute_constraint_aware_control(q, q_dot, q_desired, q_dot_desired)
        
        # Compute reward
        reward, reward_info = self.compute_reward_function(q, q_dot, q_desired, q_dot_desired, effort)
        
        return {
            'effort': effort,
            'barrier_gradient': barrier_grad,
            'reward': reward,
            'reward_info': reward_info
        }

class ConstraintAwareRLEnvironment:
    """
    RL Environment with constraint-aware Lagrangian Neural Network
    """
    
    def __init__(self, n_joints=6):
        self.n_joints = n_joints
        self.lnn = ConstraintAwareLagrangianNN(n_joints)
        
        # Joint limits
        self.joint_limits = torch.tensor([
            [-6.283185307179586, 6.283185307179586],  # shoulder_pan_joint
            [-6.283185307179586, 6.283185307179586],  # shoulder_lift_joint
            [-3.141592653589793, 3.141592653589793],  # elbow_joint
            [-6.283185307179586, 6.283185307179586],  # wrist_1_joint
            [-6.283185307179586, 6.283185307179586],  # wrist_2_joint
            [-6.283185307179586, 6.283185307179586]   # wrist_3_joint
        ], dtype=torch.float32)
    
    def reset(self):
        """Reset environment to random valid state"""
        # Sample random joint positions within limits
        q = torch.zeros(self.n_joints)
        for i in range(self.n_joints):
            lower = self.joint_limits[i, 0] + 0.2  # Safety margin
            upper = self.joint_limits[i, 1] - 0.2
            q[i] = torch.uniform(lower, upper)
        
        q_dot = torch.zeros(self.n_joints)
        
        return q, q_dot
    
    def step(self, q, q_dot, q_desired, q_dot_desired):
        """Execute one step with constraint-aware control"""
        # Compute control action
        with torch.no_grad():
            result = self.lnn(q, q_dot, q_desired, q_dot_desired)
            effort = result['effort']
            reward = result['reward']
            reward_info = result['reward_info']
        
        # Simulate dynamics (simplified)
        # In real implementation, this would integrate with Gazebo
        dt = 0.01
        q_new = q + q_dot * dt
        q_dot_new = q_dot + effort * dt  # Simplified dynamics
        
        # Check for constraint violations
        constraint_violations = self.check_constraints(q_new)
        
        # Done if constraints violated
        done = len(constraint_violations) > 0
        
        return q_new, q_dot_new, reward, done, reward_info
    
    def check_constraints(self, q):
        """Check for constraint violations"""
        violations = []
        for i in range(self.n_joints):
            if q[i] < self.joint_limits[i, 0] or q[i] > self.joint_limits[i, 1]:
                violations.append(f"Joint {i}: {q[i]:.3f} outside [{self.joint_limits[i, 0]:.3f}, {self.joint_limits[i, 1]:.3f}]")
        
        return violations

def test_constraint_aware_lnn():
    """Test the constraint-aware Lagrangian Neural Network"""
    print("🧪 Testing Constraint-Aware Lagrangian Neural Network")
    
    # Create network
    lnn = ConstraintAwareLagrangianNN(n_joints=6)
    
    # Test data
    q = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    q_dot = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    q_desired = torch.tensor([1.0, 0.5, -1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    q_dot_desired = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    
    # Test forward pass
    result = lnn(q, q_dot, q_desired, q_dot_desired)
    
    print(f"✅ Effort: {result['effort']}")
    print(f"✅ Reward: {result['reward']:.4f}")
    print(f"✅ Reward info: {result['reward_info']}")
    
    # Test constraint violation
    q_violation = torch.tensor([7.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # Beyond limit
    barrier_grad = lnn.compute_barrier_gradient(q_violation)
    print(f"⚠️  Barrier gradient for violation: {barrier_grad}")
    
    print("🎉 Constraint-aware LNN test completed!")

if __name__ == '__main__':
    test_constraint_aware_lnn()
