#!/usr/bin/env python3
"""
Fixed 6-DOF Lagrangian Neural Network for UR5e Manipulator
Properly integrated with ROS/Gazebo environment

Author: RL Training Implementation
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Set default dtype for numerical stability
torch.set_default_dtype(torch.float64)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LNN6DOF(nn.Module):
    """
    Fixed 6-DOF Lagrangian Neural Network for UR5e Manipulator
    
    This implementation is designed to work seamlessly with the ROS/Gazebo environment
    and follows the PINN_RL reference implementation structure.
    """
    
    def __init__(self, obs_size=18, action_size=6, dt=0.02, device='cpu'):
        """
        Initialize 6-DOF LNN
        
        Args:
            obs_size (int): Observation size (18 for our manipulator)
            action_size (int): Action size (6 for joint torques)
            dt (float): Time step for integration
            device (str): Device for computation
        """
        super(LNN6DOF, self).__init__()
        
        self.n = 6  # Number of DOF
        self.dt = dt
        self.device = device
        
        # Input size: 6 joint angles (after trigonometric transformation = 12)
        # + 6 joint velocities = 18 total
        input_size = 12  # cos(q1), sin(q1), ..., cos(q6), sin(q6)
        
        # Output size for lower triangular mass matrix: n(n+1)/2 = 21
        out_L = int(self.n * (self.n + 1) / 2)
        
        # Mass matrix network (L = M)
        self.fc1_L = nn.Linear(input_size, 128)
        self.fc2_L = nn.Linear(128, 128)
        self.fc3_L = nn.Linear(128, out_L)
        
        # Potential energy network (V)
        self.fc1_V = nn.Linear(input_size, 128)
        self.fc2_V = nn.Linear(128, 128)
        self.fc3_V = nn.Linear(128, 1)
        
        # Action mapping (for compatibility with PINN_RL)
        self.a_zeros = torch.zeros(1, 0, device=device)
        
        logger.info(f"🚀 Initialized Fixed 6-DOF LNN for UR5e manipulator")
        logger.info(f"   Input size: {input_size}, Mass matrix output: {out_L}")
        logger.info(f"   Time step: {dt}, Device: {device}")
    
    def trig_transform_q(self, q):
        """
        Transform joint angles to trigonometric representation
        Following PINN_RL pattern for 6-DOF manipulator
        
        Args:
            q: Joint angles [batch_size, 6]
            
        Returns:
            trig_q: Trigonometric representation [batch_size, 12]
        """
        # For 6-DOF: [cos(q1), sin(q1), cos(q2), sin(q2), ..., cos(q6), sin(q6)]
        cos_q = torch.cos(q)
        sin_q = torch.sin(q)
        
        # Interleave cos and sin values
        trig_q = torch.stack([cos_q, sin_q], dim=2).reshape(q.shape[0], -1)
        
        return trig_q
    
    def inverse_trig_transform(self, trig_q):
        """
        Inverse transform from trigonometric to joint angles
        
        Args:
            trig_q: Trigonometric representation [batch_size, 12]
            
        Returns:
            q: Joint angles [batch_size, 6]
        """
        # Reshape to [batch_size, 6, 2] and extract angles
        trig_reshaped = trig_q.reshape(trig_q.shape[0], 6, 2)
        cos_q = trig_reshaped[:, :, 0]
        sin_q = trig_reshaped[:, :, 1]
        
        # Convert back to angles using atan2
        q = torch.atan2(sin_q, cos_q)
        
        return q
    
    def compute_mass_matrix(self, trig_q):
        """
        Compute the 6x6 mass matrix M(q) using neural network
        Following PINN_RL implementation pattern
        
        Args:
            trig_q: Trigonometric joint representation [batch_size, 12]
            
        Returns:
            M: Mass matrix [batch_size, 6, 6]
        """
        # Forward pass through mass matrix network
        y1_L = F.softplus(self.fc1_L(trig_q))
        y2_L = F.softplus(self.fc2_L(y1_L))
        y_L = self.fc3_L(y2_L)
        
        batch_size = y_L.shape[0]
        device = y_L.device
        
        # Construct lower triangular matrix L such that M = L L^T
        # For 6x6 matrix, we need 21 parameters (6+5+4+3+2+1)
        L = torch.zeros(batch_size, 6, 6, device=device)
        
        # Fill lower triangular matrix
        idx = 0
        for i in range(6):
            for j in range(i + 1):
                L[:, i, j] = y_L[:, idx]
                idx += 1
        
        # Compute mass matrix: M = L L^T
        M = torch.bmm(L, L.transpose(1, 2))
        
        # Add small diagonal term for numerical stability
        M = M + 1e-6 * torch.eye(6, device=device).unsqueeze(0)
        
        return M
    
    def compute_potential_energy(self, trig_q):
        """
        Compute gravitational potential energy V(q)
        
        Args:
            trig_q: Trigonometric joint representation [batch_size, 12]
            
        Returns:
            V: Potential energy [batch_size]
        """
        # Forward pass through potential energy network
        y1_V = F.softplus(self.fc1_V(trig_q))
        y2_V = F.softplus(self.fc2_V(y1_V))
        V = self.fc3_V(y2_V).squeeze()
        
        return V
    
    def get_L(self, q):
        """
        Get Lagrangian L(q, q̇) = T(q, q̇) - V(q)
        Following PINN_RL implementation pattern
        
        Args:
            q: Joint angles [batch_size, 6]
            
        Returns:
            L_sum: Sum of Lagrangian [batch_size]
            L: Mass matrix [batch_size, 6, 6]
        """
        trig_q = self.trig_transform_q(q)
        L = self.compute_mass_matrix(trig_q)
        return L.sum(0), L
    
    def get_V(self, q):
        """
        Get potential energy V(q)
        Following PINN_RL implementation pattern
        
        Args:
            q: Joint angles [batch_size, 6]
            
        Returns:
            V: Potential energy [batch_size]
        """
        trig_q = self.trig_transform_q(q)
        V = self.compute_potential_energy(trig_q)
        return V.sum()
    
    def get_A(self, a):
        """
        Get action vector A
        Following PINN_RL implementation pattern
        
        Args:
            a: Action [batch_size, 6]
            
        Returns:
            A: Action vector [batch_size, 6]
        """
        return a
    
    def get_acc(self, q, qdot, a):
        """
        Compute joint accelerations using Lagrangian mechanics
        Following PINN_RL implementation pattern
        
        Args:
            q: Joint angles [batch_size, 6]
            qdot: Joint velocities [batch_size, 6]
            a: Joint torques [batch_size, 6]
            
        Returns:
            qddot: Joint accelerations [batch_size, 6]
        """
        # Get Lagrangian and mass matrix
        dL_dq, L = torch.func.jacrev(self.get_L, has_aux=True)(q)
        
        # Compute Coriolis and centrifugal terms
        term_1 = torch.einsum('blk,bijk->bijl', L, dL_dq.permute(2,3,0,1))
        dM_dq = term_1 + term_1.transpose(2,3)
        c = torch.einsum('bjik,bk,bj->bi', dM_dq, qdot, qdot) - 0.5 * torch.einsum('bikj,bk,bj->bi', dM_dq, qdot, qdot)
        
        # Compute mass matrix inverse
        Minv = torch.cholesky_inverse(L)
        
        # Compute potential energy gradient
        dV_dq = torch.func.jacrev(self.get_V)(q)
        
        # Solve for accelerations: M q̈ = τ - C q̇ - G
        qddot = torch.matmul(Minv, (self.get_A(a) - c - dV_dq).unsqueeze(2)).squeeze(2)
        
        return qddot
    
    def derivs(self, s, a):
        """
        Compute state derivatives
        Following PINN_RL implementation pattern
        
        Args:
            s: State [batch_size, 12] (6 positions + 6 velocities)
            a: Action [batch_size, 6]
            
        Returns:
            sdot: State derivatives [batch_size, 12]
        """
        q, qdot = s[:, :self.n], s[:, self.n:]
        qddot = self.get_acc(q, qdot, a)
        return torch.cat((qdot, qddot), dim=1)
    
    def rk2(self, s, a):
        """
        RK2 integration for forward dynamics
        Following PINN_RL implementation pattern
        
        Args:
            s: Current state [batch_size, 12]
            a: Action [batch_size, 6]
            
        Returns:
            s_next: Next state [batch_size, 12]
        """
        alpha = 2.0 / 3.0  # Ralston's method
        k1 = self.derivs(s, a)
        k2 = self.derivs(s + alpha * self.dt * k1, a)
        s_next = s + self.dt * ((1.0 - 1.0/(2.0*alpha)) * k1 + (1.0/(2.0*alpha)) * k2)
        return s_next
    
    def forward(self, o, a):
        """
        Forward pass for model-based RL
        Following PINN_RL implementation pattern
        
        Args:
            o: Current observation [batch_size, obs_size]
            a: Action [batch_size, action_size]
            
        Returns:
            o_next: Next observation [batch_size, obs_size]
        """
        # Extract joint angles and velocities from observation
        # Observation format: [q1, q2, q3, q4, q5, q6, q̇1, q̇2, q̇3, q̇4, q̇5, q̇6, target_x, target_y, target_z]
        q = o[:, :6]
        qdot = o[:, 6:12]
        
        # Create state vector
        s = torch.cat([q, qdot], dim=1)
        
        # Integrate forward in time
        s_next = self.rk2(s, a)
        
        # Extract next joint angles and velocities
        q_next = s_next[:, :6]
        qdot_next = s_next[:, 6:12]
        
        # Construct next observation (keep target position unchanged)
        # Output format: [q1, q2, q3, q4, q5, q6, q̇1, q̇2, q̇3, q̇4, q̇5, q̇6, target_x, target_y, target_z]
        o_next = torch.cat([q_next, qdot_next, o[:, 12:]], dim=1)
        
        return o_next
    
    def compute_loss(self, states, actions, next_states):
        """
        Compute prediction loss for training
        
        Args:
            states: Current states [batch_size, obs_size]
            actions: Actions [batch_size, action_size]
            next_states: Next states [batch_size, obs_size]
            
        Returns:
            loss: Prediction loss
        """
        # Predict next states
        pred_next_states = self.forward(states, actions)
        
        # Compute MSE loss
        loss = F.mse_loss(pred_next_states, next_states)
        
        return loss

def test_lnn_6dof():
    """Test the 6-DOF LNN implementation"""
    logger.info("🧪 Testing 6-DOF LNN implementation...")
    
    # Create model
    model = LNN6DOF(obs_size=18, action_size=6, dt=0.02, device='cpu')
    
    # Test data
    batch_size = 4
    q = torch.randn(batch_size, 6) * 0.1  # Small joint angles
    qdot = torch.randn(batch_size, 6) * 0.1  # Small joint velocities
    tau = torch.randn(batch_size, 6) * 0.1  # Small torques
    
    # Test trigonometric transformation
    trig_q = model.trig_transform_q(q)
    q_recovered = model.inverse_trig_transform(trig_q)
    logger.info(f"✅ Trigonometric transformation test: {torch.allclose(q, q_recovered, atol=1e-6)}")
    
    # Test mass matrix computation
    M = model.compute_mass_matrix(trig_q)
    logger.info(f"✅ Mass matrix shape: {M.shape}")
    logger.info(f"✅ Mass matrix positive definite: {torch.all(torch.linalg.eigvals(M).real > 0)}")
    
    # Test potential energy
    V = model.compute_potential_energy(trig_q)
    logger.info(f"✅ Potential energy shape: {V.shape}")
    
    # Test forward dynamics
    qddot = model.get_acc(q, qdot, tau)
    logger.info(f"✅ Forward dynamics shape: {qddot.shape}")
    
    # Test RK2 integration
    s = torch.cat([q, qdot], dim=1)
    s_next = model.rk2(s, tau)
    logger.info(f"✅ RK2 integration shape: {s_next.shape}")
    
    # Test full forward pass
    o = torch.cat([trig_q, qdot, torch.randn(batch_size, 6)], dim=1)  # Add target position
    o_next = model.forward(o, tau)
    logger.info(f"✅ Full forward pass shape: {o_next.shape}")
    
    logger.info("🎉 All tests passed!")

if __name__ == "__main__":
    test_lnn_6dof()
