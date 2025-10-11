#!/usr/bin/env python3

"""
Residual Controller for Franka Panda
Implements hybrid classical + learned control combining inverse dynamics with residual policies

Based on: Johannink et al., "Residual Reinforcement Learning for Robot Control", RSS 2019
Author: Physics-Informed RL Implementation
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import rospy
import logging
from typing import Tuple, Optional, Dict
import argparse
import os

from lagrangian_utils import FrankaLagrangianDynamics


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualPolicy(nn.Module):
    """
    Neural network policy for learning residual corrections to classical control
    """
    
    def __init__(self, state_dim: int = 21, action_dim: int = 7, hidden_dim: int = 128):
        """
        Initialize residual policy network
        
        Args:
            state_dim: Input dimension (joint states + task context)
            action_dim: Output dimension (residual torques)
            hidden_dim: Hidden layer dimension
        """
        super(ResidualPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Residual policy network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Bounded output for safety
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Residual policy initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute residual torques
        
        Args:
            state: Input state (batch_size, state_dim)
            
        Returns:
            residual_torques: Residual torques (batch_size, action_dim)
        """
        
        residual_torques = self.network(state)
        
        # Scale residual torques to reasonable range
        max_residual = 20.0  # Maximum residual torque
        residual_torques = residual_torques * max_residual
        
        return residual_torques


class ResidualController:
    """
    Residual controller combining classical inverse dynamics with learned residual corrections
    """
    
    def __init__(self, state_dim: int = 21, action_dim: int = 7):
        """
        Initialize residual controller
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
        """
        
        # Classical controller (inverse dynamics)
        self.classical_dynamics = FrankaLagrangianDynamics()
        
        # Residual policy
        self.residual_policy = ResidualPolicy(state_dim, action_dim)
        
        # Training parameters
        self.learning_rate = 3e-4
        self.batch_size = 64
        self.num_epochs = 100
        
        # Optimizer for residual policy
        self.optimizer = optim.Adam(self.residual_policy.parameters(), lr=self.learning_rate)
        
        # Safety parameters
        self.max_torque = 150.0  # Franka Panda torque limits
        self.residual_weight = 0.1  # Weight for residual corrections
        
        # Training history
        self.training_history = {
            'epoch': [],
            'total_loss': [],
            'classical_loss': [],
            'residual_loss': []
        }
        
        logger.info("Residual controller initialized")
    
    def compute_classical_torques(self, q: np.ndarray, qdot: np.ndarray, 
                                 q_target: np.ndarray, qdot_target: np.ndarray) -> np.ndarray:
        """
        Compute classical torques using inverse dynamics
        
        Args:
            q: Current joint positions
            qdot: Current joint velocities
            q_target: Target joint positions
            qdot_target: Target joint velocities
            
        Returns:
            tau_classical: Classical torques
        """
        
        # Compute desired accelerations using PD control
        kp = 100.0  # Position gain
        kd = 10.0   # Velocity gain
        
        qddot_desired = kp * (q_target - q) + kd * (qdot_target - qdot)
        
        # Clamp desired accelerations
        qddot_desired = np.clip(qddot_desired, -20.0, 20.0)
        
        # Compute torques using inverse dynamics
        try:
            tau_classical = self.classical_dynamics.compute_inverse_dynamics(
                q, qdot, qddot_desired
            )
        except Exception as e:
            logger.warning(f"Classical dynamics failed: {e}")
            # Fallback to simple PD control
            tau_classical = kp * (q_target - q) + kd * (qdot_target - qdot)
        
        # Apply torque limits
        tau_classical = np.clip(tau_classical, -self.max_torque, self.max_torque)
        
        return tau_classical
    
    def compute_residual_torques(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute residual torques using learned policy
        
        Args:
            state: Current state
            
        Returns:
            residual_torques: Residual torques
        """
        
        with torch.no_grad():
            residual_torques = self.residual_policy(state)
        
        return residual_torques
    
    def compute_total_torques(self, q: np.ndarray, qdot: np.ndarray,
                             q_target: np.ndarray, qdot_target: np.ndarray,
                             task_context: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute total torques combining classical and residual components
        
        Args:
            q: Current joint positions
            qdot: Current joint velocities
            q_target: Target joint positions
            qdot_target: Target joint velocities
            task_context: Task context (gripper state, phase, etc.)
            
        Returns:
            tau_total: Total torques
            tau_classical: Classical torques
            tau_residual: Residual torques
        """
        
        # Compute classical torques
        tau_classical = self.compute_classical_torques(q, qdot, q_target, qdot_target)
        
        # Prepare state for residual policy
        state = np.concatenate([q, qdot, q_target, qdot_target, task_context])
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Compute residual torques
        tau_residual = self.compute_residual_torques(state_tensor).numpy().flatten()
        
        # Combine classical and residual torques
        tau_total = tau_classical + self.residual_weight * tau_residual
        
        # Apply final torque limits
        tau_total = np.clip(tau_total, -self.max_torque, self.max_torque)
        
        return tau_total, tau_classical, tau_residual
    
    def train_residual_policy(self, training_data: Dict[str, np.ndarray]):
        """
        Train residual policy on collected data
        
        Args:
            training_data: Dictionary containing training data
                - 'states': Input states
                - 'tau_target': Target torques
                - 'tau_classical': Classical torques
        """
        
        logger.info("Training residual policy...")
        
        states = torch.tensor(training_data['states'], dtype=torch.float32)
        tau_target = torch.tensor(training_data['tau_target'], dtype=torch.float32)
        tau_classical = torch.tensor(training_data['tau_classical'], dtype=torch.float32)
        
        # Create data loader
        dataset = TensorDataset(states, tau_target, tau_classical)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_total_loss = 0.0
            epoch_classical_loss = 0.0
            epoch_residual_loss = 0.0
            num_batches = 0
            
            for batch_states, batch_tau_target, batch_tau_classical in dataloader:
                self.optimizer.zero_grad()
                
                # Compute residual torques
                tau_residual = self.residual_policy(batch_states)
                
                # Compute total torques
                tau_total = batch_tau_classical + self.residual_weight * tau_residual
                
                # Compute losses
                total_loss = nn.MSELoss()(tau_total, batch_tau_target)
                classical_loss = nn.MSELoss()(batch_tau_classical, batch_tau_target)
                residual_loss = nn.MSELoss()(tau_residual, (batch_tau_target - batch_tau_classical) / self.residual_weight)
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.residual_policy.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Accumulate losses
                epoch_total_loss += total_loss.item()
                epoch_classical_loss += classical_loss.item()
                epoch_residual_loss += residual_loss.item()
                num_batches += 1
            
            # Record training history
            avg_total_loss = epoch_total_loss / num_batches
            avg_classical_loss = epoch_classical_loss / num_batches
            avg_residual_loss = epoch_residual_loss / num_batches
            
            self.training_history['epoch'].append(epoch)
            self.training_history['total_loss'].append(avg_total_loss)
            self.training_history['classical_loss'].append(avg_classical_loss)
            self.training_history['residual_loss'].append(avg_residual_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Total={avg_total_loss:.4f}, "
                          f"Classical={avg_classical_loss:.4f}, Residual={avg_residual_loss:.4f}")
        
        logger.info("Residual policy training completed!")
    
    def generate_training_data(self, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Generate training data for residual policy
        
        Args:
            num_samples: Number of training samples
            
        Returns:
            training_data: Dictionary containing training data
        """
        
        logger.info(f"Generating {num_samples} training samples...")
        
        states = []
        tau_targets = []
        tau_classicals = []
        
        for i in range(num_samples):
            # Generate random joint states
            q = np.random.uniform(-np.pi, np.pi, 7)
            qdot = np.random.uniform(-2.0, 2.0, 7)
            q_target = np.random.uniform(-np.pi, np.pi, 7)
            qdot_target = np.random.uniform(-2.0, 2.0, 7)
            task_context = np.random.uniform(-1.0, 1.0, 3)  # Phase, gripper, progress
            
            # Compute classical torques
            tau_classical = self.compute_classical_torques(q, qdot, q_target, qdot_target)
            
            # Generate target torques (classical + noise + correction)
            noise = np.random.normal(0, 5.0, 7)  # Simulate modeling errors
            correction = np.random.normal(0, 10.0, 7)  # Simulate learned corrections
            tau_target = tau_classical + noise + correction
            
            # Clamp target torques
            tau_target = np.clip(tau_target, -self.max_torque, self.max_torque)
            
            # Prepare state
            state = np.concatenate([q, qdot, q_target, qdot_target, task_context])
            
            states.append(state)
            tau_targets.append(tau_target)
            tau_classicals.append(tau_classical)
        
        training_data = {
            'states': np.array(states),
            'tau_target': np.array(tau_targets),
            'tau_classical': np.array(tau_classicals)
        }
        
        logger.info(f"Training data generation completed. Shape: {training_data['states'].shape}")
        
        return training_data
    
    def save_model(self, filepath: str):
        """Save residual controller model"""
        
        torch.save({
            'residual_policy_state_dict': self.residual_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'residual_weight': self.residual_weight
            }
        }, filepath)
        
        logger.info(f"Residual controller model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load residual controller model"""
        
        checkpoint = torch.load(filepath)
        self.residual_policy.load_state_dict(checkpoint['residual_policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Residual controller model loaded from {filepath}")


def test_residual_controller():
    """Test function for residual controller"""
    
    logger.info("Testing residual controller...")
    
    # Initialize controller
    controller = ResidualController()
    
    # Generate test data
    training_data = controller.generate_training_data(num_samples=1000)
    
    # Train residual policy
    controller.train_residual_policy(training_data)
    
    # Test on new data
    q = np.random.uniform(-np.pi, np.pi, 7)
    qdot = np.random.uniform(-2.0, 2.0, 7)
    q_target = np.random.uniform(-np.pi, np.pi, 7)
    qdot_target = np.random.uniform(-2.0, 2.0, 7)
    task_context = np.array([0.5, 1.0, 0.8])  # Phase, gripper, progress
    
    tau_total, tau_classical, tau_residual = controller.compute_total_torques(
        q, qdot, q_target, qdot_target, task_context
    )
    
    logger.info(f"Test joint positions: {q}")
    logger.info(f"Test joint velocities: {qdot}")
    logger.info(f"Classical torques: {tau_classical}")
    logger.info(f"Residual torques: {tau_residual}")
    logger.info(f"Total torques: {tau_total}")
    
    logger.info("Residual controller test completed!")


def main():
    """Main function for residual controller training"""
    
    parser = argparse.ArgumentParser(description='Residual Controller Training')
    parser.add_argument('--train', action='store_true',
                       help='Train residual controller')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--model_path', type=str, default='models/residual_controller.pth',
                       help='Path to save/load model')
    
    args = parser.parse_args()
    
    logger.info("Starting residual controller training...")
    
    try:
        # Initialize controller
        controller = ResidualController()
        
        if args.train:
            # Generate training data
            training_data = controller.generate_training_data(num_samples=args.episodes)
            
            # Train residual policy
            controller.train_residual_policy(training_data)
            
            # Save model
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            controller.save_model(args.model_path)
            
            logger.info("Residual controller training completed!")
        
        else:
            # Test mode
            test_residual_controller()
        
    except Exception as e:
        logger.error(f"Residual controller training failed: {e}")
        raise


if __name__ == "__main__":
    main()
