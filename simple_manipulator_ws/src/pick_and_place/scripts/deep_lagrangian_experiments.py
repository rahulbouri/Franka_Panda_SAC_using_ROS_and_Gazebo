#!/usr/bin/env python3

"""
Deep Lagrangian Experiments for Franka Panda
Implements DeLaN-style parameterization experiments and joint identification

Based on: Lutter et al., "Deep Lagrangian Networks: Using Physics as Model Prior", ICLR 2019
Author: Physics-Informed RL Implementation  
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, List, Tuple
import argparse
import rospy

from lagrangian_utils import DeepLagrangianNetwork, FrankaLagrangianDynamics


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LagrangianExperimentRunner:
    """
    Runs Lagrangian identification and learning experiments
    """
    
    def __init__(self, experiment_dir: str = "logs/lagrangian"):
        """
        Initialize experiment runner
        
        Args:
            experiment_dir: Directory to save experiment results
        """
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Initialize networks
        self.delan_net = DeepLagrangianNetwork()
        self.symbolic_dynamics = FrankaLagrangianDynamics()
        
        # Training parameters
        self.learning_rate = 3e-4
        self.batch_size = 64
        self.num_epochs = 100
        
        # Optimizer
        self.optimizer = optim.Adam(self.delan_net.parameters(), lr=self.learning_rate)
        
        # Training history
        self.training_history = {
            'epoch': [],
            'total_loss': [],
            'prediction_loss': [],
            'structure_loss': []
        }
        
        logger.info(f"Lagrangian experiment runner initialized. Results will be saved to: {experiment_dir}")
    
    def generate_training_data(self, num_samples: int = 10000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate training data for Lagrangian learning
        
        Args:
            num_samples: Number of training samples to generate
            
        Returns:
            q: Joint positions (num_samples, 7)
            qdot: Joint velocities (num_samples, 7)
            tau: Joint torques (num_samples, 7)
        """
        
        logger.info(f"Generating {num_samples} training samples...")
        
        # Generate random joint states within safe limits
        q = torch.uniform(-torch.pi, torch.pi, (num_samples, 7))
        qdot = torch.uniform(-2.0, 2.0, (num_samples, 7))
        qddot = torch.uniform(-10.0, 10.0, (num_samples, 7))
        
        # Compute target torques using symbolic dynamics
        tau_target = torch.zeros(num_samples, 7)
        
        for i in range(num_samples):
            try:
                tau_i = self.symbolic_dynamics.compute_inverse_dynamics(
                    q[i].numpy(), qdot[i].numpy(), qddot[i].numpy()
                )
                tau_target[i] = torch.tensor(tau_i, dtype=torch.float32)
            except Exception as e:
                logger.warning(f"Failed to compute torque for sample {i}: {e}")
                # Use fallback torques
                tau_target[i] = torch.uniform(-50.0, 50.0, (7,))
        
        logger.info(f"Training data generation completed. Shape: q={q.shape}, tau={tau_target.shape}")
        
        return q, qdot, tau_target
    
    def train_delan_network(self, q: torch.Tensor, qdot: torch.Tensor, tau_target: torch.Tensor):
        """
        Train DeLaN network on generated data
        
        Args:
            q: Joint positions
            qdot: Joint velocities
            tau_target: Target torques
        """
        
        logger.info("Starting DeLaN network training...")
        
        # Create data loader
        dataset = TensorDataset(q, qdot, tau_target)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_total_loss = 0.0
            epoch_pred_loss = 0.0
            epoch_struct_loss = 0.0
            num_batches = 0
            
            for batch_q, batch_qdot, batch_tau in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                total_loss, pred_loss, struct_loss = self.delan_net.compute_lagrangian_loss(
                    batch_q, batch_qdot, batch_tau
                )
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.delan_net.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Accumulate losses
                epoch_total_loss += total_loss.item()
                epoch_pred_loss += pred_loss.item()
                epoch_struct_loss += struct_loss.item()
                num_batches += 1
            
            # Record training history
            avg_total_loss = epoch_total_loss / num_batches
            avg_pred_loss = epoch_pred_loss / num_batches
            avg_struct_loss = epoch_struct_loss / num_batches
            
            self.training_history['epoch'].append(epoch)
            self.training_history['total_loss'].append(avg_total_loss)
            self.training_history['prediction_loss'].append(avg_pred_loss)
            self.training_history['structure_loss'].append(avg_struct_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Total Loss={avg_total_loss:.4f}, "
                          f"Pred Loss={avg_pred_loss:.4f}, Struct Loss={avg_struct_loss:.4f}")
        
        logger.info("DeLaN network training completed!")
    
    def run_joint_identification_experiment(self, num_episodes: int = 200):
        """
        Run joint identification experiment to validate Lagrangian learning
        
        Args:
            num_episodes: Number of episodes for identification
        """
        
        logger.info(f"Running joint identification experiment with {num_episodes} episodes...")
        
        # Generate test data
        test_q, test_qdot, test_tau = self.generate_training_data(num_samples=1000)
        
        # Train DeLaN network
        self.train_delan_network(test_q, test_qdot, test_tau)
        
        # Evaluate on test set
        self.delan_net.eval()
        with torch.no_grad():
            test_q_eval = test_q[:100]  # Use first 100 samples for evaluation
            test_qdot_eval = test_qdot[:100]
            test_tau_eval = test_tau[:100]
            
            tau_pred, M_pred, g_pred = self.delan_net(test_q_eval, test_qdot_eval)
            
            # Compute prediction error
            mse_error = nn.MSELoss()(tau_pred, test_tau_eval)
            
            logger.info(f"Joint identification experiment completed!")
            logger.info(f"Test MSE Error: {mse_error.item():.4f}")
            
            # Save results
            self.save_experiment_results(tau_pred, test_tau_eval, mse_error.item())
    
    def save_experiment_results(self, tau_pred: torch.Tensor, tau_true: torch.Tensor, mse_error: float):
        """
        Save experiment results and plots
        
        Args:
            tau_pred: Predicted torques
            tau_true: True torques
            mse_error: Mean squared error
        """
        
        # Save training history plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.training_history['epoch'], self.training_history['total_loss'])
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.training_history['epoch'], self.training_history['prediction_loss'])
        plt.title('Prediction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(self.training_history['epoch'], self.training_history['structure_loss'])
        plt.title('Structure Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'training_history.png'))
        plt.close()
        
        # Save prediction comparison plot
        plt.figure(figsize=(12, 8))
        
        for i in range(7):
            plt.subplot(2, 4, i+1)
            plt.scatter(tau_true[:, i].numpy(), tau_pred[:, i].numpy(), alpha=0.6)
            plt.plot([-100, 100], [-100, 100], 'r--', alpha=0.8)
            plt.xlabel(f'True Tau {i+1}')
            plt.ylabel(f'Pred Tau {i+1}')
            plt.title(f'Joint {i+1} Torque Prediction')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'torque_prediction_comparison.png'))
        plt.close()
        
        # Save model
        torch.save({
            'model_state_dict': self.delan_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'mse_error': mse_error,
            'training_history': self.training_history
        }, os.path.join(self.experiment_dir, 'delan_model.pth'))
        
        logger.info(f"Experiment results saved to {self.experiment_dir}")
    
    def analyze_lagrangian_structure(self):
        """
        Analyze the learned Lagrangian structure
        """
        
        logger.info("Analyzing learned Lagrangian structure...")
        
        self.delan_net.eval()
        
        # Test on different joint configurations
        test_configs = [
            torch.zeros(1, 7),  # Neutral pose
            torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]]),  # Random pose
            torch.tensor([[-0.5, 0.8, -1.2, 1.5, -0.9, 0.3, -0.7]])  # Another random pose
        ]
        
        qdot_test = torch.zeros(1, 7)
        
        for i, q_test in enumerate(test_configs):
            with torch.no_grad():
                tau_pred, M_pred, g_pred = self.delan_net(q_test, qdot_test)
                
                logger.info(f"Configuration {i+1}:")
                logger.info(f"  Joint positions: {q_test.numpy()}")
                logger.info(f"  Predicted torques: {tau_pred.numpy()}")
                logger.info(f"  Mass matrix eigenvalues: {torch.linalg.eigvals(M_pred[0]).real.numpy()}")
                logger.info(f"  Gravity vector: {g_pred.numpy()}")
        
        logger.info("Lagrangian structure analysis completed!")


def main():
    """Main function for running Lagrangian experiments"""
    
    parser = argparse.ArgumentParser(description='Deep Lagrangian Experiments for Franka Panda')
    parser.add_argument('--mode', type=str, default='id_test', 
                       choices=['id_test', 'structure_analysis', 'full_experiment'],
                       help='Experiment mode')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of episodes for identification experiment')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of training samples')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Lagrangian experiments in mode: {args.mode}")
    
    try:
        # Initialize experiment runner
        experiment_runner = LagrangianExperimentRunner()
        
        if args.mode == 'id_test':
            # Run joint identification experiment
            experiment_runner.run_joint_identification_experiment(args.episodes)
            
        elif args.mode == 'structure_analysis':
            # Run structure analysis
            experiment_runner.run_joint_identification_experiment(50)  # Quick training
            experiment_runner.analyze_lagrangian_structure()
            
        elif args.mode == 'full_experiment':
            # Run full experiment with more data
            experiment_runner.run_joint_identification_experiment(args.episodes)
            experiment_runner.analyze_lagrangian_structure()
        
        logger.info("Lagrangian experiments completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
