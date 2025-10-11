#!/usr/bin/env python3

"""
Logging Configuration for Hierarchical SAC Training
Provides centralized logging for physics-informed control experiments

Author: Physics-Informed RL Implementation
Date: 2024
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


class ExperimentLogger:
    """
    Centralized logging system for physics-informed RL experiments
    
    Features:
    - Hierarchical logging for different components
    - File and console output
    - Experiment-specific log directories
    - Performance metrics logging
    """
    
    def __init__(self, experiment_name: str = "hierarchical_sac", log_dir: str = "logs"):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Base directory for logs
        """
        
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.experiment_log_dir, exist_ok=True)
        
        # Create subdirectories for different components
        self.lagrangian_log_dir = os.path.join(self.experiment_log_dir, "lagrangian")
        self.sac_log_dir = os.path.join(self.experiment_log_dir, "sac")
        self.residual_log_dir = os.path.join(self.experiment_log_dir, "residual")
        self.control_log_dir = os.path.join(self.experiment_log_dir, "control")
        
        os.makedirs(self.lagrangian_log_dir, exist_ok=True)
        os.makedirs(self.sac_log_dir, exist_ok=True)
        os.makedirs(self.residual_log_dir, exist_ok=True)
        os.makedirs(self.control_log_dir, exist_ok=True)
        
        # Setup loggers
        self._setup_loggers()
        
        print(f"Experiment logger initialized: {self.experiment_log_dir}")
    
    def _setup_loggers(self):
        """Setup hierarchical loggers for different components"""
        
        # Root experiment logger
        self.root_logger = self._create_logger(
            "experiment",
            os.path.join(self.experiment_log_dir, "experiment.log"),
            level=logging.INFO
        )
        
        # Lagrangian dynamics logger
        self.lagrangian_logger = self._create_logger(
            "lagrangian",
            os.path.join(self.lagrangian_log_dir, "lagrangian.log"),
            level=logging.DEBUG
        )
        
        # SAC training logger
        self.sac_logger = self._create_logger(
            "sac",
            os.path.join(self.sac_log_dir, "sac_training.log"),
            level=logging.INFO
        )
        
        # Residual controller logger
        self.residual_logger = self._create_logger(
            "residual",
            os.path.join(self.residual_log_dir, "residual_control.log"),
            level=logging.INFO
        )
        
        # Control performance logger
        self.control_logger = self._create_logger(
            "control",
            os.path.join(self.control_log_dir, "control_performance.log"),
            level=logging.DEBUG
        )
    
    def _create_logger(self, name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
        """
        Create a logger with both file and console handlers
        
        Args:
            name: Logger name
            log_file: Path to log file
            level: Logging level
            
        Returns:
            Configured logger
        """
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler (only for INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_experiment_start(self, config: dict):
        """Log experiment start with configuration"""
        self.root_logger.info("=" * 80)
        self.root_logger.info(f"Starting experiment: {self.experiment_name}")
        self.root_logger.info("=" * 80)
        self.root_logger.info("Experiment configuration:")
        for key, value in config.items():
            self.root_logger.info(f"  {key}: {value}")
        self.root_logger.info("=" * 80)
    
    def log_experiment_end(self, results: dict):
        """Log experiment end with results"""
        self.root_logger.info("=" * 80)
        self.root_logger.info(f"Experiment completed: {self.experiment_name}")
        self.root_logger.info("=" * 80)
        self.root_logger.info("Experiment results:")
        for key, value in results.items():
            self.root_logger.info(f"  {key}: {value}")
        self.root_logger.info("=" * 80)
        self.root_logger.info(f"Logs saved to: {self.experiment_log_dir}")
    
    def log_episode(self, episode: int, reward: float, length: int, 
                   success: bool, additional_info: Optional[dict] = None):
        """Log episode information"""
        self.sac_logger.info(
            f"Episode {episode}: Reward={reward:.3f}, Length={length}, Success={success}"
        )
        if additional_info:
            for key, value in additional_info.items():
                self.sac_logger.debug(f"  {key}: {value}")
    
    def log_lagrangian_metrics(self, epoch: int, total_loss: float, 
                              pred_loss: float, struct_loss: float):
        """Log Lagrangian training metrics"""
        self.lagrangian_logger.info(
            f"Epoch {epoch}: Total Loss={total_loss:.4f}, "
            f"Pred Loss={pred_loss:.4f}, Struct Loss={struct_loss:.4f}"
        )
    
    def log_residual_metrics(self, epoch: int, total_loss: float,
                           classical_loss: float, residual_loss: float):
        """Log residual controller training metrics"""
        self.residual_logger.info(
            f"Epoch {epoch}: Total={total_loss:.4f}, "
            f"Classical={classical_loss:.4f}, Residual={residual_loss:.4f}"
        )
    
    def log_control_performance(self, step: int, method: str, 
                               position_error: float, success: bool):
        """Log control performance metrics"""
        self.control_logger.debug(
            f"Step {step}: Method={method}, Error={position_error:.4f}, Success={success}"
        )
    
    def get_logger(self, component: str) -> logging.Logger:
        """
        Get logger for specific component
        
        Args:
            component: Component name (lagrangian, sac, residual, control)
            
        Returns:
            Logger for the component
        """
        
        loggers = {
            'experiment': self.root_logger,
            'lagrangian': self.lagrangian_logger,
            'sac': self.sac_logger,
            'residual': self.residual_logger,
            'control': self.control_logger
        }
        
        return loggers.get(component, self.root_logger)


# Global experiment logger instance
_experiment_logger: Optional[ExperimentLogger] = None


def setup_experiment_logging(experiment_name: str = "hierarchical_sac", 
                            log_dir: str = "logs") -> ExperimentLogger:
    """
    Setup global experiment logging
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Base directory for logs
        
    Returns:
        Experiment logger instance
    """
    
    global _experiment_logger
    _experiment_logger = ExperimentLogger(experiment_name, log_dir)
    return _experiment_logger


def get_experiment_logger() -> ExperimentLogger:
    """
    Get global experiment logger
    
    Returns:
        Experiment logger instance
    """
    
    global _experiment_logger
    if _experiment_logger is None:
        _experiment_logger = setup_experiment_logging()
    return _experiment_logger


if __name__ == "__main__":
    # Test logging system
    logger = setup_experiment_logging("test_experiment")
    
    # Test logging
    logger.log_experiment_start({
        'max_episodes': 20000,
        'batch_size': 256,
        'learning_rate': 3e-4
    })
    
    logger.log_episode(1, reward=15.5, length=150, success=False, 
                      additional_info={'phase': 'approach', 'objects': 4})
    
    logger.log_lagrangian_metrics(10, total_loss=0.523, 
                                  pred_loss=0.501, struct_loss=0.022)
    
    logger.log_residual_metrics(10, total_loss=0.234,
                               classical_loss=0.212, residual_loss=0.022)
    
    logger.log_control_performance(100, method='lagrangian', 
                                  position_error=0.015, success=True)
    
    logger.log_experiment_end({
        'total_episodes': 20000,
        'success_rate': 0.30,
        'avg_reward': 125.5
    })
    
    print(f"\nTest logging completed. Check logs in: {logger.experiment_log_dir}")

