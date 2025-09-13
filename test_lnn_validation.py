#!/usr/bin/env python3
"""
Comprehensive LNN Validation with Multiple Episodes
Tests LNN prediction accuracy against Gazebo physics with detailed logging

Author: RL Training Implementation
Date: 2024
"""

import rospy
import numpy as np
import torch
import time
import logging
import os
import sys
from datetime import datetime
from env.manipulator_env_simple import ManipulatorEnvironmentSimple
from models.lnn_6dof import LNN6DOF

# Configure comprehensive logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create detailed logger
logger = logging.getLogger('lnn_validation')
logger.setLevel(logging.DEBUG)

# Create file handler
log_file = os.path.join(log_dir, f"lnn_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def validate_lnn_comprehensive():
    """Comprehensive LNN validation with multiple episodes and detailed logging"""
    logger.info("🚀 Starting Comprehensive LNN Validation")
    logger.info("=" * 60)
    logger.info(f"📝 Log file: {log_file}")
    
    try:
        # Initialize ROS
        logger.info("🔧 Initializing ROS...")
        rospy.init_node('validate_lnn_comprehensive', anonymous=True)
        logger.info("✅ ROS node initialized successfully")
        
        # Create environment and LNN
        logger.info("🔧 Creating environment and LNN model...")
        env = ManipulatorEnvironmentSimple()
        lnn_model = LNN6DOF(obs_size=18, action_size=6, dt=0.02, device='cpu')
        logger.info("✅ Environment and LNN model created successfully")
        
        # Test configurations - more diverse scenarios
        test_configs = [
            {
                "name": "Zero Configuration", 
                "joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "description": "All joints at zero position"
            },
            {
                "name": "Small Angles", 
                "joints": [0.1, -0.1, 0.2, -0.2, 0.1, -0.1],
                "description": "Small joint angles for linear region"
            },
            {
                "name": "Medium Angles", 
                "joints": [0.5, -0.3, 0.4, -0.2, 0.3, -0.1],
                "description": "Medium joint angles for non-linear region"
            },
            {
                "name": "Large Angles", 
                "joints": [1.0, -0.5, 0.8, -0.4, 0.6, -0.2],
                "description": "Large joint angles for extreme dynamics"
            },
            {
                "name": "Random Configuration 1", 
                "joints": [0.3, 0.7, -0.4, 0.1, -0.6, 0.2],
                "description": "Random configuration for generalizability"
            },
            {
                "name": "Random Configuration 2", 
                "joints": [-0.8, 0.2, 0.9, -0.3, 0.4, -0.7],
                "description": "Another random configuration"
            }
        ]
        
        logger.info(f"📊 Testing {len(test_configs)} different configurations")
        
        # Results storage
        all_results = []
        episode_count = 0
        total_predictions = 0
        total_error_sum = 0.0
        
        # Process each configuration
        for config_idx, config in enumerate(test_configs):
            logger.info(f"\\n{'='*60}")
            logger.info(f"🔬 Configuration {config_idx + 1}/{len(test_configs)}: {config['name']}")
            logger.info(f"📝 Description: {config['description']}")
            logger.info(f"🎯 Joint angles: {config['joints']}")
            logger.info(f"{'='*60}")
            
            # Test multiple episodes for this configuration
            num_episodes = 3  # Test 3 episodes per configuration
            config_results = []
            
            for episode in range(num_episodes):
                episode_count += 1
                logger.info(f"\\n🎬 Episode {episode + 1}/{num_episodes} for {config['name']}")
                
                try:
                    # Reset environment
                    logger.debug("🔄 Resetting environment...")
                    target_position = [0.5, 0.0, 0.3]  # Fixed target for consistency
                    state = env.reset(target_position=target_position)
                    logger.debug(f"✅ Environment reset. Initial state shape: {state.shape}")
                    
                    # Set specific joint configuration
                    logger.debug("🔧 Setting joint configuration...")
                    env._set_joint_configuration(config['joints'])
                    logger.debug("⏳ Waiting for joint configuration to stabilize...")
                    rospy.sleep(2.0)  # Wait for stabilization
                    
                    # Get initial state after configuration
                    state = env._get_observation()
                    logger.info(f"📊 Initial joint angles: {state[:6]}")
                    logger.info(f"📊 Initial joint velocities: {state[6:12]}")
                    logger.info(f"📊 Target position: {state[12:15]}")
                    
                    # Test prediction accuracy over multiple steps
                    episode_errors = []
                    joint_angle_errors = []
                    joint_velocity_errors = []
                    step_predictions = 0
                    
                    # Test 15 steps per episode
                    for step in range(15):
                        logger.debug(f"\\n  🎮 Step {step + 1}/15")
                        
                        # Generate random action (scaled appropriately)
                        action = np.random.uniform(-0.3, 0.3, 6)
                        logger.debug(f"  🎯 Action: {action}")
                        
                        # Get ground truth from Gazebo
                        logger.debug("  🔍 Getting ground truth from Gazebo...")
                        next_state_gazebo, reward, done, info = env.step(action)
                        logger.debug(f"  ✅ Gazebo step completed. Reward: {reward:.4f}, Done: {done}")
                        
                        # Predict using LNN
                        logger.debug("  🧠 Making LNN prediction...")
                        state_tensor = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
                        action_tensor = torch.tensor(action, dtype=torch.float64).unsqueeze(0)
                        
                        with torch.no_grad():
                            next_state_lnn = lnn_model.forward(state_tensor, action_tensor)
                        
                        # Convert to numpy
                        next_state_lnn = next_state_lnn.squeeze(0).numpy()
                        logger.debug(f"  ✅ LNN prediction completed")
                        
                        # Compute detailed errors
                        total_error = np.linalg.norm(next_state_lnn - next_state_gazebo)
                        joint_angle_error = np.linalg.norm(next_state_lnn[:6] - next_state_gazebo[:6])
                        joint_velocity_error = np.linalg.norm(next_state_lnn[6:12] - next_state_gazebo[6:12])
                        
                        # Store errors
                        episode_errors.append(total_error)
                        joint_angle_errors.append(joint_angle_error)
                        joint_velocity_errors.append(joint_velocity_error)
                        
                        # Update global counters
                        total_predictions += 1
                        total_error_sum += total_error
                        step_predictions += 1
                        
                        # Log step results
                        logger.info(f"    Step {step + 1}: Total error = {total_error:.4f}, "
                                   f"Joint angle error = {joint_angle_error:.4f}, "
                                   f"Joint velocity error = {joint_velocity_error:.4f}")
                        
                        # Log detailed comparison
                        logger.debug(f"    Gazebo joint angles: {next_state_gazebo[:6]}")
                        logger.debug(f"    LNN joint angles:    {next_state_lnn[:6]}")
                        logger.debug(f"    Gazebo velocities:   {next_state_gazebo[6:12]}")
                        logger.debug(f"    LNN velocities:      {next_state_lnn[6:12]}")
                        
                        # Update state for next iteration
                        state = next_state_gazebo
                        
                        # Check if episode terminated early
                        if done:
                            logger.info(f"    ⚠️ Episode terminated early at step {step + 1}")
                            break
                    
                    # Episode statistics
                    if step_predictions > 0:
                        avg_episode_error = np.mean(episode_errors)
                        avg_joint_angle_error = np.mean(joint_angle_errors)
                        avg_joint_velocity_error = np.mean(joint_velocity_errors)
                        
                        episode_result = {
                            'config_name': config['name'],
                            'episode': episode + 1,
                            'steps': step_predictions,
                            'avg_total_error': avg_episode_error,
                            'avg_joint_angle_error': avg_joint_angle_error,
                            'avg_joint_velocity_error': avg_joint_velocity_error,
                            'errors': episode_errors,
                            'joint_angle_errors': joint_angle_errors,
                            'joint_velocity_errors': joint_velocity_errors
                        }
                        
                        config_results.append(episode_result)
                        
                        logger.info(f"  📊 Episode {episode + 1} Results:")
                        logger.info(f"    Steps completed: {step_predictions}")
                        logger.info(f"    Average total error: {avg_episode_error:.4f}")
                        logger.info(f"    Average joint angle error: {avg_joint_angle_error:.4f}")
                        logger.info(f"    Average joint velocity error: {avg_joint_velocity_error:.4f}")
                    else:
                        logger.warning(f"  ⚠️ Episode {episode + 1} completed with no predictions")
                
                except Exception as e:
                    logger.error(f"  ❌ Episode {episode + 1} failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # Configuration summary
            if config_results:
                config_avg_total = np.mean([r['avg_total_error'] for r in config_results])
                config_avg_joint_angle = np.mean([r['avg_joint_angle_error'] for r in config_results])
                config_avg_joint_velocity = np.mean([r['avg_joint_velocity_error'] for r in config_results])
                config_total_steps = sum([r['steps'] for r in config_results])
                
                logger.info(f"\\n📈 {config['name']} Summary:")
                logger.info(f"  Episodes completed: {len(config_results)}")
                logger.info(f"  Total steps: {config_total_steps}")
                logger.info(f"  Average total error: {config_avg_total:.4f}")
                logger.info(f"  Average joint angle error: {config_avg_joint_angle:.4f}")
                logger.info(f"  Average joint velocity error: {config_avg_joint_velocity:.4f}")
                
                all_results.extend(config_results)
            else:
                logger.warning(f"⚠️ No successful episodes for {config['name']}")
        
        # Overall validation results
        logger.info(f"\\n{'='*60}")
        logger.info("🎯 OVERALL VALIDATION RESULTS")
        logger.info(f"{'='*60}")
        
        if all_results:
            # Compute overall statistics
            overall_avg_total = np.mean([r['avg_total_error'] for r in all_results])
            overall_avg_joint_angle = np.mean([r['avg_joint_angle_error'] for r in all_results])
            overall_avg_joint_velocity = np.mean([r['avg_joint_velocity_error'] for r in all_results])
            total_steps = sum([r['steps'] for r in all_results])
            
            # Error distribution analysis
            all_errors = []
            for result in all_results:
                all_errors.extend(result['errors'])
            
            error_std = np.std(all_errors)
            error_min = np.min(all_errors)
            error_max = np.max(all_errors)
            error_median = np.median(all_errors)
            
            logger.info(f"📊 Validation Statistics:")
            logger.info(f"  Total episodes: {episode_count}")
            logger.info(f"  Total predictions: {total_predictions}")
            logger.info(f"  Total steps: {total_steps}")
            logger.info(f"  Configurations tested: {len(test_configs)}")
            
            logger.info(f"\\n📈 Error Analysis:")
            logger.info(f"  Average total error: {overall_avg_total:.4f}")
            logger.info(f"  Average joint angle error: {overall_avg_joint_angle:.4f}")
            logger.info(f"  Average joint velocity error: {overall_avg_joint_velocity:.4f}")
            logger.info(f"  Error standard deviation: {error_std:.4f}")
            logger.info(f"  Minimum error: {error_min:.4f}")
            logger.info(f"  Maximum error: {error_max:.4f}")
            logger.info(f"  Median error: {error_median:.4f}")
            
            # Validation criteria
            logger.info(f"\\n🎯 Validation Assessment:")
            if overall_avg_total < 0.1:
                logger.info("✅ EXCELLENT: LNN validation PASSED with high accuracy!")
                validation_status = "EXCELLENT"
            elif overall_avg_total < 0.3:
                logger.info("✅ GOOD: LNN validation PASSED with good accuracy!")
                validation_status = "GOOD"
            elif overall_avg_total < 0.5:
                logger.info("⚠️ ACCEPTABLE: LNN validation PARTIAL - acceptable accuracy but needs improvement.")
                validation_status = "ACCEPTABLE"
            elif overall_avg_total < 1.0:
                logger.info("⚠️ POOR: LNN validation PARTIAL - low accuracy, significant improvement needed.")
                validation_status = "POOR"
            else:
                logger.info("❌ FAILED: LNN validation FAILED - very low accuracy, major improvement required.")
                validation_status = "FAILED"
            
            # Save detailed results
            results_summary = {
                'validation_status': validation_status,
                'total_episodes': episode_count,
                'total_predictions': total_predictions,
                'total_steps': total_steps,
                'configurations_tested': len(test_configs),
                'overall_avg_total_error': overall_avg_total,
                'overall_avg_joint_angle_error': overall_avg_joint_angle,
                'overall_avg_joint_velocity_error': overall_avg_joint_velocity,
                'error_std': error_std,
                'error_min': error_min,
                'error_max': error_max,
                'error_median': error_median,
                'detailed_results': all_results
            }
            
            # Save results to file
            results_file = os.path.join(log_dir, f"lnn_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy")
            np.save(results_file, results_summary)
            logger.info(f"💾 Detailed results saved to: {results_file}")
            
            return results_summary
            
        else:
            logger.error("❌ No successful validation results obtained")
            return None
            
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Main validation function"""
    logger.info("🚀 Starting Comprehensive LNN Validation")
    logger.info(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run comprehensive validation
    results = validate_lnn_comprehensive()
    
    logger.info(f"\\n{'='*60}")
    if results:
        logger.info("🎉 LNN validation completed successfully!")
        logger.info(f"📊 Validation status: {results['validation_status']}")
        logger.info(f"📈 Overall accuracy: {results['overall_avg_total_error']:.4f}")
        logger.info(f"📝 Check log file for detailed results: {log_file}")
    else:
        logger.error("❌ LNN validation failed")
    
    logger.info(f"📅 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
