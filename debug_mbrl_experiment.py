#!/usr/bin/env python3
"""
Debug and Monitoring Script for MBRL Experiment
Helps identify issues and monitor experiment progress.
"""

import rospy
import numpy as np
import torch
import time
import logging
from typing import Dict, List, Tuple, Optional
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_mbrl_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MBRLExperimentDebugger:
    """
    Debug and monitoring tool for MBRL experiment.
    Helps identify issues and monitor system health.
    """
    
    def __init__(self):
        logger.info("🔍 Initializing MBRL Experiment Debugger")
        
        # Initialize ROS
        rospy.init_node('mbrl_experiment_debugger', anonymous=True)
        
        # Joint configuration
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # State tracking
        self.current_joint_states = None
        self.joint_states_received = False
        self.joint_states_history = []
        
        # Initialize ROS components
        self._initialize_ros()
        
        # Debug parameters
        self.debug_interval = 1.0  # seconds
        self.max_history = 1000
        
        logger.info("✅ MBRL Experiment Debugger initialized")
    
    def _initialize_ros(self):
        """Initialize ROS components"""
        # Joint states subscriber
        self.joint_states_sub = rospy.Subscriber(
            '/manipulator/joint_states', 
            JointState, 
            self._joint_states_callback
        )
        
        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Wait for joint states
        self._wait_for_joint_states()
    
    def _joint_states_callback(self, msg):
        """Callback for joint state messages"""
        self.current_joint_states = msg
        self.joint_states_received = True
        
        # Store history
        if len(self.joint_states_history) >= self.max_history:
            self.joint_states_history.pop(0)
        
        self.joint_states_history.append({
            'timestamp': time.time(),
            'positions': list(msg.position),
            'velocities': list(msg.velocity),
            'efforts': list(msg.effort)
        })
    
    def _wait_for_joint_states(self, timeout=10.0):
        """Wait for joint states to be received"""
        logger.info("⏳ Waiting for joint states...")
        start_time = time.time()
        while not self.joint_states_received and (time.time() - start_time) < timeout:
            rospy.sleep(0.1)
        
        if self.joint_states_received:
            logger.info("✅ Joint states received!")
        else:
            logger.error("❌ Timeout waiting for joint states!")
            raise RuntimeError("Failed to receive joint states")
    
    def check_ros_master(self) -> bool:
        """Check if ROS master is running"""
        try:
            rospy.get_master().getSystemState()
            logger.info("✅ ROS master is running")
            return True
        except Exception as e:
            logger.error(f"❌ ROS master not running: {e}")
            return False
    
    def check_gazebo_services(self) -> Dict[str, bool]:
        """Check if Gazebo services are available"""
        services = [
            '/gazebo/get_physics_properties',
            '/gazebo/set_physics_properties',
            '/gazebo/pause_physics',
            '/gazebo/unpause_physics',
            '/gazebo/get_model_state',
            '/gazebo/set_model_configuration'
        ]
        
        results = {}
        for service in services:
            try:
                rospy.wait_for_service(service, timeout=1.0)
                results[service] = True
                logger.info(f"✅ {service} available")
            except rospy.ROSException:
                results[service] = False
                logger.error(f"❌ {service} not available")
        
        return results
    
    def check_controllers(self) -> Dict[str, bool]:
        """Check if controllers are running"""
        try:
            from controller_manager_msgs.srv import ListControllers
            rospy.wait_for_service('/manipulator/controller_manager/list_controllers', timeout=2.0)
            list_controllers = rospy.ServiceProxy('/manipulator/controller_manager/list_controllers', ListControllers)
            response = list_controllers()
            
            controller_status = {}
            for controller in response.controller:
                controller_status[controller.name] = controller.state == 'running'
                status = "✅" if controller.state == 'running' else "❌"
                logger.info(f"{status} Controller {controller.name}: {controller.state}")
            
            return controller_status
        except Exception as e:
            logger.error(f"❌ Failed to check controllers: {e}")
            return {}
    
    def check_joint_states(self) -> Dict[str, any]:
        """Check joint states and identify issues"""
        if self.current_joint_states is None:
            return {'error': 'No joint states received'}
        
        joint_info = {}
        for i, joint in enumerate(self.joint_names):
            try:
                idx = self.current_joint_states.name.index(joint)
                position = self.current_joint_states.position[idx]
                velocity = self.current_joint_states.velocity[idx]
                effort = self.current_joint_states.effort[idx] if len(self.current_joint_states.effort) > idx else 0.0
                
                joint_info[joint] = {
                    'position': position,
                    'velocity': velocity,
                    'effort': effort,
                    'position_valid': not np.isnan(position),
                    'velocity_valid': not np.isnan(velocity),
                    'effort_valid': not np.isnan(effort)
                }
                
                # Check for issues
                if np.isnan(position) or np.isnan(velocity):
                    logger.warning(f"⚠️  {joint}: Invalid values (pos={position}, vel={velocity})")
                
            except ValueError:
                joint_info[joint] = {'error': 'Joint not found in message'}
                logger.error(f"❌ {joint}: Not found in joint states message")
        
        return joint_info
    
    def check_end_effector_pose(self) -> Dict[str, any]:
        """Check end-effector pose and identify issues"""
        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rospy.Time())
            
            position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            orientation = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
            
            # Check for valid values
            position_valid = not np.any(np.isnan(position))
            orientation_valid = not np.any(np.isnan(orientation))
            
            pose_info = {
                'position': position.tolist(),
                'orientation': orientation.tolist(),
                'position_valid': position_valid,
                'orientation_valid': orientation_valid,
                'distance_from_origin': np.linalg.norm(position)
            }
            
            if not position_valid:
                logger.warning("⚠️  End-effector position contains NaN values")
            if not orientation_valid:
                logger.warning("⚠️  End-effector orientation contains NaN values")
            
            return pose_info
            
        except Exception as e:
            logger.error(f"❌ Failed to get end-effector pose: {e}")
            return {'error': str(e)}
    
    def check_effort_topics(self) -> Dict[str, bool]:
        """Check if effort command topics are available"""
        topics = [f'/manipulator/{joint}_effort/command' for joint in self.joint_names]
        
        results = {}
        for topic in topics:
            try:
                # Check if topic exists
                topic_info = rospy.get_published_topics()
                topic_exists = any(topic in published_topic for published_topic, _ in topic_info)
                results[topic] = topic_exists
                
                status = "✅" if topic_exists else "❌"
                logger.info(f"{status} {topic} {'available' if topic_exists else 'not available'}")
            except Exception as e:
                results[topic] = False
                logger.error(f"❌ Error checking {topic}: {e}")
        
        return results
    
    def test_effort_commands(self) -> Dict[str, any]:
        """Test sending effort commands"""
        logger.info("🧪 Testing effort commands...")
        
        # Create publishers
        effort_publishers = {}
        for joint in self.joint_names:
            topic = f'/manipulator/{joint}_effort/command'
            effort_publishers[joint] = rospy.Publisher(topic, Float64, queue_size=1)
        
        # Wait for publishers to connect
        rospy.sleep(1.0)
        
        # Test commands
        test_results = {}
        for joint in self.joint_names:
            try:
                # Send small test command
                msg = Float64()
                msg.data = 1.0  # Small test torque
                effort_publishers[joint].publish(msg)
                
                test_results[joint] = {
                    'command_sent': True,
                    'command_value': 1.0
                }
                logger.info(f"✅ {joint}: Test command sent")
                
            except Exception as e:
                test_results[joint] = {
                    'command_sent': False,
                    'error': str(e)
                }
                logger.error(f"❌ {joint}: Failed to send command: {e}")
        
        return test_results
    
    def monitor_joint_movement(self, duration: float = 10.0) -> Dict[str, any]:
        """Monitor joint movement over time"""
        logger.info(f"📊 Monitoring joint movement for {duration} seconds...")
        
        start_time = time.time()
        initial_positions = None
        final_positions = None
        
        while (time.time() - start_time) < duration:
            if self.current_joint_states is not None:
                positions = []
                for joint in self.joint_names:
                    try:
                        idx = self.current_joint_states.name.index(joint)
                        positions.append(self.current_joint_states.position[idx])
                    except ValueError:
                        positions.append(0.0)
                
                if initial_positions is None:
                    initial_positions = np.array(positions)
                
                final_positions = np.array(positions)
            
            rospy.sleep(0.1)
        
        if initial_positions is not None and final_positions is not None:
            movement = final_positions - initial_positions
            max_movement = np.max(np.abs(movement))
            
            movement_info = {
                'initial_positions': initial_positions.tolist(),
                'final_positions': final_positions.tolist(),
                'movement': movement.tolist(),
                'max_movement': max_movement,
                'movement_detected': max_movement > 0.001
            }
            
            if max_movement > 0.001:
                logger.info(f"✅ Joint movement detected: max = {max_movement:.4f} rad")
            else:
                logger.warning("⚠️  No significant joint movement detected")
            
            return movement_info
        else:
            logger.error("❌ Failed to monitor joint movement")
            return {'error': 'No joint states received'}
    
    def run_comprehensive_check(self) -> Dict[str, any]:
        """Run comprehensive system check"""
        logger.info("🔍 Running comprehensive system check...")
        
        results = {
            'timestamp': time.time(),
            'ros_master': self.check_ros_master(),
            'gazebo_services': self.check_gazebo_services(),
            'controllers': self.check_controllers(),
            'joint_states': self.check_joint_states(),
            'end_effector_pose': self.check_end_effector_pose(),
            'effort_topics': self.check_effort_topics(),
            'effort_commands': self.test_effort_commands()
        }
        
        # Overall health score
        health_checks = [
            results['ros_master'],
            all(results['gazebo_services'].values()),
            len([c for c in results['controllers'].values() if c]) > 0,
            'error' not in results['joint_states'],
            'error' not in results['end_effector_pose'],
            all(results['effort_topics'].values()),
            all([r['command_sent'] for r in results['effort_commands'].values()])
        ]
        
        health_score = sum(health_checks) / len(health_checks)
        results['health_score'] = health_score
        
        logger.info(f"📊 System Health Score: {health_score:.2f} ({health_score*100:.1f}%)")
        
        if health_score >= 0.8:
            logger.info("✅ System is healthy and ready for experiment")
        elif health_score >= 0.5:
            logger.warning("⚠️  System has some issues but may work")
        else:
            logger.error("❌ System has critical issues - experiment may fail")
        
        return results
    
    def run_continuous_monitoring(self, duration: float = 60.0):
        """Run continuous monitoring for specified duration"""
        logger.info(f"📊 Starting continuous monitoring for {duration} seconds...")
        
        start_time = time.time()
        monitoring_data = []
        
        while (time.time() - start_time) < duration:
            timestamp = time.time()
            
            # Collect current state
            joint_info = self.check_joint_states()
            pose_info = self.check_end_effector_pose()
            
            monitoring_data.append({
                'timestamp': timestamp,
                'joint_states': joint_info,
                'end_effector_pose': pose_info
            })
            
            # Log status
            if len(monitoring_data) % 10 == 0:
                logger.info(f"Monitoring... {len(monitoring_data)} samples collected")
            
            rospy.sleep(1.0)
        
        logger.info(f"📊 Monitoring completed. Collected {len(monitoring_data)} samples")
        return monitoring_data

def main():
    """Main function to run debugger"""
    try:
        # Create debugger
        debugger = MBRLExperimentDebugger()
        
        # Run comprehensive check
        results = debugger.run_comprehensive_check()
        
        # Print results
        print("\n" + "="*60)
        print("🔍 MBRL EXPERIMENT DEBUG RESULTS")
        print("="*60)
        print(f"ROS Master: {'✅' if results['ros_master'] else '❌'}")
        print(f"Gazebo Services: {sum(results['gazebo_services'].values())}/{len(results['gazebo_services'])}")
        print(f"Controllers: {sum(results['controllers'].values())}/{len(results['controllers'])}")
        print(f"Joint States: {'✅' if 'error' not in results['joint_states'] else '❌'}")
        print(f"End-Effector Pose: {'✅' if 'error' not in results['end_effector_pose'] else '❌'}")
        print(f"Effort Topics: {sum(results['effort_topics'].values())}/{len(results['effort_topics'])}")
        print(f"Effort Commands: {sum([r['command_sent'] for r in results['effort_commands'].values()])}/{len(results['effort_commands'])}")
        print(f"Health Score: {results['health_score']:.2f}")
        print("="*60)
        
        # Run continuous monitoring if requested
        if rospy.get_param('~monitor', False):
            debugger.run_continuous_monitoring(60.0)
        
    except rospy.ROSInterruptException:
        logger.info("Debugger interrupted by user")
    except Exception as e:
        logger.error(f"Debugger failed: {e}")
        raise

if __name__ == "__main__":
    main()
