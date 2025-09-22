#!/usr/bin/env python3

"""
ROS-based controller for Franka Panda robot manipulation.
Compatible with RL training goals.

Author: Based on pick-and-place repository
Date: 2024
"""

import rospy
import numpy as np
import actionlib
import sys
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
import moveit_commander
from moveit_commander import PlanningSceneInterface
from std_msgs.msg import Float64MultiArray
# Optional gripper services (may be unavailable in some sim setups)
try:
    from franka_gripper.srv import Move, MoveRequest, Grasp, GraspRequest
except Exception:
    Move = None
    Grasp = None
    class MoveRequest:  # minimal placeholders
        def __init__(self):
            self.width = 0.0
            self.speed = 0.1
    class GraspRequest:
        class _Epsilon:
            def __init__(self):
                self.inner = 0.0
                self.outer = 0.0
        def __init__(self):
            self.width = 0.0
            self.speed = 0.1
            self.force = 10.0
            self.epsilon = GraspRequest._Epsilon()

from pick_and_place.msg import DetectedObjectsStamped, DetectedObject


class ROSController:
    def __init__(self):
        # Only initialize node if not already initialized
        try:
            rospy.init_node('ros_controller', anonymous=True)
        except rospy.exceptions.ROSException:
            # Node already initialized, continue
            pass
        
        # Robot configuration
        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 
                           'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        
        # Bin positions (same as original)
        self.red_bin = (-0.5, -0.25)
        self.green_bin = (-0.5, 0.0)
        self.blue_bin = (-0.5, 0.25)
        
        # Offsets (same as original)
        self.workbench_height = 0.2
        self.x_offset = 0.01
        self.z_offset = 0.105
        self.z_pre_pick_offset = 0.2
        self.z_pre_place_offset = 0.2
        
        # Current state
        self.current_joint_states = None
        self.objects_on_workbench = []
        
        # Trajectory action client (use position_joint_trajectory_controller)
        self.trajectory_client = actionlib.SimpleActionClient(
            '/position_joint_trajectory_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )
        self.trajectory_available = self.trajectory_client.wait_for_server(rospy.Duration(2.0))
        if self.trajectory_available:
            rospy.loginfo("Trajectory action server connected!")
        else:
            rospy.logwarn("Trajectory action server not available; will try position controller publisher")
        
        # Position JointTrajectory publisher fallback
        from trajectory_msgs.msg import JointTrajectory
        self.jt_pub = rospy.Publisher('/position_joint_trajectory_controller/command', JointTrajectory, queue_size=1)

        # Gazebo link attacher services (optional)
        self.attach_srv = None
        self.detach_srv = None
        try:
            rospy.wait_for_service('/link_attacher_node/attach', timeout=2.0)
            rospy.wait_for_service('/link_attacher_node/detach', timeout=2.0)
            from gazebo_ros_link_attacher.srv import Attach
            self.attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
            self.detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
            rospy.loginfo("Link attacher services found")
        except Exception as e:
            rospy.logwarn(f"Link attacher services unavailable: {e}")
        
        # MoveIt setup - disabled for now to focus on joint control
        self.group = None
        self.scene = None
        rospy.loginfo("Using joint-based control only (MoveIt disabled for RL compatibility)")
        
        # Subscribers
        self.joint_state_sub = rospy.Subscriber('/franka_state_controller/joint_states', 
                                              JointState, self.joint_state_callback)
        self.object_detection_sub = rospy.Subscriber('/object_detection', 
                                                   DetectedObjectsStamped, 
                                                   self.object_detection_callback)
        
        # Gripper control - using available services
        self.gripper_open_srv = None
        self.gripper_close_srv = None
        self.gripper_grasp_srv = None
        self.gripper_move_srv = None
        
        # Try to find gripper services
        try:
            rospy.wait_for_service('/franka_gripper/grasp', timeout=2.0)
            self.gripper_grasp_srv = rospy.ServiceProxy('/franka_gripper/grasp', Grasp)
            rospy.loginfo("Gripper grasp service found")
        except Exception as e:
            rospy.logwarn(f"Gripper grasp service not available: {e}")
        
        try:
            rospy.wait_for_service('/franka_gripper/move', timeout=2.0)
            self.gripper_move_srv = rospy.ServiceProxy('/franka_gripper/move', Move)
            rospy.loginfo("Gripper move service found")
        except Exception as e:
            rospy.logwarn(f"Gripper move service not available: {e}")
        
        # Add collision objects
        self.add_collision_objects()
        
        rospy.loginfo("ROS Controller initialized successfully!")
    
    def joint_state_callback(self, msg):
        """Callback for joint state updates"""
        self.current_joint_states = msg
    
    def object_detection_callback(self, msg):
        """Callback for object detection updates"""
        self.objects_on_workbench = msg.detected_objects
    
    def get_current_joint_positions(self):
        """Get current joint positions"""
        if self.current_joint_states is not None:
            return list(self.current_joint_states.position)
        return [0.0] * 7
    
    def move_to_joint_positions(self, joint_positions, duration=2.0):
        """Move robot to specified joint positions via available interface"""
        if self.trajectory_available:
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = self.joint_names
            point = JointTrajectoryPoint()
            point.positions = joint_positions
            point.time_from_start = rospy.Duration(duration)
            goal.trajectory.points = [point]
            self.trajectory_client.send_goal(goal)
            self.trajectory_client.wait_for_result()
            return self.trajectory_client.get_result()
        else:
            # Publish a JointTrajectory directly to the controller command topic
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            pt = JointTrajectoryPoint()
            pt.positions = joint_positions
            pt.time_from_start = rospy.Duration(duration)
            traj.points = [pt]
            self.jt_pub.publish(traj)
            rospy.sleep(duration)
            return True
    
    def move_to_cartesian_pose(self, position, orientation, duration=2.0):
        """Move robot to specified Cartesian pose using joint positions"""
        rospy.logwarn("Cartesian control not available, using joint positions")
        # For now, use a simple joint position based on the target
        # This is a simplified approach for RL training
        target_joints = self.cartesian_to_joints(position, orientation)
        if target_joints is not None:
            return self.move_to_joint_positions(target_joints, duration)
        return False
    
    def cartesian_to_joints(self, position, orientation):
        """Simple inverse kinematics approximation for RL training"""
        # This is a simplified approach - in real RL training, 
        # the policy would learn to output joint positions directly
        x, y, z = position
        
        # Simple joint configuration based on position
        # This is just for testing - the RL policy will learn proper IK
        joint1 = np.arctan2(y, x) * 0.5  # Base rotation
        joint2 = -0.785 + (z - 0.5) * 0.5  # Shoulder
        joint3 = 0.0  # Elbow
        joint4 = -2.356 + (x - 0.5) * 0.3  # Forearm
        joint5 = 0.0  # Wrist 1
        joint6 = 1.57 + (y - 0.0) * 0.5  # Wrist 2
        joint7 = 0.785  # Wrist 3
        
        return [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
    
    def move_to_neutral(self):
        """Move robot to neutral position"""
        neutral_positions = [0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785]
        return self.move_to_joint_positions(neutral_positions)
    
    def pick(self, x, y, z, roll=0, pitch=np.pi, yaw=0, object_width=0.025):
        """Pick up object at given position"""
        rospy.loginfo(f"Picking object at ({x:.3f}, {y:.3f}, {z:.3f})")
        
        # Define poses
        pre_pick_position = np.array([x + self.x_offset, y, z + self.z_offset + self.z_pre_pick_offset])
        pick_position = np.array([x + self.x_offset, y, z + self.z_offset])
        pick_orientation = quaternion_from_euler(roll, pitch, yaw)
        
        # Pre-pick
        self.move_to_cartesian_pose(pre_pick_position, pick_orientation)
        self.open_gripper()
        
        # Pick
        self.move_to_cartesian_pose(pick_position, pick_orientation)
        self.close_gripper()
        
        # Post-pick
        self.move_to_cartesian_pose(pre_pick_position, pick_orientation)
        
        rospy.loginfo("Pick completed")
    
    def place(self, x, y, z, roll=0, pitch=np.pi, yaw=0):
        """Place object at given position"""
        rospy.loginfo(f"Placing object at ({x:.3f}, {y:.3f}, {z:.3f})")
        
        # Define poses
        pre_place_position = np.array([x, y, z + self.z_pre_place_offset])
        place_position = np.array([x, y, z])
        place_orientation = quaternion_from_euler(roll, pitch, yaw)
        
        # Pre-place
        self.move_to_cartesian_pose(pre_place_position, place_orientation)
        
        # Place
        self.move_to_cartesian_pose(place_position, place_orientation)
        self.open_gripper()
        
        # Post-place
        self.move_to_cartesian_pose(pre_place_position, place_orientation)
        
        rospy.loginfo("Place completed")
    
    def open_gripper(self):
        """Open gripper"""
        rospy.loginfo("Opening gripper")
        if self.gripper_move_srv is not None:
            try:
                req = MoveRequest()
                req.width = 0.04
                req.speed = 0.1
                self.gripper_move_srv(req)
                rospy.sleep(0.5)
            except Exception as e:
                rospy.logwarn(f"Failed to open gripper: {e}")
        else:
            rospy.logwarn("Gripper service not available")
            rospy.sleep(0.5)
    
    def close_gripper(self):
        """Close gripper"""
        rospy.loginfo("Closing gripper")
        if self.gripper_grasp_srv is not None:
            try:
                req = GraspRequest()
                req.width = 0.0
                req.speed = 0.1
                req.force = 20.0
                req.epsilon.inner = 0.005
                req.epsilon.outer = 0.005
                self.gripper_grasp_srv(req)
                rospy.sleep(0.5)
            except Exception as e:
                rospy.logwarn(f"Failed to close gripper: {e}")
        else:
            rospy.logwarn("Gripper service not available")
            rospy.sleep(0.5)
    
    def select_bin(self, color):
        """Select bin position based on color"""
        if color == "red":
            return self.red_bin
        elif color == "green":
            return self.green_bin
        elif color == "blue":
            return self.blue_bin
        else:
            rospy.loginfo('Unknown color, using green bin')
            return self.green_bin
    
    def move_object(self, object):
        """Move object from workbench to appropriate bin"""
        x = object.x_world
        y = object.y_world
        z = object.height + self.workbench_height
        color = object.color
        
        rospy.loginfo(f"Moving {color} object from ({x:.3f}, {y:.3f}) to bin")
        
        # Pick object
        self.pick(x, y, z)
        
        # Place in appropriate bin
        bin_pos = self.select_bin(color)
        self.place(bin_pos[0], bin_pos[1], 0.5)

    def attach_block(self, block_model: str, block_link: str = 'link', robot_link: str = 'panda_hand'):
        """Attach block to gripper using Gazebo link attacher.
        Tries multiple likely robot/block link names and returns True on success.
        """
        if self.attach_srv is None:
            rospy.logwarn("Attach service not available")
            return False
        try:
            from gazebo_ros_link_attacher.srv import AttachRequest
            candidate_robot_links = [
                robot_link,
                'panda_link7',
                'panda_link8',
                'panda_leftfinger',
                'panda_rightfinger',
                'panda_hand',
                'panda_hand_tcp'
            ]
            candidate_block_links = [block_link, 'link', 'base_link']
            for rl in candidate_robot_links:
                for bl in candidate_block_links:
                    req = AttachRequest()
                    req.model_name_1 = 'panda'
                    req.link_name_1 = rl
                    req.model_name_2 = block_model
                    req.link_name_2 = bl
                    try:
                        resp = self.attach_srv(req)
                        success = getattr(resp, 'success', bool(resp))
                        rospy.loginfo(f"Attach attempt robot_link={rl}, block_link={bl} -> {success}")
                        if success:
                            return True
                    except Exception as inner_e:
                        rospy.logdebug(f"Attach attempt failed for rl={rl}, bl={bl}: {inner_e}")
            rospy.logwarn("All attach attempts failed")
            return False
        except Exception as e:
            rospy.logwarn(f"Attach failed: {e}")
            return False

    def detach_block(self, block_model: str, block_link: str = 'link', robot_link: str = 'panda_hand'):
        """Detach block from gripper using Gazebo link attacher"""
        if self.detach_srv is None:
            rospy.logwarn("Detach service not available")
            return False
        try:
            from gazebo_ros_link_attacher.srv import AttachRequest
            candidate_robot_links = [robot_link, 'panda_hand', 'panda_link8', 'panda_hand_tcp']
            candidate_block_links = [block_link, 'link', 'base_link']
            for rl in candidate_robot_links:
                for bl in candidate_block_links:
                    req = AttachRequest()
                    req.model_name_1 = 'panda'
                    req.link_name_1 = rl
                    req.model_name_2 = block_model
                    req.link_name_2 = bl
                    try:
                        resp = self.detach_srv(req)
                        success = getattr(resp, 'success', bool(resp))
                        rospy.loginfo(f"Detach attempt robot_link={rl}, block_link={bl} -> {success}")
                        if success:
                            return True
                    except Exception as inner_e:
                        rospy.logdebug(f"Detach attempt failed for rl={rl}, bl={bl}: {inner_e}")
            rospy.logwarn("All detach attempts failed")
            return False
        except Exception as e:
            rospy.logwarn(f"Detach failed: {e}")
            return False
    
    def are_objects_on_workbench(self):
        """Check if objects are on workbench"""
        return len(self.objects_on_workbench) > 0
    
    def select_random_object(self):
        """Select random object from workbench"""
        if self.objects_on_workbench:
            return np.random.choice(self.objects_on_workbench)
        return None
    
    def add_collision_objects(self):
        """Add collision objects to MoveIt scene"""
        if self.scene is None:
            rospy.logwarn("MoveIt scene not available, skipping collision objects")
            return
        
        try:
            # Add workbench
            workbench_pose = PoseStamped()
            workbench_pose.header.frame_id = "world"
            workbench_pose.pose.position.x = 0.7
            workbench_pose.pose.position.y = 0.0
            workbench_pose.pose.position.z = 0.1
            workbench_size = (1.0, 3.0, 0.2)
            self.scene.add_box("workbench", workbench_pose, workbench_size)
            
            # Add bin bench
            bin_bench_pose = PoseStamped()
            bin_bench_pose.header.frame_id = "world"
            bin_bench_pose.pose.position.x = -0.55
            bin_bench_pose.pose.position.y = 0.0
            bin_bench_pose.pose.position.z = 0.1
            binbench_size = (0.4, 1.5, 0.2)
            self.scene.add_box("binbench", bin_bench_pose, binbench_size)
            
            rospy.loginfo("Collision objects added to scene")
        except Exception as e:
            rospy.logwarn(f"Failed to add collision objects: {e}")


def main():
    """Test the controller"""
    controller = ROSController()
    
    rospy.loginfo("Testing robot controller...")
    
    # Test neutral position
    rospy.loginfo("Moving to neutral position...")
    controller.move_to_neutral()
    
    # Test joint movement
    rospy.loginfo("Testing joint movement...")
    test_positions = [0.1, -0.5, 0.0, -2.0, 0.0, 1.0, 0.5]
    controller.move_to_joint_positions(test_positions)
    
    # Return to neutral
    controller.move_to_neutral()
    
    rospy.loginfo("Controller test completed!")
    
    # Keep node running
    rospy.spin()


if __name__ == "__main__":
    main()
