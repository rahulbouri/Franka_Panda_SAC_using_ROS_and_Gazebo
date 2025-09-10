#!/usr/bin/env python3
import os
import random
import rospy
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelConfiguration
from geometry_msgs.msg import Pose


def wait_for_services():
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    rospy.wait_for_service('/gazebo/delete_model')
    rospy.wait_for_service('/gazebo/set_model_configuration')


def delete_if_exists(name: str):
    try:
        rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)(name)
    except rospy.ServiceException:
        pass


def spawn_coke_can(x: float, y: float, z: float):
    sdf_path = os.path.join(rospy.get_param('/coke_can_sdf_path'))
    with open(sdf_path, 'r') as f:
        sdf_xml = f.read()
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    spawn = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    spawn(model_name='coke_can', model_xml=sdf_xml, robot_namespace='/', initial_pose=pose, reference_frame='world')


def randomize_manipulator_joints(model_name: str, joint_names, positions):
    set_config = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
    set_config(model_name=model_name, urdf_param_name='robot_description', joint_names=joint_names, joint_positions=positions)


def main():
    rospy.init_node('episode_randomizer', anonymous=False)
    
    # Wait longer for Gazebo to be ready
    rospy.sleep(5.0)
    
    try:
        wait_for_services()
    except rospy.ROSException as e:
        rospy.logerr('Failed to connect to Gazebo services: %s', str(e))
        return

    # Table specifications from training_world.world:
    # Table center at z=0.375, table height=0.75, so table top is at z=0.75
    # Table size is 0.8x0.8, so bounds are x: 0.2 to 1.0, y: -0.4 to 0.4
    table_top_z = 0.75
    table_center_x = 0.6
    table_center_y = 0.0
    table_half_size = 0.4  # half of 0.8
    can_radius = 0.033  # radius of coke can
    
    # Place can on table surface with proper margins to avoid edge placement
    # Ensure can is fully on table with margin from edges
    margin = can_radius + 0.05  # can radius + safety margin
    x = random.uniform(table_center_x - table_half_size + margin, table_center_x + table_half_size - margin)
    y = random.uniform(table_center_y - table_half_size + margin, table_center_y + table_half_size - margin)
    z = table_top_z + can_radius  # table top + can radius to place on surface

    # Delete existing coke can if it exists
    delete_if_exists('coke_can')
    rospy.sleep(1.0)  # Wait for deletion to complete
    
    # Spawn new coke can
    try:
        spawn_coke_can(x, y, z)
        rospy.loginfo('Successfully spawned coke can at (%.3f, %.3f, %.3f)', x, y, z)
    except rospy.ServiceException as e:
        rospy.logerr('Failed to spawn coke can: %s', str(e))

    # Randomize arm joints (within limits)
    joint_names = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]
    positions = [
        random.uniform(-1.0, 1.0),
        random.uniform(-1.2, 0.2),
        random.uniform(-2.5, -0.5),
        random.uniform(-2.0, 2.0),
        random.uniform(-2.0, 2.0),
        random.uniform(-3.14, 3.14),
    ]
    try:
        randomize_manipulator_joints('manipulator', joint_names, positions)
        rospy.loginfo('Successfully randomized manipulator joints')
    except rospy.ServiceException as e:
        rospy.logerr('Failed to set model configuration: %s', str(e))

    rospy.loginfo('Episode randomized: can at (%.3f, %.3f, %.3f) - on table surface', x, y, z)
    rospy.loginfo('Table bounds: x=[%.3f, %.3f], y=[%.3f, %.3f], z=%.3f', 
                  table_center_x - table_half_size, table_center_x + table_half_size,
                  table_center_y - table_half_size, table_center_y + table_half_size, table_top_z)
    
    rospy.loginfo('Episode randomizer completed. Node will exit.')


if __name__ == '__main__':
    main()


