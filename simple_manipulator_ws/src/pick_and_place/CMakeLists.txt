cmake_minimum_required(VERSION 3.5)
project(pick_and_place)

## Compile as C++14, supported in ROS Kinetic and newer
add_definitions(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  sensor_msgs
  geometry_msgs
  control_msgs
  trajectory_msgs
  gazebo_msgs
  tf
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  DetectedObject.msg
  DetectedObjectsStamped.msg
)

## Generate added messages
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime sensor_msgs geometry_msgs control_msgs trajectory_msgs gazebo_msgs tf
)

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/perception_module.py
  scripts/ros_controller.py
  scripts/pick_place_sac_env.py
  scripts/sac_pick_place_trainer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch worlds models config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
