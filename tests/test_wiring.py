#!/usr/bin/env python3

import rospy
import time
from std_msgs.msg import String


def main():
    try:
        rospy.init_node('test_wiring', anonymous=True)
    except rospy.exceptions.ROSException:
        pass

    print("\n=== ROS Wiring Test ===")

    # List critical topics/services
    try:
        import rosnode
        import rostopic
        import rosservice
    except Exception as e:
        print(f"ros tooling not available: {e}")

    # Topics
    topics = [
        '/franka_state_controller/joint_states',
        '/object_detection',
        '/position_joint_trajectory_controller/command',
    ]

    print("\nTopics availability:")
    # Allow some time for topics to appear
    rospy.sleep(3.0)
    try:
        published = rospy.get_published_topics()
    except Exception:
        published = []
    pub_names = set(name for name, _ in (published or []))
    for t in topics:
        print(f"  {t}: {'FOUND' if t in pub_names else 'MISSING'}")

    # Services
    services = [
        '/franka_gripper/move',
        '/franka_gripper/grasp',
        '/gazebo/set_model_state',
    ]

    print("\nServices availability:")
    try:
        import rosservice
        available = rosservice.get_service_list()
        for s in services:
            print(f"  {s}: {'FOUND' if s in available else 'MISSING'}")
        # Link attacher services
        print("  /link_attacher_node/attach:", 'FOUND' if '/link_attacher_node/attach' in available else 'MISSING')
        print("  /link_attacher_node/detach:", 'FOUND' if '/link_attacher_node/detach' in available else 'MISSING')
    except Exception as e:
        print(f"Service check failed: {e}")

    # Controller manager presence
    try:
        import rosservice
        cms = [s for s in rosservice.get_service_list() if s.startswith('/controller_manager/')]
        print(f"\ncontroller_manager services: {'FOUND' if cms else 'MISSING'} ({len(cms)} services)")
    except Exception:
        pass

    print("\nDone.")


if __name__ == '__main__':
    main()


