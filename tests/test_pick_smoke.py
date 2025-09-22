#!/usr/bin/env python3

import os
import sys
import rospy
import time

sys.path.append('/home/bouri/roboset/simple_manipulator_ws/src/pick_and_place/scripts')
from ros_controller import ROSController


def main():
    try:
        rospy.init_node('test_pick_smoke', anonymous=True)
    except rospy.exceptions.ROSException:
        pass

    print("\n=== Pick Smoke Test ===")
    ctrl = ROSController()

    # Attach/detach via link attacher instead of gripper services
    print("Attempt an attach to block_red_1 (simulated grasp)...")
    try:
        ctrl.attach_block('block_red_1')
        time.sleep(0.5)
    except Exception:
        pass

    # Try a short move
    print("Moving to neutral...")
    ctrl.move_to_neutral()
    print("Moving small offset...")
    current = ctrl.get_current_joint_positions()
    target = list(current)
    target[0] += 0.2
    ctrl.move_to_joint_positions(target, duration=2.0)

    print("Detaching block_red_1 (release)...")
    try:
        ctrl.detach_block('block_red_1')
    except Exception:
        pass

    print("\nDone.")


if __name__ == '__main__':
    main()


