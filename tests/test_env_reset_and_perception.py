#!/usr/bin/env python3

import os
import sys
import rospy
import time

sys.path.append('/home/bouri/roboset/simple_manipulator_ws/src/pick_and_place/scripts')
from pick_place_sac_env import PickPlaceSACEnvironment


def main():
    try:
        rospy.init_node('test_env_reset_and_perception', anonymous=True)
    except rospy.exceptions.ROSException:
        pass

    print("\n=== Env Reset & Perception Test ===")
    env = PickPlaceSACEnvironment(max_episode_steps=100)
    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    # Wait up to 8s for detections
    start = rospy.Time.now().to_sec()
    while (rospy.Time.now().to_sec() - start) < 8.0 and len(env.detected_objects) == 0 and not rospy.is_shutdown():
        rospy.sleep(0.5)
    print(f"Detected objects after reset: {len(env.detected_objects)}")

    # Step a few times to see state machine progress and reward logs
    for i in range(5):
        action = (0.1 * (i + 1)) * (1.0 - 2.0 * (i % 2))
        import numpy as np
        act = np.ones(7) * action
        _, reward, done, info = env.step(act)
        print(f"Step {i+1}: reward={reward:.3f}, state={info['state_machine_state']}, done={done}")
        if done:
            break

    print("\nDone.")


if __name__ == '__main__':
    main()


