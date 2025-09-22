#!/usr/bin/env python3

import rospy
import time
from gazebo_ros_link_attacher.srv import Attach, AttachRequest

def test_link_attacher():
    try:
        rospy.init_node('test_link_attacher', anonymous=True)
    except rospy.exceptions.ROSException:
        pass

    print("Testing link attacher...")
    
    # Wait for services
    rospy.loginfo("Waiting for link attacher services...")
    rospy.wait_for_service('/link_attacher_node/attach', timeout=10.0)
    rospy.wait_for_service('/link_attacher_node/detach', timeout=10.0)
    
    attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
    detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
    
    # Test different link combinations
    robot_links = ['panda_hand', 'panda_link8', 'panda_leftfinger', 'panda_rightfinger']
    block_links = ['link', 'base_link']
    
    for robot_link in robot_links:
        for block_link in block_links:
            print(f"Testing attach: robot={robot_link}, block={block_link}")
            try:
                req = AttachRequest()
                req.model_name_1 = 'panda'
                req.link_name_1 = robot_link
                req.model_name_2 = 'block_red_1'
                req.link_name_2 = block_link
                
                resp = attach_srv(req)
                print(f"  Attach result: {resp}")
                if hasattr(resp, 'success') and resp.success:
                    print(f"  ✅ SUCCESS: {robot_link} -> {block_link}")
                    # Try detach
                    detach_resp = detach_srv(req)
                    print(f"  Detach result: {detach_resp}")
                    return True
                else:
                    print(f"  ❌ Failed: {robot_link} -> {block_link}")
            except Exception as e:
                print(f"  ❌ Exception: {e}")
    
    print("No working link combination found")
    return False

if __name__ == '__main__':
    test_link_attacher()
