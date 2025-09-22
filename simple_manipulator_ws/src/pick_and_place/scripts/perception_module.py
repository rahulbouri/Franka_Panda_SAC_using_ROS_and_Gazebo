#!/usr/bin/env python3

"""
Modular Perception Module for Pick-and-Place Task
Integrates object detection and pose estimation for RL training

Author: Based on pick-and-place repository
Date: 2024
"""

import numpy as np
import cv2
import cv_bridge
import rospy
import tf.transformations
from gazebo_msgs.srv import GetModelState
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Image, CameraInfo
from typing import List, Tuple, Optional
from pick_and_place.msg import DetectedObjectsStamped, DetectedObject


class PerceptionModule:
    """
    Modular perception module for object detection and pose estimation.
    Designed for RL training integration.
    """
    
    def __init__(self, camera_topic='/camera/color/image_raw', 
                 depth_topic='/camera/depth/image_raw',
                 camera_info_topic='/camera/color/camera_info',
                 detection_topic='/object_detection'):
        """
        Initialize the perception module.
        
        Args:
            camera_topic: ROS topic for color images
            depth_topic: ROS topic for depth images  
            camera_info_topic: ROS topic for camera info
            detection_topic: ROS topic to publish detections
        """
        # Only initialize node if not already initialized
        try:
            rospy.init_node('perception_module', anonymous=True)
        except rospy.exceptions.ROSException:
            # Node already initialized, continue
            pass
        
        # Camera topics
        self.camera_topic = camera_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.detection_topic = detection_topic
        
        # Color detection parameters
        self.color_ranges = {
            "blue": [np.array([110, 50, 50]), np.array([130, 255, 255])],
            "green": [np.array([36, 25, 25]), np.array([70, 255, 255])],
            "red": [np.array([0, 100, 100]), np.array([10, 255, 255])],
            "black": [np.array([0, 0, 0]), np.array([180, 255, 40])]
        }
        
        # Detection parameters
        self.contour_area_threshold = 200
        self.detected_objects = []
        
        # Camera setup
        self.bridge = cv_bridge.CvBridge()
        self.camera_model = None
        self.camera_transform = None
        self.world_transform = None
        
        # Initialize camera
        self._setup_camera()
        
        # ROS publishers and subscribers
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self._image_callback)
        self.detection_pub = rospy.Publisher(self.detection_topic, DetectedObjectsStamped, queue_size=10)
        
        rospy.loginfo("Perception module initialized successfully!")
    
    def _setup_camera(self):
        """Setup camera model and transforms"""
        try:
            # Get camera info
            camera_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo, timeout=5.0)
            self.camera_model = PinholeCameraModel()
            self.camera_model.fromCameraInfo(camera_info)
            
            # Get camera transforms
            self.camera_transform, self.world_transform = self._get_camera_transforms()
            
            rospy.loginfo("Camera setup completed successfully!")
            
        except Exception as e:
            rospy.logerr(f"Camera setup failed: {e}")
            self.camera_model = None
    
    def _get_camera_transforms(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get camera transformation matrices"""
        try:
            # Get camera position from Gazebo
            camera_pos = self._get_model_position("kinect")
            
            # Camera rotation matrix (OpenCV convention)
            rot_matrix = np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0], 
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            
            # Translation matrix
            trans_matrix = tf.transformations.translation_matrix(camera_pos)
            
            # Combined transform
            camera_transform = np.dot(trans_matrix, rot_matrix)
            world_transform = tf.transformations.inverse_matrix(camera_transform)
            
            return camera_transform, world_transform
            
        except Exception as e:
            rospy.logerr(f"Failed to get camera transforms: {e}")
            return None, None
    
    def _get_model_position(self, model_name: str) -> Tuple[float, float, float]:
        """Get model position from Gazebo"""
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=5.0)
            get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            response = get_model_state(model_name, "world")
            
            if response.success:
                pos = response.pose.position
                return pos.x, pos.y, pos.z
            else:
                rospy.logwarn(f"Failed to get position for model: {model_name}")
                return (0.0, 0.0, 0.0)
                
        except Exception as e:
            rospy.logerr(f"Error getting model position: {e}")
            return (0.0, 0.0, 0.0)
    
    def _image_callback(self, msg: Image):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect objects
            detected_objects = self.detect_objects(cv_image)
            
            # Publish detections
            self._publish_detections(detected_objects)
            
        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")
    
    def detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in the image and return their properties.
        
        Args:
            image: OpenCV BGR image
            
        Returns:
            List of detected objects
        """
        detected_objects = []
        
        if self.camera_model is None:
            rospy.logwarn("Camera model not available")
            return detected_objects
        
        try:
            # Convert to HSV for better color detection
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Detect objects for each color
            for color_name, (lower, upper) in self.color_ranges.items():
                # Create mask for this color
                mask = cv2.inRange(hsv_image, lower, upper)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    if area > self.contour_area_threshold:
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Get depth
                        depth = self._get_depth_at_pixel(center_x, center_y)
                        
                        # Convert to 3D world coordinates
                        world_pos = self._pixel_to_world(center_x, center_y, depth)
                        
                        if world_pos is not None:
                            # Create detected object
                            obj = DetectedObject()
                            obj.color = color_name
                            obj.x_world = world_pos[0]
                            obj.y_world = world_pos[1]
                            obj.height = world_pos[2]
                            obj.area = area
                            obj.confidence = min(area / 1000.0, 1.0)  # Simple confidence based on area
                            
                            detected_objects.append(obj)
                            
        except Exception as e:
            rospy.logerr(f"Object detection error: {e}")
        
        return detected_objects
    
    def _get_depth_at_pixel(self, x: int, y: int) -> float:
        """Get depth value at pixel (x, y)"""
        try:
            depth_msg = rospy.wait_for_message(self.depth_topic, Image, timeout=1.0)
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            
            # Clamp coordinates to image bounds
            x = max(0, min(x, depth_image.shape[1] - 1))
            y = max(0, min(y, depth_image.shape[0] - 1))
            
            depth = depth_image[y, x]
            
            # Handle invalid depth values
            if np.isnan(depth) or np.isinf(depth) or depth <= 0:
                return 0.5  # Default depth
                
            return float(depth)
            
        except Exception as e:
            rospy.logwarn(f"Failed to get depth: {e}")
            return 0.5  # Default depth
    
    def _pixel_to_world(self, u: int, v: int, depth: float) -> Optional[Tuple[float, float, float]]:
        """Convert pixel coordinates to world coordinates"""
        try:
            if self.camera_model is None or self.world_transform is None:
                return None
            
            # Project pixel to 3D point in camera frame
            ray = self.camera_model.projectPixelTo3dRay((u, v))
            
            # Scale by depth
            point_3d = np.array([ray[0] * depth, ray[1] * depth, depth, 1.0])
            
            # Transform to world frame
            world_point = np.dot(self.world_transform, point_3d)
            
            return (world_point[0], world_point[1], world_point[2])
            
        except Exception as e:
            rospy.logwarn(f"Pixel to world conversion failed: {e}")
            return None
    
    def _publish_detections(self, detected_objects: List[DetectedObject]):
        """Publish detected objects"""
        try:
            msg = DetectedObjectsStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.detected_objects = detected_objects
            
            self.detection_pub.publish(msg)
            
        except Exception as e:
            rospy.logerr(f"Failed to publish detections: {e}")
    
    def get_detected_objects(self) -> List[DetectedObject]:
        """Get current detected objects"""
        return self.detected_objects
    
    def is_object_detected(self, color: str) -> bool:
        """Check if object of specific color is detected"""
        return any(obj.color == color for obj in self.detected_objects)
    
    def get_objects_by_color(self, color: str) -> List[DetectedObject]:
        """Get all objects of specific color"""
        return [obj for obj in self.detected_objects if obj.color == color]


def main():
    """Test the perception module"""
    try:
        perception = PerceptionModule()
        rospy.loginfo("Perception module started. Waiting for camera data...")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Perception module stopped")
    except Exception as e:
        rospy.logerr(f"Perception module error: {e}")


if __name__ == "__main__":
    main()
