#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera', 10)
        self.timer = self.create_timer(0.03, self.timer_callback)  # ~30 FPS
        self.cap = cv2.VideoCapture(0)  # 0 = default laptop camera
        self.bridge = CvBridge()

        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert OpenCV image (BGR) to ROS Image message
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(msg)
        else:
            self.get_logger().warn("Failed to capture frame")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
