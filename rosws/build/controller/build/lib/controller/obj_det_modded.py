#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import torch
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # --- Subscribers and Publishers ---
        self.subscription = self.create_subscription(
            Image, 'camera', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Image, 'camera/detections', 10)

        # --- Load YOLO model ---
        self.model = YOLO('/home/pass_is_queens/Developer/WeedEater/rosws/best.pt')
        self.bridge = CvBridge()
        self.get_logger().info("YOLOv11 model loaded and detector node started")

        # --- Frame counter ---
        self.frame_count = 0
        self.skip_rate = 5   # run YOLO every 5th frame

    def image_callback(self, msg):
        # Convert ROS Image → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Increment frame count
        self.frame_count += 1

        # Check if this frame should run YOLO
        if self.frame_count % self.skip_rate != 0:
            # Pass-through: publish original frame without detections
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(img_msg)
            return

        # Run YOLOv11 inference every 5th frame
        results = self.model.predict(frame, imgsz=640, conf=0.5, verbose=False)

        # Draw bounding boxes
        annotated_frame = frame.copy()
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{self.model.names[cls]} {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish annotated frame
        img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.publisher_.publish(img_msg)


    def destroy_node(self):
        self.get_logger().info("Shutting down YOLO detector node")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
