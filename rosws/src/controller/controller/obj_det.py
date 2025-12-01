#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import torch
from ultralytics import YOLO
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # --- Subscribers and Publishers ---
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        self.image_pub = self.create_publisher(Image, 'camera/detections', 10)

        # NEW: bounding box publisher
        self.bbox_pub = self.create_publisher(Float32MultiArray,
                                              'camera/bounding_boxes', 10)

        # --- Load YOLO model ---
        self.model = YOLO('/home/pass_is_queens/Developer/WeedEater/rosws/yolo_weed_large.pt')
        self.bridge = CvBridge()
        self.get_logger().info("YOLOv11 model loaded and detector node started")

    def image_callback(self, msg):
        # Convert ROS Image → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # YOLO inference
        results = self.model.predict(frame, imgsz=640, conf=0.2, verbose=False)

        annotated_frame = frame.copy()

        # Loop over all detections
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            label = f"{self.model.names[cls]} {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Publish bounding box coordinates
            bbox_msg = Float32MultiArray()
            bbox_msg.data = [float(x1), float(y1), float(x2), float(y2),
                             float(cls), float(conf)]
            self.bbox_pub.publish(bbox_msg)

        # Publish annotated image
        img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.image_pub.publish(img_msg)

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
