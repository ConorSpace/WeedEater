#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray


class LineDetector(Node):
    def __init__(self):
        super().__init__('line_detector')
        self.bridge = CvBridge()

        # === Publishers & subscribers ===
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.line_state_pub = self.create_publisher(Float32MultiArray, '/line_state', 10)
        self.mask_pub = self.create_publisher(Image, '/line_detector/mask', 10)
        self.annotated_pub = self.create_publisher(Image, '/line_detector/annotated', 10)

        # === HSV thresholds for blue tape ===
        self.lower_blue = np.array([95, 100, 50])
        self.upper_blue = np.array([130, 255, 255])

        # === Parameters ===
        self.declare_parameter('roi_fraction', 0.3)
        self.declare_parameter('blur_kernel', 5)
        self.show_debug = True

        self.get_logger().info('Line detector node (blue HSV, stable angle) started.')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CVBridge error: {e}')
            return

        height, width = frame.shape[:2]
        roi_fraction = self.get_parameter('roi_fraction').get_parameter_value().double_value
        blur_kernel = self.get_parameter('blur_kernel').get_parameter_value().integer_value

        # --- Step 1: Convert to HSV and blur ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (blur_kernel, blur_kernel), 0)

        # --- Step 2: Mask for blue regions ---
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # --- Step 3: Morphological cleanup ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # --- Step 4: Restrict to middle ROI horizontally ---
        center_x = width // 2
        half_width = int(width * roi_fraction / 2)
        mask[:, :center_x - half_width] = 0
        mask[:, center_x + half_width:] = 0

        # --- Step 5: Find line & compute offset + angle ---
        offset = 0.0
        angle = 0.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)

            if len(largest) >= 2:
                [vx, vy, x0, y0] = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)

                # Force direction to be consistent (vy always positive)
                if vy < 0:
                    vx = -vx
                    vy = -vy

                y_bottom = height - 1
                if abs(vy) > 1e-5:
                    x_bottom = int(x0 + (y_bottom - y0) * (vx / vy))
                    offset = float(width // 2 - x_bottom)

                    if not hasattr(self, 'prev_offset'):
                        self.prev_offset = offset
                    filtered_offset = 0.8 * self.prev_offset + 0.2 * offset  # 80% old, 20% new
                    self.prev_offset = filtered_offset
                    offset = filtered_offset

                    # Compute stable angle relative to vertical
                    angle = np.arctan2(vx, vy)

                    # Normalize to [-pi/2, pi/2] to prevent wraparound
                    if angle > np.pi / 2:
                        angle -= np.pi
                    elif angle < -np.pi / 2:
                        angle += np.pi

                    # Optional smoothing (low-pass filter)
                    if not hasattr(self, 'prev_angle'):
                        self.prev_angle = angle
                    filtered_angle = 0.7 * self.prev_angle + 0.3 * angle
                    self.prev_angle = filtered_angle
                    angle = filtered_angle

                    # Clamp small noise
                    if abs(angle) < 0.02:
                        angle = 0.0

                    # Debug draw
                    if self.show_debug:
                        y_top = 0
                        x_top = int(x0 + (y_top - y0) * (vx / vy))
                        cv2.line(frame, (x_top, y_top), (x_bottom, y_bottom), (0, 255, 0), 2)
                        cv2.circle(frame, (x_bottom, y_bottom), 6, (0, 0, 255), -1)
                        cv2.line(frame, (width // 2, y_bottom),
                                 (x_bottom, y_bottom), (255, 0, 0), 2)

        # --- Step 6: Publish array [offset, angle] ---
        msg_out = Float32MultiArray()
        msg_out.data = [float(offset), float(angle)]
        self.line_state_pub.publish(msg_out)

        # Optional: log to terminal
        # self.get_logger().info(f"Offset: {offset:.1f}, Angle: {np.degrees(angle):.1f}°")

        # --- Step 7: Publish imagery ---
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            self.mask_pub.publish(mask_msg)
            annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.annotated_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'CVBridge publish error: {e}')

        if self.show_debug:
            cv2.imshow('mask', mask)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
