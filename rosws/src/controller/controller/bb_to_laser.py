#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import math
import serial
import time

# ---------------- Calibration and Geometry ---------------- #
CROP_U_MIN, CROP_U_MAX = 126, 445
CROP_V_MIN, CROP_V_MAX = 100, 440
CROP_W = CROP_U_MAX - CROP_U_MIN
CROP_H = CROP_V_MAX - CROP_V_MIN

box_inches_x = 14.75
box_inches_y = 14.5
box_inches_height = 11

max_a = 95
min_b = 0
max_b = 75


class LaserOverlay(Node):
    def __init__(self):
        super().__init__('laser_overlay')

        self.bridge = CvBridge()

        # Store last received frame (but don't process immediately)
        self.current_frame = None
        self.frame_count = 0

        # Store last bbox
        self.latest_bbox = None

        # SUBSCRIBE to camera (high-rate)
        self.create_subscription(Image, "/camera/image_raw", self.camera_callback, 10)

        # SUBSCRIBE to YOLO bbox (low-rate)
        self.create_subscription(Float32MultiArray, "camera/bounding_boxes", self.bbox_callback, 10)

        # Timers
        self.create_timer(0.05, self.update_display)   # 20 Hz display update
        self.create_timer(0.2, self.send_serial)       # 5 Hz serial command

        # Serial (non-blocking)
        self.ser = serial.Serial("/dev/ttyACM1", 115200, timeout=0)

        cv2.namedWindow("Laser Overlay")

        self.get_logger().info("LaserOverlay running (non-blocking, throttled).")

    # ---------------- Camera Callback (FAST) ---------------- #
    def camera_callback(self, msg):
        """Runs at camera FPS (likely 30 FPS). Keep extremely light."""
        self.frame_count += 1

        # Skip frames (process only 1 out of every 3 camera frames)
        if self.frame_count % 3 != 0:
            return

        self.current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    # ---------------- Bounding Box Callback ---------------- #
    def bbox_callback(self, msg):
        if len(msg.data) < 4:
            return

        x1, y1, x2, y2 = msg.data[:4]
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        self.latest_bbox = (u, v)

    # ---------------- Update Display Timer ---------------- #
    def update_display(self):
        """Runs at 20 Hz. Safe GUI update outside callbacks."""
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()

        # Draw crop
        cv2.rectangle(frame, (CROP_U_MIN, CROP_V_MIN),
                      (CROP_U_MAX, CROP_V_MAX), (0, 255, 0), 2)

        # Draw bbox center
        if self.latest_bbox is not None:
            u, v = self.latest_bbox
            cv2.circle(frame, (u, v), 6, (0, 0, 255), -1)

        cv2.imshow("Laser Overlay", frame)
        cv2.waitKey(1)

    # ---------------- Convert Pixel → Inches ---------------- #
    def pix_to_xy(self, u, v):
        if not (CROP_U_MIN <= u <= CROP_U_MAX and CROP_V_MIN <= v <= CROP_V_MAX):
            return None, None

        x_norm = (u - CROP_U_MIN) / CROP_W
        y_norm = (v - CROP_V_MIN) / CROP_H

        x_in = x_norm * box_inches_x
        y_in = y_norm * box_inches_y + 5
        return x_in, y_in

    # ---------------- Vertical Interpolation ---------------- #
    def y_pix_to_B(self, y_pix):
        table = [
            (429, 0),(410,5),(393,10),(374,15),(361,20),(346,25),(328,30),
            (308,35),(284,40),(261,45),(233,50),(205,55),(181,60),
            (156,65),(116,70),(69,75)
        ]
        if y_pix >= table[0][0]: return table[0][1]
        if y_pix <= table[-1][0]: return table[-1][1]

        for i in range(len(table)-1):
            y1, B1 = table[i]
            y2, B2 = table[i+1]
            if y1 >= y_pix >= y2:
                t = (y_pix - y2) / (y1 - y2)
                return B2 + t*(B1-B2)

        return table[-1][1]

    # ---------------- Throttled Serial Sender ---------------- #
    def send_serial(self):
        if self.latest_bbox is None:
            return

        u, v = self.latest_bbox
        x_in, y_in = self.pix_to_xy(u, v)
        if x_in is None:
            return

        # Compute A/B
        A = max_a * (1 - (x_in / box_inches_x))
        A = max(0, min(max_a, A))

        B = self.y_pix_to_B(v)
        B = max(min_b, min(max_b, B))

        cmd = f"A{int(A)} B{int(B)}\n"

        try:
            self.ser.write(cmd.encode())
            # self.get_logger().info(f"[5Hz] Sent: {cmd.strip()}")
        except Exception as e:
            self.get_logger().error(f"Serial write failed: {e}")

        self.get_logger().info(f"[5Hz] Sent: {cmd.strip()}")

    def destroy_node(self):
        # Send safe-reset command on shutdown
        try:
            reset_cmd = "A0 B0\n"
            self.ser.write(reset_cmd.encode('utf-8'))
            self.get_logger().info("Sent shutdown reset: A0 B0")
        except Exception as e:
            self.get_logger().error(f"Error sending shutdown reset: {e}")

        # Close serial cleanly
        try:
            if self.ser.is_open:
                self.ser.close()
        except:
            pass

        super().destroy_node()



# ---------------- MAIN ---------------- #
def main(args=None):
    rclpy.init(args=args)
    node = LaserOverlay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted — shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
