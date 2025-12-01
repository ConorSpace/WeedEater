#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import math
import serial


# --------------------- Geometry & Limits ---------------------
H = 0.25
X_MAX_USER = 0.50
MAX_ANGLE_DEG = 45
MAX_ANGLE = math.radians(MAX_ANGLE_DEG)
X_GEOM_MAX = H * math.tan(MAX_ANGLE)

# --------------------- Pixel Crop Region ----------------------
CROP_U_MIN = 126
CROP_U_MAX = 445
CROP_V_MIN = 100
CROP_V_MAX = 440

CROP_W = CROP_U_MAX - CROP_U_MIN
CROP_H = CROP_V_MAX - CROP_V_MIN

# Box geometry
box_inches_x = 14.75
box_inches_y = 14.5
box_inches_height = 11

# Arduino motion limits
max_a = 95
min_b = 0
max_b = 75


class LaserOverlay(Node):
    def __init__(self):
        super().__init__('laser_overlay')

        self.bridge = CvBridge()
        self.frame = None
        self.latest_bbox = None  # store latest bbox center

        # ROS Subscribers
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.bbox_sub = self.create_subscription(
            Float32MultiArray, 'camera/bounding_boxes', self.bbox_callback, 10
        )

        # THROTTLE: send A/B at 5 Hz (0.2s)
        self.timer = self.create_timer(0.2, self.throttled_send)

        # Serial
        self.ser = serial.Serial("/dev/ttyACM1", 9600, timeout=1)

        cv2.namedWindow("Laser Overlay")
        self.get_logger().info("LaserOverlay node started (throttled mode, bbox-driven).")

    # ----------------------------------------------------------------------
    # Convert pixel → inches (same as your click version)
    # ----------------------------------------------------------------------
    def pix_to_xy(self, u_pix, v_pix):
        if not (CROP_U_MIN <= u_pix <= CROP_U_MAX and
                CROP_V_MIN <= v_pix <= CROP_V_MAX):
            return None, None

        x_norm = (u_pix - CROP_U_MIN) / CROP_W
        y_norm = (v_pix - CROP_V_MIN) / CROP_H

        x_in = x_norm * box_inches_x
        y_in = y_norm * box_inches_y + 5

        return x_in, y_in

    # --------------------- B angle interpolation --------------------------
    def y_pix_to_B(self, y_pix):
        table = [
            (429, 0),
            (410, 5),
            (393, 10),
            (374, 15),
            (361, 20),
            (346, 25),
            (328, 30),
            (308, 35),
            (284, 40),
            (261, 45),
            (233, 50),
            (205, 55),
            (181, 60),
            (156, 65),
            (116, 70),
            (69,  75)
        ]

        if y_pix >= table[0][0]:
            return table[0][1]
        if y_pix <= table[-1][0]:
            return table[-1][1]

        for i in range(len(table) - 1):
            y1, B1 = table[i]
            y2, B2 = table[i + 1]

            if y1 >= y_pix >= y2:
                t = (y_pix - y2) / (y1 - y2)
                return B2 + t * (B1 - B2)

        return table[-1][1]

    # ----------------------------------------------------------------------
    # STORE BBOX CENTER (do NOT send anything here)
    # ----------------------------------------------------------------------
    def bbox_callback(self, msg):
        if len(msg.data) < 4:
            return

        x1, y1, x2, y2 = msg.data[:4]
        u_center = int((x1 + x2) / 2)
        v_center = int((y1 + y2) / 2)

        self.latest_bbox = (u_center, v_center)

    # ----------------------------------------------------------------------
    # THROTTLED SENDING OF A/B COMMANDS (5 Hz)
    # ----------------------------------------------------------------------
    def throttled_send(self):
        if self.latest_bbox is None:
            return

        u_center, v_center = self.latest_bbox

        x_in, y_in = self.pix_to_xy(u_center, v_center)
        if x_in is None:
            return

        # Compute A
        A = max_a * (1 - (x_in / box_inches_x))
        A = max(0, min(max_a, A))

        # Compute B
        B = self.y_pix_to_B(v_center)
        B = max(min_b, min(max_b, B))

        try:
            cmd = f"A{int(A)} B{int(B)}\n"
            self.ser.write(cmd.encode("utf-8"))
            self.get_logger().info(f"[5 Hz] Sent: {cmd.strip()}")
        except Exception as e:
            self.get_logger().error(f"Serial error: {e}")

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


    # ----------------------------------------------------------------------
    # IMAGE DISPLAY (unchanged)
    # ----------------------------------------------------------------------
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.frame = frame

        cv2.rectangle(frame, (CROP_U_MIN, CROP_V_MIN),
                      (CROP_U_MAX, CROP_V_MAX), (0, 255, 0), 2)

        cv2.imshow("Laser Overlay", frame)
        cv2.waitKey(1)


# --------------------------- MAIN -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LaserOverlay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted — shutting down")
    finally:
        # ALWAYS runs, even on CTRL+C
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
