#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math
import numpy as np

# --------------------- Geometry & Limits ---------------------
H = 0.25
X_MAX_USER = 0.50
MAX_ANGLE_DEG = 45
MAX_ANGLE = math.radians(MAX_ANGLE_DEG)
X_GEOM_MAX = H * math.tan(MAX_ANGLE)

# --------------------- Y Calibration -------------------------
Y_MIN_M = -0.35
Y_MAX_M =  0.35

# --------------------- Pixel Crop Region ----------------------
CROP_U_MIN = 126   # left
CROP_U_MAX = 445   # right
CROP_V_MIN = 100    # top
CROP_V_MAX = 440   # bottom

CROP_W = CROP_U_MAX - CROP_U_MIN
CROP_H = CROP_V_MAX - CROP_V_MIN

box_inches_x = 14.75 
box_inches_y = 14.5
box_inches_height = 11 

#arduino 

max_a = 95

min_b = 0
max_b = 75

#sending b angle
lastvalue_sent = 0


# --------------------- Mapping Functions ---------------------

def angle_from_vertical_for_x(x, h=H):
    if x <= 0:
        return 0.0, False
    alpha = math.atan(x / h)
    exceeds = alpha > MAX_ANGLE
    alpha_cmd = min(alpha, MAX_ANGLE)
    return alpha_cmd, exceeds

def plan_command(x_target, y_target, h=H):
    alpha_cmd, exceeded = angle_from_vertical_for_x(x_target, h)
    y_cmd = y_target
    return y_cmd, alpha_cmd, exceeded

def alpha_to_A(alpha_cmd):
    if alpha_cmd <= 0:
        return 0
    frac = alpha_cmd / MAX_ANGLE
    return int(round(max(0, min(100, 100 * frac))))

def y_to_B(y_cmd):
    if y_cmd <= Y_MIN_M:
        return 0
    if y_cmd >= Y_MAX_M:
        return 100
    frac = (y_cmd - Y_MIN_M) / (Y_MAX_M - Y_MIN_M)
    return int(round(max(0, min(100, 100 * frac))))

# --------------------- Grid Generation -----------------------

def generate_grid(x_min, x_max, y_min, y_max, nx, ny, serpentine=True):
    xs = [x_min + i * (x_max - x_min) / (nx - 1) for i in range(nx)]
    ys = [y_min + j * (y_max - y_min) / (ny - 1) for j in range(ny)]

    pts = []
    for i, x in enumerate(xs):
        row = ys[::-1] if (serpentine and i % 2 == 1) else ys
        for y in row:
            pts.append((x, y))
    return pts

# --------------------- ROS2 Overlay Node ----------------------

class LaserOverlay(Node):
    def __init__(self):
        super().__init__('laser_overlay')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.grid_pts = generate_grid(
            0.05,
            min(X_MAX_USER, X_GEOM_MAX),
            Y_MIN_M,
            Y_MAX_M,
            nx=4,
            ny=4,
            serpentine=True
        )

        cv2.namedWindow("Laser Overlay")
        cv2.setMouseCallback("Laser Overlay", self.mouse_callback)

        self.frame = None
        self.img_h = None
        self.img_w = None

    # ----------------- Convert world → pixel in cropped area ------------------
    def xy_to_pix(self, x, y):
        x_norm = (x - 0.05) / (min(X_MAX_USER, X_GEOM_MAX) - 0.05)
        y_norm = (y - Y_MIN_M) / (Y_MAX_M - Y_MIN_M)

        u = int(CROP_U_MIN + x_norm * CROP_W)
        v = int(CROP_V_MIN + y_norm * CROP_H)

        return u, v

    def pix_to_xy(self, u_pix, v_pix):
        # Ignore clicks outside the crop region
        if not (CROP_U_MIN <= u_pix <= CROP_U_MAX and
                CROP_V_MIN <= v_pix <= CROP_V_MAX):
            return None, None

        # Normalize cropped pixel coordinates → [0, 1]
        x_norm = (u_pix - CROP_U_MIN) / CROP_W
        y_norm = (v_pix - CROP_V_MIN) / CROP_H

        # Convert to inches
        x_in = x_norm * box_inches_x       # width direction
        y_in = y_norm * box_inches_y + 5    # height/depth direction


        self.get_logger().info(
            f"x_in = {x_in:.2f},y_in={y_in:.2f} "
        )

        return x_in, y_in


    def grid_2_angle(self, y_in):
        theta = math.atan(y_in / box_inches_height)
        return theta

    # def y_pix_2_grid(self, y_pix):
    #     # Ensure y stays in valid range
    #     y_clamped = max(CROP_V_MIN, min(CROP_V_MAX, y_pix))

    #     # Compute normalized position (0 = top, 1 = bottom)
    #     frac = (y_clamped - CROP_V_MIN) / (CROP_V_MAX - CROP_V_MIN)


    #     angle_table = [75, 65, 56, 48, 40, 32, 24, 16, 8, 0]

    #     # Convert fraction → index
    #     idx = int(frac * (len(angle_table) - 1))

    #     return angle_table[idx]

    def y_pix_to_B(self, y_pix):
        # Ordered from high y (B=0) to low y (B=75)
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

        # Clamp to range
        if y_pix >= table[0][0]:
            return table[0][1]
        if y_pix <= table[-1][0]:
            return table[-1][1]

        # Find the correct segment for interpolation
        for i in range(len(table) - 1):
            y1, B1 = table[i]
            y2, B2 = table[i + 1]

            # y1 > y_pix > y2 because y decreases as B increases
            if y1 >= y_pix >= y2:
                # Linear interpolation
                t = (y_pix - y2) / (y1 - y2)
                return B2 + t * (B1 - B2)

        # fallback (should never hit)
        return table[-1][1]

            

    def mouse_callback(self, event, x_pix, y_pix, flags, param):
        max_angle_deg = 64.8
        if event != cv2.EVENT_LBUTTONDOWN or self.frame is None:
            return

        # 1. Convert pixel → inches
        x_in, y_in = self.pix_to_xy(x_pix, y_pix)
        if x_in is None:
            return

        # 3. Convert inches → A (REVERSED: left=90, right=0)
        A = max_a * (1 - (x_in / box_inches_x))
        A = max(0, min(max_a, A))


        # y_norm = (y_pix - CROP_V_MIN) / CROP_H
        # y_norm = max(0.0, min(1.0, y_norm))

        # # Reversed mapping for B
        # B = max_b - y_norm * (max_b - min_b)
        # B = max(min_b, min(max_b, B))

        B = self.y_pix_to_B(y_pix)

        # 5. Print
        # 4. Print
        self.get_logger().info(
            f"CLICK → inches: x={x_in:.2f}, y={y_in:.2f} → A={A:.1f}, B={B:.1f}"
        )



    # ----------------- Draw grid ---------------------------
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.frame = frame

        if self.img_h is None:
            self.img_h, self.img_w = frame.shape[:2]

        # Draw crop box
        cv2.rectangle(frame, (CROP_U_MIN, CROP_V_MIN),
                      (CROP_U_MAX, CROP_V_MAX),
                      (0,255,0), 2)

        # Draw grid points inside crop
        for (x, y) in self.grid_pts:
            u, v = self.xy_to_pix(x, y)
            cv2.circle(frame, (u, v), 6, (0, 0, 255), -1)

        cv2.imshow("Laser Overlay", frame)
        cv2.waitKey(1)

# --------------------------- MAIN -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LaserOverlay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
