#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import math
import serial
import serial.tools.list_ports


class SteeringPID(Node):
    def __init__(self):
        super().__init__('steering_pid')

        # === Parameters ===
        self.declare_parameter('Kp_offset', 0.5)
        self.declare_parameter('Kp_angle', 0.3)
        self.declare_parameter('v_base', 0.20)
        self.declare_parameter('wheel_base', 0.66)
        self.declare_parameter('max_speed', 2.0)
        self.declare_parameter('max_omega', 1.0)

        # === Subscribers / Publishers ===
        self.sub = self.create_subscription(
            Float32MultiArray, '/line_state', self.line_callback, 10)
        self.cmd_pub = self.create_publisher(Float32MultiArray, '/wheel_cmds', 10)

        self.get_logger().info("Steering PID node started — sending serial L/R commands")

        # === Optional smoothing ===
        self.prev_omega = 0.0
        self.alpha = 0.6
        self.min_cmd_period = 0.25  # seconds between wheel commands
        self.last_cmd_time_ns = None

        # === Serial setup ===
        self.serial_port = None
        self.open_serial()

    # ---------------------------------------------------------
    # Try to open serial /dev/ttyACM1 @ 115200
    # ---------------------------------------------------------
    def open_serial(self):
        try:
            self.serial_port = serial.Serial('/dev/ttyACM0', 115200, timeout=0.01)
            self.get_logger().info("Serial connected: /dev/ttyACM0 @115200")
        except Exception as e:
            self.serial_port = None
            self.get_logger().error(f"Failed to open /dev/ttyACM0: {e}")

    # ---------------------------------------------------------
    # Send "<left_mps>,<right_mps>\n" over USB
    # ---------------------------------------------------------
    def send_serial_cmd(self, left, right):
        cmd = f"{left:.3f},{right:.3f}\n"

        # Print exactly what we're *attempting* to send
        self.get_logger().info(f"SEND_SERIAL: '{cmd.strip()}'")

        if not self.serial_port:
            self.get_logger().warn("Serial not open — attempting reopen…")
            self.open_serial()
            if not self.serial_port:
                self.get_logger().error("Failed to open serial — cannot send")
                return

        try:
            n = self.serial_port.write(cmd.encode('utf-8'))
            self.get_logger().info(f"SENT_BYTES: {n}")
        except Exception as e:
            self.get_logger().error(f"Serial write failed: {e}")
            self.serial_port = None  # force reopen next cycle

    # ---------------------------------------------------------
    # Main callback
    # ---------------------------------------------------------
    def line_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn("Malformed /line_state message — expected [offset, angle]")
            return

        offset = msg.data[0]
        angle = msg.data[1]

        # Parameters
        Kp_offset = self.get_parameter('Kp_offset').get_parameter_value().double_value
        Kp_angle = self.get_parameter('Kp_angle').get_parameter_value().double_value
        v_base = self.get_parameter('v_base').get_parameter_value().double_value
        wheel_base = self.get_parameter('wheel_base').get_parameter_value().double_value
        max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        max_omega = self.get_parameter('max_omega').get_parameter_value().double_value

        # Steering control
        omega_raw = (Kp_offset * offset) + (Kp_angle * angle)
        omega = max(-max_omega, min(max_omega, omega_raw))

        # Smooth it
        omega = self.alpha * self.prev_omega + (1 - self.alpha) * omega
        self.prev_omega = omega

        # Differential wheel velocities
        forward = max(-0.2, min(0.2, v_base))
        diff = omega * wheel_base / 2
        diff = max(-max_speed, min(max_speed, diff))
        v_left = forward - diff
        v_right = forward + diff

        v_left = max(-2.0, min(2.0, v_left))
        v_right = max(-2.0, min(2.0, v_right))

        now_ns = self.get_clock().now().nanoseconds
        if self.last_cmd_time_ns is not None:
            elapsed = (now_ns - self.last_cmd_time_ns) / 1e9
            if elapsed < self.min_cmd_period:
                return
        self.last_cmd_time_ns = now_ns

        # === Publish ROS topic ===
        msg_out = Float32MultiArray()
        msg_out.data = [v_left, v_right]
        self.cmd_pub.publish(msg_out)

        # === SEND SERIAL COMMAND ===
        self.send_serial_cmd(v_left, v_right)

        # Debug print
        self.get_logger().info(
            f"offset={offset:.1f}, angle={math.degrees(angle):.1f}°, "
            f"omega={omega:.3f}, L={v_left:.3f}, R={v_right:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = SteeringPID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
