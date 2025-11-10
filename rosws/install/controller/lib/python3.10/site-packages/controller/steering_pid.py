#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import math


class SteeringPID(Node):
    def __init__(self):
        super().__init__('steering_pid')

        # === Parameters ===
        self.declare_parameter('Kp_offset', 0.002)
        self.declare_parameter('Kp_angle', 0.2)
        self.declare_parameter('v_base', 0.4)
        self.declare_parameter('wheel_base', 0.66)
        self.declare_parameter('max_speed', 0.6)
        self.declare_parameter('max_omega', 1.0)

        # === Subscribers & Publishers ===
        self.sub = self.create_subscription(
            Float32MultiArray, '/line_state', self.line_callback, 10)
        self.cmd_pub = self.create_publisher(Float32MultiArray, '/wheel_cmds', 10)

        self.get_logger().info('Steering PID (differential drive) node started — listening to /line_state')

        # === Optional smoothing ===
        self.prev_omega = 0.0
        self.alpha = 0.6

    def line_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn("Malformed /line_state message — expected [offset, angle]")
            return

        offset = msg.data[0]
        angle = msg.data[1]

        # === Parameters ===
        Kp_offset = self.get_parameter('Kp_offset').get_parameter_value().double_value
        Kp_angle = self.get_parameter('Kp_angle').get_parameter_value().double_value
        v_base = self.get_parameter('v_base').get_parameter_value().double_value
        wheel_base = self.get_parameter('wheel_base').get_parameter_value().double_value
        max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        max_omega = self.get_parameter('max_omega').get_parameter_value().double_value

        # === Steering control law ===
        omega_raw = (Kp_offset * offset) + (Kp_angle * angle)
        omega = max(-max_omega, min(max_omega, omega_raw))

        # Optional smoothing (avoid oscillations)
        omega = self.alpha * self.prev_omega + (1 - self.alpha) * omega
        self.prev_omega = omega

        # === Convert angular velocity → differential wheel speeds ===
        # v_left  = v_base - (omega * wheel_base / 2)
        # v_right = v_base + (omega * wheel_base / 2)
        v_left = v_base - (omega * wheel_base / 2)
        v_right = v_base + (omega * wheel_base / 2)

        # Clamp speeds
        v_left = max(-max_speed, min(max_speed, v_left))
        v_right = max(-max_speed, min(max_speed, v_right))

        # === Publish as [left, right] ===
        msg_out = Float32MultiArray()
        msg_out.data = [v_left, v_right]
        self.cmd_pub.publish(msg_out)

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
