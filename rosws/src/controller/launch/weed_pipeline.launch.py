#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='controller',
            executable='camera',
            name='camera_publisher',
            output='screen'
        ),
        Node(
            package='controller',
            executable='obj_det',
            name='yolo_detector',
            output='screen'
        ),
        Node(
            package='controller',
            executable='bb_to_laser',
            name='laser_overlay',
            output='screen'
        ),
    ])
