from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='odometry_recorder',
            executable='odometry_recorder_node',
            name='odometry_recorder',
            parameters=[os.path.join(get_package_share_directory("odometry_recorder"), 'params', 'params.yaml')],
            output='screen',
        ),
    ])
