import launch
from launch_ros.actions import Node


def generate_launch_description():
    return launch.LaunchDescription(
        Node(
            package="carla_driver",
            executable="carla_client_node",
            name="Ego"
        ),
        Node(
            package="carla_driver",
            executable="carla_other_vehicles",
            name="Cyclist"
        )
    )

