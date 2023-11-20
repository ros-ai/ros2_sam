from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()
    ld.add_action(
        DeclareLaunchArgument(
            "checkpoint_dir",
            default_value="",
            description="Path to the model directory. If empty, the model will be downloaded.",
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "model_type",
            default_value="vit_h",
            choices=["vit_h", "vit_b", "vit_l"],
            description="Type of the model to use",
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "device",
            default_value="cuda",
            description="Device to run the model on. Example: 'cuda' or 'cpu'. To specify a specific GPU, e.g. use 'cuda:0'.",
        )
    )
    ld.add_action(
        Node(
            package="ros2_sam",
            executable="sam_server_node",
            name="sam_server",
            output="screen",
            parameters=[
                {
                    "checkpoint_dir": LaunchConfiguration("checkpoint_dir"),
                    "model_type": LaunchConfiguration("model_type"),
                    "device": LaunchConfiguration("device"),
                }
            ],
        )
    )
    return ld
