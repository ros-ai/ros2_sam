from typing import List

import rclpy

from ros2_sam.sam_server import SAMServer


def main(args: List[str] = None) -> None:
    rclpy.init(args=args)
    sam_server = SAMServer()
    rclpy.spin(sam_server)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
