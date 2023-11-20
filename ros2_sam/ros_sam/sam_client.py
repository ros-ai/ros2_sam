from typing import List, Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point as PointMsg
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray as Int32MultiArrayMsg

from ros2_sam_msgs.srv import Segmentation as SegmentationSrv


class SAMClient(Node):
    """Client for the SAM segmentation service"""

    def __init__(
        self, node_name: str = "sam_client", service_name: str = "sam_server/segment"
    ) -> None:
        """Initialize connection to the SAM segmentation service
        Args:
            node_name (string): Node name, defaults to 'sam_client'.
            service_name (string): Service name, defaults to 'segment'.
        """
        super().__init__(node_name)
        self._bridge = CvBridge()

        self._sam_segment_client = self.create_client(
            SegmentationSrv, f"{service_name}"
        )
        while not self._sam_segment_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                self.get_logger().info(
                    "Interrupted while waiting for service. Exiting."
                )
                return
            self.get_logger().info(
                f"Waiting for '{service_name}' service to become available..."
            )
        self.get_logger().info(f"Established client for '{service_name}' service.")

    def sync_segment_request(
        self,
        img_rgb: np.ndarray,
        points: np.ndarray,
        labels: List[np.int32],
        boxes: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Performs a synchronous (blocking) segmentation request to ROS 2 SAM server.
        Takes the input image and the input prompts with points and bounding boxes.
        Input points can be positive that represent the object to be segmented or
        negative that do not represent the object to be segmented

        Args:
            img_rgb (np.ndarray): RGB image that will be segmented
            points (np.ndarray): Input prompt points for the segmentation
            labels (List[int32]): Labels corresponging to the input prompt points
                            1 for positive and 0 for negative
            boxes (np.ndarray, optional): Bounding boxes covering the objects to be
                                            segmented. Defaults to None
        Returns:
            Masks (List(np.ndarray)): Segmentation masks of the segmented object.
            Scores (List(float)): Confidence scores of the segmentation masks.

            SAM outputs 3 masks, where scores gives the model's own estimation of the quality of these masks
        """
        msg_boxes = Int32MultiArrayMsg()
        if boxes is not None:
            msg_boxes.data = boxes.flatten().astype(int).tolist()
        future = self._sam_segment_client.call_async(
            SegmentationSrv.Request(
                image=self._bridge.cv2_to_imgmsg(img_rgb),
                query_points=[
                    PointMsg(x=float(x), y=float(y), z=0.0) for (x, y) in points
                ],
                query_labels=labels,
                boxes=msg_boxes,
                multimask=(boxes is None),
                logits=False,
            )
        )
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError("Segmentation service call failed")
        res = future.result()
        return [self._bridge.imgmsg_to_cv2(m) for m in res.masks], res.scores
