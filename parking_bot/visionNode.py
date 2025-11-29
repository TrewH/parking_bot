# =============================================================================
# File: visionNode.py
# Description: Vision node for spot and object detection
#
# Authors:   Trew Hoffman
# Created:     2025-11-26
#
# Notes:
#   - This is a ROS2 node that integrates DepthAI (Oak-D), OpenCV, and AprilTag
#     detection
#   - Color thresholds and tag size placeholders must be tuned later with real hardware

# =============================================================================

"""
Vision Node

This node will:
- Grab RGB frames from OAK-D
- Detect a taped parking spot using color segmentation
- Determine "long" and short sides of the spot
- Perform basic occupancy detection (is there an obstacle)
- Detect an AprilTag placed near the pre-parking pose

Outputs: ROS Topics
- /parking_bot/spot_info
- /parking_bot/tag_pose
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import cv2
import depthai as dai
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose2D


class ApproachSide:
    UNKNOWN = 0
    LEFT = -1   # long side / approach from left
    RIGHT = 1   # long side / approach from right


class ParkingSpotDetection:
    """
    Internal container for parking spot info
    """
    def __init__(
        self,
        detected: bool = False,
        is_open: bool = True,
        center_px: Tuple[int, int] = (0, 0),
        yaw_rad: float = 0.0,
        approach_side: int = ApproachSide.UNKNOWN,
    ) -> None:
        self.detected = detected
        self.is_open = is_open
        self.center_px = center_px
        self.yaw_rad = yaw_rad
        self.approach_side = approach_side


class ParkingVisionNode(Node):
    """
    ROS2 node that handles vision
    """

    def __init__(self) -> None:
        super().__init__("parking_vision_node")

        self.get_logger().info("Initializing ParkingVisionNode...")

        # Publishers
        self.spot_open_pub = self.create_publisher(
            Bool, "/parking_bot/spot_open", 10
        )
        self.tag_pose_pub = self.create_publisher(
            Pose2D, "/parking_bot/tag_pose", 10
        )

        # DepthAI / OAK-D initialization
        self.pipeline = self._create_pipeline()
        self.device = dai.Device(self.pipeline)

        # Get RGB output queue
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        # Camera intrinsics
        # Attempts to grab values from camera config, otherwise estimates from known values
        try: 
            calib = self.device.readCalibration()
            intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 640, 480)
            self.fx = intr[0][0]
            self.fy = intr[1][1]
            self.cx = intr[0][2]
            self.cy = intr[1][2]
        except Exception as e: 
            self.fx = 3009.0 # focal length / pixel size
            self.fy = 3009.0 # focal length / pixel size
            self.cx = 320.0 # Principal x-coordinate, x-pixels/2
            self.cy = 240.0 # Principal y coordinate, y-pixels/2

        # Main loop timer (e.g. 20 Hz)
        self.timer = self.create_timer(0.05, self._process_frame)

        self.get_logger().info("ParkingVisionNode initialized successfully.")


    def _create_pipeline(self) -> dai.Pipeline:
        """
        Helper to create a simple RGB pipeline for the OAK-D
        """
        pipeline = dai.Pipeline()

        cam_rgb = pipeline.create(dai.node.ColorCamera)
        xout_rgb = pipeline.create(dai.node.XLinkOut)

        xout_rgb.setStreamName("rgb")

        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)

        cam_rgb.preview.link(xout_rgb.input)

        return pipeline

    def _process_frame(self) -> None:
        """
        Helper called periodically by the ROS2 timer.
        Grabs the latest frame, runs detection, publishes results.
        """
        in_rgb = self.q_rgb.tryGet()
        if in_rgb is None:
            return

        frame = in_rgb.getCvFrame()  # BGR image (np.ndarray)

        # Detect parking spot
        spot = self._detect_parking_spot(frame)

        # publish occupancy as a Bool
        open_msg = Bool()
        open_msg.data = bool(spot.detected and spot.is_open)
        self.spot_open_pub.publish(open_msg)


    def _detect_parking_spot(self, frame_bgr: np.ndarray) -> ParkingSpotDetection:
        """
        Detect parking spot via colored tape.

        High-level idea:
            1. Convert to HSV
            2. Threshold for the tape color
            3. Find contours, pick the largest plausible rectangle
            4. Compute bounding box, center, and orientation (yaw)
            5. Infer approach side from the long vs short edges and their colors
            6. Do a basic occupancy check by looking at the interior region

        Must be tuned with hardware later!!
        """
        detection = ParkingSpotDetection(detected=False)

        # Convert to HSV
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Tune me
        lower_tape = np.array([90, 80, 80])
        upper_tape = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, lower_tape, upper_tape)

        # Morphological cleanup :devil-emoji: 
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return detection

        # Pick  largest contour by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest = contours[0]

        area = cv2.contourArea(largest)
        h, w, _ = frame_bgr.shape

        # Filter out tiny or huge blobs
        if area < 500 or area > 0.5 * w * h:
            return detection

        # Compute minimum area rectangle (oriented box)
        rect = cv2.minAreaRect(largest)  # ((cx, cy), (width, height), angle)
        (cx, cy), (bw, bh), angle_deg = rect

        # Convert to a consistent yaw (in radians)
        # OpenCV angle: rotation of the rectangle, normalize
        if bw < bh:
            bw, bh = bh, bw
            angle_deg += 90.0

        yaw_rad = math.radians(angle_deg)

        detection.detected = True
        detection.center_px = (int(cx), int(cy))
        detection.yaw_rad = yaw_rad

        # If angle is between -45 and +45 deg, we say approach from "bottom"
        # and map that to RIGHT or LEFT 
        # pick RIGHT if angle is roughly horizontal
        norm_angle = (angle_deg + 180.0) % 180.0  # [0, 180)
        if 45.0 <= norm_angle <= 135.0:
            # Mostly vertical rectangle
            detection.approach_side = ApproachSide.RIGHT
        else:
            # Mostly horizontal rectangle
            detection.approach_side = ApproachSide.LEFT

        # Basic object detection
        # Extract a slightly shrunken interior ROI and see if it's mostly
        # "floor color" or if there's a big blob of something else.
        box_points = cv2.boxPoints(rect).astype(int)
        mask_spot = np.zeros_like(mask)
        cv2.drawContours(mask_spot, [box_points], -1, 255, thickness=-1)

        # Shrink it a bit
        kernel_small = np.ones((15, 15), np.uint8)
        mask_spot_inner = cv2.erode(mask_spot, kernel_small, iterations=1)

        # Grab the pixels inside the inner mask
        spot_pixels = frame_bgr[mask_spot_inner == 255]

        if spot_pixels.size == 0:
            detection.is_open = True
            return detection

        # Convert to grayscale and threshold.
        gray_spot = cv2.cvtColor(spot_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY)
        # Look at variance / distribution for peak (object present)
        intensity_std = np.std(gray_spot)

        # TODO: tune this on real data of object in spot
        detection.is_open = intensity_std < 20.0

        return detection


# ROS2 entry point
def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = ParkingVisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down ParkingVisionNode...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()