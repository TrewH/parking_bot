# =============================================================================
# File: visionNode.py
# Description: Vision node for spot and object detection with alignment
#
# Authors:   Trew Hoffman, Terri Tai
# Created:     2025-11-26
# Updated:     2025-11-29
#
# Notes:
#   - This is a ROS2 node that integrates DepthAI (Oak-D), OpenCV, and AprilTag
#     detection
#   - Color thresholds and tag size placeholders must be tuned later with real hardware
#   - Now includes alignment guidance for parallel parking
# =============================================================================

"""
Vision Node

This node will:
- Grab RGB frames from OAK-D
- Detect a taped parking spot using color segmentation
- Determine "long" and short sides of the spot
- Perform basic occupancy detection (is there an obstacle)
- Detect an AprilTag placed near the pre-parking pose
- Provide alignment guidance for positioning alongside the spot

Outputs: ROS Topics
- /parking_bot/spot_info
- /parking_bot/tag_pose
- /parking_bot/spot_open
- /parking_bot/alignment_command
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import cv2
import depthai as dai
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Pose2D, Twist


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
        width: float = 0.0,
        height: float = 0.0,
    ) -> None:
        self.detected = detected
        self.is_open = is_open
        self.center_px = center_px
        self.yaw_rad = yaw_rad
        self.approach_side = approach_side
        self.width = width
        self.height = height


class AlignmentCommand:
    """
    Alignment command types
    """
    ALIGNED = "ALIGNED"
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    ADJUST_CLOSER = "ADJUST_CLOSER"
    ADJUST_FARTHER = "ADJUST_FARTHER"
    LOST_SPOT = "LOST_SPOT"


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
        self.alignment_cmd_pub = self.create_publisher(
            String, "/parking_bot/alignment_command", 10
        )
        self.alignment_error_pub = self.create_publisher(
            Twist, "/parking_bot/alignment_error", 10
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

        # Alignment thresholds (need to tune these with hardware!)
        self.ALIGNMENT_TOLERANCE_PX = 50  # How close to center is "aligned"
        self.ANGLE_TOLERANCE_RAD = 0.1    # ~5.7 degrees
        self.TARGET_LATERAL_OFFSET_PX = 100  # Desired lateral distance from spot
        self.LATERAL_TOLERANCE_PX = 30

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

        # Publish occupancy as a Bool
        open_msg = Bool()
        open_msg.data = bool(spot.detected and spot.is_open)
        self.spot_open_pub.publish(open_msg)

        # Compute and publish alignment command
        if spot.detected:
            alignment_cmd, error = self._compute_alignment(spot, frame.shape)
            
            cmd_msg = String()
            cmd_msg.data = alignment_cmd
            self.alignment_cmd_pub.publish(cmd_msg)

            # Publish alignment error as Twist (linear.x = forward/back, angular.z = rotation)
            error_msg = Twist()
            error_msg.linear.x = error[0]   # longitudinal error
            error_msg.linear.y = error[1]   # lateral error
            error_msg.angular.z = error[2]  # angular error
            self.alignment_error_pub.publish(error_msg)
        else:
            # No spot detected
            cmd_msg = String()
            cmd_msg.data = AlignmentCommand.LOST_SPOT
            self.alignment_cmd_pub.publish(cmd_msg)


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

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return detection

        # Pick largest contour by area
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
        detection.width = bw
        detection.height = bh

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

    def _compute_alignment(
        self, 
        spot: ParkingSpotDetection, 
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[str, Tuple[float, float, float]]:
        """
        Compute alignment command based on spot position relative to camera.
        
        For parallel parking, we want to:
        1. Be alongside the spot (lateral alignment)
        2. Be at the correct longitudinal position (front/back)
        3. Be parallel to the spot (angular alignment)
        
        Returns:
            (alignment_command, (longitudinal_error, lateral_error, angular_error))
        """
        h, w, _ = frame_shape
        frame_center_x = w / 2
        frame_center_y = h / 2
        
        spot_x, spot_y = spot.center_px
        
        # Compute errors
        # Longitudinal: how far forward/back the spot is from frame center
        longitudinal_error = spot_y - frame_center_y  # positive = spot is below center
        
        # Lateral: how far left/right the spot is from desired offset
        # For parallel parking, we want the spot to be offset to one side
        lateral_error = spot_x - (frame_center_x + self.TARGET_LATERAL_OFFSET_PX)
        
        # Angular: how parallel we are to the spot
        angular_error = spot.yaw_rad  # 0 = parallel, non-zero = need rotation
        
        # Normalize angular error to [-pi, pi]
        while angular_error > math.pi:
            angular_error -= 2 * math.pi
        while angular_error < -math.pi:
            angular_error += 2 * math.pi
        
        # Decision logic
        # Priority: 1) Angular alignment, 2) Lateral position, 3) Longitudinal position
        
        # Check if angular alignment is off
        if abs(angular_error) > self.ANGLE_TOLERANCE_RAD:
            if angular_error > 0:
                return AlignmentCommand.TURN_RIGHT, (longitudinal_error, lateral_error, angular_error)
            else:
                return AlignmentCommand.TURN_LEFT, (longitudinal_error, lateral_error, angular_error)
        
        # Check lateral position (distance from spot)
        if abs(lateral_error) > self.LATERAL_TOLERANCE_PX:
            if lateral_error > 0:
                # Spot is too far right, need to move closer (left)
                return AlignmentCommand.ADJUST_CLOSER, (longitudinal_error, lateral_error, angular_error)
            else:
                # Spot is too far left, need to move farther (right)
                return AlignmentCommand.ADJUST_FARTHER, (longitudinal_error, lateral_error, angular_error)
        
        # Check longitudinal position
        if abs(longitudinal_error) > self.ALIGNMENT_TOLERANCE_PX:
            if longitudinal_error > 0:
                # Spot is below center, need to move forward
                return AlignmentCommand.MOVE_FORWARD, (longitudinal_error, lateral_error, angular_error)
            else:
                # Spot is above center, need to move backward
                return AlignmentCommand.MOVE_BACKWARD, (longitudinal_error, lateral_error, angular_error)
        
        # If we're here, we're aligned!
        return AlignmentCommand.ALIGNED, (longitudinal_error, lateral_error, angular_error)


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