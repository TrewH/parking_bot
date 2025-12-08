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
- /parking_bot/no_parking_detected
"""


from __future__ import annotations

from typing import Optional, Tuple, List

import cv2
import depthai as dai
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, String, Float32


class SpotId:
    UNKNOWN = "UNKNOWN"
    NEAR = "NEAR"
    FAR = "FAR"


class ParkingSpotDetection:
    """
    Container for one parking spot's info.
    """
    def __init__(
        self,
        spotted: bool = False,
        spot_id: str = SpotId.UNKNOWN,
        center_px: Tuple[int, int] = (0, 0),
        width_px: float = 0.0,
        height_px: float = 0.0,
        is_open: bool = True,
        has_no_parking_sign: bool = False,
        has_obstacle: bool = False,
        distance_m: float = 0.0,
        rect_box_points: np.ndarray | None = None,
    ) -> None:
        self.spotted = spotted
        self.spot_id = spot_id
        self.center_px = center_px
        self.width_px = width_px
        self.height_px = height_px
        self.is_open = is_open
        self.has_no_parking_sign = has_no_parking_sign
        self.has_obstacle = has_obstacle
        self.distance_m = distance_m
        self.rect_box_points = rect_box_points  # 4x2 array of box points (int)


class ParkingVisionNode(Node):
    """
    ROS2 node that detects two parallel parking spots on the right and
    publishes which spot is available and the distance to it.
    """

    def __init__(self) -> None:
        super().__init__("parking_vision_node")

        self.get_logger().info("Initializing ParkingVisionNode...")

        # ---------------------------------------------------------------------
        # Publishers for orchestrator
        # ---------------------------------------------------------------------
        self.near_spot_open_pub = self.create_publisher(
            Bool, "/parking_bot/near_spot_open", 10
        )
        self.far_spot_open_pub = self.create_publisher(
            Bool, "/parking_bot/far_spot_open", 10
        )

        self.near_spot_distance_pub = self.create_publisher(
            Float32, "/parking_bot/near_spot_distance_m", 10
        )
        self.far_spot_distance_pub = self.create_publisher(
            Float32, "/parking_bot/far_spot_distance_m", 10
        )

        self.near_no_parking_pub = self.create_publisher(
            Bool, "/parking_bot/near_no_parking_sign", 10
        )
        self.far_no_parking_pub = self.create_publisher(
            Bool, "/parking_bot/far_no_parking_sign", 10
        )

        self.near_obstacle_pub = self.create_publisher(
            Bool, "/parking_bot/near_obstacle", 10
        )
        self.far_obstacle_pub = self.create_publisher(
            Bool, "/parking_bot/far_obstacle", 10
        )

        # Main orchestrator topics:
        self.open_spot_id_pub = self.create_publisher(
            String, "/parking_bot/open_spot_id", 10
        )
        self.open_spot_distance_pub = self.create_publisher(
            Float32, "/parking_bot/open_spot_distance_m", 10
        )

        # No parking sign template
        template_path = "no_parking_sign.jpeg"
        self.no_parking_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.no_parking_template is None:
            self.get_logger().error(
                f"Failed to load no parking sign template from: {template_path}"
            )
        else:
            self.get_logger().info(
                f"Loaded no parking sign template from: {template_path}"
            )

        self.NO_PARKING_THRESHOLD = 0.6

        # DepthAI / OAK-D setup: RGB + stereo depth aligned to RGB
        self.pipeline = self._create_pipeline()
        self.device = dai.Device(self.pipeline)

        # Queues
        self.q_rgb = self.device.getOutputQueue(
            name="rgb", maxSize=4, blocking=False
        )
        self.q_depth = self.device.getOutputQueue(
            name="depth", maxSize=4, blocking=False
        )

        # Main processing timer (20 Hz)
        self.timer = self.create_timer(0.05, self._process_frame)

        self.get_logger().info("ParkingVisionNode initialized successfully.")

    # DepthAI pipeline: ColorCamera + Mono + StereoDepth aligned to RGB
    def _create_pipeline(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()

        # RGB camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)

        # Mono cameras for stereo
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Stereo depth
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)  # align depth to RGB
        stereo.setSubpixel(False)  # you can turn this on if you want subpixel depth

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # Outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

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

        # Detect no parking sign
        no_parking_detected = self._detect_no_parking_sign(frame, spot)
        
        # Update spot status if no parking sign detected
        if no_parking_detected:
            spot.no_parking_sign = True
            spot.is_open = False  # Mark as unavailable

        # Publish no parking detection status
        no_parking_msg = Bool()
        no_parking_msg.data = no_parking_detected
        self.no_parking_pub.publish(no_parking_msg)

        # Publish occupancy as a Bool (False if no parking sign present)
        open_msg = Bool()
        open_msg.data = bool(spot.detected and spot.is_open and not no_parking_detected)
        self.spot_open_pub.publish(open_msg)

        # Compute and publish alignment command
        if spot.detected:
            if no_parking_detected:
                # Don't provide alignment if no parking sign detected
                cmd_msg = String()
                cmd_msg.data = AlignmentCommand.NO_PARKING
                self.alignment_cmd_pub.publish(cmd_msg)
                self.get_logger().info("No parking sign detected - skipping this spot")
            else:
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

    def _detect_no_parking_sign(
        self, 
        frame_bgr: np.ndarray, 
        spot: ParkingSpotDetection
    ) -> bool:
        """
        Detect no parking sign using template matching at multiple scales
        
        Returns True if no parking sign is detected
        """
        if self.no_parking_template is None:
            return False

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Template matching at multiple scales
        return self._template_matching(gray_frame, self.no_parking_template)

    def _template_matching(
        self, 
        gray_frame: np.ndarray, 
        template: np.ndarray
    ) -> bool:
        """
        Perform template matching at multiple scales
        """
        # Try multiple scales
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        for scale in scales:
            # Resize template
            width = int(template.shape[1] * scale)
            height = int(template.shape[0] * scale)
            
            # Skip if template is larger than frame
            if width > gray_frame.shape[1] or height > gray_frame.shape[0]:
                continue
                
            resized_template = cv2.resize(template, (width, height))
            
            # Perform template matching
            result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Check if match exceeds threshold
            if max_val > self.NO_PARKING_THRESHOLD:
                self.get_logger().info(f"No parking sign detected (template match: {max_val:.2f} at scale {scale})")
                return True
        
        return False

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