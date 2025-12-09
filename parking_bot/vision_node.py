#!/usr/bin/env python3

from typing import List, Tuple, Optional

import apriltag
import cv2
import depthai as dai
import numpy as np
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger

Point = Tuple[int, int]

MAX_VALID_DEPTH_MM = 8000  # treat >= 8 m as invalid for our use-case


class ParkingVisionNode(Node):
    """
    ROS2 node that:

      - Builds a DepthAI pipeline using:
          * Mono left & right @ 400p
          * StereoDepth in HIGH_ACCURACY mode with subpixel, LR check,
            3x3 median filter, depth+rectified outputs.
          * Color camera for RGB frames (for red/blue detection).

      - Detects:
          * A red "NO PARKING" region (circle-ish + fallback red imbalance)
          * (Optionally) blue-tape spots for logging.
          * An AprilTag (tag36h11) and measures its distance using the
            median depth over the tag polygon.

      - On each service call:
          * Samples up to N frames from rectified-left + depth.
          * For each detection, computes depth from the tag area.
          * Returns the median of valid per-frame medians.
    """

    def __init__(self) -> None:
        super().__init__("parking_vision_node")

        # Build pipeline and connect to device
        self.pipeline = self._create_pipeline()
        self.device = dai.Device(self.pipeline)

        # Output queues
        # Depth in mm (rectified-left coordinate system)
        self.q_depth = self.device.getOutputQueue(
            name="depth", maxSize=1, blocking=True
        )
        # Rectified left mono for AprilTag detection
        self.q_rect_left = self.device.getOutputQueue(
            name="rectifiedLeft", maxSize=1, blocking=True
        )
        # RGB for red/blue detection and spot finding
        self.q_rgb = self.device.getOutputQueue(
            name="rgb", maxSize=1, blocking=True
        )
        # Disparity (optional; for debugging only)
        self.q_disparity = self.device.getOutputQueue(
            name="disparity", maxSize=1, blocking=False
        )

        # AprilTag detector (family tag36h11), tuned like test file
        options = apriltag.DetectorOptions(
            families="tag36h11",
            quad_decimate=1.0,     # keep full 400p resolution
            refine_edges=True,
            refine_decode=True,
            refine_pose=True,
        )
        self.apriltag_detector = apriltag.Detector(options)

        # Service: one-shot detection
        self.srv = self.create_service(
            Trigger,
            "get_parking_spots",
            self.handle_get_parking_spots,
        )

        self.get_logger().info("ParkingVisionNode ready. Service: /get_parking_spots")

    # -------------------------------------------------------------------------
    # DepthAI pipeline setup (matching test file stereo config)
    # -------------------------------------------------------------------------
    def _create_pipeline(self) -> dai.Pipeline:
        """
        Build a pipeline that uses:

          - Mono left & right @ 400p
          - StereoDepth with HIGH_ACCURACY preset, subpixel, LR check,
            3x3 median filter, depth + rectified + disparity outputs.
          - Color camera for RGB preview.
        """
        pipeline = dai.Pipeline()

        # --- Mono cameras (same as getMonoCamera) ---
        mono_left = pipeline.createMonoCamera()
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setFps(30)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

        mono_right = pipeline.createMonoCamera()
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setFps(30)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # --- Stereo depth (same as getStereoPair) ---
        stereo = pipeline.createStereoDepth()

        # Link mono cameras to stereo node
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # Configure for accuracy
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.setSubpixel(True)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setMedianFilter(dai.MedianFilter.KERNEL_3x3)

        stereo.setOutputDepth(True)          # depth in mm (rectified-left space)
        stereo.setOutputRectified(True)      # rectified mono images
        stereo.setConfidenceThreshold(200)   # filter low-confidence matches

        # --- Color camera for RGB (for red / blue detection) ---
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(640, 400)  # 400p-ish to roughly match mono height
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)

        # XLink outputs
        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        xout_rectified_left = pipeline.createXLinkOut()
        xout_rectified_left.setStreamName("rectifiedLeft")
        stereo.rectifiedLeft.link(xout_rectified_left.input)

        xout_disparity = pipeline.createXLinkOut()
        xout_disparity.setStreamName("disparity")
        stereo.disparity.link(xout_disparity.input)

        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        return pipeline

    # -------------------------------------------------------------------------
    # Frame helpers
    # -------------------------------------------------------------------------
    def _get_rect_left_and_depth(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Blocking read of a rectified-left frame + depth frame.
        Returns (rect_left, depth_frame), or (None, None) on failure.
        """
        try:
            in_rect = self.q_rect_left.get()
            in_depth = self.q_depth.get()
        except RuntimeError as e:
            self.get_logger().error(f"Error getting rectified/depth frames: {e}")
            return None, None

        rect_left = in_rect.getCvFrame()   # 400p mono, uint8
        depth_frame = in_depth.getFrame()  # HxW uint16, mm

        return rect_left, depth_frame

    def _get_rgb_frame(self) -> Optional[np.ndarray]:
        """
        Blocking read of a single RGB frame (BGR order).
        """
        try:
            in_rgb = self.q_rgb.get()
        except RuntimeError as e:
            self.get_logger().error(f"Error getting RGB frame: {e}")
            return None
        frame_bgr = in_rgb.getCvFrame()  # already BGR
        return frame_bgr

    # -------------------------------------------------------------------------
    # AprilTag depth logic (matching test file)
    # -------------------------------------------------------------------------
    def _get_tag_depth_from_area(
        self,
        depth_frame: np.ndarray,
        det: apriltag.Detection,
    ) -> Optional[float]:
        """
        Compute depth as the median of all valid depth pixels inside the
        AprilTag polygon (more stable than a single pixel).
        """
        h, w = depth_frame.shape[:2]

        corners = det.corners.astype(np.int32)  # (4, 2)

        # Clamp to image bounds
        corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

        pts = corners.reshape(-1, 1, 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)

        patch = depth_frame[mask == 255].astype(np.int32)
        valid = patch[(patch > 0) & (patch < MAX_VALID_DEPTH_MM)]

        if valid.size == 0:
            self.get_logger().warn(
                "[WARN] No valid depth values inside AprilTag area."
            )
            return None

        median_mm = float(np.median(valid))
        return median_mm / 1000.0  # mm -> m

    def sample_tag_depth(
        self,
        num_samples: int = 10,
    ) -> Tuple[Optional[np.ndarray], Optional[Point], Optional[float]]:
        """
        Take up to num_samples frames, detect AprilTag (tag36h11) in each,
        and collect depth measurements over the *area* of the tag, using the
        same technique as the test script (rectified-left + HIGH_ACCURACY stereo).

        Returns:
            (frame_bgr, tag_point, depth_median)

            - frame_bgr: RGB BGR frame for color-based detection.
            - tag_point: (x, y) of the tag center (rectified-left coords)
              from the LAST valid depth sample, or None if no tag/depth valid.
            - depth_median: median of all valid per-sample medians in meters,
              or None if no valid samples.
        """
        depths: List[float] = []
        tag_point: Optional[Point] = None

        for i in range(num_samples):
            rect_left, depth_frame = self._get_rect_left_and_depth()
            if rect_left is None or depth_frame is None:
                self.get_logger().warn(
                    f"[WARN] sample {i+1}/{num_samples}: missing rectified/depth frame."
                )
                continue

            # rect_left is already mono; boost contrast like test
            gray = cv2.equalizeHist(rect_left)

            detections = self.apriltag_detector.detect(gray)

            if not detections:
                continue

            det = detections[0]  # Use first detection for now
            cx, cy = det.center
            x = int(round(cx))
            y = int(round(cy))
            tag_point = (x, y)

            depth_m = self._get_tag_depth_from_area(depth_frame, det)
            if depth_m is None:
                self.get_logger().warn(
                    f"[WARN] No valid depth inside AprilTag area for "
                    f"sample {i+1}."
                )
                continue

            depths.append(depth_m)
            self.get_logger().info(
                f"[INFO] Sample {i+1}: Tag ID={det.tag_id}, "
                f"center_pixel=({x},{y}), median_depth={depth_m:.3f} m"
            )

        # Always grab a fresh RGB frame for downstream color detection
        frame_bgr = self._get_rgb_frame()
        if frame_bgr is None:
            self.get_logger().error(
                "No RGB frame received from OAK-D in sample_tag_depth()."
            )

        if not depths:
            self.get_logger().error(
                f"No valid AprilTag depth samples after {num_samples} attempts."
            )
            return frame_bgr, None, None

        median_depth_m = float(np.median(depths))
        self.get_logger().info(
            f"Median depth over {len(depths)} valid AprilTag area samples: "
            f"{median_depth_m:.3f} m"
        )
        return frame_bgr, tag_point, median_depth_m

    # -------------------------------------------------------------------------
    # Color masks (red and blue) – unchanged
    # -------------------------------------------------------------------------
    def _red_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return cleaned-up binary mask of red regions (two HSV lobes)."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        RED1_LOWER = np.array([0,   70, 50])
        RED1_UPPER = np.array([10,  255, 255])
        RED2_LOWER = np.array([170, 70, 50])
        RED2_UPPER = np.array([179, 255, 255])

        red_mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
        red_mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        return red_mask

    def _blue_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return cleaned-up binary mask of blue tape (still here for logging/debug)."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        BLUE_LOWER = np.array([93, 45,  104], dtype=np.uint8)
        BLUE_UPPER = np.array([111, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    # -------------------------------------------------------------------------
    # Red / "NO PARKING" detection – unchanged
    # -------------------------------------------------------------------------
    def detect_no_parking_side_by_circle(
        self,
        frame_bgr: np.ndarray
    ) -> Optional[str]:
        """
        Detect a red, mostly circular region (approximating a NO PARKING sign).
        """
        red_mask = self._red_mask(frame_bgr)
        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        h, w = red_mask.shape[:2]
        img_area = h * w
        min_area = img_area * 0.005
        max_area = img_area * 0.5

        best_cnt = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
            if radius <= 0:
                continue

            circle_area = float(np.pi * radius * radius)
            if circle_area <= 0:
                continue

            circularity = area / circle_area
            if circularity < 0.5:
                continue

            if area > best_area:
                best_area = area
                best_cnt = cnt

        if best_cnt is None:
            return None

        M = cv2.moments(best_cnt)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        mid_x = w // 2
        sign_side = "left" if cx < mid_x else "right"

        self.get_logger().info(
            f"Circle-like red region detected at x={cx}, classified as side={sign_side}"
        )
        return sign_side

    def error_correction(self, measured_m: float) -> float:
        """
        Linear depth calibration:

            corrected = a * measured + b
        """
        return measured_m * 0.834028 + 0.0863219

    def detect_no_parking_side_by_red_imbalance(
        self,
        frame_bgr: np.ndarray
    ) -> Optional[str]:
        """
        Fallback: red pixel imbalance between left and right halves.
        """
        red_mask = self._red_mask(frame_bgr)
        h, w = red_mask.shape[:2]
        mid_x = w // 2

        left_mask = red_mask[:, :mid_x]
        right_mask = red_mask[:, mid_x:]

        red_left = int(cv2.countNonZero(left_mask))
        red_right = int(cv2.countNonZero(right_mask))
        total_red = red_left + red_right

        if total_red == 0:
            return None

        diff = abs(red_left - red_right)
        diff_ratio = diff / float(total_red)

        if diff_ratio < 0.2:
            return None

        if red_left > red_right:
            side = "left"
        else:
            side = "right"

        self.get_logger().info(
            f"Red imbalance detected: left={red_left}, right={red_right}, "
            f"diff_ratio={diff_ratio:.2f}, side={side}"
        )
        return side

    # -------------------------------------------------------------------------
    # Service handler
    # -------------------------------------------------------------------------
    def handle_get_parking_spots(self, request, response):
        self.get_logger().info(
            "GetParkingSpots (Trigger) called, sampling AprilTag depth..."
        )

        # Sample stereo / rectified-left for AprilTag distance
        frame_bgr, tag_point, depth_m = self.sample_tag_depth(num_samples=10)

        if frame_bgr is None:
            self.get_logger().error("No RGB frame available; cannot proceed.")
            response.success = False
            response.message = "Camera failure: no RGB frame."
            return response

        # --- NO PARKING detection (circle first, then red imbalance) ---
        no_parking_side: Optional[str] = self.detect_no_parking_side_by_circle(frame_bgr)
        if no_parking_side is None:
            no_parking_side = self.detect_no_parking_side_by_red_imbalance(frame_bgr)

        if no_parking_side is None:
            self.get_logger().info(
                "No NO PARKING sign detected (neither circle-like nor red imbalance). "
                "Defaulting side=right."
            )
            selected_side = "right"
        else:
            selected_side = "left" if no_parking_side == "right" else "right"
            self.get_logger().info(
                f"NO PARKING region on {no_parking_side}; returning opposite side={selected_side}"
            )

        # --- Depth + correction ---
        no_tag = False
        corrected_depth: float

        if depth_m is None:
            no_tag = True
            depth_m = 1000.0
            corrected_depth = depth_m
            self.get_logger().error(
                "No valid AprilTag depth from any sample; "
                "sending sentinel depth_m=1000.0 and marking success=False."
            )
        else:
            raw_depth = depth_m
            corrected_depth = self.error_correction(depth_m)
            self.get_logger().info(
                f"Raw AprilTag depth={raw_depth:.3f} m, "
                f"corrected depth={corrected_depth:.3f} m"
            )

        response.success = not no_tag
        msg = f"side={selected_side} depth_m={corrected_depth:.2f}"
        if no_tag:
            msg += " no_tag=1"
        response.message = msg

        return response


def main(args=None):
    rclpy.init(args=args)
    node = ParkingVisionNode()
    try:
        node.get_logger().info("Starting up ParkingVisionNode!")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down ParkingVisionNode...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
