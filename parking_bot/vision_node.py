#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import depthai as dai
import numpy as np
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger

Point = Tuple[int, int]


@dataclass
class ParkingSpotResult:
    corners: List[Point]          # 4 points in image space
    center: Point                 # (cx, cy)
    has_no_parking_sign: bool
    has_obstacle: bool
    depth_m: Optional[float]      # distance at center in meters


class ParkingVisionNode(Node):
    """
    ROS2 node that:
      - Connects to an OAK-D camera
      - Gets RGB, mono-left, mono-right, and depth frames
      - Detects two parking spots (blue tape rectangles)
      - Checks for a red "NO PARKING" sign and white obstacle inside each
      - Chooses the best spot according to:
          * If one has a NO PARKING sign and the other is empty → choose the empty one
          * If one has an obstacle and the other has a NO PARKING sign → choose the obstacle
          * If both are empty → choose the right-most spot
    """

    def __init__(self) -> None:
        super().__init__("parking_vision_node")

        # Optional: template image for NO PARKING sign (grayscale)
        self.sign_template_gray: Optional[np.ndarray] = cv2.imread(
            'no_parking_sign.jpeg', cv2.IMREAD_GRAYSCALE
        )

        # Build pipeline and connect to device
        self.pipeline = self._create_pipeline()
        self.device = dai.Device(self.pipeline)

        # Output queues
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=1, blocking=True)
        self.q_mono_left = self.device.getOutputQueue(name="mono_left", maxSize=1, blocking=True)
        self.q_mono_right = self.device.getOutputQueue(name="mono_right", maxSize=1, blocking=True)

        # Service: one-shot parking spot detection
        self.srv = self.create_service(
            Trigger,
            "get_parking_spots",
            self.handle_get_parking_spots,
        )

        self.get_logger().info("ParkingVisionNode ready. Service: /get_parking_spots")

    # -------------------------------------------------------------------------
    # DepthAI pipeline setup
    # -------------------------------------------------------------------------
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
        stereo.setSubpixel(False)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # Outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        xout_mono_left = pipeline.create(dai.node.XLinkOut)
        xout_mono_left.setStreamName("mono_left")
        mono_left.out.link(xout_mono_left.input)

        xout_mono_right = pipeline.create(dai.node.XLinkOut)
        xout_mono_right.setStreamName("mono_right")
        mono_right.out.link(xout_mono_right.input)

        return pipeline

    def get_single_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blocking wait for one RGB and one depth frame from the OAK-D.
        Returns (frame_bgr, depth_frame).
        """
        in_rgb = self.q_rgb.get()    # blocking until a frame arrives
        in_depth = self.q_depth.get()

        frame_bgr = in_rgb.getCvFrame()     # HxWx3 BGR
        depth_frame = in_depth.getFrame()   # HxW uint16 (mm)

        return frame_bgr, depth_frame

    # -------------------------------------------------------------------------
    # Parking spot detection & analysis
    # -------------------------------------------------------------------------
    def detect_parking_spots_bgr(self, frame_bgr: np.ndarray) -> List[List[Point]]:
        """
        Detects parking spots outlined in blue tape.
        Returns a list of spots, each spot is a list of 4 (x, y) corner points.
        Spots are sorted left-to-right by center x.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        BLUE_LOWER = np.array([85, 55,  70])
        BLUE_UPPER = np.array([110, 255, 255])

        blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

        # Clean up mask a bit
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        spots: List[List[Point]] = []
        h, w = frame_bgr.shape[:2]
        min_area = (w * h) * 0.01      # tweak: min area as fraction of image
        max_area = (w * h) * 0.4       # tweak: max area

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            # Polygon approximation
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # We want rectangles (4 corners)
            if len(approx) != 4:
                continue

            # Optionally enforce rectangularity by checking bounding box ratio
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w_rect, h_rect), angle = rect
            if w_rect == 0 or h_rect == 0:
                continue
            aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
            if aspect_ratio < 0.5 or aspect_ratio > 4.0:
                continue

            corners = [(int(p[0][0]), int(p[0][1])) for p in approx]
            spots.append(corners)

        # If more than 2 are found, keep the two largest by area
        if len(spots) > 2:
            def spot_area(corners: List[Point]) -> float:
                cnt = np.array(corners).reshape(-1, 1, 2)
                return cv2.contourArea(cnt)

            spots = sorted(spots, key=spot_area, reverse=True)[:2]

        # Sort left-to-right by center x
        def spot_center_x(corners: List[Point]) -> float:
            xs = [p[0] for p in corners]
            return float(sum(xs)) / len(xs)

        spots = sorted(spots, key=spot_center_x)

        return spots

    def analyze_spot(
        self,
        frame_bgr: np.ndarray,
        corners: List[Point],
        sign_template_gray: Optional[np.ndarray] = None,
        sign_template_thresh: float = 0.6,
    ) -> Tuple[bool, bool]:
        """
        Given the frame and 4 corners of a parking spot,
        returns (has_no_parking_sign, has_obstacle).
        Uses color + optional template matching.
        """
        h, w = frame_bgr.shape[:2]

        # Create mask for the polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        # Shrink mask a bit to avoid including blue borders
        kernel = np.ones((15, 15), np.uint8)  # tweak size
        mask_eroded = cv2.erode(mask, kernel, iterations=1)

        # Extract ROI using the mask
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # --- Red detection (NO PARKING sign) ---
        RED1_LOWER = np.array([0,   100, 80])
        RED1_UPPER = np.array([10,  255, 255])
        RED2_LOWER = np.array([170, 100, 80])
        RED2_UPPER = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
        red_mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        red_mask_spot = cv2.bitwise_and(red_mask, red_mask, mask=mask_eroded)

        red_pixels = cv2.countNonZero(red_mask_spot)
        spot_pixels = cv2.countNonZero(mask_eroded)
        red_ratio = red_pixels / spot_pixels if spot_pixels > 0 else 0.0

        HAS_SIGN_BY_COLOR = red_ratio > 0.10  # tweak threshold

        # Optional template matching for extra confidence
        HAS_SIGN_BY_TEMPLATE = False
        if sign_template_gray is not None:
            x, y, w_box, h_box = cv2.boundingRect(pts)
            roi_bgr = frame_bgr[y:y+h_box, x:x+w_box]
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

            roi_mask = mask_eroded[y:y+h_box, x:x+w_box]
            roi_gray_masked = roi_gray.copy()
            roi_gray_masked[roi_mask == 0] = 0

            res = cv2.matchTemplate(roi_gray_masked, sign_template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            HAS_SIGN_BY_TEMPLATE = max_val > sign_template_thresh

        has_sign = HAS_SIGN_BY_COLOR or HAS_SIGN_BY_TEMPLATE

        # --- White obstacle detection ---
        WHITE_LOWER = np.array([0,   0,   180])
        WHITE_UPPER = np.array([180, 60,  255])

        white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
        white_mask_spot = cv2.bitwise_and(white_mask, white_mask, mask=mask_eroded)

        # Remove red region from white (if sign has white text etc.)
        white_mask_spot = cv2.bitwise_and(
            white_mask_spot,
            cv2.bitwise_not(red_mask_spot)
        )

        contours, _ = cv2.findContours(white_mask_spot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0.0
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a > max_area:
                max_area = a

        white_ratio = max_area / spot_pixels if spot_pixels > 0 else 0.0
        has_obstacle = white_ratio > 0.02  # tweak threshold (~2% area)

        return has_sign, has_obstacle

    def get_depth_at_pixel(
        self,
        depth_frame: np.ndarray,
        x: int,
        y: int,
        use_median_3x3: bool = True
    ) -> Optional[float]:
        """
        Returns depth in meters at (x, y).
        Assumes depth_frame is uint16 in millimeters.
        If depth is invalid (0 or NaN region), returns None.
        """
        h, w = depth_frame.shape[:2]

        if x < 0 or x >= w or y < 0 or y >= h:
            return None

        if not use_median_3x3:
            raw_val = int(depth_frame[y, x])
            if raw_val <= 0:
                return None
            return raw_val / 1000.0  # mm -> m

        # 3x3 neighborhood around (x, y)
        x0 = max(0, x - 1)
        x1 = min(w, x + 2)
        y0 = max(0, y - 1)
        y1 = min(h, y + 2)

        patch = depth_frame[y0:y1, x0:x1].astype(np.int32)
        valid = patch[patch > 0]  # 0 usually means invalid depth

        if valid.size == 0:
            return None

        median_mm = float(np.median(valid))
        return median_mm / 1000.0  # mm -> m

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        depth_frame: np.ndarray,
        sign_template_gray: Optional[np.ndarray] = None
    ) -> List[ParkingSpotResult]:
        """
        Full pipeline for a single RGB + depth frame.
        frame_bgr: HxWx3 BGR image
        depth_frame: HxW uint16 depth image (mm), aligned with frame_bgr
        """
        spots_corners = self.detect_parking_spots_bgr(frame_bgr)
        results: List[ParkingSpotResult] = []

        for corners in spots_corners:
            has_sign, has_obstacle = self.analyze_spot(frame_bgr, corners, sign_template_gray)

            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))
            center = (cx, cy)

            depth_m = self.get_depth_at_pixel(depth_frame, cx, cy, use_median_3x3=True)

            result = ParkingSpotResult(
                corners=corners,
                center=center,
                has_no_parking_sign=has_sign,
                has_obstacle=has_obstacle,
                depth_m=depth_m,
            )
            results.append(result)

        return results

    # -------------------------------------------------------------------------
    # Best-spot selection logic
    # -------------------------------------------------------------------------
    def _classify_spot(self, spot: ParkingSpotResult) -> str:
        """
        Classify a spot as 'no_parking', 'obstacle', or 'empty'.

        NOTE: no_parking overrides obstacle if both are detected.
        """
        if spot.has_no_parking_sign:
            return "no_parking"
        if spot.has_obstacle:
            return "obstacle"
        return "empty"

    def choose_best_spot(
        self, results: List[ParkingSpotResult]
    ) -> Optional[Tuple[ParkingSpotResult, str]]:
        """
        Apply the rules:

        - If there is a no parking sign in one, the best spot should always be the
          other, empty spot.
        - If there is an obstacle in one and a no parking sign in the other,
          choose the one with an obstacle.
        - If both spots are empty, choose the right-most spot.
        - Tie-breakers and extra cases:
          * If one empty and one obstacle → choose empty.
          * If both obstacle → choose right-most.
          * If both no_parking → no valid spot.
        Returns (best_spot, side_str) where side_str is 'left' or 'right'.
        """
        if not results:
            return None

        # Spots are sorted left-to-right already
        if len(results) == 1:
            spot = results[0]
            status = self._classify_spot(spot)
            if status == "no_parking":
                return None
            # Only one spot visible; treat it as 'left' logically
            return spot, "left"

        # We only care about the first two (left, right)
        left_spot = results[0]
        right_spot = results[1]

        left_status = self._classify_spot(left_spot)
        right_status = self._classify_spot(right_spot)

        self.get_logger().info(
            f"Spot statuses: left={left_status}, right={right_status}"
        )

        # 1) exactly one empty and one no_parking → choose empty
        if {"empty", "no_parking"} == {left_status, right_status}:
            if left_status == "empty":
                return left_spot, "left"
            else:
                return right_spot, "right"

        # 2) one obstacle and one no_parking → choose obstacle
        if {"obstacle", "no_parking"} == {left_status, right_status}:
            if left_status == "obstacle":
                return left_spot, "left"
            else:
                return right_spot, "right"

        # 3) both empty → choose right-most
        if left_status == "empty" and right_status == "empty":
            return right_spot, "right"

        # 4) one empty, one obstacle → choose empty
        if {"empty", "obstacle"} == {left_status, right_status}:
            if left_status == "empty":
                return left_spot, "left"
            else:
                return right_spot, "right"

        # 5) both obstacle → choose right-most
        if left_status == "obstacle" and right_status == "obstacle":
            return right_spot, "right"

        # 6) both no_parking → no valid spot
        if left_status == "no_parking" and right_status == "no_parking":
            return None

        # Fallback (shouldn't really hit this, but just in case):
        # prefer any non-no_parking, else None
        for spot, side, status in [
            (left_spot, "left", left_status),
            (right_spot, "right", right_status),
        ]:
            if status != "no_parking":
                return spot, side

        return None

    # -------------------------------------------------------------------------
    # Service handler
    # -------------------------------------------------------------------------
    def handle_get_parking_spots(self, request, response):
        """
        Service callback for /get_parking_spots using example_interfaces/Trigger.

        Grabs ONE frame, runs detection once, and encodes ONLY:
          - which side ('left' or 'right') the best spot is on
          - the distance (depth_m) to that spot

        Response format on success:
          "side=left depth_m=1.23"
        """
        self.get_logger().info("GetParkingSpots (Trigger) called, grabbing one frame...")

        frame_bgr, depth_frame = self.get_single_frame()
        results = self.process_frame(frame_bgr, depth_frame, self.sign_template_gray)

        if not results:
            response.success = False
            response.message = "No spots detected"
            self.get_logger().info("Detection complete: no spots.")
            return response

        best = self.choose_best_spot(results)

        if best is None:
            response.success = False
            response.message = "No valid parking spot found"
            self.get_logger().info("Detection complete: no valid spot.")
            return response

        best_spot, side = best
        depth_val = best_spot.depth_m if best_spot.depth_m is not None else -1.0

        response.success = True
        response.message = f"side={side} depth_m={depth_val:.2f}"

        self.get_logger().info(
            f"Detection complete: {len(results)} spots, best side={side}, depth_m={depth_val:.2f}"
        )
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
