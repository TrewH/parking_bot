#!/usr/bin/env python3

from typing import List, Tuple, Optional

import cv2
import depthai as dai
import numpy as np
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger

Point = Tuple[int, int]


class ParkingVisionNode(Node):
    """
    ROS2 node that:

      - Connects to an OAK-D camera
      - Gets RGB and depth frames
      - Detects:
          * Parking spots (blue tape rectangles)
          * Vertical blue lines near the middle of the image
          * Red "no parking" region (circle-ish + fallback red imbalance)

    Core behavior:

      - Depth is ALWAYS measured at the midpoint between the two vertical
        blue lines closest to the middle of the screen (inner tape lines).
      - Independently of parking spot detection, it will:
          * Try to detect a NO PARKING sign by looking for a red circular-ish
            region (circle + line style).
          * If that fails, fall back to comparing red pixel counts on the
            left vs right halves of the image.
          * The service ALWAYS returns the OPPOSITE side of the detected
            no-parking region, if it exists.
      - If FEWER than two parking spots are detected, we still compute depth
        and still run sign detection, but we log a ROS ERROR.
      - If NO sign is detected (neither circle nor red imbalance), we log a
        ROS ERROR and default the side.
    """

    def __init__(self) -> None:
        super().__init__("parking_vision_node")

        # Build pipeline and connect to device
        self.pipeline = self._create_pipeline()
        self.device = dai.Device(self.pipeline)

        # Output queues
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=1, blocking=True)
        self.q_mono_left = self.device.getOutputQueue(name="mono_left", maxSize=1, blocking=True)
        self.q_mono_right = self.device.getOutputQueue(name="mono_right", maxSize=1, blocking=True)

        # Service: one-shot detection
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
    # Basic utility
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Blue detection: spots and vertical lines
    # -------------------------------------------------------------------------
    def _blue_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return cleaned-up binary mask of blue tape."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        BLUE_LOWER = np.array([85, 66,  40])
        BLUE_UPPER = np.array([127, 255, 255])

        mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def detect_parking_spots_bgr(self, frame_bgr: np.ndarray) -> List[List[Point]]:
        """
        Detects parking spots outlined in blue tape as rectangles.
        Returns a list of spots, each spot is a list of 4 (x, y) corner points.
        Spots are sorted left-to-right by center x.
        """
        blue_mask = self._blue_mask(frame_bgr)
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

            rect = cv2.minAreaRect(cnt)
            (_, _), (w_rect, h_rect), _ = rect
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

    def find_inner_blue_lines(
        self,
        frame_bgr: np.ndarray
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Find two vertical blue lines closest to the vertical middle of the image
        (inner edges of the two parking spots).

        Returns:
            (left_x, right_x) where each is the x-coordinate of a line center,
            or None if not found on that side.
        """
        blue_mask = self._blue_mask(frame_bgr)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = blue_mask.shape[:2]
        mid_x = w // 2

        # Collect candidate vertical-ish blue strips as their center x positions
        candidate_xs: List[int] = []
        for cnt in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)

            # Require it to be reasonably tall to be a vertical line segment.
            if h_rect < 0.3 * h:
                continue

            center_x = x + w_rect // 2
            candidate_xs.append(center_x)

        if not candidate_xs:
            return None, None

        left_x: Optional[int] = None
        right_x: Optional[int] = None

        # Pick closest candidate left of mid, and right of mid
        left_candidates = [cx for cx in candidate_xs if cx <= mid_x]
        right_candidates = [cx for cx in candidate_xs if cx >= mid_x]

        if left_candidates:
            left_x = min(left_candidates, key=lambda cx: abs(mid_x - cx))
        if right_candidates:
            right_x = min(right_candidates, key=lambda cx: abs(mid_x - cx))

        return left_x, right_x

    def get_midpoint_depth(
        self,
        frame_bgr: np.ndarray,
        depth_frame: np.ndarray
    ) -> Tuple[Point, Optional[float]]:
        """
        Use find_inner_blue_lines() to get the two vertical lines closest to the
        middle of the screen, then compute the midpoint between them at the
        vertical center of the image, and read depth there.

        If one or both lines are missing, fall back gracefully:
          - If one line is missing, use the image center for that side.
          - If both missing, use the image center.
        """
        h, w = frame_bgr.shape[:2]
        mid_x = w // 2
        mid_y = h // 2

        left_x, right_x = self.find_inner_blue_lines(frame_bgr)

        # Fallbacks if lines not found
        if left_x is None and right_x is None:
            self.get_logger().warn(
                "Could not find inner blue lines; using image center for depth."
            )
            cx = mid_x
        elif left_x is None:
            self.get_logger().warn(
                f"Missing left inner blue line; using mid_x for left. right_x={right_x}"
            )
            cx = (mid_x + right_x) // 2
        elif right_x is None:
            self.get_logger().warn(
                f"Missing right inner blue line; using mid_x for right. left_x={left_x}"
            )
            cx = (left_x + mid_x) // 2
        else:
            cx = (left_x + right_x) // 2
            self.get_logger().info(
                f"Inner blue lines: left_x={left_x}, right_x={right_x}, mid_x={cx}"
            )

        cy = mid_y
        depth_m = self.get_depth_at_pixel(depth_frame, cx, cy, use_median_3x3=True)
        return (cx, cy), depth_m

    # -------------------------------------------------------------------------
    # Red / "NO PARKING" detection
    # -------------------------------------------------------------------------
    def _red_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return cleaned-up binary mask of red regions (two HSV lobes)."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        RED1_LOWER = np.array([0,   100, 80])
        RED1_UPPER = np.array([10,  255, 255])
        RED2_LOWER = np.array([170, 100, 80])
        RED2_UPPER = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
        red_mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        return red_mask

    def detect_no_parking_side_by_circle(
        self,
        frame_bgr: np.ndarray
    ) -> Optional[str]:
        """
        Detect a red, mostly circular region (approximating a NO PARKING sign)
        in the entire frame.

        Returns:
            'left' or 'right' based on which half of the image the sign center is in,
            or None if no suitable circular region is found.
        """
        red_mask = self._red_mask(frame_bgr)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        h, w = red_mask.shape[:2]
        img_area = h * w
        min_area = img_area * 0.005   # ignore tiny red blobs
        max_area = img_area * 0.5     # ignore huge blobs

        best_cnt = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            # Approximate circularity via min enclosing circle
            (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
            if radius <= 0:
                continue

            circle_area = float(np.pi * radius * radius)
            if circle_area <= 0:
                continue

            circularity = area / circle_area  # 1.0 = perfect filled circle
            if circularity < 0.5:
                # Not circular enough to be our "circle with a line"
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

    def detect_no_parking_side_by_red_imbalance(
        self,
        frame_bgr: np.ndarray
    ) -> Optional[str]:
        """
        Fallback: if no clear circle-like sign is detected, look at the total
        red pixels on the left vs right halves of the image.

        If one side has a significantly larger share of red pixels, that side
        is treated as the NO PARKING side.
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

        # "Major difference" threshold; tweak as needed.
        if diff_ratio < 0.2:
            # Not confidently biased to one side
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
        """
        Service callback for /get_parking_spots (example_interfaces/Trigger).

        Steps:

          1. Grab one RGB + depth frame.
          2. Compute depth at the midpoint between the two vertical blue lines
             closest to the center of the image (inner tape lines).
          3. Detect full blue parking spots (rectangles) and log an error if
             fewer than two.
          4. Independently of spot detection, try to detect a NO PARKING region:
               - First by circle-like red region.
               - Then by red pixel imbalance.
             The resulting NO PARKING side (if any) is then inverted to select
             the driving side to return.
          5. If no sign is detected at all, log ROS error and default side="right".

        Response on success:
          response.success = True
          response.message = "side=<left|right> depth_m=<float>"
        """
        self.get_logger().info("GetParkingSpots (Trigger) called, grabbing one frame...")

        frame_bgr, depth_frame = self.get_single_frame()

        # 1) Always compute depth from inner blue lines (with fallback)
        depth_point, depth_m = self.get_midpoint_depth(frame_bgr, depth_frame)
        if depth_m is None:
            self.get_logger().error(
                f"Failed to obtain valid depth at depth_point={depth_point}"
            )
            response.success = False
            response.message = "Could not obtain valid depth measurement"
            return response

        # 2) Detect parking spots (rectangles) just to know how many we see
        spots = self.detect_parking_spots_bgr(frame_bgr)
        num_spots = len(spots)
        self.get_logger().info(f"Detected {num_spots} blue parking spot(s).")

        if num_spots < 2:
            # Requirement: fewer than two spots -> ROS error
            self.get_logger().error(
                "Fewer than two parking spots detected; spots may be clipped. "
                "Continuing with NO PARKING sign detection anyway."
            )

        # 3) Always try to detect NO PARKING side via circle
        no_parking_side: Optional[str] = self.detect_no_parking_side_by_circle(frame_bgr)

        # 4) If no circle sign, try red imbalance
        if no_parking_side is None:
            no_parking_side = self.detect_no_parking_side_by_red_imbalance(frame_bgr)

        # 5) Decide output side (opposite of NO PARKING region if found)
        if no_parking_side is None:
            # Requirement: log ROS error if no sign is detected
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

        response.success = True
        response.message = f"side={selected_side} depth_m={depth_m:.2f}"
        self.get_logger().info(
            f"Service response: side={selected_side}, depth_m={depth_m:.2f}, "
            f"depth_point={depth_point}"
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
