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
          * Blue-tape parking spots on the ground (as blobs).
          * A red "NO PARKING" region (circle-ish + fallback red imbalance).

    Core behavior:

      - Depth is measured at the front edge of the parking spots:
          * We find blue blobs in the lower part of the image (the two spots).
          * For each blob, we take its bottom-most y-pixel.
          * We average those two y-values to get the front edge y.
          * x is always the horizontal image center (car is centered).
      - Independently of depth, we:
          * Try to detect a NO PARKING sign by looking for a red circular-ish
            region (circle + line style).
          * If that fails, fall back to comparing red pixel counts on the
            left vs right halves of the image.
          * The service ALWAYS returns the OPPOSITE side of the detected
            no-parking region, if it exists.
      - We also detect how many blue parking spots are visible and log an
        error if fewer than two are found.
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
        frame_rgb = in_rgb.getCvFrame()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
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
    # Color masks
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
        """Return cleaned-up binary mask of blue tape."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        BLUE_LOWER = np.array([93, 45,  104], dtype=np.uint8)
        BLUE_UPPER = np.array([111, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    # -------------------------------------------------------------------------
    # Blue spot geometry (for depth + spot count)
    # -------------------------------------------------------------------------
    
    def _find_spot_bottoms_from_blobs(
        self,
        frame_bgr: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Find blue parking-spot blobs in the entire image and return
        a list of (cx, y_bottom) pairs:

            - cx       = approximate horizontal center of the blob
            - y_bottom = bottom-most y pixel of that blob (in full-image coords)

        The list is sorted left-to-right by cx.
        """
        blue_mask = self._blue_mask(frame_bgr)
        h, w = blue_mask.shape[:2]

        contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        img_area = h*w
        min_area = img_area * 0.03

        blobs: List[Tuple[int, int]] = []

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)

            if area < min_area:
                continue

            ys = cnt[:, 0, 1]  # full-image coords
            xs = cnt[:, 0, 0]

            y_bottom = int(ys.max())
            cx = int(xs.mean())

            blobs.append((cx, y_bottom))
            self.get_logger().info(
                f"[DEBUG] contour {i} accepted -> cx={cx}, y_bottom={y_bottom}"
            )

        blobs.sort(key=lambda p: p[0])
        self.get_logger().info(f"[DEBUG] blobs detected (cx, y_bottom): {blobs}")
        return blobs



    def detect_parking_spots_bgr(self, frame_bgr: np.ndarray) -> List[List[Point]]:
        """
        Detect parking spots outlined in blue tape as blobs/rectangles.

        Returns a list of spots, each as a list of 4 (x, y) corner points.
        Only used for logging how many spots are visible; depth comes from
        _find_spot_bottoms_from_blobs().
        """
        blue_mask = self._blue_mask(frame_bgr)
        contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

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

            # We want quadrilateral-ish shapes (4 corners)
            if len(approx) != 4:
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

    def get_midpoint_depth(
        self,
        frame_bgr: np.ndarray,
        depth_frame: np.ndarray
    ) -> Tuple[Point, Optional[float]]:
        """
        Compute depth at the point where the parking spots START (front edge).

        Assumptions:
          - Car is centered horizontally between the two spots.
          - We only care about the y at which the tape blobs begin.
          - So x is always image_center_x; y is based on the bottoms of the blobs.

        Steps:
          1. Find blue blobs in the lower part of the image.
          2. For each blob, compute bottom-most y.
          3. If we have two blobs, average their y_bottom values.
             If we have one, just use its y_bottom (with a warning).
             If none, fall back to image center.
          4. Measure depth at (x_mid, y_front).
        """
        h, w = frame_bgr.shape[:2]
        x_mid = w // 2

        blobs = self._find_spot_bottoms_from_blobs(frame_bgr)

        if len(blobs) >= 2:
            (_, y_bottom_left), (_, y_bottom_right) = blobs[0], blobs[1]
            y_front = int((y_bottom_left + y_bottom_right) / 2.0)
            self.get_logger().info(
                f"Front edge from blobs: y_bottom_left={y_bottom_left}, "
                f"y_bottom_right={y_bottom_right}, y_front={y_front}"
            )
        elif len(blobs) == 1:
            (_, y_front) = blobs[0]
            self.get_logger().warn(
                f"Only one parking-spot blob detected; using its bottom y={y_front}"
            )
        else:
            # Total fallback if we see no blobs at all
            y_front = h // 2
            self.get_logger().warn(
                "No parking-spot blobs detected; using image center for depth."
            )

        # Clamp just in case
        y_front = max(0, min(h - 1, y_front))

        depth_m = self.get_depth_at_pixel(
            depth_frame,
            x_mid,
            y_front,
            use_median_3x3=True,
        )

        return (x_mid, y_front), depth_m

    # -------------------------------------------------------------------------
    # Red / "NO PARKING" detection
    # -------------------------------------------------------------------------
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
        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

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
          2. Compute depth at the front edge of the parking spots using
             blob geometry (bottoms of blue blobs, averaged).
          3. Detect blue parking spots and log how many we see (error if < 2).
          4. Independently, try to detect a NO PARKING region:
               - First by circle-like red region.
               - Then by red pixel imbalance.
             The resulting NO PARKING side (if any) is then inverted to select
             the driving side to return.
          5. If no sign is detected at all, default side="right".

        Response on success:
          response.success = True
          response.message = "side=<left|right> depth_m=<float>"
        """
        self.get_logger().info("GetParkingSpots (Trigger) called, grabbing one frame...")

        frame_bgr, depth_frame = self.get_single_frame()

        # 1) Compute depth at the front edge of the spots
        depth_point, depth_m = self.get_midpoint_depth(frame_bgr, depth_frame)
        if depth_m is None:
            self.get_logger().error(
                f"Failed to obtain valid depth at depth_point={depth_point}"
            )
            response.success = False
            response.message = "Could not obtain valid depth measurement"
            return response

        # 2) Detect parking spots (rectangles/blobs) just to know how many we see
        spots = self.detect_parking_spots_bgr(frame_bgr)
        num_spots = len(spots)
        self.get_logger().info(f"Detected {num_spots} blue parking spot(s).")

        if num_spots < 2:
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
