#!/usr/bin/env python3

from typing import List, Tuple, Optional

import cv2
import depthai as dai
import numpy as np
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger
import apriltag  # NEW

Point = Tuple[int, int]


class ParkingVisionNode(Node):
    """
    ROS2 node that:

      - Connects to an OAK-D camera
      - Gets RGB and depth frames
      - Detects:
          * A red "NO PARKING" region (circle-ish + fallback red imbalance)
          * (Optionally) blue-tape spots for logging.
          * An AprilTag used as a distance marker.

    Depth behavior (NEW):

      - We ONLY trust depth at a vertical AprilTag (family tag36h11).
      - On each service call, we:
          * Grab up to N frames (default 10).
          * For each frame, detect AprilTags.
          * If a tag is found, take its center pixel (cx, cy),
            look up depth there, and store it.
          * At the end, we take the MEDIAN of all valid depth samples.

      - If no valid tag depth is obtained from any sample:
          * We set depth_m = 1000.0 (sentinel "no tag" distance).
          * We set response.success = False.
          * We append "no_tag=1" in the message string so the orchestrator
            can detect this condition.

    Side selection:

      - Independently of depth, we still:
          * Try to detect a NO PARKING sign via circle-like red region.
          * If that fails, fall back to red pixel imbalance.
          * The resulting NO PARKING side (if any) is inverted to select
            the driving side to return.
          * If no sign at all, default side="right".
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

        # AprilTag detector (family tag36h11)
        options = apriltag.DetectorOptions(
            families="tag36h11",
            quad_decimate=1.0,
            refine_edges=True,
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

        OpenCV / DepthAI convention:
            depth_frame[row, col] = depth_frame[y, x]
            origin (0, 0) is top-left
            x increases to the RIGHT
            y increases DOWN
        """
        h, w = depth_frame.shape[:2]

        if x < 0 or x >= w or y < 0 or y >= h:
            self.get_logger().warn(f"Requested depth at out-of-bounds pixel ({x}, {y})")
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
            self.get_logger().warn(
                f"No valid depth values in 3x3 patch around ({x}, {y}). Patch:\n{patch}"
            )
            return None

        median_mm = float(np.median(valid))
        self.get_logger().debug(
            f"Depth 3x3 patch around ({x},{y}) (mm):\n{patch}\n"
            f"Using median depth {median_mm:.1f} mm"
        )
        return median_mm / 1000.0  # mm -> m

    # -------------------------------------------------------------------------
    # AprilTag-based depth sampling
    # -------------------------------------------------------------------------
    def sample_tag_depth(
        self,
        num_samples: int = 10
    ) -> Tuple[Optional[np.ndarray], Optional[Point], Optional[float]]:
        """
        Take up to num_samples frames, detect AprilTag (tag36h11) in each,
        and collect depth measurements at the tag center.

        Returns:
            (last_frame_bgr, tag_point, depth_median)

            - last_frame_bgr: the last RGB frame captured (always non-None
              if camera is working).
            - tag_point: (x, y) of the tag center from the LAST valid depth sample,
              or None if no tag/depth was valid.
            - depth_median: median of all valid depth samples in meters,
              or None if no valid samples.
        """
        depths: List[float] = []
        tag_point: Optional[Point] = None
        last_frame_bgr: Optional[np.ndarray] = None

        for i in range(num_samples):
            frame_bgr, depth_frame = self.get_single_frame()
            last_frame_bgr = frame_bgr

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            detections = self.apriltag_detector.detect(gray)
            self.get_logger().info(
                f"[DEBUG] sample {i+1}/{num_samples}: detected {len(detections)} AprilTag(s)"
            )

            if not detections:
                continue

            det = detections[0]  # just use the first for now
            cx, cy = det.center
            x = int(round(cx))
            y = int(round(cy))

            depth_m = self.get_depth_at_pixel(depth_frame, x, y, use_median_3x3=True)
            if depth_m is None:
                self.get_logger().warn(
                    f"[WARN] No valid depth at AprilTag center for sample {i+1}. "
                    f"Pixel=({x},{y})"
                )
                continue

            depths.append(depth_m)
            tag_point = (x, y)
            self.get_logger().info(
                f"[INFO] Sample {i+1}: Tag ID={det.tag_id}, "
                f"pixel=({x},{y}), depth_m={depth_m:.3f}"
            )

        if last_frame_bgr is None:
            # Should not happen if the camera is running, but guard just in case.
            self.get_logger().error("No frames received from OAK-D in sample_tag_depth().")
            return None, None, None

        if not depths:
            self.get_logger().error(
                f"No valid AprilTag depth samples after {num_samples} attempts."
            )
            return last_frame_bgr, None, None

        median_depth_m = float(np.median(depths))
        self.get_logger().info(
            f"Median depth over {len(depths)} valid AprilTag samples: {median_depth_m:.3f} m"
        )
        return last_frame_bgr, tag_point, median_depth_m

    # -------------------------------------------------------------------------
    # Color masks (red and blue)
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
    # Blue spot geometry (for spot count / logging only)
    # -------------------------------------------------------------------------
    def detect_parking_spots_bgr(self, frame_bgr: np.ndarray) -> List[List[Point]]:
        """
        Detect parking spots outlined in blue tape as blobs/rectangles.

        Returns a list of spots, each as a list of 4 (x, y) corner points.
        Only used for logging how many spots are visible; depth is now
        determined solely from the AprilTag, not the floor.
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

          1. Take up to N frames (N=10) and:
               - Detect AprilTag in each.
               - Get depth at tag center for each valid detection.
               - Aggregate median depth.
          2. Use the last RGB frame from that sampling to:
               - Detect blue parking spots (for logging only).
               - Detect a NO PARKING region (circle-like or red imbalance).
          3. Decide output side (opposite of NO PARKING region if found).
          4. If no valid tag depth:
               - depth_m = 1000.0 (sentinel "no tag").
               - success = False.
               - message includes "no_tag=1".
          5. If we do have a valid depth:
               - success = True.

        Response format:
          response.success = True/False
          response.message = "side=<left|right> depth_m=<float> [no_tag=1]"
        """
        self.get_logger().info("GetParkingSpots (Trigger) called, sampling AprilTag depth...")

        frame_bgr, tag_point, depth_m = self.sample_tag_depth(num_samples=10)

        if frame_bgr is None:
            # Catastrophic camera failure
            self.get_logger().error("No RGB frame available; cannot proceed.")
            response.success = False
            response.message = "Camera failure: no RGB frame."
            return response

        # Logging: # of blue spots (still optional, independent of depth)
        spots = self.detect_parking_spots_bgr(frame_bgr)
        num_spots = len(spots)
        self.get_logger().info(f"Detected {num_spots} blue parking spot(s).")

        # Detect NO PARKING side via circle or red imbalance
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

        # Handle depth / tag status
        no_tag = False
        if depth_m is None:
            no_tag = True
            depth_m = 1000.0  # sentinel distance
            self.get_logger().error(
                "No valid AprilTag depth from any sample; "
                "sending sentinel depth_m=1000.0 and marking success=False."
            )

        # Build response
        response.success = not no_tag
        msg = f"side={selected_side} depth_m={depth_m:.2f}"
        if no_tag:
            msg += " no_tag=1"
        response.message = msg

        self.get_logger().info(
            f"Service response: success={response.success}, "
            f"side={selected_side}, depth_m={depth_m:.2f}, tag_point={tag_point}, "
            f"message='{msg}'"
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
