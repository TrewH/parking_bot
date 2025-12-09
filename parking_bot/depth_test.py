#!/usr/bin/env python3

import time
from typing import Optional

import apriltag
import cv2
import depthai as dai
import numpy as np


MAX_VALID_DEPTH_MM = 8000  # treat >= 8 m as "invalid / infinity" for our use-case


def get_depth_at_pixel(
    depth_frame: np.ndarray,
    x: int,
    y: int,
    use_median_3x3: bool = True,
    max_radius: int = 5,
) -> Optional[float]:
    """
    Returns depth in meters at (x, y) using an expanding window search.
    Filters out invalid values (0 or >= MAX_VALID_DEPTH_MM).
    """
    h, w = depth_frame.shape[:2]

    if x < 0 or x >= w or y < 0 or y >= h:
        print(f"[WARN] Requested depth at out-of-bounds pixel ({x}, {y})")
        return None

    if not use_median_3x3:
        raw_mm = int(depth_frame[y, x])
        if 0 < raw_mm < MAX_VALID_DEPTH_MM:
            return raw_mm / 1000.0
        return None

    # Try 3×3, then 5×5, ... up to max_radius
    for radius in range(1, max_radius + 1):
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)

        patch = depth_frame[y0:y1, x0:x1].astype(np.int32)
        valid = patch[(patch > 0) & (patch < MAX_VALID_DEPTH_MM)]

        if valid.size > 0:
            return float(np.median(valid)) / 1000.0

    print(f"[WARN] No valid depth around ({x},{y}) within radius={max_radius}.")
    return None


def create_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    # ==========================================================
    # RGB camera (aligned with stereo depth)
    # Use 640x400 to match mono 400p aspect ratio for alignment.
    # ==========================================================
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(640, 400)
    cam_rgb.setPreviewKeepAspectRatio(False)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(15)

    # ==========================================================
    # Mono cameras
    # ==========================================================
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)

    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # 400p gives good compromise of precision and speed, and
    # matches our RGB preview aspect ratio.
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setFps(15)
    mono_right.setFps(15)

    # ==========================================================
    # Stereo depth node tuned for precision (not density/range)
    # ==========================================================
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)

    # Enable subpixel for higher depth precision
    stereo.setSubpixel(True)

    # Better accuracy near 1–5 m (we don't need ultra-close or 20 m range)
    stereo.setExtendedDisparity(False)

    # Left-right check helps reject bad matches (improves reliability)
    stereo.setLeftRightCheck(True)

    # Small median filter to clean up noise while preserving edges
    stereo.setMedianFilter(dai.MedianFilter.KERNEL_3x3)

    # Align depth to RGB camera
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Optionally tweak confidence; lower threshold -> more points, higher -> fewer but cleaner
    stereo.setConfidenceThreshold(200)

    # Link mono to stereo
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # ==========================================================
    # XLink outputs
    # ==========================================================
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


def main() -> None:
    # AprilTag detector
    options = apriltag.DetectorOptions(
        families="tag36h11",
        quad_decimate=1.0,
        refine_edges=True,
    )
    detector = apriltag.Detector(options)

    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=1, blocking=True)
        q_depth = device.getOutputQueue("depth", maxSize=1, blocking=True)

        print("[INFO] Starting depth test... Move AprilTag around. Ctrl+C quits.")

        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RGB", 960, 600)

        try:
            while True:
                try:
                    in_rgb = q_rgb.get()
                    in_depth = q_depth.get()
                except RuntimeError as e:
                    print(f"[ERROR] OAK-D read error (X_LINK_ERROR): {e}")
                    print("[ERROR] Unplug/replug the OAK-D and restart the script.")
                    break

                frame_bgr = in_rgb.getCvFrame()
                depth_frame = in_depth.getFrame()

                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(gray)

                if detections:
                    # Use the first detection (you can loop if needed)
                    det = detections[0]
                    x, y = int(det.center[0]), int(det.center[1])
                    print(f"[INFO] Tag {det.tag_id} at pixel ({x}, {y})")

                    depth_m = get_depth_at_pixel(
                        depth_frame,
                        x,
                        y,
                        use_median_3x3=True,
                        max_radius=5,
                    )

                    if depth_m is not None:
                        print(f"[INFO] Depth: {depth_m:.3f} m\n")
                    else:
                        print("[WARN] No valid depth at tag center (or depth was infinite).\n")

                    # Visualize the tag center on the RGB image
                    cv2.circle(frame_bgr, (x, y), 5, (0, 0, 255), 2)
                else:
                    print("[INFO] No AprilTag detected.\n")

                # Show RGB window
                cv2.imshow("RGB", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n[INFO] Exiting...")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
