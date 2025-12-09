#!/usr/bin/env python3

import time
from typing import Optional

import apriltag
import cv2
import depthai as dai
import numpy as np

MAX_VALID_DEPTH_MM = 8000  # treat >= 8 m as "invalid / infinity" for our use-case


def getMonoCamera(pipeline: dai.Pipeline, is_left: bool):
    """Create and configure a mono camera (left or right)."""
    mono = pipeline.createMonoCamera()
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono.setFps(30)

    if is_left:
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    return mono


def getStereoPair(pipeline: dai.Pipeline, mono_left, mono_right):
    """
    Create a stereo depth node and link mono cameras to it.

    Tuned for higher accuracy:
      - HIGH_ACCURACY preset
      - subpixel enabled
      - left-right check
      - 3x3 median filter
    """
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

    stereo.setOutputDepth(True)          # depth in mm
    stereo.setOutputRectified(True)      # rectified mono images
    stereo.setConfidenceThreshold(200)   # filter low-confidence matches

    return stereo


def get_tag_depth_from_area(
    depth_frame: np.ndarray,
    det: apriltag.Detection,
) -> Optional[float]:
    """
    Compute depth as the median of all valid depth pixels inside the
    AprilTag polygon (more stable than a single pixel).
    """
    h, w = depth_frame.shape[:2]

    # det.corners is (4, 2) float32: [[x0,y0], [x1,y1], ...]
    corners = det.corners.astype(np.int32)

    # Clamp to image bounds
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

    # Build mask for tag polygon
    pts = corners.reshape(-1, 1, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)

    patch = depth_frame[mask == 255].astype(np.int32)
    valid = patch[(patch > 0) & (patch < MAX_VALID_DEPTH_MM)]

    if valid.size == 0:
        print("[WARN] No valid depth values inside AprilTag area.")
        return None

    median_mm = float(np.median(valid))
    return median_mm / 1000.0


def create_pipeline() -> dai.Pipeline:
    """
    Build a pipeline that uses:

      - Mono left & right @ 400p
      - StereoDepth with rectified outputs + depth + disparity

    We do NOT create an RGB camera here; AprilTag detection runs
    on the rectified left mono image.
    """
    pipeline = dai.Pipeline()

    # Mono cameras
    mono_left = getMonoCamera(pipeline, is_left=True)
    mono_right = getMonoCamera(pipeline, is_left=False)

    # Stereo depth
    stereo = getStereoPair(pipeline, mono_left, mono_right)

    # XLink outputs
    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    xout_rectified_left = pipeline.createXLinkOut()
    xout_rectified_left.setStreamName("rectifiedLeft")
    stereo.rectifiedLeft.link(xout_rectified_left.input)

    # Disparity (optional; not used for distance here, but available)
    xout_disparity = pipeline.createXLinkOut()
    xout_disparity.setStreamName("disparity")
    stereo.disparity.link(xout_disparity.input)

    return pipeline


def get_latest_msg(queue: dai.DataOutputQueue):
    """
    Non-blocking: drain the queue and return only the most recent message.
    Returns None if the queue is empty.
    """
    latest = None
    while True:
        msg = queue.tryGet()
        if msg is None:
            break
        latest = msg
    return latest


def main() -> None:
    # AprilTag detector, tuned to be a bit more aggressive for small/far tags
    options = apriltag.DetectorOptions(
        families="tag36h11",
        quad_decimate=1.0,     # keep full 400p resolution
        refine_edges=True,
        refine_decode=True,
        refine_pose=True,
    )
    detector = apriltag.Detector(options)

    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        # Non-blocking, maxSize=1 to avoid backlog
        depth_queue = device.getOutputQueue(
            name="depth", maxSize=1, blocking=False
        )
        rectified_left_queue = device.getOutputQueue(
            name="rectifiedLeft", maxSize=1, blocking=False
        )
        disparity_queue = device.getOutputQueue(
            name="disparity", maxSize=1, blocking=False
        )

        print("[INFO] Starting AprilTag depth measurement test (1 Hz)...")
        print("[INFO] Move the AprilTag in front of the camera. Ctrl+C to quit.\n")

        try:
            while True:
                start_t = time.time()

                # Get the most recent frames (if any)
                in_depth = get_latest_msg(depth_queue)
                in_rect_left = get_latest_msg(rectified_left_queue)
                # disparity_msg = get_latest_msg(disparity_queue)  # unused

                if in_depth is None or in_rect_left is None:
                    print("[WARN] No new frames available yet.")
                else:
                    depth_frame = in_depth.getFrame()          # uint16, mm
                    rect_left = in_rect_left.getCvFrame()      # 400p mono (uint8)

                    # Preprocess: boost contrast for better long-range detection
                    gray = rect_left  # already mono
                    gray = cv2.equalizeHist(gray)

                    detections = detector.detect(gray)

                    if detections:
                        det = detections[0]  # Use first detection
                        cx, cy = det.center
                        x = int(round(cx))
                        y = int(round(cy))

                        print(
                            f"[INFO] Tag ID={det.tag_id} at rectified-left pixel ({x}, {y})"
                        )

                        depth_m = get_tag_depth_from_area(depth_frame, det)

                        if depth_m is not None:
                            corrected = depth_m * 0.834028 + 0.0863219
                            print(
                                f"[INFO] Estimated distance (area median): {corrected:.3f} m\n"
                            )
                        else:
                            print(
                                "[WARN] No valid depth inside tag area for this frame.\n"
                            )
                    else:
                        print("[INFO] No AprilTag detected in this frame.\n")

                # Enforce ~1 Hz rate
                elapsed = time.time() - start_t
                sleep_time = max(0.0, 1.0 - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[INFO] Exiting...")

    # No windows were opened, so nothing to destroy.


if __name__ == "__main__":
    main()
