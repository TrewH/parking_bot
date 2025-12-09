#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import apriltag
from typing import Optional


def get_depth_at_pixel(
    depth_frame: np.ndarray,
    x: int,
    y: int,
    use_median_3x3: bool = True,
) -> Optional[float]:
    """
    Returns depth in meters at (x, y) from a uint16 depth_frame in millimeters.
    If depth is invalid (0) in the neighborhood, returns None.

    OpenCV / DepthAI convention:
        depth_frame[row, col] = depth_frame[y, x]
        origin (0, 0) is top-left
        x increases to the RIGHT
        y increases DOWN
    """
    h, w = depth_frame.shape[:2]

    if x < 0 or x >= w or y < 0 or y >= h:
        print(f"[WARN] Requested depth at out-of-bounds pixel ({x}, {y})")
        return None

    if not use_median_3x3:
        raw_mm = int(depth_frame[y, x])
        if raw_mm <= 0:
            return None
        return raw_mm / 1000.0

    # 3×3 patch around (x, y)
    x0 = max(0, x - 1)
    x1 = min(w, x + 2)
    y0 = max(0, y - 1)
    y1 = min(h, y + 2)

    patch = depth_frame[y0:y1, x0:x1].astype(np.int32)
    valid = patch[patch > 0]

    if valid.size == 0:
        print(
            f"[WARN] No valid depth values in 3x3 patch around ({x}, {y}). Patch:\n{patch}"
        )
        return None

    median_mm = float(np.median(valid))
    return median_mm / 1000.0


def create_pipeline() -> dai.Pipeline:
    """
    Create a DepthAI pipeline with:
      - Color camera (RGB) for AprilTag detection
      - Stereo depth aligned to RGB for depth lookup at tag center
    """
    pipeline = dai.Pipeline()

    # Color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    # Mono cameras for stereo
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # Stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setSubpixel(False)  # depth in integer millimeters
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)  # align depth to RGB

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


def main() -> None:
    # Build pipeline
    pipeline = create_pipeline()

    # AprilTag detector (family 36h11)
    options = apriltag.DetectorOptions(
        families="tag36h11",
        quad_decimate=1.0,
        refine_edges=True,
    )
    detector = apriltag.Detector(options)

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=1, blocking=True)
        q_depth = device.getOutputQueue("depth", maxSize=1, blocking=True)

        print("[INFO] Waiting for frames... Press 'q' to quit.")

        while True:
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()

            frame_bgr = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()  # uint16 (mm), aligned to RGB

            h, w = depth_frame.shape[:2]

            # AprilTag detection on RGB frame
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)

            tag_depth_m = None

            if detections:
                # For now, just use the first detected tag
                det = detections[0]
                cx, cy = det.center  # floats

                x = int(round(cx))
                y = int(round(cy))

                print(f"[INFO] Detected AprilTag ID={det.tag_id} at pixel ({x}, {y})")

                # Draw detection on the RGB frame
                corners = det.corners.astype(int)
                cv2.polylines(frame_bgr, [corners], True, (0, 255, 0), 2)
                cv2.circle(frame_bgr, (x, y), 5, (0, 0, 255), -1)

                # Get depth at that pixel
                tag_depth_m = get_depth_at_pixel(depth_frame, x, y, use_median_3x3=True)
                if tag_depth_m is not None:
                    print(f"[INFO] Depth at tag center: {tag_depth_m:.3f} m")
                    cv2.putText(
                        frame_bgr,
                        f"{tag_depth_m:.2f} m",
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                else:
                    print("[WARN] Could not obtain valid depth at tag center.")

            else:
                print("[INFO] No AprilTag detected in this frame.")

            # Optional: visualize RGB
            cv2.imshow("RGB + AprilTag + Depth", frame_bgr)

            # Optional: visualize depth as a colormap (for debugging)
            # Normalize for display
            # (Clip far distances so nearby stuff is visible)
            depth_vis = depth_frame.copy().astype(np.float32)
            depth_vis[depth_vis == 0] = np.nan  # treat invalid as NaN
            max_display_m = 3.0
            depth_vis = np.clip(depth_vis / 1000.0, 0.0, max_display_m)
            depth_vis = (depth_vis / max_display_m * 255.0).astype(np.uint8)
            depth_vis[np.isnan(depth_vis)] = 0
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            cv2.imshow("Depth (visualized)", depth_color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
