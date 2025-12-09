#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np


def getFrame(queue: dai.DataOutputQueue) -> np.ndarray:
    """Grab a frame from a DepthAI output queue and convert to OpenCV image."""
    frame = queue.get()
    return frame.getCvFrame()


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
    """Create a stereo depth node and link mono cameras to it."""
    stereo = pipeline.createStereoDepth()

    # Link mono cameras to stereo node
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Configure stereo output
    stereo.setOutputDepth(False)          # We don't use raw depth in this script
    stereo.setOutputRectified(True)       # We DO want rectified left/right images
    stereo.setConfidenceThreshold(200)    # Filter out low-confidence disparity

    # You can tweak these if needed
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)             # Keep it simple; disparity is uint8

    return stereo


def mouseCallback(event, x, y, flags, param):
    """Track mouse position when left button is clicked."""
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y


if __name__ == "__main__":
    # Initial mouse position
    mouseX = 0
    mouseY = 200  # within 400p height

    # Build pipeline
    pipeline = dai.Pipeline()

    mono_left = getMonoCamera(pipeline, is_left=True)
    mono_right = getMonoCamera(pipeline, is_left=False)
    stereo = getStereoPair(pipeline, mono_left, mono_right)

    # Output nodes (XLinkOut) – these match your DepthAI version
    xout_disp = pipeline.createXLinkOut()
    xout_disp.setStreamName("disparity")

    xout_rectified_left = pipeline.createXLinkOut()
    xout_rectified_left.setStreamName("rectifiedLeft")

    xout_rectified_right = pipeline.createXLinkOut()
    xout_rectified_right.setStreamName("rectifiedRight")

    # Link stereo outputs to XLinkOuts
    stereo.disparity.link(xout_disp.input)
    stereo.rectifiedLeft.link(xout_rectified_left.input)
    stereo.rectifiedRight.link(xout_rectified_right.input)

    # Start device
    with dai.Device(pipeline) as device:
        disparity_queue = device.getOutputQueue(
            name="disparity", maxSize=1, blocking=False
        )
        rectified_left_queue = device.getOutputQueue(
            name="rectifiedLeft", maxSize=1, blocking=False
        )
        rectified_right_queue = device.getOutputQueue(
            name="rectifiedRight", maxSize=1, blocking=False
        )

        # Normalize disparity for visualization
        max_disp = stereo.getMaxDisparity()
        disparity_multiplier = 255 / max_disp if max_disp != 0 else 1.0

        # Only TWO windows: "Stereo Pair" and "Disparity"
        cv2.namedWindow("Stereo Pair")
        cv2.setMouseCallback("Stereo Pair", mouseCallback)

        side_by_side = False

        while True:
            # Get frames
            disparity = getFrame(disparity_queue)
            left_frame = getFrame(rectified_left_queue)
            right_frame = getFrame(rectified_right_queue)

            # Scale disparity up to 8-bit for color mapping
            disparity_vis = (disparity * disparity_multiplier).astype(np.uint8)
            disparity_vis = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)

            # Combine stereo pair either side-by-side or averaged
            if side_by_side:
                im_out = np.hstack((left_frame, right_frame))
            else:
                im_out = np.uint8(left_frame / 2 + right_frame / 2)

            im_out = cv2.cvtColor(im_out, cv2.COLOR_GRAY2BGR)

            # Draw line + marker at last mouse click
            h, w = im_out.shape[:2]
            y_clamped = max(0, min(mouseY, h - 1))
            x_clamped = max(0, min(mouseX, w - 1))

            im_out = cv2.line(im_out, (x_clamped, y_clamped), (w - 1, y_clamped),
                              (0, 0, 255), 2)
            im_out = cv2.circle(im_out, (x_clamped, y_clamped), 3,
                                (25, 255, 120), 2)

            # Show only these two windows
            cv2.imshow("Stereo Pair", im_out)
            cv2.imshow("Disparity", disparity_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                side_by_side = not side_by_side

        cv2.destroyAllWindows()
