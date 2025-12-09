#!/usr/bin/env python3

import cv2
import depthai as dai


def create_pipeline() -> dai.Pipeline:
    """
    Minimal pipeline: just RGB preview out.
    """
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    return pipeline


def main() -> None:
    pipeline = create_pipeline()

    print("[INFO] Connecting to OAK-D...")
    with dai.Device(pipeline) as device:
        print("[INFO] Connected. Press 'q' or ESC in the window to quit.")

        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)

        win_name = "OAK-D RGB"

        while True:
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()

            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] 'q' or ESC pressed, exiting viewer loop.")
                break

            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window closed by user, exiting viewer loop.")
                break

    cv2.destroyAllWindows()
    print("[INFO] Viewer closed.")


if __name__ == "__main__":
    main()
