#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai


def create_pipeline() -> dai.Pipeline:
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

def nothing(x):
    pass

def main() -> None:
    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        rgb_win = "RGB"
        mask_win = "Mask"

        cv2.namedWindow(rgb_win)
        cv2.namedWindow(mask_win)

        cv2.createTrackbar("H_min", mask_win, 0,   179, nothing)
        cv2.createTrackbar("H_max", mask_win, 179, 179, nothing)
        cv2.createTrackbar("S_min", mask_win, 0,   255, nothing)
        cv2.createTrackbar("S_max", mask_win, 255, 255, nothing)
        cv2.createTrackbar("V_min", mask_win, 0,   255, nothing)
        cv2.createTrackbar("V_max", mask_win, 255, 255, nothing)

        print("Adjust HSV sliders until only the blue tape is white in the mask.")
        print("Press 'q' or ESC to quit, or close the windows.")

        while True:
            in_rgb = q_rgb.tryGet()
            if in_rgb is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

                if (cv2.getWindowProperty(rgb_win, cv2.WND_PROP_VISIBLE) < 1 or
                        cv2.getWindowProperty(mask_win, cv2.WND_PROP_VISIBLE) < 1):
                    break
                continue

            frame = in_rgb.getCvFrame()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            h_min = cv2.getTrackbarPos("H_min", mask_win)
            h_max = cv2.getTrackbarPos("H_max", mask_win)
            s_min = cv2.getTrackbarPos("S_min", mask_win)
            s_max = cv2.getTrackbarPos("S_max", mask_win)
            v_min = cv2.getTrackbarPos("V_min", mask_win)
            v_max = cv2.getTrackbarPos("V_max", mask_win)

            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])

            mask = cv2.inRange(hsv, lower, upper)


            cv2.imshow(rgb_win, frame)
            cv2.imshow(mask_win, mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("Final HSV ranges:")
                print("lower =", lower)
                print("upper =", upper)
                break

            if (cv2.getWindowProperty(rgb_win, cv2.WND_PROP_VISIBLE) < 1 or
                    cv2.getWindowProperty(mask_win, cv2.WND_PROP_VISIBLE) < 1):
                print("Window closed by user. Exiting.")
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
