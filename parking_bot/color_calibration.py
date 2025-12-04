import cv2
import numpy as np
import depthai as dai

# ---------------------------
# Helper: build pipeline
# ---------------------------
def create_pipeline():
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

# ---------------------------
# Trackbar callback (no-op)
# ---------------------------
def nothing(x):
    pass

def main():
    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        # Create windows
        cv2.namedWindow("RGB")
        cv2.namedWindow("Mask")

        # Create trackbars for HSV min/max
        cv2.createTrackbar("H_min", "Mask", 0,   179, nothing)
        cv2.createTrackbar("H_max", "Mask", 179, 179, nothing)
        cv2.createTrackbar("S_min", "Mask", 0,   255, nothing)
        cv2.createTrackbar("S_max", "Mask", 255, 255, nothing)
        cv2.createTrackbar("V_min", "Mask", 0,   255, nothing)
        cv2.createTrackbar("V_max", "Mask", 255, 255, nothing)

        print("Adjust HSV sliders until only the blue tape is white in the mask.")
        print("Press 'q' to quit.")

        while True:
            in_rgb = q_rgb.tryGet()
            if in_rgb is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            frame = in_rgb.getCvFrame()  # BGR
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Read trackbars
            h_min = cv2.getTrackbarPos("H_min", "Mask")
            h_max = cv2.getTrackbarPos("H_max", "Mask")
            s_min = cv2.getTrackbarPos("S_min", "Mask")
            s_max = cv2.getTrackbarPos("S_max", "Mask")
            v_min = cv2.getTrackbarPos("V_min", "Mask")
            v_max = cv2.getTrackbarPos("V_max", "Mask")

            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])

            mask = cv2.inRange(hsv, lower, upper)

            # Optional: masked image to visualize where tape is
            masked = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("RGB", frame)
            cv2.imshow("Mask", mask)      # white = detected
            # cv2.imshow("Masked", masked)  # uncomment if you want

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Final HSV ranges:")
                print("lower =", lower)
                print("upper =", upper)
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
