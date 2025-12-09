import cv2
from cv2 import aruco
import os

# Map from tag id to human label
TAG_CONFIG = {
    0: "left_parallel",
    1: "left_perpendicular",
    2: "right_parallel",
    3: "right_perpendicular",
}

def make_output_dir(dirname="apriltags_out"):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

def generate_tag_images(side_pixels=400, out_dir="apriltags_out"):
    """
    side_pixels is the image size in pixels (square).
    400 is fine for printing on normal paper.
    """
    out_dir = make_output_dir(out_dir)

    # Use the AprilTag 36h11 dictionary inside OpenCV
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

    for tag_id, label in TAG_CONFIG.items():
        # Some OpenCV builds use drawMarker, some use generateImageMarker
        try:
            img = aruco.drawMarker(dictionary, tag_id, side_pixels)
        except AttributeError:
            # Fallback if drawMarker is not available
            img = aruco.generateImageMarker(dictionary, tag_id, side_pixels)

        filename = os.path.join(out_dir, f"tag36h11_id{tag_id}_{label}.png")
        cv2.imwrite(filename, img)
        print(f"Saved {filename}")

if __name__ == "__main__":
    generate_tag_images()
