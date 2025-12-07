# =============================================================================
# File: sweeper_arm.py
# Description: Sweeper Arm Controller using PCA9685 + DS3225MG Servo
#
# Author:      Manan Tuteja
# Created:     2025-12-5
#
# Notes:
#   - Controls a DS3225MG digital servo using a PCA9685 driver.
#   - Provides methods to set an angle and sweep.
# =============================================================================

import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo


# -----------------------------
# I2C + PCA9685 Setup
# -----------------------------
i2c = busio.I2C(board.SCL, board.SDA)

pca = PCA9685(i2c)
pca.frequency = 50   # Standard servo frequency


# =============================================================================
# SweeperServo Class
# =============================================================================
class SweeperServo:
    def __init__(self, channel=0, min_pulse=500, max_pulse=2500, home_angle=0):
        """
        Initialize servo and automatically move it to the home angle.
        """
        self.servo = servo.Servo(
            pca.channels[channel],
            min_pulse=min_pulse,
            max_pulse=max_pulse
        )

        self.home_angle = home_angle
        self.current_angle = home_angle

        # Move to home on startup
        print(f"[Servo] Initializing… moving to home angle {home_angle}°")
        self.set_angle(home_angle)

    # ------------------------------------------------------------------
    def set_angle(self, angle):
        """

        """
        angle = angle
        self.servo.angle = angle
        self.current_angle = angle
        print(f"[Servo] Set angle → {angle}°")
        time.sleep(0.3)

    # ------------------------------------------------------------------
    def full_sweep(self, speed=0.01):
        """
        Sweep from home_angle → (home_angle + 180°) and back,
        automatically clamping to safe range.
        """
        print("[Servo] Starting home-based 180° sweep")

        start = int(self.home_angle)
        end = self.home_angle + 180  # clamp max to 180°

        # Sweep forward
        for a in range(start, end + 1, 2):
            self.servo.angle = a
            time.sleep(speed)

        # Sweep backward
        for a in range(end, start - 1, -2):
            self.servo.angle = a
            time.sleep(speed)

        # Return to home
        print(f"[Servo] Returning to home angle {self.home_angle}°")
        self.set_angle(self.home_angle)


if __name__ == "__main__":
    print(" Sweep Test Starting...")

    # Initialize at right-most angle
    while True:
        arm = SweeperServo(channel=0, home_angle=7.5)
        arm.full_sweep()
