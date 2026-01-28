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


i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50


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

    def set_angle(self, angle):
        angle = angle
        self.servo.angle = angle
        self.current_angle = angle
        print(f"[Servo] Set angle → {angle}°")
        time.sleep(0.3)

    def full_sweep(self, sweeprange, speed=0.02):
        """
        Sweep from home_angle → (home_angle + sweeprange°) and back,
        """
        print("[Servo] Starting home-based ", sweeprange,"° sweep")

        start = int(self.home_angle)
        end = self.home_angle - sweeprange

        #servo only goes to 180
        if end < 0:
            end = 0

        #Sweep forward
        print("[Servo] Sweeping Forward")
        for a in range(start,end-1, -2):
            self.servo.angle = a
            time.sleep(speed)

        # Sweep backward
        print("[Servo] Sweeping Backward")
        for a in range(end, start + 1, 2):
            self.servo.angle = a
            time.sleep(speed)

        # Return to home
        print(f"[Servo] Returning to home angle {self.home_angle}°")
        self.set_angle(self.home_angle)


if __name__ == "__main__":
    print(" Sweep Test Starting...")
    
    arm = SweeperServo(channel=0, home_angle=180)
    arm.full_sweep(sweeprange=120)
