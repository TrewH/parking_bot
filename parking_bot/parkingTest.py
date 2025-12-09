# =============================================================================
# File: robot_control.py
# Description: Simple wrapper around HAL for testing robot movement.
# =============================================================================

from HAL import ParkingHAL
import time

def park_right():
    # RIGHT PARALLEL PARK

    hal = ParkingHAL()

    # Forward 1
    hal.set_steering(0.5)
    hal.drive(0.6)
    time.sleep(0.5)

    # Reverse 1
    hal.set_steering(1.0)
    hal.drive(-0.425)
    time.sleep(0.5)

    # Reverse 2
    hal.set_steering(0.0)
    hal.drive(-0.465)
    time.sleep(0.5)

    # Forward 2
    hal.set_steering(0.5)
    hal.drive(0.125)
    time.sleep(0.5)

    hal.shutdown()

def park_left():
    # LEFT PARALLEL PARK

    hal = ParkingHAL()

    # Forward 1
    hal.set_steering(0.5)
    hal.drive(0.6)
    time.sleep(0.5)

    # Reverse 1
    hal.set_steering(0.0)
    hal.drive(-0.45)
    time.sleep(0.5)

    # Reverse 2
    hal.set_steering(1.0)
    hal.drive(-0.5)
    time.sleep(0.5)

    # Forward 2
    hal.set_steering(0.5)
    hal.drive(0.18)
    time.sleep(0.5)
    
    hal.shutdown()

if __name__ == "__main__":
    park_left()
