#!/usr/bin/env python3
# =============================================================================
# File: orchestrator.py
# Description: Process orchestrator for self parking
#
# Authors:   Trew Hoffman
# Created:     2025-12-2
#
# Notes:
#   - This is a ROS2 node that will take in vision information to make decisions on calling HAL commands


# =============================================================================


"""
Process Orchestrator


This node will:
- Use vision information from vision node to determine how far to move and in which direction
- Call HAL movement commands based on


Outputs:
- Robot movement
"""




from HAL import ParkingHAL
import time


def distance_test() -> None:
    hal = ParkingHAL(serial_port="/dev/ttyACM0")


    print("Reading tach once just to make sure it works...")
    try:
        t = hal._get_tach()
        print(f"Initial tach: {t}")
    except Exception as e:
        print(f"Error reading tach: {e}")
        return


    input("prepare for driving, then press Enter...")


    hal.drive(0.5)  # 50 cm


    print("Done. Reading tach again...")
    t2 = hal._get_tach()
    print(f"Final tach: {t2}, delta={t2 - t}")


def parallel_park() -> None:
    hal = ParkingHAL(serial_port="/dev/ttyACM0")


    input("Prepare for driving, then press Enter...")


    # reverse parking procedure


    # 1) GO FORWARD
    print("Going forward...")
    hal.set_steering(0.5)  # straight (0.5), right (1.0), left (0.0)
    hal.drive(0.3)
    time.sleep(1)


    # 2) REVERSE LEFT
    print("Going back...")
    hal.set_steering(0.9)  # straight (0.5), right (1.0), left (0.0)
    hal.drive(-0.2)
    time.sleep(1)


    # 2) REVERSE RIGHT
    print("Going back...")
    hal.set_steering(0.0)  # straight (0.5), right (1.0), left (0.0)
    hal.drive(-0.2)
    time.sleep(1)


    # END WITH WHEELS STRAIGHT
    print("Straightening out...")
    hal.set_steering(0.5)
    hal.drive(0.1)


    # Step 3: Stop
    hal.stop() # END OF CODE
    print("Reverse Parked!")




if __name__ == "__main__":
    #distance_test()
    parallel_park()
