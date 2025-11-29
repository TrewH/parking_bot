# =============================================================================
# File: HAL.py
# Description: Hardware Abstraction Layer
#
# Authors:   Trew Hoffman, Terri Tai
# Created:     2025-11-26
#
# Notes:
#   - This HAL uses a fixed duty cycle and a measured constant speed (m/s).
#   - All higher-level logic should use drive_forward(), drive_backward(),
#     and turn() rather than interfacing with the VESC directly.
# =============================================================================

"""
Hardware Abstraction Layer (HAL).

This module is intended wrap VESC / steering control so the rest of the project can make calls based on distances and 
steering commands rather than duty cycles and on-durations.

We will measure how far the car moves under 0.05 duty cycle after 1 second of driving. This will give us a m/s velocity
that we can use to drive a desired distance by scaling the time that we are sending the duty signal. 0.05 
(or whatever otherduty cycle we deem a good speed) should be the only duty cycle used to reduce complexity. 

The rest of the robot's code should treat this HAL
as the only interface to the physical car.

Typical use:
    hal = ParkingHAL()
    hal.drive_forward(1.0)      # ~1 meter
    hal.drive_backward(0.3)
    hal.turn(0.5)               # slight right

Configuration:
    Edit HALConfig() to match your car's servo values, duty cycle, and
    measured straight-line speed.
"""


from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

# Remove type ignore after getting vesc setup in venv
from pyvesc.VESC import VESC  # type: ignore


@dataclass
class SteeringConfig:
    """
    Edit these values later; maximum and minimum usable servo positions
    """
    servo_center: float = 0.0
    servo_min: float = -0.5
    servo_max: float = 0.5


@dataclass
class HALConfig:
    """
    Base parameters for HAL class
    """
    serial_port: str = "/dev/ttyACM0"
    base_duty_cycle: float = 0.05
    base_speed_mps: float = 0.6   # Change after measuring
    steering: SteeringConfig = SteeringConfig()
    max_drive_duration_s: float = 5.0  # safety cap


class ParkingHAL:
    """
    Internally we convert distance to time assuming constant speed and
    apply a fixed duty cycle. This class will produce callable methods we can use
    after determining what distances and angles we want to drive.
    """

    def __init__(self, config: Optional[HALConfig] = None) -> None:
        self.config = config or HALConfig()
        self._vesc = VESC(serial_port=self.config.serial_port)

        # Center on startup always
        self.set_steering(0.0)
        self.stop()

    # ---------------------------------------------------------------------
    # Low-level callable functions 
    # ---------------------------------------------------------------------

    def _set_duty_cycle(self, duty_cycle: float) -> None:
        """Helper to set motor duty cycle"""
        self._vesc.set_duty_cycle(duty_cycle)

    def set_steering(self, steering: float) -> None:
        """
        Set steering in normalized range [-1.0, 1.0].

        -1.0 = full left, 0.0 = straight, +1.0 = full right.
        """
        steering_clamped = max(-1.0, min(1.0, steering))
        cfg = self.config.steering

        if steering_clamped >= 0.0:
            # center -> right
            servo_value = cfg.servo_center + steering_clamped * (cfg.servo_max - cfg.servo_center)
        else:
            # center -> left
            servo_value = cfg.servo_center + steering_clamped * (cfg.servo_center - cfg.servo_min)

        self._vesc.set_servo(servo_value)

    # Alias bc i dont wanna type set_steering always
    def turn(self, steering: float) -> None:
        self.set_steering(steering)

    def stop(self) -> None:
        """Stop!!!"""
        self._set_duty_cycle(0.0)

    # ---------------------------------------------------------------------
    # Distance-based movement
    # ---------------------------------------------------------------------

    def _distance_to_duration(self, distance_m: float) -> float:
        """
        Helper to convert desired distance to time based on measured mps
        """
        speed = self.config.base_speed_mps
        if speed <= 0.0:
            raise ValueError(
                "mps must be > 0 "
            )
        duration = abs(distance_m) / speed
        return min(duration, self.config.max_drive_duration_s)

    def drive_straight(self, distance_m: float, steering: float = 0.0) -> None:
        """
        Drive approximately straight for a given distance in meters
        Positive distance for forward
        Negative distance for backward
        """
        if distance_m == 0.0:
            return

        duty = self.config.base_duty_cycle
        if duty == 0.0:
            raise ValueError("base_duty_cycle cannot be 0")

        # Choose duty sign based on desired direction
        duty_signed = duty if distance_m > 0.0 else -duty
        duration_s = self._distance_to_duration(distance_m)

        # Apply steering and go
        self.set_steering(steering)
        self._set_duty_cycle(duty_signed)

        time.sleep(duration_s)

        self.stop()

    def drive_forward(self, distance_m: float, steering: float = 0.0) -> None:
        """
        Essentially an alias for drive_straight with more usable name
        """
        if distance_m < 0.0:
            raise ValueError("distance_m must be >= 0 in drive_forward().")
        self.drive_straight(distance_m=distance_m, steering=steering)

    def drive_backward(self, distance_m: float, steering: float = 0.0) -> None:
        """
        Essentially an alias for drive_straight with more usable name
        """
        if distance_m < 0.0:
            raise ValueError("distance_m must be >= 0 in drive_backward().")
        self.drive_straight(distance_m=-distance_m, steering=steering)
