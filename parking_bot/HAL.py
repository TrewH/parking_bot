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

We will use 0.05 duty cycle as the only speed in this project, and measure distance based on tachometry readings from the VESC to
ensure accurate distance movements.  

The rest of the robot's code should treat this HAL as the only interface to the physical car.

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
    steering: SteeringConfig = SteeringConfig()
    max_drive_duration_s: float = 5.0  # safety cap


class ParkingHAL:
    """
    Internally we convert distance to time assuming constant speed and
    apply a fixed duty cycle. This class will produce callable methods we can use
    after determining what distances and angles we want to drive.
    """

    # rotate wheel exactly one full turn by hand
    COUNTS_PER_REV = 1 # placeholder

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
    # Tachometry-based movement
    # ---------------------------------------------------------------------

    def drive_straight(self, distance_m: float, steering: float = 0.0) -> None:
        """
        Drive approximately straight for a given distance in meters
        Positive distance for forward
        Negative distance for backward
        """

        # Zero protections
        if distance_m == 0.0:
            return

        duty = self.config.base_duty_cycle
        if duty == 0.0:
            raise ValueError("base_duty_cycle cannot be 0")

        # Get starting tachometer value
        tach_start = self._vesc.get_values().tachometer

        # Set signed duty and steering
        duty_signed = duty if distance_m > 0.0 else -duty
        self.set_steering(steering)
        self._set_duty_cycle(duty_signed)

        # Continue moving until
        target_counts = distance_m / WHEEL_CIRCUMFERENCE_M * COUNTS_PER_REV
        while rclpy.ok():
            values = self._vesc.get_values()
            delta_counts = values.tachometer - tach_start
            if abs(delta_counts) >= (target_counts):
                break
            
            time.sleep(0.01) 
        
        self.stop()