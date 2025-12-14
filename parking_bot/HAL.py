# =============================================================================
# File: HAL.py
# Description: Hardware Abstraction Layer
#
# Authors:   Trew Hoffman, Terri Tai
# Created:     2025-11-26
#
# Notes:
#   - This HAL uses a fixed duty cycle and known tachometer per meter values.
#   - All higher-level logic should use drive() and set_steering(),
#     rather than interfacing with the VESC directly.
# =============================================================================


"""
Hardware Abstraction Layer (HAL).


This module is intended wrap VESC so the rest of the project can make calls based on distance commands
rather than duty cycles and on-durations.


We will use 0.05 duty cycle as the only speed in this project, and measure distance based on tachometry readings
from the VESC to ensure accurate distance movements.  


The rest of the robot's code should treat this HAL as the only interface to the physical car.


Typical use:
    hal = ParkingHAL()
    hal.drive(1.0)
    hal.set_steering(0.8)
    hal.drive(-4.0)
"""


from __future__ import annotations
import time
from typing import Optional, Any
import math
from pyvesc.VESC import VESC




class ParkingHAL:
    """
    Internally we convert distance to time assuming constant speed and
    apply a fixed duty cycle. This class will produce callable methods we can use
    after determining what distances and angles we want to drive.
    """


    WHEEL_DIAMETER_M: float = 0.101  # 101 mm
    TACH_COUNTS_PER_REV: float = 143.0
    DIST_PER_TACH: float = (
        math.pi * WHEEL_DIAMETER_M / TACH_COUNTS_PER_REV
    )  # ≈ 0.0022189 m/count


    def __init__(self, serial_port: str = "/dev/ttyACM0") -> None:
        self.vesc = VESC(serial_port=serial_port)
        self.DUTY_CYCLE = 0.03


    def _get_tach(self) -> int:
        """
        Read the tachometer counter from the VESC.
        """
        values: Any = None
        while values is None:
            values = self.vesc.get_measurements()
        return int(values.tachometer)


    def _set_duty(self, duty: float) -> None:
        self.vesc.set_duty_cycle(duty)


    def stop(self) -> None:
        """Immediately stop the motor."""
        self._set_duty(0.0)


    def set_steering(self, angle: float = 0.0) -> None:
        """
        Set steering, ensure wheels are straight to begin
        """
        self.vesc.set_servo(angle)
        time.sleep(0.25)


    def drive(self, distance_m: float) -> None:
        """
        Drive forward (distance_m > 0) or backward (distance_m < 0)
        by a given distance in meters, using tachometer feedback.

        This is a blocking method.
        """
        if abs(distance_m) < 1e-4:
            print("[HAL] drive(): tiny distance, returning nothing")
            return

        if abs(distance_m) > 4:
            print("[HAL] Distance request exceeded 4 meters, returning nothing")
            return
        

        poll_hz: float = 50.0
        period = 1.0 / poll_hz

        # Direction: +1 for forward, -1 for backward
        direction = 1 if distance_m > 0 else -1

        # Find target tach counts
        target_counts = abs(distance_m) / self.DIST_PER_TACH

        start_tach = self._get_tach()
        print(f"[HAL] drive(): distance={distance_m}, direction={direction}")
        print(f"[HAL] drive(): start_tach={start_tach}, target_counts={target_counts}")

        # Safety timeout (seconds)
        max_time_s = 10.0
        start_time = time.time()

        # Turn on motor
        self._set_duty(direction * self.DUTY_CYCLE)
        print(f"[HAL] drive(): set duty = {direction * self.DUTY_CYCLE}")

        try:
            while True:
                current_tach = self._get_tach()
                delta_counts = (current_tach - start_tach) * direction

                elapsed = time.time() - start_time
                print(f"[HAL] drive(): tach={current_tach}, delta={delta_counts}, "
                    f"elapsed={elapsed:.2f}s")

                if delta_counts >= target_counts:
                    print("[HAL] drive(): reached target counts, stopping")
                    break

                if elapsed > max_time_s:
                    print("[HAL] drive(): TIMEOUT, stopping to avoid infinite loop")
                    break

                time.sleep(period)
        finally:
            self.stop()
            print("[HAL] drive(): stop() called")


    def shutdown(self) -> None:
        """
        Cleanly shut down the VESC so Python can exit.
        """
        try:
            print("[HAL] SHUTTING DOWN!")
            # Make sure motor is off
            self.stop()
        except Exception:
            pass

        # Try to stop any heartbeat / background stuff if available
        for attr_name in ("stop_heartbeat", "close", "disconnect"):
            func = getattr(self.vesc, attr_name, None)
            if callable(func):
                try:
                    func()
                except Exception:
                    pass

        # Close the underlying serial object
        for serial_attr in ("serial_port", "ser", "_ser"):
            ser = getattr(self.vesc, serial_attr, None)
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass