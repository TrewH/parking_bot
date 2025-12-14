#!/usr/bin/env python3

import time
from typing import Any

from pyvesc.VESC import VESC

COUNTER_ATTR = "tachometer_abs"


def get_counter(vesc: VESC) -> int:
    values: Any = vesc.get_measurements()
    if values is None:
        raise RuntimeError("No measurements received from VESC")
    return int(getattr(values, COUNTER_ATTR))


def main() -> None:
    # Connect to VESC
    vesc = VESC(serial_port="/dev/ttyACM0")

    print("=== Ticks per Revolution Measurement ===")
    print("IMPORTANT:")
    print(" - Lift the driven wheels off the ground.")
    print(" - Put a piece of tape or a clear mark on the tire.")
    print(" - Make sure nothing can get caught in the spinning wheel.\n")

    input("When ready, press Enter to start the wheel spinning slowly...")

    # Start slow rotation
    duty = 0.015  
    vesc.set_duty_cycle(duty)
    print(f"Wheel spinning at duty_cycle = {duty}.")
    print("Let it spin for a second to stabilize...\n")
    time.sleep(1.5)

    # First reference pass (start of revolution)
    print("Watch your tape mark on the tire.")
    input("Press Enter when the mark is at your reference point (START)...")
    start = get_counter(vesc)
    print(f"Start {COUNTER_ATTR}: {start}")

    # Second reference pass (end of revolution)
    print("\nLet the wheel keep spinning.")
    input("Press Enter again when the SAME mark comes back to that same spot (END)...")
    end = get_counter(vesc)
    print(f"End {COUNTER_ATTR}: {end}")

    vesc.set_duty_cycle(0.0)
    print("Wheel stopped.\n")

    delta = end - start
    delta_abs = abs(delta)

    print("=== Result ===")
    print(f"Raw delta: {delta}")
    print(f"Ticks per one wheel revolution (abs): {delta_abs}")


if __name__ == "__main__":
    main()
