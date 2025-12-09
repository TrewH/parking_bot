#!/usr/bin/env python3

import time
from typing import Any

from pyvesc.VESC import VESC

# If you discover the correct field is "tachometer_abs", change this:
COUNTER_ATTR = "tachometer_abs"


def get_counter(vesc: VESC) -> int:
    values: Any = vesc.get_measurements()
    if values is None:
        raise RuntimeError("No measurements received from VESC")
    return int(getattr(values, COUNTER_ATTR))


def main() -> None:
    # 1. Connect to VESC
    vesc = VESC(serial_port="/dev/ttyACM0")

    print("=== Ticks per Revolution Measurement ===")
    print("IMPORTANT:")
    print(" - Lift the driven wheels off the ground.")
    print(" - Put a piece of tape or a clear mark on the tire.")
    print(" - Make sure nothing can get caught in the spinning wheel.\n")

    input("When ready, press Enter to start the wheel spinning slowly...")

    # 2. Start slow rotation
    duty = 0.015  # adjust if needed (smaller if too fast, larger if it doesn't move)
    vesc.set_duty_cycle(duty)
    print(f"Wheel spinning at duty_cycle = {duty}.")
    print("Let it spin for a second to stabilize...\n")
    time.sleep(1.5)

    # 3. First reference pass (start of revolution)
    print("Watch your tape mark on the tire.")
    input("Press Enter when the mark is at your reference point (START)...")
    start = get_counter(vesc)
    print(f"Start {COUNTER_ATTR}: {start}")

    # 4. Second reference pass (end of revolution)
    print("\nLet the wheel keep spinning.")
    input("Press Enter again when the SAME mark comes back to that same spot (END)...")
    end = get_counter(vesc)
    print(f"End {COUNTER_ATTR}: {end}")

    # 5. Stop the wheel
    vesc.set_duty_cycle(0.0)
    print("Wheel stopped.\n")

    # 6. Compute delta
    delta = end - start
    delta_abs = abs(delta)

    print("=== Result ===")
    print(f"Raw delta: {delta}")
    print(f"Ticks per one wheel revolution (abs): {delta_abs}")


if __name__ == "__main__":
    main()
