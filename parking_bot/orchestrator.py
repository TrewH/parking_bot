#!/usr/bin/env python3

import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger

from .HAL import ParkingHALr

PAUSE_BETWEEN_MOVES_S: float = 0.5


class Orchestrator(Node):
    """
    High-level orchestrator node.

    - Does NOT instantiate the vision node.
    - Communicates with the vision node only via the /get_parking_spots service.
    - Uses the returned side + distance to run a fixed parallel parking maneuver.
    """

    def __init__(self) -> None:
        super().__init__('orchestrator')

        # Service client for the vision node
        self.cli = self.create_client(Trigger, 'get_parking_spots')
        self.get_logger().info('Waiting for /get_parking_spots service...')
        time.sleep(2)

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.get_logger().info('Service is available.')

        # Iterative depth-correction mode (set via ROS param)
        #   ros2 run ... --ros-args -p iterative:=true
        param = self.declare_parameter('iterative', True)
        self.iterative: bool = param.value
        self.get_logger().info(f'Iterative mode: {self.iterative}')

    # ------------------------------------------------------------------
    # Service interaction
    # ------------------------------------------------------------------
    def get_best_spot_position(self) -> Tuple[Optional[float], Optional[str]]:
        """
        Call /get_parking_spots and return (distance_m, side).

        Returns:
            (distance_m, side) on success
            (None, None) on any error
        """
        request = Trigger.Request()
        future = self.cli.call_async(request)

        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error(f'Service call failed: {future.exception()}')
            return None, None

        response: Trigger.Response = future.result()
        self.get_logger().info(
            f"Service response: success={response.success}, message='{response.message}'"
        )

        if not response.success:
            self.get_logger().warning(f"Vision node reported failure: {response.message}")
            return None, None

        distance_m, side = self._parse_message(response.message)
        if distance_m is None or side is None:
            self.get_logger().error("Failed to parse response.message for side/depth.")
            return None, None

        return distance_m, side

    # ------------------------------------------------------------------
    # Parking maneuvers
    # ------------------------------------------------------------------
    def parallel_park_right(self, hal: ParkingHAL) -> None:
        """
        Fixed parallel parking maneuver into a RIGHT-hand spot.
        """
        self.get_logger().info("=== Starting RIGHT parallel park ===")

        # Forward 1
        hal.set_steering(0.5)
        hal.drive(0.6)
        time.sleep(PAUSE_BETWEEN_MOVES_S)

        # Reverse 1
        hal.set_steering(1.0)
        hal.drive(-0.425)
        time.sleep(PAUSE_BETWEEN_MOVES_S)

        # Reverse 2
        hal.set_steering(0.0)
        hal.drive(-0.465)
        time.sleep(PAUSE_BETWEEN_MOVES_S)

        # Forward 2
        hal.set_steering(0.5)
        hal.drive(0.125)
        time.sleep(PAUSE_BETWEEN_MOVES_S)

        hal.stop()

    def parallel_park_left(self, hal: ParkingHAL) -> None:
        """
        Fixed parallel parking maneuver into a LEFT-hand spot.
        """
        self.get_logger().info("=== Starting LEFT parallel park ===")

        # Forward 1
        hal.set_steering(0.5)
        hal.drive(0.6)
        time.sleep(PAUSE_BETWEEN_MOVES_S)

        # Reverse 1
        hal.set_steering(0.0)
        hal.drive(-0.45)
        time.sleep(PAUSE_BETWEEN_MOVES_S)

        # Reverse 2
        hal.set_steering(1.0)
        hal.drive(-0.50)
        time.sleep(PAUSE_BETWEEN_MOVES_S)

        # Forward 2
        hal.set_steering(0.5)
        hal.drive(0.18)
        time.sleep(PAUSE_BETWEEN_MOVES_S)

        hal.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_message(self, msg: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse a message of the form:

            "side=left depth_m=1.23"

        into (distance_m, side).
        """
        parts = msg.split()
        kv = {}
        for p in parts:
            if '=' in p:
                k, v = p.split('=', 1)
                kv[k.strip()] = v.strip()

        side = kv.get('side')
        depth_str = kv.get('depth_m')

        # Validate side
        if side not in ('left', 'right'):
            self.get_logger().error(f"Invalid side in message: '{side}'")
            return None, None

        # Parse depth
        try:
            distance_m = float(depth_str) if depth_str is not None else None
        except (TypeError, ValueError):
            self.get_logger().error(f"Invalid depth_m in message: '{depth_str}'")
            distance_m = None

        if distance_m is None:
            return None, None

        return distance_m, side


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Orchestrator()
    hal = ParkingHAL()

    node.get_logger().info("HAL and Orchestrator running")

    try:
        ok = True

        # First measurement
        dist, side = node.get_best_spot_position()

        # Subtract space between apriltag and spots
        dist = dist - 0.38

        if dist is None or side is None:
            node.get_logger().error("No valid best spot returned; aborting.")
            ok = False
        else:
            # -------------------------------
            # Iterative depth correction step
            # -------------------------------
            if node.iterative and dist > 3.0:
                approach = dist - 2.0
                node.get_logger().info(
                    f"Iterative mode: initial distance {dist:.2f} m > 1.0 m; "
                    f"driving {approach:.2f} m to get to ~1.0 m, then re-measuring."
                )

                hal.set_steering(0.5)
                approach = approach # (distance from spots to apriltag)
                hal.drive(approach)
                time.sleep(PAUSE_BETWEEN_MOVES_S)

                # Second measurement at ~1 m
                dist2, side2 = node.get_best_spot_position()
                if dist2 is None or side2 is None:
                    node.get_logger().error(
                        "Second measurement failed in iterative mode; aborting."
                    )
                    ok = False
                else:
                    dist, side = dist2, side2
                    node.get_logger().info(
                        f"Iterative mode: updated distance={dist:.2f} m, side={side}"
                    )

        # ---------------------------------
        # Final approach + parallel parking
        # ---------------------------------
        if ok and dist is not None and side is not None:
            node.get_logger().info(
                f"Driving forward {dist:.2f} m toward the spot..."
            )
            hal.set_steering(0.5)
            hal.drive(dist)
            time.sleep(PAUSE_BETWEEN_MOVES_S)

            # Move apriltag between sequences
            node.get_logger().warning("Please move the AprilTag: Parallel parking in 5s")
            time.sleep(5)

            # Fixed parallel parking
            if side == "right":
                node.get_logger().info("Calling parallel_park_right()")
                node.parallel_park_right(hal)
                node.get_logger().info("=== Finished RIGHT parallel park ===")
            elif side == "left":
                node.get_logger().info("Calling parallel_park_left()")
                node.parallel_park_left(hal)
                node.get_logger().info("=== Finished LEFT parallel park ===")
            else:
                node.get_logger().error(f"Unexpected side '{side}', aborting.")
    finally:
        # Clean shutdown of HAL and ROS
        try:
            hal.shutdown()
        except Exception as e:
            node.get_logger().warning(f"Error during HAL shutdown: {e}")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
