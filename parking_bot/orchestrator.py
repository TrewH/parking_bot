#!/usr/bin/env python3

import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger

from .HAL import ParkingHAL  # Your HAL wrapper

PAUSE_BETWEEN_MOVES_S: float = 0.5  # to match your robot_control.py


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

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.get_logger().info('Service is available.')

        # # For Sweeper Mechanism
        # self.sweeper_cli = self.create_client(Trigger, 'activate_sweeper')
    
    # def trigger_sweeper(self):
    # """Blocks until the sweep is finished."""
    # if not self.sweeper_cli.wait_for_service(timeout_sec=1.0):
    #     self.get_logger().error("Sweeper service not available!")
    #     return

    # req = Trigger.Request()
    # future = self.sweeper_cli.call_async(req)
    # rclpy.spin_until_future_complete(self, future)
    # res = future.result()
    # if res.success:
    #     self.get_logger().info("Sweeper successfully cleared the path.")
    # else:
    #     self.get_logger().error(f"Sweeper failed: {res.message}")

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

        self.get_logger().info('Calling /get_parking_spots...')
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error(f'Service call failed: {future.exception()}')
            return None, None

        response: Trigger.Response = future.result()
        self.get_logger().info(
            f"Service response: success={response.success}, message='{response.message}'"
        )

        if not response.success:
            self.get_logger().warn(f"Vision node reported failure: {response.message}")
            return None, None

        distance_m, side = self._parse_message(response.message)
        if distance_m is None or side is None:
            self.get_logger().error("Failed to parse response.message for side/depth.")
            return None, None

        self.get_logger().info(f"Best spot: side={side}, distance_m={distance_m:.2f}")
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
        hal.drive(-0.45)
        time.sleep(PAUSE_BETWEEN_MOVES_S)

        # Reverse 2
        hal.set_steering(0.0)
        hal.drive(-0.49)
        time.sleep(PAUSE_BETWEEN_MOVES_S)

        # Forward 2
        hal.set_steering(0.5)
        hal.drive(0.18)
        time.sleep(PAUSE_BETWEEN_MOVES_S)

        hal.stop()
        self.get_logger().info("=== RIGHT parallel park complete ===")

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
        self.get_logger().info("=== LEFT parallel park complete ===")

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

        # # Obstacle field for sweeper
        
        # obstacle_str = kv.get('obstacle')
        # has_obstacle = (obstacle_str == 'True') 

        # return distance_m, side, has_obstacle


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Orchestrator()
    hal = ParkingHAL()

    node.get_logger().info("HAL and Orchestrator running")

    try:
        # One-shot service call + maneuver
        dist, side = node.get_best_spot_position()

        if dist is None or side is None:
            node.get_logger().error("No valid best spot returned; aborting.")
        else:
            # 1) Drive straight up to the spot
            node.get_logger().info(f"Driving forward {dist:.2f} m toward the spot...")
            hal.set_steering(0.5)
            hal.drive(dist)
            time.sleep(PAUSE_BETWEEN_MOVES_S)

            # Fixed parallel parking
            if side == "right":
                node.get_logger().info("Calling parallel_park_right()")
                node.parallel_park_right(hal)
            elif side == "left":
                node.get_logger().info("Calling parallel_park_left()")
                node.parallel_park_left(hal)
            else:
                node.get_logger().error(f"Unexpected side '{side}', aborting.")
                
    # # For sweeper
    # dist, side, has_obstacle = node.get_best_spot_position() # Update to unpack 3 values

    #         if dist is not None:
    #             # DRIVE TO POSITION FIRST
    #             node.get_logger().info(f"Driving forward {dist:.2f} m...")
    #             hal.set_steering(0.5)
    #             hal.drive(dist)
    #             time.sleep(PAUSE_BETWEEN_MOVES_S)

    #     # CHECK OBSTACLE
    #         if has_obstacle:
    #             node.get_logger().warn("Obstacle reported! Deploying sweeper...")
    #             node.trigger_sweeper()
    #  # Optional: Move slightly to ensure clearance if needed
    
    # # NOW PARK
    # if side == "right":
    #      node.parallel_park_right(hal)
    
    finally:
        # Clean shutdown of HAL and ROS
        try:
            hal.shutdown()
        except Exception as e:
            node.get_logger().warn(f"Error during HAL shutdown: {e}")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
