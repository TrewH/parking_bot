#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger
from typing import Optional, Tuple


class Orchestrator(Node):
    def __init__(self) -> None:
        super().__init__('orchestrator')

        # Using Trigger service as defined in ParkingVisionNode
        self.cli = self.create_client(Trigger, 'get_parking_spots')
        self.get_logger().info('Waiting for /get_parking_spots service...')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.get_logger().info('Service is available.')

    # ------------------------------------------------------------------
    # Public helper for other nodes
    # ------------------------------------------------------------------
    def get_best_spot_position(self) -> Tuple[Optional[float], Optional[str]]:
        """
        Call /get_parking_spots and return:

            (distance_m, side)

        where:
          - distance_m: float distance forward to the best spot in meters
          - side: "left" or "right"

        Returns (None, None) if:
          - the service call fails
          - the service reports no valid spot
          - the message cannot be parsed
        """
        request = Trigger.Request()
        future = self.cli.call_async(request)

        self.get_logger().info('Calling /get_parking_spots...')
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error(f'Service call failed: {future.exception()}')
            return None, None

        response: Trigger.Response = future.result()
        self.get_logger().info(f"Service response: success={response.success}, message='{response.message}'")

        if not response.success:
            # e.g. "No spots detected" or "No valid parking spot found"
            self.get_logger().warn(f"No valid spot: {response.message}")
            return None, None

        # Expecting message like: "side=left depth_m=1.23"
        distance_m, side = self._parse_message(response.message)
        if distance_m is None or side is None:
            self.get_logger().error("Failed to parse response.message for side/depth.")
            return None, None

        self.get_logger().info(f"Best spot: side={side}, distance_m={distance_m:.2f}")
        return distance_m, side

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
            return None, None

        # Parse depth
        try:
            distance_m = float(depth_str) if depth_str is not None else None
        except (TypeError, ValueError):
            distance_m = None

        if distance_m is None:
            return None, None

        return distance_m, side


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Orchestrator()

    # Example usage / test
    dist, side = node.get_best_spot_position()
    if dist is not None and side is not None:
        node.get_logger().info(f"[MAIN] Best spot: side={side}, distance={dist:.2f} m")
    else:
        node.get_logger().info("[MAIN] No valid best spot returned.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
