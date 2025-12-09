# =============================================================================
# File: parking_controller.py
# Description: ROS2 node for autonomous parking control
#
# Authors:   Trew Hoffman, Terri Tai
# Created:     2025-11-29
#
# Notes:
#   - Integrates HAL with vision node
#   - Implements state machine for parking procedure
#   - Supports forward and reverse parallel parking
#   - Handles no parking sign detection
# =============================================================================

"""
Parking Controller Node

This node orchestrates the autonomous parking procedure by:
- Subscribing to vision data (/parking_bot/spot_open)
- Managing parking state machine
- Commanding the HAL to execute maneuvers
- Supporting both forward and reverse parallel parking
- Avoiding spots with no parking signs
"""

from __future__ import annotations
import time
from enum import Enum, auto
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Pose2D, Twist

from HAL import ParkingHAL, HALConfig


class ParkingState(Enum):
    """State machine for parking procedure"""
    IDLE = auto()
    SEARCHING = auto()
    ALIGNING = auto()
    PRE_PARKING = auto()
    PARKING = auto()
    PARKED = auto()
    ERROR = auto()


class ParkingType(Enum):
    """Type of parking maneuver to execute"""
    FORWARD = auto()
    REVERSE = auto()


class ParkingController(Node):
    """
    ROS2 node that controls the parking procedure
    """

    def __init__(self, parking_type: ParkingType = ParkingType.FORWARD) -> None:
        super().__init__("parking_controller")
        
        self.get_logger().info("Initializing ParkingController...")

        # Initialize HAL
        try:
            hal_config = HALConfig()
            self.hal = ParkingHAL(config=hal_config)
            self.get_logger().info("HAL initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize HAL: {e}")
            raise

        # State machine
        self.state = ParkingState.IDLE
        self.parking_type = parking_type
        
        # Vision data
        self.spot_is_open = False
        self.spot_detected = False
        self.no_parking_sign_detected = False
        self.tag_pose: Optional[Pose2D] = None

        # Vision data - alignment
        self.alignment_command = "LOST_SPOT"
        self.alignment_error = (0.0, 0.0, 0.0)
        
        # Subscribers
        self.spot_sub = self.create_subscription(
            Bool,
            "/parking_bot/spot_open",
            self._spot_callback,
            10
        )
        self.tag_sub = self.create_subscription(
            Pose2D,
            "/parking_bot/tag_pose",
            self._tag_callback,
            10
        )
        self.alignment_sub = self.create_subscription(
            String,
            "/parking_bot/alignment_command",
            self._alignment_callback,
            10
        )
        self.alignment_error_sub = self.create_subscription(
            Twist,
            "/parking_bot/alignment_error",
            self._alignment_error_callback,
            10
        )
        self.no_parking_sub = self.create_subscription(
            Bool,
            "/parking_bot/no_parking_detected",
            self._no_parking_callback,
            10
        )

        # Control loop timer (10 Hz)
        self.timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f"ParkingController initialized - Mode: {parking_type.name}"
        )

    def _spot_callback(self, msg: Bool) -> None:
        """Callback for parking spot occupancy updates"""
        self.spot_is_open = msg.data
        if msg.data:
            self.spot_detected = True
            self.get_logger().info("Open parking spot detected!")
        else:
            if self.spot_detected:
                self.get_logger().warn("Lost spot or spot now occupied")

    def _tag_callback(self, msg: Pose2D) -> None:
        """Callback for AprilTag pose updates"""
        self.tag_pose = msg
        self.get_logger().debug(f"Tag pose: x={msg.x}, y={msg.y}, theta={msg.theta}")

    def _alignment_callback(self, msg: String) -> None:
        """Callback for alignment command updates"""
        self.alignment_command = msg.data
        self.get_logger().debug(f"Alignment command: {msg.data}")

    def _alignment_error_callback(self, msg: Twist) -> None:
        """Callback for alignment error updates"""
        self.alignment_error = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.get_logger().debug(
            f"Alignment error - Long: {msg.linear.x:.1f}, Lat: {msg.linear.y:.1f}, Ang: {msg.angular.z:.3f}"
        )

    def _no_parking_callback(self, msg: Bool) -> None:
        """Callback for no parking sign detection"""
        self.no_parking_sign_detected = msg.data
        if msg.data:
            self.get_logger().warn("No parking sign detected - will skip this spot")

    def _control_loop(self) -> None:
        """Main control loop - runs state machine"""
        if self.state == ParkingState.IDLE:
            self._handle_idle()
        elif self.state == ParkingState.SEARCHING:
            self._handle_searching()
        elif self.state == ParkingState.ALIGNING:
            self._handle_aligning()
        elif self.state == ParkingState.PRE_PARKING:
            self._handle_pre_parking()
        elif self.state == ParkingState.PARKING:
            self._handle_parking()
        elif self.state == ParkingState.PARKED:
            self._handle_parked()
        elif self.state == ParkingState.ERROR:
            self._handle_error()

    def _handle_idle(self) -> None:
        """IDLE state - waiting to start"""
        pass

    def _handle_searching(self) -> None:
        """SEARCHING state - looking for open spot"""
        if self.no_parking_sign_detected:
            self.get_logger().warn("Spot has no parking sign - continuing search...")
            # Continue driving slowly to find another spot
            self.hal.drive_forward(distance_m=0.2, steering=0.0)
            time.sleep(0.5)
            return
            
        if self.spot_is_open:
            self.get_logger().info("Found open spot! Stopping to align...")
            self.hal.stop()
            self.state = ParkingState.ALIGNING

    def _handle_aligning(self) -> None:
        """ALIGNING state - positioning alongside the spot"""
        # Check for no parking sign during alignment
        if self.no_parking_sign_detected:
            self.get_logger().warn("No parking sign detected - aborting alignment")
            self.hal.stop()
            self.state = ParkingState.SEARCHING
            return
            
        self.get_logger().info("Aligning to side of spot...")
        
        try:
            self._align_to_side_of_spot()
            self.get_logger().info("Alignment complete, ready to park")
            self.state = ParkingState.PRE_PARKING
        except Exception as e:
            self.get_logger().error(f"Alignment failed: {e}")
            self.state = ParkingState.ERROR

    def _handle_pre_parking(self) -> None:
        """PRE_PARKING state - final checks before parking"""
        if self.no_parking_sign_detected:
            self.get_logger().warn("No parking sign detected! Aborting")
            self.state = ParkingState.SEARCHING
            return
            
        if not self.spot_is_open:
            self.get_logger().warn("Spot no longer open! Aborting")
            self.state = ParkingState.SEARCHING
            return

        self.get_logger().info(
            f"Executing {self.parking_type.name} parking maneuver..."
        )
        self.state = ParkingState.PARKING

    def _handle_parking(self) -> None:
        """PARKING state - executing maneuver"""
        try:
            if self.parking_type == ParkingType.FORWARD:
                self._execute_forward_parking()
            elif self.parking_type == ParkingType.REVERSE:
                self._execute_reverse_parking()
            
            self.state = ParkingState.PARKED
            self.get_logger().info("Successfully parked!")
        except Exception as e:
            self.get_logger().error(f"Parking failed: {e}")
            self.state = ParkingState.ERROR

    def _handle_parked(self) -> None:
        """PARKED state - parking complete"""
        pass

    def _handle_error(self) -> None:
        """ERROR state - something went wrong"""
        self.get_logger().error("In ERROR state, stopping vehicle")
        self.hal.stop()

    def _align_to_side_of_spot(self) -> None:
        """
        Align the car to be positioned alongside the parking spot.
        Uses vision feedback to achieve proper alignment.
        """
        self.get_logger().info("Performing vision-based alignment...")
        
        max_iterations = 100  # Safety limit
        iteration = 0
        aligned = False
        
        while not aligned and iteration < max_iterations:
            iteration += 1
            
            # Check for no parking sign
            if self.no_parking_sign_detected:
                self.get_logger().warn("No parking sign detected during alignment - aborting")
                raise Exception("No parking sign detected")
            
            # Get current alignment command from vision
            cmd = self.alignment_command
            long_err, lat_err, ang_err = self.alignment_error
            
            self.get_logger().info(
                f"Alignment iter {iteration}: {cmd} "
                f"(long={long_err:.0f}px, lat={lat_err:.0f}px, ang={ang_err:.2f}rad)"
            )
            
            if cmd == "ALIGNED":
                self.get_logger().info("Alignment achieved!")
                aligned = True
                break
            
            elif cmd == "NO_PARKING":
                self.get_logger().warn("No parking sign detected - aborting alignment")
                raise Exception("No parking sign detected")
            
            elif cmd == "LOST_SPOT":
                self.get_logger().warn("Lost sight of spot during alignment")
                time.sleep(0.5)  # Wait a bit for vision to reacquire
                continue
            
            elif cmd == "TURN_LEFT":
                # Small rotation left
                self.hal.set_steering(-0.5)
                self.hal.drive_forward(distance_m=0.1, steering=-0.5)
                time.sleep(0.2)
            
            elif cmd == "TURN_RIGHT":
                # Small rotation right
                self.hal.set_steering(0.5)
                self.hal.drive_forward(distance_m=0.1, steering=0.5)
                time.sleep(0.2)
            
            elif cmd == "MOVE_FORWARD":
                # Move forward to position spot correctly
                self.hal.drive_forward(distance_m=0.15, steering=0.0)
                time.sleep(0.2)
            
            elif cmd == "MOVE_BACKWARD":
                # Move backward to position spot correctly
                self.hal.drive_backward(distance_m=0.15, steering=0.0)
                time.sleep(0.2)
            
            elif cmd == "ADJUST_CLOSER":
                # Move left (closer to spot)
                self.hal.drive_forward(distance_m=0.1, steering=-0.3)
                time.sleep(0.2)
            
            elif cmd == "ADJUST_FARTHER":
                # Move right (farther from spot)
                self.hal.drive_forward(distance_m=0.1, steering=0.3)
                time.sleep(0.2)
            
            else:
                self.get_logger().warn(f"Unknown alignment command: {cmd}")
                time.sleep(0.2)
            
            # Brief pause to let vision update
            time.sleep(0.3)
        
        if not aligned:
            self.get_logger().warn("Alignment timeout - proceeding anyway")
        
        # Ensure wheels are straight
        self.hal.set_steering(0.0)
        self.hal.stop()
        time.sleep(0.2)

    def _execute_forward_parking(self) -> None:
        """
        Execute the FORWARD parallel parking maneuver using HAL.
        
        Based on your original procedure:
        1. Back up straight
        2. Turn sharp right while moving forward
        3. Turn left while moving forward
        4. Straighten out
        """
        self.get_logger().info("=== FORWARD PARKING ===")
        
        try:
            # Step 1: Go back straight
            self.get_logger().info("Step 1: Backing up straight...")
            self.hal.drive_backward(distance_m=1.02, steering=0.0)
            time.sleep(0.2)

            # Step 2: Turn sharp right while going forward
            self.get_logger().info("Step 2: Forward with sharp right turn...")
            self.hal.drive_forward(distance_m=0.44, steering=1.0)
            time.sleep(0.2)

            # Step 3: Turn left while going forward
            self.get_logger().info("Step 3: Forward with left turn...")
            self.hal.drive_forward(distance_m=0.35, steering=-1.0)
            time.sleep(0.2)

            # Step 4: Straighten wheels
            self.get_logger().info("Step 4: Straightening wheels...")
            self.hal.set_steering(0.0)
            time.sleep(0.1)

            # Final stop
            self.hal.stop()
            self.get_logger().info("Forward parking maneuver complete!")

        except Exception as e:
            self.get_logger().error(f"Error during forward parking: {e}")
            self.hal.stop()
            raise

    def _execute_reverse_parking(self) -> None:
        """
        Execute the REVERSE parallel parking maneuver using HAL.
        
        Based on your original procedure:
        1. Go forward straight
        2. Reverse with sharp right turn
        3. Reverse with left turn
        4. Straighten out
        """
        self.get_logger().info("=== REVERSE PARKING ===")
        
        try:
            # Step 1: Go forward straight
            self.get_logger().info("Step 1: Going forward...")
            self.hal.drive_forward(distance_m=1.2, steering=0.0)
            time.sleep(0.2)

            # Step 2: Reverse with sharp right turn
            self.get_logger().info("Step 2: Reversing with right turn...")
            self.hal.drive_backward(distance_m=0.41, steering=1.0)
            time.sleep(0.2)

            # Step 3: Reverse with left turn
            self.get_logger().info("Step 3: Reversing with left turn...")
            self.hal.drive_backward(distance_m=0.35, steering=-1.0)
            time.sleep(0.2)

            # Step 4: Straighten wheels
            self.get_logger().info("Step 4: Straightening wheels...")
            self.hal.set_steering(0.0)
            time.sleep(0.1)

            # Final stop
            self.hal.stop()
            self.get_logger().info("Reverse parking maneuver complete!")

        except Exception as e:
            self.get_logger().error(f"Error during reverse parking: {e}")
            self.hal.stop()
            raise

    def start_parking(self) -> None:
        """Public method to initiate parking procedure"""
        if self.state == ParkingState.IDLE:
            self.get_logger().info("Starting parking procedure...")
            self.state = ParkingState.SEARCHING
        else:
            self.get_logger().warn(f"Cannot start - already in state: {self.state}")

    def set_parking_type(self, parking_type: ParkingType) -> None:
        """Change parking type (only allowed when IDLE)"""
        if self.state == ParkingState.IDLE:
            self.parking_type = parking_type
            self.get_logger().info(f"Parking type set to: {parking_type.name}")
        else:
            self.get_logger().warn("Cannot change parking type while active")

    def shutdown(self) -> None:
        """Clean shutdown"""
        self.get_logger().info("Shutting down controller...")
        self.hal.stop()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    
    # Get parking type from command line or user input
    print("Select parking type:")
    print("  'f' - Forward parallel parking")
    print("  'r' - Reverse parallel parking")
    choice = input("Enter choice: ").lower()
    
    if choice == 'f':
        parking_type = ParkingType.FORWARD
    elif choice == 'r':
        parking_type = ParkingType.REVERSE
    else:
        print("Invalid choice, defaulting to FORWARD")
        parking_type = ParkingType.FORWARD
    
    controller = ParkingController(parking_type=parking_type)

    try:
        # Start the parking procedure
        controller.start_parking()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()