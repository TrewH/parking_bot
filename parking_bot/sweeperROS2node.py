# =============================================================================
# File: sweeperROS2node.py
# Description: ROS2 Node for Sweeper Arm Controller (PCA9685 + DS3225MG)
#
# Author:      Owen Hanenian (Ported from logic by Manan Tuteja)
# Created:     2025-12-07
#
# Notes:
#   - Wraps servo logic in a ROS2 lifecycle (Timer-based, non-blocking).
#   - Hybrid Architecture: Auto-detects simulation (Mac) vs. Hardware (Car).
#   - Configurable ROS2 parameters for sweep speed and step size.
# =============================================================================

import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    # Define Mock Classes so the code doesn't crash on your Mac
    class Node:
        def __init__(self, name): self.name = name
        def declare_parameter(self, n, v): pass
        def get_parameter(self, n): 
            class V: value = 0.01 if n == 'sweep_speed' else 2.0
            return V()
        def create_timer(self, p, c): 
            return {'period': p, 'callback': c, 'last': time.time()}
        def get_logger(self):
            class L: 
                def info(self, m): print(f"[INFO]: {m}")
                def warn(self, m): print(f"[WARN]: {m}")
                def error(self, m): print(f"[ERROR]: {m}")
            return L()
        def destroy_node(self): pass

    class rclpy:
        @staticmethod
        def init(args=None): print("--- MOCK ROS2 STARTED ---")
        @staticmethod
        def shutdown(): print("--- MOCK ROS2 STOPPED ---")
        @staticmethod
        def spin(node):
            try:
                while True:
                    time.sleep(node.get_parameter('sweep_speed').value)
                    # Manually trigger callback for simulation
                    node.timer_callback()
            except KeyboardInterrupt: pass

# =============================================================================
# HARDWARE SETUP
# =============================================================================
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo
    HARDWARE_AVAILABLE = True
except:
    HARDWARE_AVAILABLE = False

# =============================================================================
# THE ACTUAL NODE LOGIC
# =============================================================================
class SweeperNode(Node):
    def __init__(self):
        super().__init__('sweeper_node')
        
        # PARAMETERS
        self.declare_parameter('i2c_channel', 0)
        self.declare_parameter('sweep_speed', 0.01)
        self.declare_parameter('step_size', 2.0)
        
        # HARDWARE INIT
        self.servo = None
        if HARDWARE_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                pca = PCA9685(i2c)
                pca.frequency = 50
                channel = self.get_parameter('i2c_channel').value if ROS_AVAILABLE else 0
                self.servo = servo.Servo(pca.channels[channel])
                self.get_logger().info('✅ Hardware Connected')
            except Exception as e:
                self.get_logger().error(f'❌ Hardware Error: {e}')
        else:
            self.get_logger().warn('⚠️  Running in Virtual Hardware Mode')

        # STATE
        self.current_angle = 7.5
        self.direction = 1

        # TIMER
        period = self.get_parameter('sweep_speed').value
        self.timer = self.create_timer(period, self.timer_callback)

    def timer_callback(self):
        step = self.get_parameter('step_size').value
        next_angle = self.current_angle + (step * self.direction)

        if next_angle >= 180.0:
            next_angle = 180.0
            self.direction = -1
        elif next_angle <= 0.0:
            next_angle = 0.0
            self.direction = 1

        self.current_angle = next_angle

        # Move Real Servo or Print Mock
        if self.servo:
            self.servo.angle = self.current_angle
        else:
            # Only print every 10th step to avoid spamming the console
            if int(self.current_angle) % 20 == 0: 
                self.get_logger().info(f'Sweeping: {self.current_angle:.1f}')

def main(args=None):
    rclpy.init(args=args)
    node = SweeperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
