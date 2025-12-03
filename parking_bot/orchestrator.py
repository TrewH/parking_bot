# =============================================================================
# File: orchestrator.py
# Description: Process orchestrator for self parking 
#
# Authors:   Trew Hoffman
# Created:     2025-12-2
#
# Notes:
#   - This is a ROS2 node that will take in vision information to make decisions on calling HAL commands

# =============================================================================

"""
Process Orchestrator 

This node will:
- Use vision information from vision node to determine how far to move and in which direction
- Call HAL movement commands based on 

Outputs: 
- Robot movement
"""