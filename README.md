# MAE148 Fall 2025 - Team 9 - Smart Parallel Parking Robot
## Parallel Parking Car with Sign Identification and Obstacle Remover

### Team Members
* **Trew Hoffman** - Mechanical Engineering: Software - Class of 2026
* **Manan Tuteja** - Aerospace Engineering: Hardware - Class of 2026
* **Terri Tai** - Computer Engineering: Software - Class of 2027
* **Owen Hanenian** - Mechanical Engineering: Hardware - Class of 2026

---

## Overview
This project implements an autonomous parallel-parking robot that identifies which of two parallel parking spots is available and executes a full parking maneuver accordingly. The robot uses a vision-based decision pipeline, detecting no-parking signs, measuring depth to an AprilTag, and chooses the correct side to park.

The full system runs on ROS2, uses an OAK-D Lite for stereo depth + RGB vision, controls the car through a custom Hardware Abstraction Layer (HAL) connected to a VESC, and includes a servo-driven sweeper mechanism for obstacle removal.

The design intentionally follows a “Look → Measure → Execute” architecture, ensuring that every parking action is informed by sensor data.

## Abstract
Our goal was to create an autonomous ground robot that travels along a straight hallway with two potential parking spaces—one on the left and one on the right. A red NO PARKING sign declares one spot as occupied, and the robot must determine which side is open, measure its distance to the AprilTag marking the correct parking spot, and perform a tight parallel-parking sequence.

The system relies on the OAK-D Lite’s depth and rectified stereo output to estimate the true distance to an AprilTag using a median-over-area approach. The RGB stream is used to detect large red circular regions (NO PARKING signs) or, in ambiguous cases, a left-versus-right red-pixel imbalance.

Once the spot is selected, the orchestrator executes a multi-step parking routine using the HAL, steering, reversing, and straightening until the robot is aligned within its chosen spot. A sweeper arm powered by a servo motor controlled via a PCA9685 removes debris blocking the parking path.

## Key Features
* **Vision-based sign detection:** Identifies red circular no-parking signs or left/right red distribution imbalance to determine which spot is closed via HSV mask. 
* **AprilTag depth measurement:** [Computes tag distance using OAK-D rectified stereo depth, median-filtered for accuracy.
* **Parallel parking maneuver generator:** Performs a multi-stage, pre-tuned parking sequence via HAL steering and motor control.
* **ROS2 Modular architecture:** Clean separation of HAL, vision, orchestrator, and servo control nodes.

## Core Objectives
* **Objective A:** Identify the correct parking spot
    * *Measure:* Detect presence and location of no-parking sign
    * *Action:* Return via ROS2 service which side is available, with no sign present
* **Objective B:** Execute a parallel parking maneuver to the chosen spot
    * *Measure:* Receive desired side from vision node
    * *Action:* Execute hard-coded parallel parking manever with distances and steering dependent on chosen parking spot side.
* **Objective C:** Determine distance to parking spots
    * *Measure:* Use OAK-D stereo depth to determine distance to AprilTag (parking spots)
    * *Action:* Drive forward the distance determined by OAK-D measurement to appraoch parking spots. Use an iterative approach to get closer if the AprilTag is too far to determine distance. 

## System Architecture
### Node Descriptions
* **`vision_node.py`**
    * **Inputs:**
        * RGB preview, mono left/right, stereo depth from OAK-D
        * Trigger service call /get_parking_spots
    * **Logic:**
        * Detect red circular NO PARKING signs or red imbalance.
        * Detect AprilTag (tag36h11) in rectified-left frame.
        * Compute median depth inside tag polygon.
        * Apply linear depth correction.
        * Return available side and corrected tag depth.
    * **Outputs:**
        * Trigger response: "side={left/right} depth_m={float}"
* **`orchestrator.py`**
    * **Inputs:**
        * Output from parking_vision_node
        * ROS 2 timer or driver logic to initiate maneuver
    * **Logic:**
        * Call vision service.
        * Parse "side" and "depth_m".
        * Execute parallel_park_left() or parallel_park_right() with correct HAL commands.
    * **Outputs:**
        * Motion commands to HAL for steering and driving.
* **`HAL.py`**
    * **Inputs:**
        * Calls to drive() and set_steering() commands of certain distances and normalized steering angle.
    * **Logic:**
        * Convert high-level movement commands into VESC duty cycle and servo commands.
    * **Outputs:**
        * Motor power and steering PWM to the car. 

## Hardware & Embedded Systems
* **Compute Unit:** Raspberry Pi 5 running Rasperry Pi OS (Bookworm)
* **Sensors:**
  * OAK-D Lite
  * VESC Hall sensors
  * AprilTag fiducial marker
* **Connectivity:**
  * Ethernet SSH, wireless SSH
* **CAD/Mechanical:**
    * *Custom Parts:*
      * Rotary Servo Arm using a DS3225MG Servo: https://a360.co/4ah9HVK
      * PC9685 i2c to pwm driver assembly: https://a360.co/3MY30xW
      * Mount for OAK-D Lite
    * *Stock Parts:*
      * DC-DC converter
      * VESC
      * 2 Servos (steering + arm)
      * Car chassis

## Setup & Execution
1. **Environment:** SSH to Raspberry Pi
   * `docker start container`
   * `docker exec -it container bash`
   * `source_ros2`
2. **Build:**
   * `cd home/projects/ros2_ws/src/parking_bot`
   * `colcon build --packages-select parking_bot`
   * `source ~/ros2_ws/install/setup.bash`
3. **Launch:** `ros2 launch parking_bot parking.launch.py`

***NOTE:*** Execution of this project's code requires using https://github.com/LiamBindle/PyVESC/blob/master/pyvesc/VESC/VESC.py with lines 40-43 removed.

## Demonstration Videos

* **Park Right with No Sign:** https://drive.google.com/file/d/129XeshiFcHtZDFLckgNlD_xzgmjZ26v2/view?usp=drive_link
* **Park Left with Sign:** https://drive.google.com/file/d/13skkytyhFNDIrWCEbx_Y8rJ3Qe-3wAON/view?usp=drive_link
* **Park Right with Sign:** https://drive.google.com/file/d/1NbXFlpDSn8gsvdqW_95T_S3DBBavCneX/view?usp=drive_link

## Challenges & Solutions

1. **OAK-D Color Channel Format Mismatch**
   * **The Issue:** The OAK-D camera outputs images in RGB format by default, but OpenCV's HSV color detection functions expect BGR format. This caused our red color mask to fail completely, preventing sign detection.
   * **The Solution:** We implemented a channel remapping step in our vision pipeline to convert RGB frames to BGR before HSV conversion, ensuring proper color detection.

2. **Inaccurate OAK-D Depth Measurements**
   * **The Issue:** Raw depth values from the OAK-D stereo camera showed systematic errors that increased with distance, making precise parking distance measurements unreliable.
   * **The Solution:** We characterized the depth error by measuring known distances and created a correction curve. We applied linear compensation to depth measurements (depth_corrected = depth_raw × 0.9 + 0.1) to achieve centimeter-level accuracy.

3. **Unstable Depth Map Noise**
   * **The Issue:** The stereo depth map contained significant noise and outliers, especially at tag boundaries, leading to inconsistent distance measurements when using simple averaging.
   * **The Solution:** Instead of averaging, we implemented a 9×9 region of interest (ROI) median filter over the AprilTag area. This robust statistic rejected outliers and provided stable depth estimates.

4. **AprilTag Detection Range Limitations**
   * **The Issue:** The AprilTag was often too far from the camera to be reliably detected at the start of the parking sequence, preventing initial distance measurement.
   * **The Solution:** We implemented an iterative approach where the robot moves forward in small increments until the AprilTag becomes detectable, then proceeds with the full measurement and parking maneuver.

5. **Power Distribution Issues**
   * **The Issue:** With the OAK-D Lite, Raspberry Pi 5, VESC, and servos all drawing power simultaneously, we experienced brownouts and unstable operation during high-current maneuvers.
   * **The Solution:** We added a dedicated DC-DC converter with sufficient current capacity and implemented proper power sequencing to ensure stable voltage during all operating conditions.

6. **ROS2 Service Timing Synchronization**
   * **The Issue:** The vision node's service response sometimes arrived before depth processing was complete, causing the orchestrator to receive stale or invalid measurements.
   * **The Solution:** We implemented a callback-based synchronization system with proper service request validation, ensuring the orchestrator only proceeds when all sensor data is fresh and validated.

## Future Improvements

1. **Yaw Alignment Correction**
   * **Current Limitation:** Any misalignment at the start (robot not parallel with parking spots) results in the same misalignment angle after parking.
   * **Improvement:** Implement real-time yaw control using IMU data to correct misalignment during approach. This would make the final parallel parking pose less dependent on initial orientation variation.

2. **Camera Calibration & 3D Pose Estimation**
   * **Current Limitation:** Fitting a correction curve to OAK-D depth measurements is a band-aid solution.
   * **Improvement:** Perform proper stereo camera intrinsic calibration and implement a Perspective-n-Point (PnP) solver using AprilTag corners to get more accurate 6DOF pose estimation.

3. **Obstacle Removal Mechanism Redesign**
   * **Current Limitation:** The rotary servo arm struggles to fully sweep objects out of the way due to its limited range of motion.
   * **Improvement:** Replace the rotary sweeper with a linear pushing mechanism or design a multi-servo articulated arm that can lift and push obstacles more effectively.

4. **Multi-Type Parking Capability**
   * **Current Limitation:** Only supports parallel parking in two predetermined spots.
   * **Improvement:** Extend the system to handle perpendicular and slanted parking spots (like standard parking lots) using LiDAR-based spot detection and more sophisticated path planning algorithms.

5. **Hardware Acceleration Integration**
   * **Current Limitation:** Vision processing runs entirely on the Raspberry Pi CPU, limiting frame rates and complexity.
   * **Improvement:** Fully integrate an AI Hat to offload computer vision processing, allowing for higher frame rates and more complex detection models.

6. **Improved Navigation Recovery**
   * **Current Limitation:** The system lacks robust recovery behaviors when initial conditions are suboptimal.
   * **Improvement:** Fine-tune the recovery behaviors to include backing up, spinning, or re-positioning when the robot gets stuck or when parking conditions are not ideal.

7. **Vertical Camera Panning**
   * **Current Limitation:** The camera only pans horizontally, limiting object tracking capability.
   * **Improvement:** Add a vertical servo to allow the camera to track objects on the floor more effectively without losing them from the frame when getting close.

8. **Enhanced 3D Mounting System**
   * **Current Limitation:** Some components are mounted with temporary solutions.
   * **Improvement:** Design custom 3D-printed mounts for all components (speaker, I2C extenders, DC/DC converters) to clean up wiring and improve mechanical stability.

9. **Adaptive Parking Algorithms**
   * **Current Limitation:** Uses fixed parking maneuvers regardless of environmental conditions.
   * **Improvement:** Implement machine learning-based adaptive parking that can adjust trajectories based on real-time obstacle detection and varying parking spot dimensions.

## Acknowledgments
* Dr. Jack Silberman, Winston Chou
* University of California, San Diego - MAE/ECE148 Course Staff
