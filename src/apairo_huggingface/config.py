"""Configuration for the TartanDrive -> LeRobot export.

One place for every knob: which channels feed which feature, how the sensors are
resampled, and the shape/naming of the LeRobot features.
"""

from __future__ import annotations

# --- TartanDrive channels feeding each LeRobot feature --------------------- #
CAMERA = "image_left_color"   # (H, W, 3) uint8  -- left colour camera
STATE = "gps_odom"            # nav_msgs/Odometry row: pose (7) + twist (6)
ACTION = "cmd"                # twist command [linear_velocity, angular_velocity]

# --- Synchronization ------------------------------------------------------- #
# Resample every channel onto the camera clock, so each LeRobot frame carries a
# *real* image (never interpolated) plus the state and command nearest in time.
REFERENCE = CAMERA
METHOD = "nearest"
TOLERANCE = 0.05              # seconds; drop frames with no match within +/-50 ms
FPS = 10                      # nominal rate declared to LeRobot (camera ~10 Hz)

# --- LeRobot dataset metadata ---------------------------------------------- #
ROBOT_TYPE = "tartandrive"
TASK = "drive off-road"

# --- Feature dimensions ---------------------------------------------------- #
# gps_odom holds a full Odometry row; we keep the pose (position + orientation).
POSE_DIM = 7
STATE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
ACTION_NAMES = ["linear_velocity", "angular_velocity"]
