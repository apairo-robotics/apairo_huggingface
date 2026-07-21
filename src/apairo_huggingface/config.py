"""Configuration for the TartanDrive -> LeRobot export.

One place for every knob: which channels feed which feature, how the sensors are
resampled, and the shape/naming of the LeRobot features.
"""

from __future__ import annotations

# --- TartanDrive channels feeding each LeRobot feature --------------------- #
CAMERA = "image_left_color"   # (H, W, 3) uint8  -- left colour camera
STATE = "gps_odom"            # nav_msgs/Odometry row: pose (7) + twist (6)
ACTION = "cmd"                # twist command [linear_velocity, angular_velocity]
TRAV_MASK = "trav_mask"       # (H, W) uint8 -- written by .labeling
                              # 1 driven / 0 lidar-seen, no info / 255 unobserved

# --- Traversability labeling (see .labeling) ------------------------------- #
POSES = "gicp_poses"          # per-scan (4, 4) poses on the lidar clock
TRAV_TRAJ = "trav_traj"       # per-point labels channel
LIDAR_UV = "lidar_uv"         # per-point [u, v, depth] channel
FORWARD_DISTANCE = 40.0       # look-ahead cap, metres of path (loops mislabel beyond)
# The rig is a modified Yamaha Viking (overall width 61.8in = 1.57m per the
# manufacturer spec -> 0.785m half-width; 0.8m matches within 1.5cm).
ROBOT_HALF_WIDTH = 0.8        # ATV footprint half-width (m)
# TraversabilityFromTrajectory's height_min/height_max are relative to the
# nearest future *pose* (the velodyne, ~1.77m above the ground -- see
# GROUND_OFFSET below), not the ground directly: [-2.3, -1.2] here is
# apairo_preprocess's own ground-relative default (-0.3m / +0.5m -- a low
# obstacle/rock band a 0.3m-clearance UTV would actually snag on) shifted by
# that ~1.77m rig height, i.e. effectively [-0.53m, +0.57m] around the ground.
HEIGHT_MIN = -2.3             # height window around the ground, relative to the
HEIGHT_MAX = -1.2             # velodyne (ground sits ~1.8 m below it)
GROUND_OFFSET = -1.77         # corridor height relative to the pose (m)
SPLAT_RADIUS = 2              # sparse point splat radius (px)
OCCLUSION_BIN_PX = 8          # z-buffer bin for the point mask (px)

# --- Hand-labeled ground truth (toaster; see .eval_labeling / .eval_export) - #
# Written by apairo_experiments/scripts/import_toaster_gt.py as a sparse
# per-point channel: 0 unlabeled / 1 traversable / 2 obstacle, one npy per
# hand-labeled scan -- most recordings have none of it.
GROUND_TRUTH = "ground_truth"
TRAV_MASK_GT = "trav_mask_gt"   # (H, W) uint8 -- ground_truth projected into the camera

# --- Synchronization ------------------------------------------------------- #
# Resample every channel onto the camera clock, so each LeRobot frame carries a
# *real* image (never interpolated) plus the state and command nearest in time.
REFERENCE = CAMERA
METHOD = "nearest"
TOLERANCE = 0.1               # seconds; the lidar-clocked trav_mask sits up to
                              # ~50 ms from the camera tick on real recordings
FPS = 10                      # nominal rate declared to LeRobot (camera ~10 Hz)

# --- LeRobot dataset metadata ---------------------------------------------- #
ROBOT_TYPE = "tartandrive"
TASK = "drive off-road"

# --- Feature dimensions ---------------------------------------------------- #
# gps_odom holds a full Odometry row; we keep the pose (position + orientation).
POSE_DIM = 7
STATE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
ACTION_NAMES = ["linear_velocity", "angular_velocity"]
