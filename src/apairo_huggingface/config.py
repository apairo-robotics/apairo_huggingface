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
# Poses: the vehicle's INS (gps_odom), looked up on the lidar clock by
# .reference_poses -- delivered data, no estimation. gicp_poses / kissicp_poses
# remain selectable (apairo-tartan-label --poses) for comparison only.
POSES = "ins_poses"           # per-scan (4, 4) poses on the lidar clock, NaN where invalid
POSE_VALID = "ins_valid"      # per-scan bool: the pose may be used
POSE_SOURCES = {"ins": "ins_poses", "gicp": "gicp_poses", "kissicp": "kissicp_poses"}
INS_CLOCK_OFFSET = -0.05      # s, INS clock minus lidar clock; measured -0.06 / -0.05
                              # on turnpike_afternoon_fall / down_meadows
INS_MAX_GAP = 0.1             # s; INS rows every 0.02 s, a longer hole leaves its scans without a pose
MAX_SPEED = 20.0              # m/s; top speed bound, 14.5 m/s is the fastest measured
TRAV_TRAJ = "trav_traj"       # per-point labels channel: 1 / 0 / UNKNOWN
LIDAR_UV = "lidar_uv"         # per-point [u, v, depth] channel
UNKNOWN = 255                 # label of a scan without a trusted pose (and unobserved pixels)
# Look-ahead cap, metres of path. Sources of poses agree with each other within
# 5-10 cm (median) at 10 m ahead and drift apart further on, to 1.2-1.5 m (p95)
# at 40 m, the former cap (INS vs KISS-ICP vs the release's SLAM, two
# recordings): 15 m keeps the error of the driven path well below the
# footprint's half-width.
FORWARD_DISTANCE = 15.0
# The rig is a modified Yamaha Viking (overall width 61.8in = 1.57m per the
# manufacturer spec -> 0.785m half-width; 0.8m matches within 1.5cm).
ROBOT_HALF_WIDTH = 0.8        # ATV footprint half-width (m)
# The poses are the velodyne's, POSE_HEIGHT above the ground: the labeller's
# height window is given around the ground and shifted by pose_height. These
# are the values the labels have always used ([-2.3, -1.2] around the
# velodyne); on turnpike_afternoon_fall, 504 scans and 18M points, the two
# spellings give the same label for every point.
POSE_HEIGHT = 1.77            # velodyne above the ground (m)
HEIGHT_MIN = -0.53            # height window around the ground under a future pose (m)
HEIGHT_MAX = 0.57
GROUND_OFFSET = -POSE_HEIGHT  # corridor height relative to the pose (m)
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
