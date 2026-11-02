"""Knobs of the FoMo traversability labelling (see :mod:`.labeling`)."""

from __future__ import annotations

from .calibration import LIDAR_HEIGHT

LIDAR = "robosense"
CAMERA = "basler"
POSES = "kissicp_poses"       # per-scan T_world_lidar, written by KISS-ICP
TRAV_TRAJ = "trav_traj"       # per-point labels, on the lidar clock
TRAV_MASK = "trav_mask"       # (H, W) uint8 PNG, on the *camera* clock
                              # 1 driven / 0 lidar-seen, not driven / 255 no data

# KISS-ICP on the RoboSense (128 rings, ~170k points, per-point time).
ODOM_MAX_RANGE = 80.0
ODOM_MIN_RANGE = 2.0          # drops the robot's own returns
ODOM_VOXEL = 0.5

# The platform is a Clearpath Warthog, 1.38 m wide; 0.7 m half-width.
ROBOT_HALF_WIDTH = 0.7
FORWARD_DISTANCE = 30.0       # look-ahead cap, metres of path
# Height window of TraversabilityFromTrajectory, relative to the nearest future
# pose (the lidar, LIDAR_HEIGHT above the ground): -0.3 m / +0.5 m around it.
HEIGHT_MIN = -LIDAR_HEIGHT - 0.3
HEIGHT_MAX = -LIDAR_HEIGHT + 0.5
GROUND_OFFSET = -LIDAR_HEIGHT  # corridor height relative to the pose
SPLAT_RADIUS = 2
OCCLUSION_BIN_PX = 8

# Camera <- lidar matching. Basler and RoboSense run at 10 Hz on their own
# triggers (~16 ms apart); a frame with no scan within half a period is dropped.
TOLERANCE = 0.05
POSE_TOLERANCE = 0.15        # bracketing poses of an interpolated frame
