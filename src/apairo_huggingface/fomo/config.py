"""Knobs of the FoMo traversability labelling (see :mod:`.labeling`)."""

from __future__ import annotations

from .calibration import LIDAR_HEIGHT

LIDAR = "robosense"
CAMERA = "zedx_left"          # default camera: the forward ZED X left
CAMERAS = ("zedx_left", "zedx_right", "basler")
POSES = "kissicp_poses"       # per-scan T_world_lidar, written by KISS-ICP
TRAV_TRAJ = "trav_traj"       # per-point labels, FUTURE footprint, lidar clock
TRAV_TRAJ_PAST = "trav_traj_past"  # per-point labels, PAST footprint, lidar clock
POINT_LABELS = {"future": TRAV_TRAJ, "past": TRAV_TRAJ_PAST}

# Image masks: (H, W) uint8 PNG per camera frame, named by the raw image's
# microsecond stamp, 1 driven / 0 lidar-seen, not driven / 255 no data.
# Forward cameras carry the future footprint, the rear Basler the past one
# (calibration.FOOTPRINT). ``trav_mask`` (older Basler masks built with a
# flipped extrinsic) is not one of them.
TRAV_MASKS = {
    "zedx_left": "trav_mask_zedx_left",
    "zedx_right": "trav_mask_zedx_right",
    "basler": "trav_past_mask_basler",
}
TRAV_MASK = TRAV_MASKS[CAMERA]

# KISS-ICP on the RoboSense (128 rings, ~170k points, per-point time).
ODOM_MAX_RANGE = 80.0
ODOM_MIN_RANGE = 2.0          # drops the robot's own returns
ODOM_VOXEL = 0.5

# The robot model, one footprint for the scan labels and the image masks: the
# Clearpath Warthog wheel-contact box, centred on base_link (the RoboSense
# origin). warthog_description: wheelbase 2 x 0.457 m + 0.6 m tyres = 1.51 m,
# track 2 x 0.568 m + 0.25 m tyres = 1.39 m; Clearpath manual: 1.52 x 1.38 m.
ROBOT_LENGTH = 1.52
ROBOT_WIDTH = 1.38
FORWARD_DISTANCE = 30.0       # look-ahead cap, metres of path
POSE_Z_SMOOTHING = 1.0        # gaussian sigma (m of path) on the footprint height drawn in images
# Height window of TraversabilityFromTrajectory, relative to the nearest future
# pose (the lidar, LIDAR_HEIGHT above the ground): -0.3 m / +0.5 m around it.
HEIGHT_MIN = -LIDAR_HEIGHT - 0.3
HEIGHT_MAX = -LIDAR_HEIGHT + 0.5
GROUND_OFFSET = -LIDAR_HEIGHT  # corridor height relative to the pose
SPLAT_RADIUS = 2
OCCLUSION_BIN_PX = 8

# Camera <- lidar matching. Cameras and RoboSense run on their own triggers
# (lidar 10 Hz); a frame with no scan within half a lidar period is dropped.
TOLERANCE = 0.05
POSE_TOLERANCE = 0.15        # bracketing poses of an interpolated frame
