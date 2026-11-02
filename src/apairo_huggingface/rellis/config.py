"""Knobs of the RELLIS-3D derived dataset (see :mod:`.labeling`)."""

from __future__ import annotations

SEQUENCES = ["00000", "00001", "00002", "00003", "00004"]

# Channel names: the source directories (symlinked into the view), then the
# derived ones written beside them.
CAMERA = "pylon_camera_node"
LABEL_2D = "pylon_camera_node_label_id"
LIDAR = "lidar"            # alias of os1_cloud_node_kitti_bin
LABELS = "labels"          # alias of os1_cloud_node_semantickitti_label_id
SOURCE_DIRS = [CAMERA, LABEL_2D, "os1_cloud_node_kitti_bin", "os1_cloud_node_semantickitti_label_id"]
SEM_MASK = "sem_mask"      # (H, W) uint8 PNG: ontology id of the nearest projected point, 255 no data
TRAV_MASK = "trav_mask"    # (H, W) uint8 PNG: 1 driven / 0 lidar-seen, not driven / 255 no data
NO_DATA = 255

# Scan <-> image: each scan is co-captured with the image of the same index and
# borrows its stamp, so "nearest within TOLERANCE" pairs them one to one.
TOLERANCE = 0.02           # s; the camera runs at 10 Hz

# Projection density: each point is splatted as a disc of SPLAT_RADIUS px, the
# nearest point wins a pixel, and points more than 0.5 m behind the nearest
# return of their OCCLUSION_BIN_PX bin are dropped. No dilation beyond that.
SPLAT_RADIUS = 2
OCCLUSION_BIN_PX = 8
MIN_RANGE = 2.0            # m; drops returns off the robot itself

# Traversability. The platform is a Clearpath Warthog, 1.38 m wide.
ROBOT_HALF_WIDTH = 0.7
FORWARD_DISTANCE = 30.0    # look-ahead cap, metres of path
MIN_FUTURE_PATH = 10.0     # m; frames with less path ahead get an all-255 trav mask
# The Ouster sits ~1.0 m above the ground (median ground z, 2-4 m ring, all
# sequences). Height window relative to the nearest future pose: -0.3 / +0.5 m
# around the ground.
LIDAR_HEIGHT = 1.0
HEIGHT_MIN = -LIDAR_HEIGHT - 0.3
HEIGHT_MAX = -LIDAR_HEIGHT + 0.5
GROUND_OFFSET = -LIDAR_HEIGHT

# RELLIS-3D ontology (Rellis_3D_ontology/ontology.yaml): id -> (name, RGB).
ONTOLOGY = {
    0: ("void", (0, 0, 0)),
    1: ("dirt", (108, 64, 20)),
    3: ("grass", (0, 102, 0)),
    4: ("tree", (0, 255, 0)),
    5: ("pole", (0, 153, 153)),
    6: ("water", (0, 128, 255)),
    7: ("sky", (0, 0, 255)),
    8: ("vehicle", (255, 255, 0)),
    9: ("object", (255, 0, 127)),
    10: ("asphalt", (64, 64, 64)),
    12: ("building", (255, 0, 0)),
    15: ("log", (102, 0, 0)),
    17: ("person", (204, 153, 255)),
    18: ("fence", (102, 0, 204)),
    19: ("bush", (255, 153, 204)),
    23: ("concrete", (170, 170, 170)),
    27: ("barrier", (41, 121, 255)),
    31: ("puddle", (134, 255, 239)),
    33: ("mud", (99, 66, 34)),
    34: ("rubble", (110, 22, 138)),
}

# ------------------------------------------------------------------ BEV grids
# Robot-centred bird's-eye-view grids, one per scan (see :mod:`.bev`). The grid
# frame is gravity-aligned: origin at the lidar, +x the robot's heading (the
# Ouster -x) projected on the horizontal plane, +y left, +z world up. Row 0 is
# the far front (x = BEV_X_MAX), column 0 the far left (y = BEV_Y_MAX), so the
# grid reads like a map with the robot driving up the page.
BEV_POSES = "poses"            # poses.txt as a csv channel; it carries the scan clock
BEV_ELEVATION = "bev_elevation"  # (3, H, W) float16: min / max / mean height, NaN unknown
BEV_SEMANTIC = "bev_semantic"    # (H, W) uint8: majority ontology id, 255 unknown
BEV_TRAV = "bev_trav"            # (H, W) uint8: 1 driven / 0 seen, not driven / 255 unknown
BEV_RESOLUTION = 0.2           # m per cell
BEV_X_MIN, BEV_X_MAX = -10.0, 30.0   # m, behind / ahead of the lidar (forward-biased)
BEV_Y_MIN, BEV_Y_MAX = -20.0, 20.0   # m, right / left
# Heights are stored relative to the ground under the robot: z_lidar + LIDAR_HEIGHT.
# Accumulation: the current scan plus the BEV_ACCUMULATE - 1 previous scans taken
# every BEV_STRIDE scans (past only, never future: the grid at scan j uses scans
# <= j), placed with poses.txt. 10 scans, stride 2 = the last ~2 s, ~2.6 m of
# motion at the median 1.3 m/s.
BEV_ACCUMULATE = 10
BEV_STRIDE = 2
# Driven corridor: the robot's footprint (ROBOT_HALF_WIDTH either side of the
# path) along the trajectory from BEV_PATH_BACK metres of path behind the scan to
# FORWARD_DISTANCE metres ahead. Frames with less than MIN_FUTURE_PATH ahead get
# an all-255 trav grid, as in the image masks.
BEV_PATH_BACK = 10.0
# Self-returns: besides MIN_RANGE, every sequence has robot-attached returns
# 1.7-2.9 m BEHIND the lidar (Ouster +x), mostly labelled void, at a fixed
# place in the sensor frame (00003: in 78% of scans, ~1700 points). They are
# dropped by this sensor-frame box (x along the Ouster, i.e. +x = rearward).
SELF_BOX_X = (0.0, 3.0)
SELF_BOX_Y = (-1.2, 1.2)
SELF_BOX_Z = (-0.75, 1.0)
# Classes a Warthog drives on (RELLIS ontology), for the trav-vs-semantic check.
DRIVEABLE = (1, 3, 10, 23, 31, 33)   # dirt, grass, asphalt, concrete, puddle, mud
OBSTACLE = (4, 5, 8, 9, 12, 15, 17, 18, 19, 27, 34)  # tree pole vehicle object building log person fence bush barrier rubble
