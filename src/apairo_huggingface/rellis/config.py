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
