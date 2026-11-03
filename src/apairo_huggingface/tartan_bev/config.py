"""Knobs of the TartanDrive BEV grids (see :mod:`.bev`).

The grid (cells, extent, frame) is the shared one of
:mod:`apairo_huggingface.bev_grid`; the accumulation and the driven corridor
use the same numbers as the RELLIS recipe. Only what the rig imposes differs:
the channel names, the heading axis (+x of the vehicle frame), the lidar
height and the vehicle width.
"""

from __future__ import annotations

from .. import config as tartan

# Channels of a raw TartanDrive recording (symlinked into the view).
CAMERA = tartan.CAMERA        # image_left_color: 1024x544 rectified RGB, ~10 Hz
LIDAR = "velodyne_0"          # (N, 3) float32 x y z in the vehicle frame (x forward, z up), 10 Hz
POSES = tartan.POSES          # gicp_poses: (4, 4) T_world_vehicle per scan, on the lidar clock
ACTION = tartan.ACTION        # cmd: [linear_velocity, angular_velocity] command
SOURCE_DIRS = [CAMERA, LIDAR, POSES, ACTION]

# Derived channels written in the view.
BEV_ELEVATION = "bev_elevation"  # (3, H, W) float16: min / max / mean height, NaN unknown
BEV_GEOMETRY = "bev_geometry"    # (2, H, W) float16: roughness / step (m), NaN unknown
BEV_TRAV = "bev_trav"            # (H, W) uint8: 1 driven / 0 seen, not driven / 255 unknown
NO_DATA = 255

# Rig. velodyne_0 is in the vehicle frame, whose origin sits at the lidar,
# with the ground ~1.77 m below it (median of the ground returns 3-6 m ahead
# on flat stretches of five sequences; the trav_mask labels use the same).
LIDAR_HEIGHT = -tartan.GROUND_OFFSET   # 1.77 m: heights are z + LIDAR_HEIGHT, ground under the robot ~0
MIN_RANGE = 2.0               # m; drops the roof and sensor mounts (all within 2 m of the lidar)
ROBOT_HALF_WIDTH = tartan.ROBOT_HALF_WIDTH   # 0.8 m: Yamaha Viking, 1.57 m wide
ROBOT_HALF_LENGTH = 1.5       # m, for drawing only (the Viking is ~3 m long)

# Recipe: identical to the RELLIS BEV grids.
BEV_ACCUMULATE = 10           # scans per grid, past only: j, j - 2, ..., j - 18 (~2 s)
BEV_STRIDE = 2
BEV_PATH_BACK = 10.0          # m of path behind the scan in the driven corridor
FORWARD_DISTANCE = 30.0       # m of path ahead (the grid reaches 30 m ahead)
MIN_FUTURE_PATH = 10.0        # m; frames with less path ahead get an all-255 trav grid
MIN_CELL_POINTS = 2           # roughness needs at least two points in the cell

# Driveable proxy for the quality check: a cell is "flat" when its roughness
# and its step to the neighbours are both below these. The Viking has ~0.3 m
# of ground clearance; these are deliberately stricter.
FLAT_ROUGHNESS = 0.10         # m, std of the point heights in the cell
FLAT_STEP = 0.20              # m, largest mean-height jump to a known 8-neighbour

# Export: camera clock -> nearest scan, as the published image dataset.
TOLERANCE = tartan.TOLERANCE  # 0.1 s
FPS = tartan.FPS
ROBOT_TYPE = tartan.ROBOT_TYPE
TASK = tartan.TASK

# A representative subset: every site and both seasons (Sep 2023, Oct-Nov
# 2023), 13 of the 55 recordings with scans, 28.2k of the 125.7k scans.
SEQUENCES = [
    "turnpike_2023-09-12-12-39-19",
    "turnpike_flat_2023-09-12-14-21-03",
    "turnpike_warehouse_2023-09-13-19-03-05",
    "figure_8_morning_slow_2023-09-12-11-06-32",
    "figure_8_2023-09-14-11-37-49",
    "gupta_skydio_2023-09-14-11-03-09",
    "warehouse_gupta_2023-09-13-16-28-57",
    "meadows_2023-09-14-12-18-22",
    "slag_heap_2023-09-14-12-23-05",
    "sam_loop_2023-09-27-13-04-54",
    "2023-11-02-16-00-43_down_meadows",
    "2023-11-14-14-34-53_gupta",
    "2023-11-14-15-22-01_figure_8",
]
