"""Knobs of the AMIAD Isaac Sim off-road recording (see :mod:`.extract`).

The recording is one ROS 2 mcap of the Barakuda platform driven through a
procedurally generated forest in Isaac Sim, with a perfect semantic camera.
Every rate below is in SIMULATION time (the header stamps): the sim runs at
about 0.26x real time, so the 381 s recording covers about 100 s of sim time.
"""

from __future__ import annotations

BAG = "/mnt/twotower/simulation_amiad/rosbag2_2026_06_29-12_23_00_0.mcap"
CLASSES_JSON = "/mnt/twotower/simulation_amiad/semantic_classes.json"

# Topics -> channel names of the extracted sequence.
TOPIC_LIDAR = "/ouster/points"              # PointCloud2 x y z float32, frame 'sensor'
TOPIC_RGB = "/cam2/rgb"                     # rgb8 1280x720
TOPIC_SEG = "/cam2/segmentation"            # 32SC1, RUNTIME Isaac label ids
TOPIC_SEG_LABELS = "/cam2/semantic_labels"  # String: JSON {runtime id: {"class": name}}
TOPIC_DEPTH = "/cam2/disp"                  # 32FC1, planar z-depth in metres (inf = sky)
TOPIC_CAMERA_INFO = "/cam2/camera_info"
TOPIC_POINTS_SEMANTIC = "/ouster/points_semantic"  # sim's own coloured cloud, x y z rgb
TOPIC_ODOMETRY = "/imu/odometry"            # nav_msgs/Odometry odom -> base_link
TOPIC_IMU = "/ouster/imu"
TOPIC_CMD_VEL = "/cmd_vel"                  # Twist, no header
TOPIC_CLOCK = "/clock"
TOPIC_TF = "/tf"                            # no /tf_static: fixed mounts are re-sent on /tf

LIDAR = "lidar"
RGB = "rgb"
SEMANTIC = "semantic"          # (H, W) uint8 training class id (name_to_id), 0 unlabeled
DEPTH = "depth"                # (H, W) uint16 millimetres, 0 = no return (sky, or > 65.534 m)
POINTS_SEMANTIC = "points_semantic"  # (N, 4) float32 x y z class id (255: outside the camera)
POSE = "pose"                  # (T, 7) odom -> base_link, t_xyz_q_xyzw, from /tf
ODOMETRY = "odometry"          # (T, 13) pose7 + twist6, from /imu/odometry
IMU = "imu"                    # (T, 10) acc3 gyro3 quat4
CMD_VEL = "cmd_vel"            # (T, 6) linear3 angular3, stamped with the latest /clock

# Frames. The camera headers say 'ZED2', but ZED2 is x-left / y-up / z-forward:
# the pinhole (optical: x right, y down, z forward) frame is its child
# 'ZED2_pcl' (180 deg about z). Projecting with 'ZED2' puts the ground in the
# sky; with 'ZED2_pcl' the lidar depth matches the depth image to 5 cm (median).
BASE_FRAME = "base_link"
ODOM_FRAME = "odom"
LIDAR_FRAME = "sensor"
CAMERA_HEADER_FRAME = "ZED2"
CAMERA_FRAME = "ZED2_pcl"
# /tf edges that never move (checked at extraction) -> calibration.yaml.
STATIC_EDGES = [
    ("base_link", "sensor"), ("base_link", "os_sensor"), ("base_link", "os_imu"),
    ("base_link", "ZED0"), ("base_link", "ZED1"), ("base_link", "ZED2"),
    ("ZED0", "ZED0_pcl"), ("ZED1", "ZED1_pcl"), ("ZED2", "ZED2_pcl"),
]
STATIC_TOLERANCE = 1e-3        # max |matrix difference| for an edge to count as fixed

# Two stamps belong to the same sim tick when they differ by at most this.
# Every sensor is stamped on the same 60 Hz tick (16.7 ms apart), so 5 ms is
# unambiguous. Shared by labelling, the export, agreement and the overlays.
SYNC_TOLERANCE = 0.005         # s

IMAGE_HEIGHT, IMAGE_WIDTH = 720, 1280
NO_DATA = 255
POINTS_SEMANTIC_NO_CAMERA_RGB = (40, 40, 40)  # colour of points the sim cloud leaves unlabeled

# semantic_classes.json, copied so the package does not depend on the share
# being mounted; extraction checks the file still says the same.
CLASSES = {
    0: ("unlabeled", (0, 0, 0), False),
    1: ("ground", (139, 90, 43), True),
    2: ("rock", (128, 128, 128), False),
    3: ("tree", (34, 139, 34), False),
    4: ("vegetation", (124, 252, 0), True),
    5: ("obstacle", (255, 0, 0), False),
    6: ("landmark", (255, 215, 0), False),
    7: ("object", (255, 0, 255), False),
}
NAME_TO_ID = {name: cid for cid, (name, _, _) in CLASSES.items()}
TRAVERSABLE_IDS = [cid for cid, (_, _, trav) in CLASSES.items() if trav]
