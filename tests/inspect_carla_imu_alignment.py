import numpy as np

from carla_reader import CarlaReader
from carla_imu_motion import CarlaIMUMotion


SEQ = "0006"

# Frames where the previous validation showed unusual errors
FRAMES = [
    8, 9, 10,
    69, 70, 71, 72, 73, 74, 75,
    272, 273, 274, 275, 276, 277,
    300, 302, 303, 305, 306,
    312, 313, 314
]


def rotation_matrix(rotation):
    roll = np.deg2rad(rotation["roll"])
    pitch = np.deg2rad(rotation["pitch"])
    yaw = np.deg2rad(rotation["yaw"])

    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    return np.array([
        [cp * cy,
         cy * sp * sr - sy * cr,
         -cy * sp * cr - sy * sr],

        [cp * sy,
         sy * sp * sr + cy * cr,
         -sy * sp * cr + cy * sr],

        [sp,
         -cp * sr,
         cp * cr]
    ])


reader = CarlaReader(
    carla_root="../CARLA_LVIMOT",
    sequence=SEQ
)

motion = CarlaIMUMotion(reader)

print("=" * 110)
print("CARLA IMU DETAILED DIAGNOSTIC")
print("=" * 110)

for frame in FRAMES:

    pose = reader.load_pose(frame)
    imu = reader.load_imu(frame)

    R = rotation_matrix(pose["rotation"])

    gt_body = np.array([
        pose["acceleration"]["x"],
        pose["acceleration"]["y"],
        pose["acceleration"]["z"]
    ])

    gt_world = R @ gt_body

    raw = motion.get_raw_acceleration(frame)
    gravity = motion.gravity_in_body_frame(frame)
    corrected_body = motion.corrected_acceleration(frame)
    corrected_world = motion.corrected_acceleration_world(frame)

    error = corrected_world - gt_world
    error_mag = np.linalg.norm(error)

    velocity = pose["velocity"]
    rotation = pose["rotation"]
    location = pose["location"]

    print("\n" + "=" * 110)
    print(f"FRAME {frame}")
    print("=" * 110)

    print("\n--- CARLA FRAME / TIMESTAMP ---")
    print(f"IMU : frame={imu.get('carla_frame')} timestamp={imu.get('timestamp')}")
    print(f"POSE: frame={pose.get('carla_frame')} timestamp={pose.get('timestamp')}")

    print("\n--- POSITION ---")
    print(
        f"X={location['x']:.6f} "
        f"Y={location['y']:.6f} "
        f"Z={location['z']:.6f}"
    )

    print("\n--- VELOCITY ---")
    print(
        f"X={velocity['x']:.6f} "
        f"Y={velocity['y']:.6f} "
        f"Z={velocity['z']:.6f}"
    )

    print("\n--- ROTATION ---")
    print(
        f"roll={rotation['roll']:.6f} "
        f"pitch={rotation['pitch']:.6f} "
        f"yaw={rotation['yaw']:.6f}"
    )

    print("\n--- ANGULAR VELOCITY ---")
    av = pose["angular_velocity"]
    print(
        f"X={av['x']:.6f} "
        f"Y={av['y']:.6f} "
        f"Z={av['z']:.6f}"
    )

    print("\n--- ACCELERATION ---")
    print(f"GT body       : {gt_body}")
    print(f"GT world      : {gt_world}")
    print(f"IMU raw       : {raw}")
    print(f"Gravity body  : {gravity}")
    print(f"Corrected body: {corrected_body}")
    print(f"Corrected world: {corrected_world}")

    print("\n--- ERROR ---")
    print(f"Error vector  : {error}")
    print(f"Error magnitude: {error_mag:.6f} m/s²")

print("\n" + "=" * 110)
print("DIAGNOSTIC COMPLETE")
print("=" * 110)
