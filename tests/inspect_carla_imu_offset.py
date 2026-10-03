import numpy as np

from carla_reader import CarlaReader


SEQ = "0006"
DT = 0.1

FRAMES = [
    8, 9, 10,
    69, 70, 71, 72, 73, 74, 75,
    272, 273, 274, 275, 276, 277,
    300, 302, 303, 305, 306,
    312, 313, 314
]

# IMU location relative to CARLA vehicle reference point
R = np.array([0.0, 0.0, 1.5])


def vec(d):
    return np.array([d["x"], d["y"], d["z"]], dtype=float)


def cross(a, b):
    return np.cross(a, b)


reader = CarlaReader(
    carla_root="../CARLA_LVIMOT",
    sequence=SEQ
)


def get_world_angular_velocity(frame):
    pose = reader.load_pose(frame)
    return vec(pose["angular_velocity"])


def get_world_acceleration(frame):
    pose = reader.load_pose(frame)
    return vec(pose["acceleration"])


print("=" * 120)
print("CARLA IMU MOUNTING OFFSET DIAGNOSTIC")
print("=" * 120)

for frame in FRAMES:

    if frame == 0:
        continue

    pose = reader.load_pose(frame)
    imu = reader.load_imu(frame)

    a_vehicle = vec(pose["acceleration"])

    omega = vec(pose["angular_velocity"])
    omega_prev = get_world_angular_velocity(frame - 1)

    alpha = (omega - omega_prev) / DT

    # Rotational acceleration caused by IMU offset
    tangential = cross(alpha, R)
    centripetal = cross(omega, cross(omega, R))

    offset_acceleration = tangential + centripetal

    predicted_imu_acceleration = a_vehicle + offset_acceleration

    print("\n" + "=" * 120)
    print(f"FRAME {frame}")
    print("=" * 120)

    print(f"Angular velocity     : {omega}")
    print(f"Previous omega       : {omega_prev}")
    print(f"Angular acceleration : {alpha}")

    print(f"\nVehicle acceleration : {a_vehicle}")

    print(f"\nTangential term      : {tangential}")
    print(f"Centripetal term     : {centripetal}")
    print(f"Offset contribution  : {offset_acceleration}")

    print(f"\nPredicted IMU accel  : {predicted_imu_acceleration}")

    print(f"\nActual raw IMU       : {vec(imu['accelerometer'])}")

    difference = vec(imu["accelerometer"]) - predicted_imu_acceleration

    print(f"Difference           : {difference}")
    print(f"Difference magnitude : {np.linalg.norm(difference):.6f}")


print("\n" + "=" * 120)
print("OFFSET DIAGNOSTIC COMPLETE")
print("=" * 120)
