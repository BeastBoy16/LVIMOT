import numpy as np


class CarlaIMUMotion:

    def __init__(self, reader):
        self.reader = reader

    # -------------------------------------------------------------
    # CARLA rotation matrix
    #
    # This follows CARLA's Transform::GetMatrix() convention.
    # CARLA uses a left-handed Unreal coordinate system:
    # X = forward
    # Y = right
    # Z = up
    # -------------------------------------------------------------

    def rotation_matrix(self, rotation):

        pitch = np.deg2rad(rotation["pitch"])
        yaw = np.deg2rad(rotation["yaw"])
        roll = np.deg2rad(rotation["roll"])

        cp = np.cos(pitch)
        sp = np.sin(pitch)

        cy = np.cos(yaw)
        sy = np.sin(yaw)

        cr = np.cos(roll)
        sr = np.sin(roll)

        R = np.array([
            [
                cp * cy,
                cy * sp * sr - sy * cr,
                -cy * sp * cr - sy * sr
            ],
            [
                cp * sy,
                sy * sp * sr + cy * cr,
                -sy * sp * cr + cy * sr
            ],
            [
                sp,
                -cp * sr,
                cp * cr
            ]
        ], dtype=np.float64)

        return R

    # -------------------------------------------------------------
    # Raw CARLA IMU acceleration
    # -------------------------------------------------------------

    def get_raw_acceleration(self, frame):

        imu = self.reader.load_imu(frame)

        return np.array([
            imu["accelerometer"]["x"],
            imu["accelerometer"]["y"],
            imu["accelerometer"]["z"]
        ], dtype=np.float64)

    # -------------------------------------------------------------
    # Gravity expressed in ego/body frame
    # -------------------------------------------------------------

    def gravity_in_body_frame(self, frame):

        pose = self.reader.load_pose(frame)

        R_world_ego = self.rotation_matrix(
            pose["rotation"]
        )

        gravity_world = np.array([
            0.0,
            0.0,
            -9.80665
        ], dtype=np.float64)

        gravity_body = R_world_ego.T @ gravity_world

        return gravity_body

    # -------------------------------------------------------------
    # Gravity-corrected linear acceleration
    #
    # CARLA IMU measurement behaves as specific force:
    #
    # a_corrected = a_raw + g_body
    # -------------------------------------------------------------

    def corrected_acceleration(self, frame):

        raw = self.get_raw_acceleration(frame)

        gravity = self.gravity_in_body_frame(frame)

        corrected = raw + gravity

        return corrected

    # -------------------------------------------------------------
    # Corrected acceleration converted to WORLD frame
    # -------------------------------------------------------------

    def corrected_acceleration_world(self, frame):

        pose = self.reader.load_pose(frame)

        R_world_ego = self.rotation_matrix(
            pose["rotation"]
        )

        acceleration_body = self.corrected_acceleration(frame)

        acceleration_world = R_world_ego @ acceleration_body

        return acceleration_world

    # -------------------------------------------------------------
    # Ground-truth vehicle velocity
    # -------------------------------------------------------------

    def get_velocity(self, frame):

        pose = self.reader.load_pose(frame)

        return np.array([
            pose["velocity"]["x"],
            pose["velocity"]["y"],
            pose["velocity"]["z"]
        ], dtype=np.float64)


if __name__ == "__main__":

    from carla_reader import CarlaReader

    DATA_ROOT = "/home/gowtham507/catkin_ws/src/CARLA_LVIMOT"
    SEQUENCE = "0006"

    reader = CarlaReader(DATA_ROOT, SEQUENCE)

    motion = CarlaIMUMotion(reader)

    frames = [50, 100, 150, 200]

    print("=" * 100)
    print("CARLA IMU MOTION TEST")
    print("=" * 100)

    for frame in frames:

        pose = reader.load_pose(frame)

        raw = motion.get_raw_acceleration(frame)

        gravity = motion.gravity_in_body_frame(frame)

        corrected = motion.corrected_acceleration(frame)

        corrected_world = motion.corrected_acceleration_world(frame)

        print()
        print(f"Frame {frame}")
        print("-" * 100)

        print(
            "Rotation:",
            f"roll={pose['rotation']['roll']:.6f}",
            f"pitch={pose['rotation']['pitch']:.6f}",
            f"yaw={pose['rotation']['yaw']:.6f}"
        )

        print("\nRaw acceleration:")
        print(raw)

        print("\nGravity body:")
        print(gravity)

        print("\nCorrected body:")
        print(corrected)

        print("\nCorrected world:")
        print(corrected_world)
