import numpy as np


class CarlaIMUPreprocessor:
    """
    Converts CARLA IMU JSON measurements into the motion format
    expected by the existing LVIMOT IMU preintegrator.
    """

    def __init__(self, dt=0.1):
        self.dt = dt

    def process(self, imu_data):
        """
        Convert one CARLA IMU measurement into LVIMOT motion format.
        """

        acceleration = imu_data["accelerometer"]
        gyroscope = imu_data["gyroscope"]

        accel = np.array(
            [
                acceleration["x"],
                acceleration["y"],
                acceleration["z"]
            ],
            dtype=np.float64
        )

        gyro = np.array(
            [
                gyroscope["x"],
                gyroscope["y"],
                gyroscope["z"]
            ],
            dtype=np.float64
        )

        return {
            "dt": self.dt,
            "acceleration": accel,
            "gyro": gyro,

            # CARLA does not provide KITTI-style vf/vl/vu here.
            # These are kept as None until velocity/orientation
            # information is incorporated from the CARLA pose.
            "velocity": None,
            "rotation": None,

            "timestamp": imu_data.get("timestamp"),
            "dataset_frame": imu_data.get("dataset_frame"),
            "carla_frame": imu_data.get("carla_frame")
        }

    def remove_gravity(self, acceleration, gravity=9.80665):
        """
        Remove gravity from the Z component.

        This is useful for analysing the acceleration signal.
        The raw acceleration is preserved separately.
        """

        acceleration = np.asarray(
            acceleration,
            dtype=np.float64
        ).copy()

        acceleration[2] -= gravity

        return acceleration


if __name__ == "__main__":

    import sys

    sys.path.insert(0, "src")

    from carla_reader import CarlaReader
    from imu_preintegration import IMUPreintegrator

    sequence = "0001"

    reader = CarlaReader(
        "~/catkin_ws/src/CARLA_LVIMOT",
        sequence
    )

    imu_processor = CarlaIMUPreprocessor(
        dt=0.1
    )

    preintegrator = IMUPreintegrator()

    print("=" * 60)
    print("CARLA IMU PREINTEGRATION TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Frame k
    # ---------------------------------------------------------

    imu_data_k = reader.load_imu(0)

    motion_k = imu_processor.process(
        imu_data_k
    )

    # ---------------------------------------------------------
    # Frame k+1
    # ---------------------------------------------------------

    imu_data_k1 = reader.load_imu(1)

    motion_k1 = imu_processor.process(
        imu_data_k1
    )

    # ---------------------------------------------------------
    # Print measurements
    # ---------------------------------------------------------

    print("\nFrame 0")
    print("Acceleration:", motion_k["acceleration"])
    print("Gyroscope:", motion_k["gyro"])

    print("\nFrame 1")
    print("Acceleration:", motion_k1["acceleration"])
    print("Gyroscope:", motion_k1["gyro"])

    print("\nDelta time:", motion_k["dt"])

    # ---------------------------------------------------------
    # Preintegrate
    # ---------------------------------------------------------

    result = preintegrator.integrate(
        motion_k,
        motion_k1
    )

    print("\nIMU PREINTEGRATION RESULT")
    print("-" * 60)

    print("Delta position:")
    print(result["delta_p"])

    print("\nDelta velocity:")
    print(result["delta_v"])

    print("\nDelta rotation:")
    print(result["delta_theta"])

    print("\nDelta rotation matrix:")
    print(result["delta_R"])

    print("\nCARLA IMU PREINTEGRATION TEST PASSED")
