import sys
import numpy as np

sys.path.append("src")

from carla_reader import CarlaReader


SEQUENCE = "0006"
DATA_ROOT = "/home/gowtham507/catkin_ws/src/CARLA_LVIMOT"

reader = CarlaReader(DATA_ROOT, SEQUENCE)

frames = [0, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 400, 450, 499]

print("=" * 100)
print("CARLA SEQUENCE MOTION INSPECTION")
print("=" * 100)
print()

for k in frames:

    pose = reader.load_pose(k)

    loc = pose["location"]
    rot = pose["rotation"]
    vel = pose["velocity"]

    print(f"Frame {k}")
    print("-" * 100)

    print(
        "Position:",
        f"X={loc['x']:.6f}",
        f"Y={loc['y']:.6f}",
        f"Z={loc['z']:.6f}"
    )

    print(
        "Rotation:",
        f"roll={rot['roll']:.6f}",
        f"pitch={rot['pitch']:.6f}",
        f"yaw={rot['yaw']:.6f}"
    )

    print(
        "Velocity:",
        f"X={vel['x']:.6f}",
        f"Y={vel['y']:.6f}",
        f"Z={vel['z']:.6f}"
    )

    print()


print("=" * 100)
