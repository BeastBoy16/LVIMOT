import numpy as np


class CarlaDetector:

    def __init__(self, reader, calibration):
        self.reader = reader
        self.calibration = calibration

    # ========================================================
    # Load CARLA ground-truth objects
    # ========================================================

    def load_detections(self, frame):

        labels = self.reader.load_labels(frame)

        detections = []

        for obj in labels.get("objects", []):

            bbox = obj["bounding_box"]

            dimensions = bbox["dimensions"]

            length = float(dimensions["length"])
            width = float(dimensions["width"])
            height = float(dimensions["height"])

            location = obj["location"]

            center = np.array(
                [
                    float(location["x"]),
                    float(location["y"]),
                    float(location["z"])
                ],
                dtype=np.float64
            )

            rotation = obj["rotation"]

            yaw = float(rotation["yaw"])
            pitch = float(rotation["pitch"])
            roll = float(rotation["roll"])

            detection = {
                "actor_id": int(obj["actor_id"]),
                "type": obj["type"],
                "score": 1.0,

                "location": center,

                "dimensions": np.array(
                    [
                        length,
                        width,
                        height
                    ],
                    dtype=np.float64
                ),

                "rotation": {
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll
                },

                "velocity": obj.get(
                    "velocity",
                    {}
                ),

                "acceleration": obj.get(
                    "acceleration",
                    {}
                ),

                "bounding_box": bbox
            }

            detections.append(detection)

        return detections

    # ========================================================
    # Number of detections
    # ========================================================

    def count(self, frame):

        return len(
            self.load_detections(frame)
        )
