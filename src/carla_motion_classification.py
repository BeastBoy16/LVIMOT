import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


class MotionState(Enum):
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    STOPPED_DYNAMIC = "STOPPED_DYNAMIC"
    UNCERTAIN = "UNCERTAIN"


class CarlaMotionClassifier:
    def __init__(
        self,
        static_speed_threshold: float = 0.5,
        dynamic_speed_threshold: float = 1.2,
        min_history_length: int = 3,
        stopped_dynamic_window: float = 5.0,
        variance_threshold: float = 0.2,
    ):
        self.static_speed_threshold = static_speed_threshold
        self.dynamic_speed_threshold = dynamic_speed_threshold
        self.min_history_length = min_history_length
        self.stopped_dynamic_window = stopped_dynamic_window
        self.variance_threshold = variance_threshold

    @staticmethod
    def compute_world_velocity(position_k: np.ndarray, position_k1: np.ndarray, dt: float) -> np.ndarray:
        dt = max(float(dt), 1e-6)
        p_curr = np.asarray(position_k, dtype=np.float64)
        p_prev = np.asarray(position_k1, dtype=np.float64)
        return (p_curr - p_prev) / dt

    @staticmethod
    def ego_to_world_velocity(relative_velocity: np.ndarray, ego_velocity: np.ndarray, R_world_ego: np.ndarray) -> np.ndarray:
        v_rel = np.asarray(relative_velocity, dtype=np.float64)
        v_ego = np.asarray(ego_velocity, dtype=np.float64)
        R = np.asarray(R_world_ego, dtype=np.float64)
        return R @ v_rel + v_ego

    def classify_track(self, track_history: List[Dict[str, Union[float, np.ndarray]]], class_name: str = "vehicle") -> Dict[str, Union[MotionState, float, str]]:
        if len(track_history) < 2:
            return {
                "state": MotionState.UNCERTAIN,
                "speed_mean": 0.0,
                "speed_std": 0.0,
                "total_displacement": 0.0,
                "confidence": 0.0,
                "description": "Insufficient track history (need >= 2 frames)"
            }

        if class_name.lower() in ["traffic_sign", "pole", "building", "vegetation", "static"]:
            return {
                "state": MotionState.STATIC,
                "speed_mean": 0.0,
                "speed_std": 0.0,
                "total_displacement": 0.0,
                "confidence": 1.0,
                "description": f"Class {class_name} is inherently static"
            }

        timestamps = [float(h["timestamp"]) for h in track_history]
        positions = np.array([np.asarray(h["position"], dtype=np.float64) for h in track_history])

        speeds = []
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt > 1e-6:
                v = (positions[i] - positions[i - 1]) / dt
                speed = float(np.linalg.norm(v))
                speeds.append(speed)

        if not speeds:
            speeds = [0.0]

        speeds_arr = np.array(speeds, dtype=np.float64)
        speed_mean = float(np.mean(speeds_arr))
        speed_std = float(np.std(speeds_arr)) if len(speeds_arr) > 1 else 0.0
        
        total_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        duration = max(timestamps[-1] - timestamps[0], 1e-6)
        effective_speed = total_displacement / duration

        recent_speeds = speeds_arr[-2:] if len(speeds_arr) >= 2 else speeds_arr
        recent_speed_mean = float(np.mean(recent_speeds))
        latest_speed = float(speeds_arr[-1])

        if len(track_history) < self.min_history_length:
            if recent_speed_mean > self.dynamic_speed_threshold:
                state = MotionState.DYNAMIC
                confidence = 0.6
            elif recent_speed_mean < self.static_speed_threshold:
                state = MotionState.STATIC
                confidence = 0.6
            else:
                state = MotionState.UNCERTAIN
                confidence = 0.4
        else:
            had_high_speed = bool(np.any(speeds_arr > self.dynamic_speed_threshold))
            currently_stopped = latest_speed < self.static_speed_threshold and recent_speed_mean < self.static_speed_threshold

            if had_high_speed and currently_stopped:
                state = MotionState.STOPPED_DYNAMIC
                confidence = min(0.9, 0.5 + 0.1 * len(track_history))
            elif effective_speed > self.dynamic_speed_threshold or recent_speed_mean > self.dynamic_speed_threshold:
                state = MotionState.DYNAMIC
                confidence = min(1.0, 0.7 + 0.05 * len(track_history))
            elif effective_speed < self.static_speed_threshold and recent_speed_mean < self.static_speed_threshold:
                state = MotionState.STATIC
                confidence = min(1.0, 0.7 + 0.05 * len(track_history))
            else:
                state = MotionState.UNCERTAIN
                confidence = 0.5

        return {
            "state": state,
            "speed_mean": speed_mean,
            "speed_std": speed_std,
            "recent_speed": recent_speed_mean,
            "total_displacement": total_displacement,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "description": f"Classified as {state.value} (mean speed={speed_mean:.2f} m/s)"
        }

    def classify_all_tracks(self, tracks: List[Dict]) -> List[Dict]:
        classified = []
        for track in tracks:
            history = track.get("history", [])
            class_name = track.get("class", "vehicle")
            classification = self.classify_track(history, class_name)
            res = dict(track)
            res["motion_state"] = classification["state"].value
            res["motion_confidence"] = classification["confidence"]
            res["speed_mean"] = classification["speed_mean"]
            res["total_displacement"] = classification["total_displacement"]
            classified.append(res)
        return classified
