from collections import deque
from typing import Dict, List, Tuple

import numpy as np


class Carla4DEnvironmentMapper:
    """Bounded-memory static 3D map plus dynamic-object tubes."""

    def __init__(
        self,
        static_voxel_size: float = 0.25,
        max_map_distance: float = 60.0,
        max_static_voxels: int = 300000,
        max_planar_landmarks: int = 20000,
        max_dynamic_history: int = 500,
    ):
        self.static_voxel_size = float(static_voxel_size)
        self.max_map_distance = float(max_map_distance)
        self.max_static_voxels = int(max_static_voxels)
        self.max_planar_landmarks = int(max_planar_landmarks)
        self.max_dynamic_history = int(max_dynamic_history)
        self.static_voxel_map: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.static_planar_landmarks = deque(maxlen=self.max_planar_landmarks)
        self.dynamic_object_tubes: Dict[int, deque] = {}

    @staticmethod
    def _rotation_matrix(rotation: Dict[str, float]) -> np.ndarray:
        pitch = np.deg2rad(float(rotation.get("pitch", 0.0)))
        yaw = np.deg2rad(float(rotation.get("yaw", 0.0)))
        roll = np.deg2rad(float(rotation.get("roll", 0.0)))
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cr, sr = np.cos(roll), np.sin(roll)
        return np.array(
            [
                [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
                [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
                [sp, -cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _vector3(value, default=(0.0, 0.0, 0.0)) -> np.ndarray:
        if value is None:
            return np.asarray(default, dtype=np.float64)
        if isinstance(value, dict):
            return np.array(
                [float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0))],
                dtype=np.float64,
            )
        return np.asarray(value, dtype=np.float64).reshape(3)

    @staticmethod
    def _dimensions(value) -> Dict[str, float]:
        if isinstance(value, dict):
            return {
                "length": float(value.get("length", 4.5)),
                "width": float(value.get("width", 1.8)),
                "height": float(value.get("height", 1.5)),
            }
        arr = np.asarray(value if value is not None else [4.5, 1.8, 1.5], dtype=np.float64).reshape(-1)
        padded = np.pad(arr[:3], (0, max(0, 3 - len(arr))), constant_values=1.0)
        return {"length": float(padded[0]), "width": float(padded[1]), "height": float(padded[2])}

    def separate_points(
        self,
        points: np.ndarray,
        dynamic_objects: List[Dict],
        R_world_ego: np.ndarray,
        t_world_ego: np.ndarray,
    ):
        points = np.asarray(points, dtype=np.float64)
        if len(points) == 0:
            return np.empty((0, 3)), np.empty((0, 3))
        pts_w = points[:, :3] @ np.asarray(R_world_ego, dtype=np.float64).T + np.asarray(t_world_ego, dtype=np.float64)
        if not dynamic_objects:
            return pts_w, np.empty((0, 3))

        dynamic_mask = np.zeros(len(pts_w), dtype=bool)
        for obj in dynamic_objects:
            center = self._vector3(obj.get("position", obj.get("location")))
            dims = self._dimensions(obj.get("dimensions"))
            half = np.array([dims["length"] / 2 + 0.2, dims["width"] / 2 + 0.2, dims["height"] / 2 + 0.2])
            radius2 = float(np.dot(half, half))
            delta = pts_w - center
            candidate = np.flatnonzero(np.einsum("ij,ij->i", delta, delta) <= radius2)
            if candidate.size == 0:
                continue
            R = self._rotation_matrix(obj.get("rotation", {}))
            local = (pts_w[candidate] - center) @ R
            inside = np.all(np.abs(local) <= half, axis=1)
            dynamic_mask[candidate[inside]] = True
        return pts_w[~dynamic_mask], pts_w[dynamic_mask]

    def _evict_distant_voxels(self, ego_position: np.ndarray, margin: float = 1.35):
        """Keep the live map spatially bounded around the current ego pose.

        A fixed global dictionary eventually reaches its cap and then stops
        mapping new road.  A rolling local map is more appropriate for the
        live visualization target and gives bounded RAM over long drives.
        """
        if not self.static_voxel_map:
            return
        ego = np.asarray(ego_position, dtype=np.float64)
        max_d2 = float((self.max_map_distance * margin) ** 2)
        remove = []
        for key in self.static_voxel_map.keys():
            center = (np.asarray(key, dtype=np.float64) + 0.5) * self.static_voxel_size
            d = center - ego
            if float(np.dot(d, d)) > max_d2:
                remove.append(key)
        for key in remove:
            self.static_voxel_map.pop(key, None)

    def _update_static_voxels(self, points_world: np.ndarray, ego_position: np.ndarray):
        if len(points_world) == 0:
            return
        delta = points_world - ego_position
        keep = np.einsum("ij,ij->i", delta, delta) <= self.max_map_distance ** 2
        pts = points_world[keep]
        if len(pts) == 0:
            return
        keys_arr = np.floor(pts / self.static_voxel_size).astype(np.int64)
        keys, inverse = np.unique(keys_arr, axis=0, return_inverse=True)
        sums = np.zeros((len(keys), 3), dtype=np.float64)
        np.add.at(sums, inverse, pts)
        counts = np.bincount(inverse, minlength=len(keys)).astype(np.float64)
        representatives = sums / np.maximum(counts[:, None], 1.0)
        for key_arr, pt in zip(keys, representatives):
            key = tuple(int(x) for x in key_arr)
            if key in self.static_voxel_map:
                self.static_voxel_map[key] = (0.7 * self.static_voxel_map[key] + 0.3 * pt).astype(np.float32)
            elif len(self.static_voxel_map) < self.max_static_voxels:
                self.static_voxel_map[key] = np.asarray(pt, dtype=np.float32).copy()

    def add_frame_data(
        self,
        frame_id: int,
        timestamp: float,
        lidar_points: np.ndarray,
        planar_features: List[Dict],
        active_tracks: List[Dict],
        ego_pose: Dict,
        update_static: bool = True,
        update_planar: bool = True,
        evict_static: bool = False,
    ):
        R_w_e = self._rotation_matrix(ego_pose["rotation"])
        t_w_e = self._vector3(ego_pose["location"])
        dynamic_tracks = [
            t for t in active_tracks if t.get("motion_state") in {"DYNAMIC", "STOPPED_DYNAMIC", "UNCERTAIN"}
        ]

        for track in dynamic_tracks:
            tid = int(track["track_id"])
            tube = self.dynamic_object_tubes.setdefault(tid, deque(maxlen=self.max_dynamic_history))
            tube.append(
                {
                    "frame_id": int(frame_id),
                    "timestamp": float(timestamp),
                    "position": self._vector3(track.get("position")).tolist(),
                    "velocity": self._vector3(track.get("velocity")).tolist(),
                    "speed": float(track.get("speed", np.linalg.norm(self._vector3(track.get("velocity"))))),
                    "dimensions": self._dimensions(track.get("dimensions")),
                    "rotation": track.get("rotation", {}),
                    "class": track.get("class", "object"),
                    "motion_state": track.get("motion_state", "UNCERTAIN"),
                }
            )

        if bool(update_static):
            static_pts, _ = self.separate_points(lidar_points, dynamic_tracks, R_w_e, t_w_e)
            self._update_static_voxels(static_pts, t_w_e)
        if bool(evict_static):
            self._evict_distant_voxels(t_w_e)

        # Store only a bounded representative subset of planar landmarks.
        if bool(update_planar) and planar_features:
            step = max(1, len(planar_features) // 256)
            for feat in planar_features[::step][:256]:
                c = self._vector3(feat.get("center"))
                n = self._vector3(feat.get("normal"), default=(0.0, 0.0, 1.0))
                c_w = R_w_e @ c + t_w_e
                n_w = R_w_e @ n
                n_w /= max(np.linalg.norm(n_w), 1e-6)
                self.static_planar_landmarks.append(
                    {
                        "center": c_w.tolist(),
                        "normal": n_w.tolist(),
                        "planarity": float(feat.get("planarity", 0.0)),
                        "frame_id": int(frame_id),
                        "timestamp": float(timestamp),
                    }
                )

    def get_static_map_points(self) -> np.ndarray:
        if not self.static_voxel_map:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(list(self.static_voxel_map.values()), dtype=np.float64)

    def query_dynamic_objects_at_time(self, query_timestamp: float) -> List[Dict]:
        results = []
        for tid, tube_deque in self.dynamic_object_tubes.items():
            tube = list(tube_deque)
            if not tube:
                continue
            timestamps = np.asarray([snap["timestamp"] for snap in tube], dtype=np.float64)
            if query_timestamp <= timestamps[0]:
                snap = tube[0]
                pos = np.asarray(snap["position"]) + np.asarray(snap["velocity"]) * (query_timestamp - timestamps[0])
            elif query_timestamp >= timestamps[-1]:
                snap = tube[-1]
                pos = np.asarray(snap["position"]) + np.asarray(snap["velocity"]) * (query_timestamp - timestamps[-1])
            else:
                idx = int(np.searchsorted(timestamps, query_timestamp))
                s0, s1 = tube[idx - 1], tube[idx]
                alpha = (query_timestamp - s0["timestamp"]) / max(s1["timestamp"] - s0["timestamp"], 1e-6)
                pos = (1.0 - alpha) * np.asarray(s0["position"]) + alpha * np.asarray(s1["position"])
                snap = s1
            results.append(
                {
                    "track_id": tid,
                    "timestamp": float(query_timestamp),
                    "interpolated_position": pos.tolist(),
                    "velocity": snap["velocity"],
                    "class": snap["class"],
                    "dimensions": snap["dimensions"],
                    "rotation": snap["rotation"],
                }
            )
        return results

    def export_summary(self) -> Dict:
        return {
            "total_static_voxels": len(self.static_voxel_map),
            "total_planar_landmarks": len(self.static_planar_landmarks),
            "total_dynamic_actors": len(self.dynamic_object_tubes),
            "dynamic_actor_ids": list(self.dynamic_object_tubes.keys()),
            "dynamic_tube_lengths": {int(k): len(v) for k, v in self.dynamic_object_tubes.items()},
            "memory_bounded": True,
        }
