from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class CarlaLiveVisualizer:
    """Processed live camera + cleaned rolling spatial 4-D-map visualizer.

    Important separation:
    - The estimator/tracker/map are not modified here.
    - This class only visualizes estimator outputs.
    - Short display-only track persistence is labelled PREDICTED and never fed
      back into MOT or evaluation.
    - The dashboard preview files are display outputs only.
    """

    def __init__(
        self,
        bev_size=620,
        meters=60.0,
        max_snapshot_points=5000,
        map_cell_size_m=0.35,
        display_track_hold_frames=5,
        display_candidate_hold_frames=None,
        preview_width=960,
    ):
        self.bev_size = int(bev_size)
        self.meters = float(meters)
        self.max_snapshot_points = max(800, int(max_snapshot_points))
        self.map_cell_size_m = float(np.clip(map_cell_size_m, 0.15, 1.0))
        self.display_track_hold_frames = max(0, int(display_track_hold_frames))
        if display_candidate_hold_frames is None:
            display_candidate_hold_frames = os.environ.get(
                "LVIMOT_CANDIDATE_HOLD_FRAMES", "4"
            )
        try:
            display_candidate_hold_frames = int(display_candidate_hold_frames)
        except Exception:
            display_candidate_hold_frames = 4
        self.display_candidate_hold_frames = max(0, min(12, display_candidate_hold_frames))
        self.preview_width = max(480, int(preview_width))

        map_env = os.environ.get("LVIMOT_LIVE_MAP_SNAPSHOT", "").strip()
        if map_env:
            self.snapshot_path = Path(map_env)
        else:
            self.snapshot_path = Path("outputs_carla") / "live" / "live_map_snapshot.json"
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        camera_env = os.environ.get("LVIMOT_LIVE_CAMERA_SNAPSHOT", "").strip()
        if camera_env:
            self.camera_snapshot_path = Path(camera_env)
        else:
            self.camera_snapshot_path = Path("outputs_carla") / "live" / "live_camera_preview.png"
        self.camera_snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        # Display-only cache.  It prevents the camera overlay from blinking off
        # immediately when one detector frame is missed.  These cached boxes do
        # NOT become tracker outputs; they are drawn amber as PREDICTED.
        self._display_track_cache: Dict[int, Dict] = {}

        # Display-only fused-candidate cache.  The actual detector/fusion output
        # is untouched.  A candidate that disappears for only a few frames is
        # retained as CAND HOLD so the live frontend visualization does not
        # flicker, especially for distant/static vehicles with sparse LiDAR.
        self._display_candidate_cache: Dict[int, Dict] = {}
        self._next_display_candidate_id = 1

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _pose(result):
        est = result.get("state_estimation", {}) or {}
        pose = np.asarray(est.get("pose", np.zeros(6)), dtype=np.float64).reshape(-1)
        if pose.size < 6:
            padded = np.zeros(6, dtype=np.float64)
            padded[: pose.size] = pose
            pose = padded
        return pose

    @staticmethod
    def _velocity(result):
        est = result.get("state_estimation", {}) or {}
        vel = np.asarray(est.get("velocity", np.zeros(3)), dtype=np.float64).reshape(-1)
        if vel.size < 3:
            padded = np.zeros(3, dtype=np.float64)
            padded[: vel.size] = vel
            vel = padded
        return vel

    @staticmethod
    def _world_to_heading_up(points_xy, ego_x, ego_y, yaw):
        pts = np.asarray(points_xy, dtype=np.float64)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        dx = pts[:, 0] - float(ego_x)
        dy = pts[:, 1] - float(ego_y)
        c = math.cos(float(yaw))
        s = math.sin(float(yaw))
        forward = c * dx + s * dy
        lateral = -s * dx + c * dy
        return np.column_stack([forward, lateral])

    # ------------------------------------------------------------------
    # Clean map rendering
    # ------------------------------------------------------------------

    def _bounded_local_map(self, map_points, pose):
        pts = np.asarray(map_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)

        local_xy = self._world_to_heading_up(pts[:, :2], pose[3], pose[4], pose[2])
        z_rel = pts[:, 2] - pose[5]
        dist2 = np.einsum("ij,ij->i", local_xy, local_xy)
        keep = (
            np.isfinite(local_xy).all(axis=1)
            & np.isfinite(z_rel)
            & (dist2 <= self.meters * self.meters)
            & (z_rel >= -4.0)
            & (z_rel <= 6.0)
        )
        return np.column_stack([local_xy[keep], z_rel[keep]])

    @staticmethod
    def _estimate_ground_height(local):
        if len(local) < 30:
            return -1.0
        z = np.asarray(local[:, 2], dtype=np.float64)
        z = z[np.isfinite(z) & (z >= -4.0) & (z <= 2.0)]
        if len(z) < 20:
            return float(np.median(local[:, 2])) if len(local) else -1.0
        bins = np.arange(-4.0, 2.05, 0.15)
        hist, edges = np.histogram(z, bins=bins)
        idx = int(np.argmax(hist))
        return float(0.5 * (edges[idx] + edges[idx + 1]))

    def _clean_map_cells(self, local):
        """Convert noisy 3-D points into a cleaner 2-D local occupancy map.

        The underlying mapper is unchanged.  For display only, points are
        collapsed into 35 cm XY cells, isolated single-cell speckle is removed,
        and cells are split into ground/road versus vertical structure.
        """
        pts = np.asarray(local, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            empty = np.empty((0, 3), dtype=np.float64)
            return empty, empty, {"ground_z": None, "raw_points": 0, "clean_cells": 0}

        cell = self.map_cell_size_m
        ix = np.floor((pts[:, 0] + self.meters) / cell).astype(np.int32)
        iy = np.floor((pts[:, 1] + self.meters) / cell).astype(np.int32)
        grid_n = int(math.ceil((2.0 * self.meters) / cell)) + 2
        valid = (ix >= 0) & (ix < grid_n) & (iy >= 0) & (iy < grid_n)
        pts = pts[valid]
        ix = ix[valid]
        iy = iy[valid]
        if len(pts) == 0:
            empty = np.empty((0, 3), dtype=np.float64)
            return empty, empty, {"ground_z": None, "raw_points": 0, "clean_cells": 0}

        keys = ix.astype(np.int64) * int(grid_n) + iy.astype(np.int64)
        unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        n_cells = len(unique_keys)

        # Mean height in each XY cell.  Forward/lateral use the cell centre so
        # nearby 3-D voxels appear as a clean surface instead of random speckle.
        sum_z = np.bincount(inverse, weights=pts[:, 2], minlength=n_cells)
        mean_z = sum_z / np.maximum(counts, 1)
        ux = (unique_keys // int(grid_n)).astype(np.int32)
        uy = (unique_keys % int(grid_n)).astype(np.int32)
        forward = (ux.astype(np.float64) + 0.5) * cell - self.meters
        lateral = (uy.astype(np.float64) + 0.5) * cell - self.meters

        occupied = set((int(a), int(b)) for a, b in zip(ux.tolist(), uy.tolist()))
        neighbour_count = np.zeros(n_cells, dtype=np.int16)
        for i, (a, b) in enumerate(zip(ux.tolist(), uy.tolist())):
            total = 0
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if da == 0 and db == 0:
                        continue
                    if (a + da, b + db) in occupied:
                        total += 1
            neighbour_count[i] = total

        ground_z = self._estimate_ground_height(pts)
        is_ground = np.abs(mean_z - ground_z) <= 0.32

        # Ground needs spatial support; structures may be thin (poles, walls),
        # so repeated vertical occupancy is also enough to keep them.
        keep_ground = is_ground & (neighbour_count >= 2)
        keep_structure = (~is_ground) & (
            (counts >= 2) | (neighbour_count >= 2)
        )

        ground = np.column_stack([forward[keep_ground], lateral[keep_ground], mean_z[keep_ground]])
        structure = np.column_stack(
            [forward[keep_structure], lateral[keep_structure], mean_z[keep_structure]]
        )

        # Conservative fallback for unusually sparse maps.
        if len(ground) + len(structure) < min(120, max(20, n_cells // 20)):
            keep = neighbour_count >= 1
            fallback = np.column_stack([forward[keep], lateral[keep], mean_z[keep]])
            gmask = np.abs(fallback[:, 2] - ground_z) <= 0.32 if len(fallback) else np.zeros(0, dtype=bool)
            ground = fallback[gmask]
            structure = fallback[~gmask]

        # Bound dashboard JSON size deterministically.
        total = len(ground) + len(structure)
        if total > self.max_snapshot_points:
            ratio = total / float(self.max_snapshot_points)
            step = max(1, int(math.ceil(ratio)))
            ground = ground[::step]
            structure = structure[::step]

        stats = {
            "ground_z": float(ground_z),
            "raw_points": int(len(pts)),
            "clean_cells": int(len(ground) + len(structure)),
            "ground_cells": int(len(ground)),
            "structure_cells": int(len(structure)),
            "cell_size_m": float(cell),
        }
        return ground, structure, stats

    # ------------------------------------------------------------------
    # Track display continuity
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_bbox(box):
        if box is None or len(box) != 4:
            return None
        try:
            values = [float(v) for v in box]
        except Exception:
            return None
        if not np.isfinite(values).all():
            return None
        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            return None
        return values


    @staticmethod
    def _bbox_iou(a, b):
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 1e-9 else 0.0

    @staticmethod
    def _bbox_center_distance(a, b):
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        acx, acy = 0.5 * (ax1 + ax2), 0.5 * (ay1 + ay2)
        bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
        return math.hypot(acx - bcx, acy - bcy)

    @staticmethod
    def _draw_dashed_rect(image, box, color, thickness=1, dash=8, gap=5):
        if cv2 is None:
            return
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        for x in range(x1, x2, dash + gap):
            cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
            cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
        for y in range(y1, y2, dash + gap):
            cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
            cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)

    def _update_display_candidate_cache(self, result):
        frame = int(result.get("frame", 0) or 0)
        current = []
        for obj in result.get("multimodal_objects", []) or []:
            box = self._valid_bbox(obj.get("projected_bbox"))
            if box is None:
                continue
            current.append({
                "bbox": box,
                "class": obj.get("class", obj.get("label", "candidate")),
            })

        cache_ids = list(self._display_candidate_cache.keys())
        unmatched_cache = set(cache_ids)
        matched_current = set()

        # Greedy association for display only.  Strong IoU wins; otherwise use
        # a conservative centre-distance gate scaled by box size.
        pairs = []
        for ci, cand in enumerate(current):
            cb = cand["bbox"]
            cw = max(1.0, cb[2] - cb[0])
            ch = max(1.0, cb[3] - cb[1])
            for cache_id in cache_ids:
                item = self._display_candidate_cache[cache_id]
                age = frame - int(item.get("last_frame", frame))
                if age < 0 or age > self.display_candidate_hold_frames + 1:
                    continue
                pb = item.get("bbox")
                if pb is None:
                    continue
                iou = self._bbox_iou(cb, pb)
                dist = self._bbox_center_distance(cb, pb)
                pw = max(1.0, pb[2] - pb[0])
                ph = max(1.0, pb[3] - pb[1])
                gate = max(28.0, 1.25 * max(cw, ch, pw, ph))
                if iou >= 0.08 or dist <= gate:
                    score = 4.0 * iou - dist / max(gate, 1.0)
                    pairs.append((score, ci, cache_id))

        for _score, ci, cache_id in sorted(pairs, reverse=True):
            if ci in matched_current or cache_id not in unmatched_cache:
                continue
            cand = current[ci]
            item = self._display_candidate_cache[cache_id]
            item.update({
                "bbox": cand["bbox"],
                "class": cand["class"],
                "last_frame": frame,
                "age": 0,
                "status": "ACTIVE",
            })
            matched_current.add(ci)
            unmatched_cache.remove(cache_id)

        for ci, cand in enumerate(current):
            if ci in matched_current:
                continue
            cache_id = self._next_display_candidate_id
            self._next_display_candidate_id += 1
            self._display_candidate_cache[cache_id] = {
                "display_candidate_id": cache_id,
                "bbox": cand["bbox"],
                "class": cand["class"],
                "last_frame": frame,
                "age": 0,
                "status": "ACTIVE",
            }

        for cache_id in list(self._display_candidate_cache):
            item = self._display_candidate_cache[cache_id]
            age = frame - int(item.get("last_frame", frame))
            if age <= 0:
                item["status"] = "ACTIVE"
                item["age"] = 0
            elif age <= self.display_candidate_hold_frames:
                item["status"] = "HELD"
                item["age"] = int(age)
            else:
                self._display_candidate_cache.pop(cache_id, None)

        return list(self._display_candidate_cache.values())

    def _update_display_track_cache(self, result):
        frame = int(result.get("frame", 0) or 0)
        active = result.get("tracking", {}).get("active_tracks", []) or []
        active_ids = set()

        for track in active:
            try:
                tid = int(track.get("track_id"))
            except Exception:
                continue
            box = self._valid_bbox(track.get("bbox"))
            if box is None:
                continue
            active_ids.add(tid)
            self._display_track_cache[tid] = {
                "track_id": tid,
                "class": track.get("class", "object"),
                "bbox": box,
                "last_frame": frame,
                "speed": float(track.get("speed", 0.0) or 0.0),
                "motion_state": track.get("motion_state", "-"),
                "position": track.get("position"),
                "status": "ACTIVE",
            }

        for tid in list(self._display_track_cache):
            item = self._display_track_cache[tid]
            age = frame - int(item.get("last_frame", frame))
            if tid in active_ids:
                item["status"] = "ACTIVE"
                continue
            if age <= self.display_track_hold_frames:
                item["status"] = "PREDICTED"
                item["age"] = int(age)
            else:
                self._display_track_cache.pop(tid, None)

        return list(self._display_track_cache.values())

    def _local_tracks(self, result, pose, display_tracks):
        current_by_id = {}
        for track in result.get("tracking", {}).get("active_tracks", []) or []:
            try:
                current_by_id[int(track.get("track_id"))] = track
            except Exception:
                pass

        output = []
        c = math.cos(float(pose[2]))
        s = math.sin(float(pose[2]))
        for display in display_tracks:
            tid = display.get("track_id")
            track = current_by_id.get(tid)
            position = None
            if track is not None:
                position = track.get("position")
            if position is None:
                position = display.get("position")
            p = np.asarray(position if position is not None else [], dtype=np.float64).reshape(-1)
            if p.size < 2:
                continue
            dx = float(p[0]) - float(pose[3])
            dy = float(p[1]) - float(pose[4])
            forward = c * dx + s * dy
            lateral = -s * dx + c * dy
            if not np.isfinite(forward) or not np.isfinite(lateral):
                continue
            if forward * forward + lateral * lateral > self.meters * self.meters:
                continue
            output.append(
                {
                    "track_id": tid,
                    "class": display.get("class", "object"),
                    "forward": float(forward),
                    "lateral": float(lateral),
                    "speed": float(display.get("speed", 0.0) or 0.0),
                    "motion_state": display.get("motion_state", "-"),
                    "status": display.get("status", "ACTIVE"),
                    "age": int(display.get("age", 0) or 0),
                }
            )
        return output

    # ------------------------------------------------------------------
    # Snapshot streams for the dashboard
    # ------------------------------------------------------------------

    def _write_snapshot(self, result, pose, ground, structure, local_tracks, stats):
        combined = np.vstack([ground, structure]) if len(ground) + len(structure) else np.empty((0, 3))
        snapshot = {
            "frame": int(result.get("frame", 0) or 0),
            "carla_frame": int(result.get("carla_frame", 0) or 0),
            "timestamp": float(result.get("timestamp", 0.0) or 0.0),
            "meters": float(self.meters),
            "pose": [float(v) for v in pose[:6]],
            "map_voxels": int((result.get("mapping_4d", {}) or {}).get("total_static_voxels", 0) or 0),
            "map_points_local": combined.astype(np.float32).tolist(),
            "ground_points_local": ground.astype(np.float32).tolist(),
            "structure_points_local": structure.astype(np.float32).tolist(),
            "map_render_stats": stats,
            "tracks_local": local_tracks,
        }
        temp_path = self.snapshot_path.with_suffix(self.snapshot_path.suffix + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, separators=(",", ":"))
            os.replace(str(temp_path), str(self.snapshot_path))
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

    def _write_camera_preview(self, camera):
        if cv2 is None or camera is None or np.asarray(camera).size == 0:
            return
        image = np.asarray(camera)
        h, w = image.shape[:2]
        if w > self.preview_width:
            scale = self.preview_width / float(w)
            image = cv2.resize(
                image,
                (self.preview_width, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return
        temp_path = self.camera_snapshot_path.with_suffix(self.camera_snapshot_path.suffix + ".tmp")
        try:
            with temp_path.open("wb") as handle:
                handle.write(encoded.tobytes())
            os.replace(str(temp_path), str(self.camera_snapshot_path))
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Camera overlay
    # ------------------------------------------------------------------

    def _draw_camera(self, image, result, pose, vel, display_tracks, display_candidates):
        camera = np.asarray(image).copy()
        if cv2 is None:
            return camera

        # Fused camera+LiDAR candidates.  ACTIVE is a real current-frame
        # candidate.  HELD is display-only continuity for a very short miss and
        # is deliberately dashed/dim so it cannot be mistaken for current data.
        for item in display_candidates:
            box = self._valid_bbox(item.get("bbox"))
            if box is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            if item.get("status") == "ACTIVE":
                color = (255, 255, 0)
                cv2.rectangle(camera, (x1, y1), (x2, y2), color, 2)
                label = "CAND"
            else:
                color = (150, 150, 0)
                self._draw_dashed_rect(camera, box, color, thickness=1)
                label = f"CAND HOLD +{int(item.get('age', 1))}f"
            cls = str(item.get("class", "candidate"))
            if cls and cls not in ("candidate", "object", "None"):
                label += f" {cls}"
            cv2.putText(
                camera, label, (x1, min(camera.shape[0] - 8, y2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA
            )

        # Confirmed track = solid green.  A one-to-three frame display bridge
        # is amber and explicitly labelled PRED so a missed detector frame does
        # not look like a tracker ID was deleted.
        for item in display_tracks:
            box = self._valid_bbox(item.get("bbox"))
            if box is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            active = item.get("status") == "ACTIVE"
            color = (0, 220, 0) if active else (0, 175, 255)
            thickness = 2 if active else 1
            cv2.rectangle(camera, (x1, y1), (x2, y2), color, thickness)
            label = f"ID {item.get('track_id', '?')} {item.get('class', 'object')}"
            if not active:
                label += f" PRED +{int(item.get('age', 1))}f"
            cv2.putText(
                camera,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        perf = result.get("performance", {}) or {}
        text = [
            "LVIMOT LIVE - RGB/LIDAR/IMU ESTIMATE",
            f"frame {result.get('frame', 0)}  speed {np.linalg.norm(vel):.2f} m/s",
            f"pose x={pose[3]:.2f} y={pose[4]:.2f} yaw={np.rad2deg(pose[2]):.1f} deg",
            f"published tracks {len(result.get('tracking', {}).get('active_tracks', []))}  map {(result.get('mapping_4d', {}) or {}).get('total_static_voxels', 0)} voxels",
            f"frame {perf.get('total_ms', 0):.0f} ms  RSS {perf.get('rss_mb', 0):.0f} MB",
            f"cyan=fused candidate (hold {self.display_candidate_hold_frames}f dashed) | green=published track | amber=track hold",
            "GT is evaluation only",
        ]
        for i, line in enumerate(text):
            cv2.putText(
                camera,
                line,
                (18, 28 + 23 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return camera

    # ------------------------------------------------------------------
    # BEV rendering
    # ------------------------------------------------------------------

    def _draw_bev(self, ground, structure, local_tracks):
        if cv2 is None:
            return np.empty((0, 0, 3), dtype=np.uint8)

        bev = np.zeros((self.bev_size, self.bev_size, 3), dtype=np.uint8)
        center = self.bev_size // 2
        scale = self.bev_size / (2.0 * self.meters)

        for radius_m in (10, 20, 30, 40, 50, 60):
            radius_px = int(round(radius_m * scale))
            if radius_px <= center:
                cv2.circle(bev, (center, center), radius_px, (30, 48, 65), 1)
        cv2.line(bev, (0, center), (self.bev_size - 1, center), (30, 48, 65), 1)
        cv2.line(bev, (center, 0), (center, self.bev_size - 1), (30, 48, 65), 1)

        def draw_cells(points, color, radius):
            pts = np.asarray(points, dtype=np.float64)
            if len(pts) == 0:
                return
            px = np.rint(center + pts[:, 1] * scale).astype(np.int32)
            py = np.rint(center - pts[:, 0] * scale).astype(np.int32)
            valid = (px >= 0) & (px < self.bev_size) & (py >= 0) & (py < self.bev_size)
            for x, y in zip(px[valid], py[valid]):
                cv2.circle(bev, (int(x), int(y)), radius, color, -1)

        # Ground/road is intentionally dim; structures/obstacles are bright.
        draw_cells(ground, (75, 88, 100), 1)
        draw_cells(structure, (180, 190, 200), 1)

        triangle = np.asarray(
            [[center, center - 10], [center - 7, center + 8], [center + 7, center + 8]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(bev, triangle, (0, 220, 160))

        for track in local_tracks:
            x = int(round(center + float(track["lateral"]) * scale))
            y = int(round(center - float(track["forward"]) * scale))
            if 0 <= x < self.bev_size and 0 <= y < self.bev_size:
                predicted = track.get("status") != "ACTIVE"
                color = (0, 175, 255) if predicted else (0, 165, 255)
                cv2.circle(bev, (x, y), 6 if not predicted else 5, color, -1)
                suffix = " P" if predicted else ""
                cv2.putText(
                    bev,
                    f"ID {track.get('track_id', '')}{suffix}",
                    (x + 8, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )

        cv2.putText(bev, "CLEAN ROLLING 4D LOCAL MAP", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(
            bev,
            "dark=ground/road | light=structures | orange=tracks",
            (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 190, 200),
            1,
        )
        return bev

    # ------------------------------------------------------------------
    # Public renderer
    # ------------------------------------------------------------------

    def render(self, image, result, map_points):
        pose = self._pose(result)
        vel = self._velocity(result)

        raw_local = self._bounded_local_map(map_points, pose)
        ground, structure, map_stats = self._clean_map_cells(raw_local)
        display_tracks = self._update_display_track_cache(result)
        display_candidates = self._update_display_candidate_cache(result)
        local_tracks = self._local_tracks(result, pose, display_tracks)

        self._write_snapshot(result, pose, ground, structure, local_tracks, map_stats)

        if cv2 is None:
            return image

        camera = self._draw_camera(image, result, pose, vel, display_tracks, display_candidates)
        self._write_camera_preview(camera)
        bev = self._draw_bev(ground, structure, local_tracks)

        h = max(camera.shape[0], bev.shape[0])
        cam_scale = h / camera.shape[0]
        camera = cv2.resize(camera, (int(camera.shape[1] * cam_scale), h))
        if bev.shape[0] != h:
            bev = cv2.resize(bev, (h, h))
        return np.hstack([camera, bev])
