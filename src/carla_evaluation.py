import numpy as np
from typing import Dict, List
from scipy.optimize import linear_sum_assignment


class CarlaEvaluator:
    def __init__(
        self,
        distance_threshold: float = 4.0,
        iou_threshold: float = 0.25,
        vertical_threshold: float = 2.5,
    ):
        # Object proposals are estimated from partial LiDAR surfaces rather
        # than full GT boxes, so association uses BEV centre distance plus a
        # separate vertical gate.  This is evaluation only.
        self.distance_threshold = float(distance_threshold)
        self.iou_threshold = float(iou_threshold)
        self.vertical_threshold = float(vertical_threshold)

    @staticmethod
    def _position_from_gt(gt: Dict) -> np.ndarray:
        loc = gt.get("location", [0.0, 0.0, 0.0])
        if isinstance(loc, dict):
            return np.array([loc["x"], loc["y"], loc["z"]], dtype=np.float64)
        return np.asarray(loc, dtype=np.float64)

    @staticmethod
    def _velocity_from_gt(gt: Dict):
        vel = gt.get("velocity")
        if vel is None:
            return None
        if isinstance(vel, dict):
            return np.array([vel["x"], vel["y"], vel["z"]], dtype=np.float64)
        return np.asarray(vel, dtype=np.float64)

    @staticmethod
    def _bbox_iou(a, b):
        if a is None or b is None:
            return None
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if len(a) != 4 or len(b) != 4 or not np.isfinite(a).all() or not np.isfinite(b).all():
            return None
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    def evaluate_tracking_sequence(
        self,
        tracker_tracks_per_frame: List[List[Dict]],
        ground_truth_per_frame: List[List[Dict]],
    ) -> Dict:
        total_gt = 0
        total_hyp = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0
        position_errors = []
        bev_errors = []
        velocity_errors = []
        gt_to_track_map: Dict[int, int] = {}
        gt_appearances: Dict[int, int] = {}
        gt_tracked_count: Dict[int, int] = {}
        nearest_bev_distances = []
        frame_diagnostics = []
        camera_iou_matches = 0
        bev_matches = 0
        matched_ious = []

        num_frames = min(len(tracker_tracks_per_frame), len(ground_truth_per_frame))

        for f_idx in range(num_frames):
            tracks = tracker_tracks_per_frame[f_idx]
            gts = ground_truth_per_frame[f_idx]
            total_gt += len(gts)
            total_hyp += len(tracks)

            for gt in gts:
                gid = int(gt.get("actor_id", gt.get("id", 0)))
                gt_appearances[gid] = gt_appearances.get(gid, 0) + 1

            if len(tracks) == 0:
                total_fn += len(gts)
                frame_diagnostics.append({
                    "frame_index": f_idx,
                    "gt": len(gts),
                    "outputs": 0,
                    "matches": 0,
                    "fp": 0,
                    "fn": len(gts),
                    "nearest_bev_distance": None,
                })
                continue
            if len(gts) == 0:
                total_fp += len(tracks)
                frame_diagnostics.append({
                    "frame_index": f_idx,
                    "gt": 0,
                    "outputs": len(tracks),
                    "matches": 0,
                    "fp": len(tracks),
                    "fn": 0,
                    "nearest_bev_distance": None,
                })
                continue

            cost_matrix = np.full((len(tracks), len(gts)), np.inf, dtype=np.float64)
            raw_bev = np.full((len(tracks), len(gts)), np.inf, dtype=np.float64)
            pair_iou = np.full((len(tracks), len(gts)), np.nan, dtype=np.float64)
            pair_mode = np.full((len(tracks), len(gts)), "", dtype=object)
            for i, trk in enumerate(tracks):
                p_trk = np.asarray(trk["position"], dtype=np.float64)
                for j, gt in enumerate(gts):
                    p_gt = self._position_from_gt(gt)
                    bev_dist = float(np.linalg.norm(p_trk[:2] - p_gt[:2]))
                    dz = abs(float(p_trk[2] - p_gt[2]))
                    raw_bev[i, j] = bev_dist

                    iou = self._bbox_iou(trk.get("bbox"), gt.get("bbox"))
                    if iou is not None:
                        pair_iou[i, j] = iou
                    # Camera+LiDAR tracking has a directly observed RGB box.
                    # Use image IoU as the primary identity association when
                    # both boxes exist; this is independent of any global/world
                    # coordinate convention.  BEV remains the fallback and is
                    # still reported for 3-D quality diagnostics.
                    if iou is not None and iou >= self.iou_threshold:
                        cost_matrix[i, j] = (1.0 - iou) + 0.001 * min(bev_dist, 100.0)
                        pair_mode[i, j] = "camera_iou"
                    elif bev_dist <= self.distance_threshold and dz <= self.vertical_threshold:
                        cost_matrix[i, j] = 2.0 + bev_dist / max(self.distance_threshold, 1e-6)
                        pair_mode[i, j] = "bev"

            finite_raw = raw_bev[np.isfinite(raw_bev)]
            nearest = float(np.min(finite_raw)) if finite_raw.size else None
            if nearest is not None:
                nearest_bev_distances.append(nearest)

            row_ind, col_ind = linear_sum_assignment(
                np.where(np.isfinite(cost_matrix), cost_matrix, 1e9)
            )

            matched_tracks = set()
            matched_gts = set()
            frame_position_errors = []

            for r, c in zip(row_ind, col_ind):
                if not np.isfinite(cost_matrix[r, c]):
                    continue
                matched_tracks.add(int(r))
                matched_gts.add(int(c))
                trk = tracks[r]
                gt = gts[c]
                gid = int(gt.get("actor_id", gt.get("id", 0)))
                tid = int(trk["track_id"])
                gt_tracked_count[gid] = gt_tracked_count.get(gid, 0) + 1
                mode = str(pair_mode[r, c])
                if mode == "camera_iou":
                    camera_iou_matches += 1
                    if np.isfinite(pair_iou[r, c]):
                        matched_ious.append(float(pair_iou[r, c]))
                elif mode == "bev":
                    bev_matches += 1

                if gid in gt_to_track_map and gt_to_track_map[gid] != tid:
                    total_idsw += 1
                gt_to_track_map[gid] = tid

                p_gt = self._position_from_gt(gt)
                p_trk = np.asarray(trk["position"], dtype=np.float64)
                err_3d = float(np.linalg.norm(p_trk - p_gt))
                err_bev = float(cost_matrix[r, c])
                position_errors.append(err_3d)
                bev_errors.append(err_bev)
                frame_position_errors.append(err_3d)

                v_gt = self._velocity_from_gt(gt)
                if v_gt is not None and "velocity" in trk:
                    v_trk = np.asarray(trk["velocity"], dtype=np.float64)
                    velocity_errors.append(float(np.linalg.norm(v_trk - v_gt)))

            frame_fp = len(tracks) - len(matched_tracks)
            frame_fn = len(gts) - len(matched_gts)
            total_fp += frame_fp
            total_fn += frame_fn
            frame_diagnostics.append({
                "frame_index": f_idx,
                "gt": len(gts),
                "outputs": len(tracks),
                "matches": len(matched_tracks),
                "fp": frame_fp,
                "fn": frame_fn,
                "nearest_bev_distance": nearest,
                "mean_matched_position_error": (
                    float(np.mean(frame_position_errors)) if frame_position_errors else None
                ),
            })

        mota = (
            1.0 - (total_fn + total_fp + total_idsw) / float(total_gt)
            if total_gt > 0
            else None
        )
        motp = float(np.mean(bev_errors)) if bev_errors else None
        pos_rmse = float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else None
        vel_rmse = float(np.sqrt(np.mean(np.square(velocity_errors)))) if velocity_errors else None

        mt = 0
        ml = 0
        pt = 0
        for gid, total_frames in gt_appearances.items():
            ratio = gt_tracked_count.get(gid, 0) / float(total_frames)
            if ratio >= 0.8:
                mt += 1
            elif ratio <= 0.2:
                ml += 1
            else:
                pt += 1

        num_unique_gt = max(len(gt_appearances), 1)
        return {
            "num_frames": num_frames,
            "total_gt_objects": total_gt,
            "total_tracker_outputs": total_hyp,
            "total_matches": len(position_errors),
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "id_switches": total_idsw,
            "mota": (float(mota) if mota is not None else None),
            "evaluation_valid": bool(total_gt > 0),
            "invalid_reason": (None if total_gt > 0 else "No eligible ground-truth objects in evaluation region"),
            "motp": motp,
            "position_rmse": pos_rmse,
            "velocity_rmse": vel_rmse,
            "matching_distance": "camera_IoU_primary_with_BEV_fallback",
            "evaluation_frame": "current_lidar_sensor_frame",
            "association_mode": "camera_IoU_primary_with_BEV_fallback",
            "camera_iou_matches": int(camera_iou_matches),
            "bev_matches": int(bev_matches),
            "mean_matched_iou": (float(np.mean(matched_ious)) if matched_ious else None),
            "distance_threshold_m": self.distance_threshold,
            "vertical_threshold_m": self.vertical_threshold,
            "mean_nearest_bev_distance": (
                float(np.mean(nearest_bev_distances)) if nearest_bev_distances else None
            ),
            "unique_gt_actors": len(gt_appearances),
            "mostly_tracked_count": mt,
            "mostly_tracked_ratio": float(mt / num_unique_gt),
            "mostly_lost_count": ml,
            "mostly_lost_ratio": float(ml / num_unique_gt),
            "partially_tracked_count": pt,
            "frame_diagnostics": frame_diagnostics,
        }

    def evaluate_localization(self, estimated_poses: List[np.ndarray], ground_truth_poses: List[np.ndarray]) -> Dict:
        est = np.asarray(estimated_poses, dtype=np.float64)
        gt = np.asarray(ground_truth_poses, dtype=np.float64)
        if est.ndim != 2 or gt.ndim != 2 or len(est) == 0 or len(gt) == 0:
            return {
                "num_evaluated_frames": 0,
                "ate_rmse": 0.0,
                "raw_ate_rmse": 0.0,
                "aligned_ate_rmse": 0.0,
                "ate_mean": 0.0,
                "ate_std": 0.0,
                "ate_max": 0.0,
                "rpe_rmse": 0.0,
                "rpe_mean": 0.0,
            }

        est_pos = est[:, 3:] if est.shape[1] == 6 else est[:, :3]
        gt_pos = gt[:, 3:] if gt.shape[1] == 6 else gt[:, :3]
        n = min(len(est_pos), len(gt_pos))
        est_pos = est_pos[:n]
        gt_pos = gt_pos[:n]

        raw_errors = np.linalg.norm(est_pos - gt_pos, axis=1)
        raw_rmse = float(np.sqrt(np.mean(np.square(raw_errors))))

        offset = gt_pos[0] - est_pos[0]
        aligned_est = est_pos + offset
        aligned_errors = np.linalg.norm(aligned_est - gt_pos, axis=1)
        aligned_rmse = float(np.sqrt(np.mean(np.square(aligned_errors))))

        rpe_errors = [
            np.linalg.norm((est_pos[i] - est_pos[i - 1]) - (gt_pos[i] - gt_pos[i - 1]))
            for i in range(1, n)
        ]
        rpe_rmse = float(np.sqrt(np.mean(np.square(rpe_errors)))) if rpe_errors else 0.0
        rpe_mean = float(np.mean(rpe_errors)) if rpe_errors else 0.0

        return {
            "num_evaluated_frames": n,
            "ate_rmse": raw_rmse,
            "raw_ate_rmse": raw_rmse,
            "aligned_ate_rmse": aligned_rmse,
            "ate_mean": float(np.mean(raw_errors)),
            "ate_std": float(np.std(raw_errors)),
            "ate_max": float(np.max(raw_errors)),
            "aligned_ate_mean": float(np.mean(aligned_errors)),
            "aligned_ate_max": float(np.max(aligned_errors)),
            "rpe_rmse": rpe_rmse,
            "rpe_mean": rpe_mean,
        }
