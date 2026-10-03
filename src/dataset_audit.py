import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class AuditThresholds:
    pose_step_m: float = 0.02
    pose_step_deg: float = 0.20
    frozen_run_min: int = 3
    mostly_stationary_fraction: float = 0.10
    mostly_moving_fraction: float = 0.80

    # Backward-compatible V16 near-pixel thresholds.  V17 no longer uses
    # them to infer ego motion, but keeps the diagnostic and public API.
    rgb_near_mean_abs_diff: float = 0.25
    rgb_near_changed_fraction: float = 0.002

    # RGB motion analysis.  These thresholds are deliberately based on
    # dominant/global image motion rather than raw pixel equality so a parked
    # ego vehicle with moving traffic is not mistaken for a moving camera.
    rgb_analysis_width: int = 320
    rgb_global_motion_px: float = 0.60
    rgb_global_rotation_deg: float = 0.18
    rgb_dynamic_residual_px: float = 0.70
    rgb_dynamic_feature_fraction: float = 0.015
    rgb_dense_dynamic_residual_px: float = 0.35
    rgb_dense_dynamic_fraction: float = 0.003
    rgb_dense_dynamic_p90_px: float = 0.45
    rgb_visual_freeze_ssim: float = 0.9995
    rgb_visual_freeze_phash_hamming: int = 2
    rgb_visual_freeze_motion_px: float = 0.15
    rgb_visual_freeze_dynamic_fraction: float = 0.005

    # LiDAR occupancy consistency.  CARLA ray-cast scans are highly repeatable
    # for a stationary sensor; moving traffic changes only a small portion of
    # the scene whereas ego motion changes a much larger portion.
    lidar_voxel_size_m: float = 0.50
    lidar_sample_stride: int = 8
    lidar_global_still_overlap: float = 0.90  # compatibility only
    lidar_scene_change_fraction: float = 0.025
    lidar_global_nn_median_m: float = 0.12
    lidar_global_nn_q75_m: float = 0.20
    lidar_scene_tail_distance_m: float = 0.25
    lidar_scene_tail_fraction: float = 0.02
    lidar_audit_max_points: int = 3000

    # IMU is supporting evidence only.  Constant-velocity motion is expected
    # to be detected by RGB/LiDAR instead.
    imu_gyro_moving_rad_s: float = 0.02
    imu_accel_change_m_s2: float = 0.25

    # Ground-truth object labels are used only to audit whether a sequence
    # actually contains moving traffic; they are never estimator input.
    object_motion_m: float = 0.12
    object_velocity_m_s: float = 0.50
    object_motion_min_consecutive: int = 3


def _file_digest(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.blake2b(digest_size=12)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _json_load(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _vec3(value) -> np.ndarray:
    if isinstance(value, dict):
        return np.array(
            [
                float(value.get("x", 0.0)),
                float(value.get("y", 0.0)),
                float(value.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return np.zeros(3, dtype=np.float64)
    return arr[:3]


def _pose_vectors(pose: Dict) -> Tuple[np.ndarray, np.ndarray]:
    p = _vec3(pose.get("location", {}))
    r = pose.get("rotation", {})
    euler_deg = np.array(
        [
            float(r.get("roll", 0.0)),
            float(r.get("pitch", 0.0)),
            float(r.get("yaw", 0.0)),
        ],
        dtype=np.float64,
    )
    return p, euler_deg


def _wrapped_angle_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    d = (np.asarray(b) - np.asarray(a) + 180.0) % 360.0 - 180.0
    return float(np.linalg.norm(d))


def _runs_from_equal_flags(
    equal_flags: Sequence[bool],
    start_frame: int = 0,
    min_len: int = 2,
) -> List[Tuple[int, int]]:
    """Convert pairwise True transition flags into inclusive frame runs."""
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(equal_flags)
    while i < n:
        if not equal_flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and equal_flags[j + 1]:
            j += 1
        first = start_frame + i
        last = start_frame + j + 1
        if (last - first + 1) >= min_len:
            runs.append((first, last))
        i = j + 1
    return runs


def _ranges_from_frame_flags(flags: Sequence[bool], min_len: int = 1) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        if (j - i + 1) >= min_len:
            runs.append((i, j))
        i = j + 1
    return runs


def _format_runs(runs: Sequence[Tuple[int, int]]) -> str:
    if not runs:
        return "none"
    return ", ".join(f"{a}..{b} ({b-a+1})" for a, b in runs)


def _contiguous_good_ranges(
    num_frames: int,
    bad_runs: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    bad = np.zeros(num_frames, dtype=bool)
    for a, b in bad_runs:
        a = max(0, int(a))
        b = min(num_frames - 1, int(b))
        if b >= a:
            bad[a : b + 1] = True
    return _ranges_from_frame_flags((~bad).tolist())


def _existing_frame_files(directory: str, suffix: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(suffix.lower())
    )


def _timestamps(path: str) -> List[float]:
    values: List[float] = []
    if not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            numeric: List[float] = []
            for part in line.split():
                try:
                    numeric.append(float(part))
                except ValueError:
                    continue
            if not numeric:
                values.append(float("nan"))
            elif len(numeric) == 1:
                values.append(numeric[0])
            elif len(numeric) >= 3:
                values.append(numeric[2])
            else:
                values.append(numeric[-1])
    return values


def _decode_image(path: str) -> Optional[np.ndarray]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return np.ascontiguousarray(image)


def _analysis_gray(image: np.ndarray, width: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if width > 0 and gray.shape[1] > width:
        scale = float(width) / float(gray.shape[1])
        h = max(1, int(round(gray.shape[0] * scale)))
        gray = cv2.resize(gray, (int(width), h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(gray)


def _phash_bits(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:8, :8].reshape(-1)
    # Exclude the DC component from the median, but retain its bit position so
    # all hashes have the same 64-bit layout.
    median = float(np.median(low[1:])) if low.size > 1 else 0.0
    return (low > median).astype(np.uint8)


def _phash_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(a, dtype=np.uint8) != np.asarray(b, dtype=np.uint8)))


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    x = a.astype(np.float32)
    y = b.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = cv2.GaussianBlur(x, (7, 7), 1.5)
    mu_y = cv2.GaussianBlur(y, (7, 7), 1.5)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = cv2.GaussianBlur(x * x, (7, 7), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (7, 7), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (7, 7), 1.5) - mu_xy
    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    score = np.divide(num, den, out=np.ones_like(num), where=np.abs(den) > 1e-12)
    return float(np.clip(np.mean(score), -1.0, 1.0))


def _rgb_motion_pair(prev_gray: np.ndarray, curr_gray: np.ndarray, thresholds: AuditThresholds) -> Dict:
    phash_prev = _phash_bits(prev_gray)
    phash_curr = _phash_bits(curr_gray)
    phash_hamming = _phash_distance(phash_prev, phash_curr)
    ssim = _ssim_gray(prev_gray, curr_gray)

    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=450,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
    )
    feature_count = 0
    tracked_count = 0
    global_motion_px = 0.0
    global_translation_px = 0.0
    global_rotation_deg = 0.0
    dynamic_fraction = 0.0
    median_flow_px = 0.0
    inlier_fraction = 0.0

    if p0 is not None and len(p0) >= 6:
        feature_count = int(len(p0))
        p1, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
        )
        if p1 is not None and status is not None:
            good = status.reshape(-1).astype(bool)
            a = p0.reshape(-1, 2)[good]
            b = p1.reshape(-1, 2)[good]
            tracked_count = int(len(a))
            if tracked_count:
                flow = np.linalg.norm(b - a, axis=1)
                median_flow_px = float(np.median(flow))
            if tracked_count >= 6:
                affine, inliers = cv2.estimateAffinePartial2D(
                    a,
                    b,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=1.25,
                    maxIters=500,
                    confidence=0.995,
                    refineIters=5,
                )
                if affine is not None:
                    tx = float(affine[0, 2])
                    ty = float(affine[1, 2])
                    global_translation_px = float(math.hypot(tx, ty))
                    global_rotation_deg = float(math.degrees(math.atan2(affine[1, 0], affine[0, 0])))
                    scale = float(math.hypot(affine[0, 0], affine[1, 0]))
                    diag = float(math.hypot(prev_gray.shape[1], prev_gray.shape[0]))
                    rotation_px = abs(math.radians(global_rotation_deg)) * 0.5 * diag
                    scale_px = abs(scale - 1.0) * 0.5 * diag
                    global_motion_px = global_translation_px + rotation_px + scale_px
                    pred = cv2.transform(a.reshape(-1, 1, 2), affine).reshape(-1, 2)
                    residual = np.linalg.norm(b - pred, axis=1)
                    dynamic_fraction = float(
                        np.mean(residual > float(thresholds.rgb_dynamic_residual_px))
                    )
                    if inliers is not None and len(inliers):
                        inlier_fraction = float(np.mean(inliers.reshape(-1) != 0))
                else:
                    global_motion_px = median_flow_px
                    global_translation_px = median_flow_px
                    dynamic_fraction = float(
                        np.mean(flow > float(thresholds.rgb_dynamic_residual_px))
                    )

    # Dense residual flow catches small moving vehicles that sparse corner
    # tracks can miss.  Remove the dominant affine camera motion first so
    # parked ego + moving traffic remains scene motion, not ego motion.
    dense = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.1, flags=0,
    )
    h, w = prev_gray.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if tracked_count >= 6 and 'affine' in locals() and affine is not None:
        pred_x = affine[0, 0] * xx + affine[0, 1] * yy + affine[0, 2]
        pred_y = affine[1, 0] * xx + affine[1, 1] * yy + affine[1, 2]
        global_fx = pred_x - xx
        global_fy = pred_y - yy
    else:
        med = np.median(dense.reshape(-1, 2), axis=0)
        global_fx = np.full((h, w), float(med[0]), dtype=np.float32)
        global_fy = np.full((h, w), float(med[1]), dtype=np.float32)
    residual_dense = np.sqrt(
        (dense[..., 0] - global_fx) ** 2 + (dense[..., 1] - global_fy) ** 2
    )
    dense_dynamic_fraction = float(
        np.mean(residual_dense > float(thresholds.rgb_dense_dynamic_residual_px))
    )
    dense_dynamic_p90_px = float(np.percentile(residual_dense, 90.0))

    # Uniform/feature-poor frames still need a reliable static/frozen decision.
    if tracked_count < 6:
        global_motion_px = median_flow_px
        global_translation_px = median_flow_px

    ego_moving = bool(
        global_motion_px > float(thresholds.rgb_global_motion_px)
        or abs(global_rotation_deg) > float(thresholds.rgb_global_rotation_deg)
    )
    scene_dynamic = bool(
        not ego_moving
        and (
            dynamic_fraction >= float(thresholds.rgb_dynamic_feature_fraction)
            or (
                dense_dynamic_fraction >= float(thresholds.rgb_dense_dynamic_fraction)
                and dense_dynamic_p90_px >= float(thresholds.rgb_dense_dynamic_p90_px)
            )
        )
    )
    visual_freeze = bool(
        ssim >= float(thresholds.rgb_visual_freeze_ssim)
        and phash_hamming <= int(thresholds.rgb_visual_freeze_phash_hamming)
        and global_motion_px <= float(thresholds.rgb_visual_freeze_motion_px)
        and dynamic_fraction <= float(thresholds.rgb_visual_freeze_dynamic_fraction)
    )

    return {
        "phash_hamming": phash_hamming,
        "ssim": ssim,
        "feature_count": feature_count,
        "tracked_count": tracked_count,
        "global_motion_px": global_motion_px,
        "global_translation_px": global_translation_px,
        "global_rotation_deg": global_rotation_deg,
        "median_flow_px": median_flow_px,
        "dynamic_fraction": dynamic_fraction,
        "dense_dynamic_fraction": dense_dynamic_fraction,
        "dense_dynamic_p90_px": dense_dynamic_p90_px,
        "inlier_fraction": inlier_fraction,
        "ego_moving": ego_moving,
        "scene_dynamic": scene_dynamic,
        "visual_freeze": visual_freeze,
    }


def _rgb_motion_audit(files: Sequence[str], thresholds: AuditThresholds) -> Dict:
    if not files:
        return {
            "file_equal": [],
            "pixel_equal": [],
            "near_equal": [],
            "visual_freeze": [],
            "ego_moving": [],
            "scene_dynamic": [],
            "pairs": [],
            "decode_failures": [],
        }

    file_digests = [_file_digest(p) for p in files]
    file_equal = [file_digests[i] == file_digests[i - 1] for i in range(1, len(files))]

    first = _decode_image(files[0])
    decode_failures: List[int] = []
    if first is None:
        decode_failures.append(0)
    prev_image = first
    prev_gray = _analysis_gray(first, thresholds.rgb_analysis_width) if first is not None else None

    pixel_equal: List[bool] = []
    near_equal: List[bool] = []
    visual_freeze: List[bool] = []
    ego_moving: List[bool] = []
    scene_dynamic: List[bool] = []
    pair_metrics: List[Dict] = []

    for frame_index in range(1, len(files)):
        curr_image = _decode_image(files[frame_index])
        if curr_image is None:
            decode_failures.append(frame_index)
        if (
            prev_image is None
            or curr_image is None
            or prev_image.shape != curr_image.shape
        ):
            pixel_equal.append(False)
            near_equal.append(False)
            visual_freeze.append(False)
            ego_moving.append(False)
            scene_dynamic.append(False)
            pair_metrics.append({"decode_ok": False})
            prev_image = curr_image
            prev_gray = (
                _analysis_gray(curr_image, thresholds.rgb_analysis_width)
                if curr_image is not None
                else None
            )
            continue

        exact = bool(np.array_equal(prev_image, curr_image))
        if exact:
            mad = 0.0
            changed_fraction = 0.0
        else:
            diff = np.abs(curr_image.astype(np.int16) - prev_image.astype(np.int16))
            mad = float(np.mean(diff))
            changed_fraction = float(np.mean(np.any(diff != 0, axis=2)))
        near = bool(
            exact
            or (
                mad <= float(thresholds.rgb_near_mean_abs_diff)
                and changed_fraction <= float(thresholds.rgb_near_changed_fraction)
            )
        )
        curr_gray = _analysis_gray(curr_image, thresholds.rgb_analysis_width)
        metrics = _rgb_motion_pair(prev_gray, curr_gray, thresholds)
        metrics["mean_abs_diff"] = mad
        metrics["changed_fraction"] = changed_fraction
        metrics["near_equal"] = near
        metrics["decode_ok"] = True
        metrics["pixel_equal"] = exact
        pixel_equal.append(exact)
        near_equal.append(near)
        visual_freeze.append(bool(metrics["visual_freeze"]))
        ego_moving.append(bool(metrics["ego_moving"]))
        scene_dynamic.append(bool(metrics["scene_dynamic"]))
        pair_metrics.append(metrics)
        prev_image = curr_image
        prev_gray = curr_gray

    return {
        "file_equal": file_equal,
        "pixel_equal": pixel_equal,
        "near_equal": near_equal,
        "visual_freeze": visual_freeze,
        "ego_moving": ego_moving,
        "scene_dynamic": scene_dynamic,
        "pairs": pair_metrics,
        "decode_failures": decode_failures,
    }


def _digest_flags(files: Sequence[str]) -> Tuple[List[str], List[bool]]:
    digests = [_file_digest(p) for p in files]
    equal = [digests[i] == digests[i - 1] for i in range(1, len(digests))]
    return digests, equal


def _load_lidar_xyz(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if raw.size % 4 == 0:
        return raw.reshape(-1, 4)[:, :3]
    if raw.size % 3 == 0:
        return raw.reshape(-1, 3)
    return np.empty((0, 3), dtype=np.float32)


def _voxel_centers(points: np.ndarray, voxel: float, stride: int, max_points: int) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)[:: max(1, int(stride)), :3]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64)
    q = np.floor(pts / max(float(voxel), 1e-3)).astype(np.int64)
    _uniq, idx = np.unique(q, axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    if len(pts) > int(max_points):
        take = np.linspace(0, len(pts)-1, int(max_points), dtype=np.int64)
        pts = pts[take]
    return pts


def _lidar_motion_audit(files: Sequence[str], thresholds: AuditThresholds) -> Dict:
    if not files:
        return {"exact_equal": [], "ego_moving": [], "scene_dynamic": [], "overlap": [], "median_nn_m": [], "q75_nn_m": [], "tail_fraction": []}
    _digests, exact_equal = _digest_flags(files)
    prev = _voxel_centers(
        _load_lidar_xyz(files[0]), thresholds.lidar_voxel_size_m,
        thresholds.lidar_sample_stride, thresholds.lidar_audit_max_points,
    )
    overlaps: List[Optional[float]] = []
    medians: List[Optional[float]] = []
    q75s: List[Optional[float]] = []
    tails: List[Optional[float]] = []
    ego_moving: List[bool] = []
    scene_dynamic: List[bool] = []
    for i in range(1, len(files)):
        curr = _voxel_centers(
            _load_lidar_xyz(files[i]), thresholds.lidar_voxel_size_m,
            thresholds.lidar_sample_stride, thresholds.lidar_audit_max_points,
        )
        if len(prev) < 20 or len(curr) < 20:
            overlap = median_nn = q75_nn = tail_fraction = None
            moving = dynamic = False
        else:
            tree_curr = cKDTree(curr)
            tree_prev = cKDTree(prev)
            d_pc, _ = tree_curr.query(prev, k=1, workers=-1)
            d_cp, _ = tree_prev.query(curr, k=1, workers=-1)
            d = np.concatenate([d_pc, d_cp])
            median_nn = float(np.median(d))
            q75_nn = float(np.percentile(d, 75.0))
            tail_fraction = float(np.mean(d > float(thresholds.lidar_scene_tail_distance_m)))
            # Occupancy overlap remains diagnostic only; the old overlap test
            # falsely called moving traffic global ego motion.
            qa = np.floor(prev / max(float(thresholds.lidar_voxel_size_m), 1e-3)).astype(np.int64)
            qb = np.floor(curr / max(float(thresholds.lidar_voxel_size_m), 1e-3)).astype(np.int64)
            ha = np.unique((qa[:,0]*73856093) ^ (qa[:,1]*19349663) ^ (qa[:,2]*83492791))
            hb = np.unique((qb[:,0]*73856093) ^ (qb[:,1]*19349663) ^ (qb[:,2]*83492791))
            inter = np.intersect1d(ha, hb, assume_unique=True).size
            overlap = float(inter) / float(max(1, min(len(ha), len(hb))))
            moving = bool(
                median_nn > float(thresholds.lidar_global_nn_median_m)
                and q75_nn > float(thresholds.lidar_global_nn_q75_m)
            )
            dynamic = bool(
                not moving
                and tail_fraction >= float(thresholds.lidar_scene_tail_fraction)
            )
        overlaps.append(overlap)
        medians.append(median_nn)
        q75s.append(q75_nn)
        tails.append(tail_fraction)
        ego_moving.append(bool(moving))
        scene_dynamic.append(bool(dynamic))
        prev = curr
    return {
        "exact_equal": exact_equal,
        "ego_moving": ego_moving,
        "scene_dynamic": scene_dynamic,
        "overlap": overlaps,
        "median_nn_m": medians,
        "q75_nn_m": q75s,
        "tail_fraction": tails,
    }

def _label_object_motion_audit(files: Sequence[str], thresholds: AuditThresholds) -> Dict:
    def actor_map(path: str) -> Dict[int, Dict]:
        try:
            data = _json_load(path)
        except Exception:
            return {}
        objects = data.get("objects", []) if isinstance(data, dict) else []
        output: Dict[int, Dict] = {}
        for obj in objects:
            if not isinstance(obj, dict) or "actor_id" not in obj:
                continue
            try:
                output[int(obj["actor_id"])] = obj
            except Exception:
                continue
        return output

    if not files:
        return {"moving": [], "moving_actor_counts": [], "raw_moving": []}
    prev = actor_map(files[0])
    raw_flags: List[bool] = []
    counts: List[int] = []
    for i in range(1, len(files)):
        curr = actor_map(files[i])
        moving_actors = 0
        for actor_id in set(prev).intersection(curr):
            p0 = _vec3(prev[actor_id].get("location", {}))
            p1 = _vec3(curr[actor_id].get("location", {}))
            displacement = float(np.linalg.norm(p1 - p0))
            velocity = float(np.linalg.norm(_vec3(curr[actor_id].get("velocity", {}))))
            if displacement > float(thresholds.object_motion_m) or velocity > float(thresholds.object_velocity_m_s):
                moving_actors += 1
        raw_flags.append(moving_actors > 0)
        counts.append(moving_actors)
        prev = curr

    # Reject one-frame CARLA physics jitter.  Keep only sustained motion runs.
    min_run = max(1, int(thresholds.object_motion_min_consecutive))
    persistent = [False] * len(raw_flags)
    i = 0
    while i < len(raw_flags):
        if not raw_flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(raw_flags) and raw_flags[j + 1]:
            j += 1
        if j - i + 1 >= min_run:
            for k in range(i, j + 1):
                persistent[k] = True
        i = j + 1
    return {"moving": persistent, "moving_actor_counts": counts, "raw_moving": raw_flags}

def _motion_class(moving_fraction: float, thresholds: AuditThresholds) -> str:
    if moving_fraction <= 0.0:
        return "stationary"
    if moving_fraction < float(thresholds.mostly_stationary_fraction):
        return "mostly_stationary"
    if moving_fraction >= float(thresholds.mostly_moving_fraction):
        return "moving"
    return "intermittent_motion"


def _pair_flags_to_frame_mask(flags: Sequence[bool], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for i, flag in enumerate(flags):
        if flag:
            mask[i] = True
            if i + 1 < n:
                mask[i + 1] = True
    return mask


def _consensus_pair_flags(flag_sets: Sequence[Sequence[bool]]) -> List[bool]:
    valid = [list(x) for x in flag_sets if x]
    if not valid:
        return []
    n = min(len(x) for x in valid)
    output: List[bool] = []
    for i in range(n):
        votes = sum(bool(x[i]) for x in valid)
        needed = 2 if len(valid) >= 3 else max(1, int(math.ceil(len(valid) / 2.0)))
        output.append(votes >= needed)
    return output


def audit_sequence(
    carla_root: str,
    sequence: str,
    thresholds: Optional[AuditThresholds] = None,
) -> Dict:
    thresholds = thresholds or AuditThresholds()
    seq = f"{int(sequence):04d}"
    seq_dir = os.path.join(os.path.expanduser(carla_root), "sequences", seq)
    if not os.path.isdir(seq_dir):
        return {"sequence": seq, "status": "missing", "sequence_dir": seq_dir}

    image_files = _existing_frame_files(os.path.join(seq_dir, "image_02"), ".png")
    lidar_files = _existing_frame_files(os.path.join(seq_dir, "velodyne"), ".bin")
    imu_files = _existing_frame_files(os.path.join(seq_dir, "imu"), ".json")
    pose_files = _existing_frame_files(os.path.join(seq_dir, "poses"), ".json")
    label_files = _existing_frame_files(os.path.join(seq_dir, "labels"), ".json")
    ts = _timestamps(os.path.join(seq_dir, "timestamps.txt"))

    counts = {
        "timestamps": len(ts),
        "rgb": len(image_files),
        "lidar": len(lidar_files),
        "imu": len(imu_files),
        "poses": len(pose_files),
        "labels": len(label_files),
    }
    nonzero_counts = [v for v in counts.values() if v > 0]
    n = min(nonzero_counts) if nonzero_counts else 0
    if n <= 0:
        return {"sequence": seq, "status": "empty", "counts": counts}

    image_files = image_files[:n]
    lidar_files = lidar_files[:n]
    imu_files = imu_files[:n]
    pose_files = pose_files[:n]
    label_files = label_files[:n]
    ts = ts[:n]

    rgb = _rgb_motion_audit(image_files, thresholds)
    lidar = _lidar_motion_audit(lidar_files, thresholds)
    labels = _label_object_motion_audit(label_files, thresholds)

    rgb_file_runs = _runs_from_equal_flags(rgb["file_equal"], min_len=thresholds.frozen_run_min)
    rgb_pixel_runs = _runs_from_equal_flags(rgb["pixel_equal"], min_len=thresholds.frozen_run_min)
    rgb_near_runs = _runs_from_equal_flags(rgb["near_equal"], min_len=thresholds.frozen_run_min)
    rgb_visual_freeze_runs = _runs_from_equal_flags(
        rgb["visual_freeze"], min_len=thresholds.frozen_run_min
    )
    rgb_ego_motion_ranges = _runs_from_equal_flags(rgb["ego_moving"], min_len=2)
    rgb_scene_dynamic_ranges = _runs_from_equal_flags(rgb["scene_dynamic"], min_len=2)
    lidar_runs = _runs_from_equal_flags(lidar["exact_equal"], min_len=thresholds.frozen_run_min)
    lidar_ego_motion_ranges = _runs_from_equal_flags(lidar["ego_moving"], min_len=2)
    lidar_scene_dynamic_ranges = _runs_from_equal_flags(lidar["scene_dynamic"], min_len=2)
    object_motion_ranges = _runs_from_equal_flags(labels["moving"], min_len=2)

    pose_positions: List[np.ndarray] = []
    pose_angles: List[np.ndarray] = []
    for p in pose_files:
        pos, ang = _pose_vectors(_json_load(p))
        pose_positions.append(pos)
        pose_angles.append(ang)

    translation_steps: List[float] = []
    rotation_steps: List[float] = []
    pose_moving_flags: List[bool] = []
    for i in range(1, len(pose_positions)):
        dt = float(np.linalg.norm(pose_positions[i] - pose_positions[i - 1]))
        dr = _wrapped_angle_delta_deg(pose_angles[i - 1], pose_angles[i])
        translation_steps.append(dt)
        rotation_steps.append(dr)
        pose_moving_flags.append(
            dt > float(thresholds.pose_step_m)
            or dr > float(thresholds.pose_step_deg)
        )

    pose_moving_fraction = float(np.mean(pose_moving_flags)) if pose_moving_flags else 0.0
    pose_moving_ranges = _runs_from_equal_flags(pose_moving_flags, min_len=2)
    pose_stationary_ranges = _contiguous_good_ranges(n, pose_moving_ranges)
    recorded_pose_motion = _motion_class(pose_moving_fraction, thresholds)

    imu_accel: List[np.ndarray] = []
    imu_gyro: List[np.ndarray] = []
    for p in imu_files:
        imu = _json_load(p)
        imu_accel.append(_vec3(imu.get("accelerometer", {})))
        imu_gyro.append(_vec3(imu.get("gyroscope", {})))
    accel_arr = np.asarray(imu_accel, dtype=np.float64) if imu_accel else np.empty((0, 3))
    gyro_arr = np.asarray(imu_gyro, dtype=np.float64) if imu_gyro else np.empty((0, 3))
    imu_moving_flags: List[bool] = []
    for i in range(1, len(imu_accel)):
        gyro_norm = float(np.linalg.norm(imu_gyro[i]))
        accel_change = float(np.linalg.norm(imu_accel[i] - imu_accel[i - 1]))
        imu_moving_flags.append(
            gyro_norm > float(thresholds.imu_gyro_moving_rad_s)
            or accel_change > float(thresholds.imu_accel_change_m_s2)
        )

    consensus_flags = _consensus_pair_flags(
        [pose_moving_flags, rgb["ego_moving"], lidar["ego_moving"], imu_moving_flags]
    )
    consensus_fraction = float(np.mean(consensus_flags)) if consensus_flags else 0.0
    consensus_motion = _motion_class(consensus_fraction, thresholds)
    consensus_moving_ranges = _runs_from_equal_flags(consensus_flags, min_len=2)
    consensus_stationary_ranges = _contiguous_good_ranges(n, consensus_moving_ranges)

    ts_arr = np.asarray(ts, dtype=np.float64)
    ts_diff = np.diff(ts_arr) if len(ts_arr) >= 2 else np.empty(0)
    timestamp_nonmonotonic = int(np.sum(~np.isfinite(ts_diff) | (ts_diff <= 0.0)))

    # Long visually frozen runs are legitimate when all physical modalities
    # also indicate a parked/static scene.  They are suspicious only when
    # independent motion evidence contradicts the camera.
    suspect_rgb_runs: List[Tuple[int, int]] = []
    for a, b in rgb_visual_freeze_runs:
        pair_a = max(0, a)
        pair_b = min(len(consensus_flags) - 1, b - 1)
        consensus_motion_inside = (
            any(consensus_flags[pair_a : pair_b + 1]) if pair_b >= pair_a else False
        )
        lidar_motion_inside = (
            any(lidar["ego_moving"][pair_a : pair_b + 1])
            if pair_b >= pair_a and lidar["ego_moving"]
            else False
        )
        pose_motion_inside = (
            any(pose_moving_flags[pair_a : pair_b + 1])
            if pair_b >= pair_a and pose_moving_flags
            else False
        )
        if consensus_motion_inside or lidar_motion_inside or pose_motion_inside:
            suspect_rgb_runs.append((a, b))

    suspect_lidar_runs: List[Tuple[int, int]] = []
    for a, b in lidar_runs:
        pair_a = max(0, a)
        pair_b = min(len(consensus_flags) - 1, b - 1)
        visual_motion_inside = (
            any(rgb["ego_moving"][pair_a : pair_b + 1])
            if pair_b >= pair_a and rgb["ego_moving"]
            else False
        )
        pose_motion_inside = (
            any(pose_moving_flags[pair_a : pair_b + 1])
            if pair_b >= pair_a and pose_moving_flags
            else False
        )
        if visual_motion_inside or pose_motion_inside:
            suspect_lidar_runs.append((a, b))

    # Backward-compatible conflict lists from V15/V16.
    rgb_repeat_while_pose_moves: List[int] = []
    lidar_repeat_while_pose_moves: List[int] = []
    for i in range(min(len(pose_moving_flags), len(rgb["near_equal"]))):
        if pose_moving_flags[i] and rgb["near_equal"][i]:
            rgb_repeat_while_pose_moves.append(i + 1)
    for i in range(min(len(pose_moving_flags), len(lidar["exact_equal"]))):
        if pose_moving_flags[i] and lidar["exact_equal"][i]:
            lidar_repeat_while_pose_moves.append(i + 1)

    pose_says_move_sensors_still: List[int] = []
    sensors_say_move_pose_still: List[int] = []
    for i in range(min(len(pose_moving_flags), len(rgb["ego_moving"]), len(lidar["ego_moving"]))):
        if pose_moving_flags[i] and not rgb["ego_moving"][i] and not lidar["ego_moving"][i]:
            pose_says_move_sensors_still.append(i + 1)
        if (
            not pose_moving_flags[i]
            and rgb["ego_moving"][i]
            and lidar["ego_moving"][i]
        ):
            sensors_say_move_pose_still.append(i + 1)

    bad_for_fusion = sorted(set(suspect_rgb_runs + suspect_lidar_runs))
    usable_ranges = _contiguous_good_ranges(n, bad_for_fusion)

    consensus_moving_mask = _pair_flags_to_frame_mask(consensus_flags, n)
    # Recorded pose remains audit/evaluation evidence.  Do not label a frame
    # stationary when the recorded trajectory explicitly moved, even when a
    # synthetic/feature-poor sensor pair cannot independently vote.
    pose_moving_mask = _pair_flags_to_frame_mask(pose_moving_flags, n)
    localization_motion_mask = consensus_moving_mask | pose_moving_mask
    localization_bad_mask = np.zeros(n, dtype=bool)
    for a, b in suspect_lidar_runs:
        localization_bad_mask[a : b + 1] = True
    usable_moving_mask = localization_motion_mask & ~localization_bad_mask
    usable_stationary_mask = (~localization_motion_mask) & ~localization_bad_mask
    usable_moving_ranges = _ranges_from_frame_flags(usable_moving_mask.tolist())
    usable_stationary_ranges = _ranges_from_frame_flags(usable_stationary_mask.tolist())

    # MOT activity is intentionally reported from independent scene-motion
    # evidence.  GT-label motion is audit-only and never estimator input.
    mot_mask = _pair_flags_to_frame_mask(rgb["scene_dynamic"], n)
    mot_mask |= _pair_flags_to_frame_mask(lidar["scene_dynamic"], n)
    mot_mask |= _pair_flags_to_frame_mask(labels["moving"], n)
    mot_active_ranges = _ranges_from_frame_flags(mot_mask.tolist(), min_len=2)

    issues: List[str] = []
    if len(set(counts.values())) != 1:
        issues.append("modality frame counts differ")
    if timestamp_nonmonotonic:
        issues.append(f"{timestamp_nonmonotonic} non-monotonic timestamp step(s)")
    if rgb["decode_failures"]:
        issues.append(f"{len(rgb['decode_failures'])} RGB decode failure(s)")
    if suspect_rgb_runs:
        issues.append("suspect frozen RGB run(s)")
        issues.append("camera appears frozen while independent motion evidence changes")
    if suspect_lidar_runs:
        issues.append("suspect frozen LiDAR run(s)")
        issues.append("LiDAR appears frozen while independent motion evidence changes")
    if pose_says_move_sensors_still:
        issues.append("recorded pose motion conflicts with RGB+LiDAR stillness")
    if sensors_say_move_pose_still:
        issues.append("RGB+LiDAR motion conflicts with recorded stationary pose")

    observations: List[str] = []
    if rgb_visual_freeze_runs and not suspect_rgb_runs:
        observations.append("visually unchanged RGB is consistent with stationary/static intervals")
    if object_motion_ranges:
        observations.append("GT labels contain moving traffic (audit only; not estimator input)")
    if recorded_pose_motion != consensus_motion:
        observations.append(
            f"recorded-pose classification ({recorded_pose_motion}) differs from sensor-consensus ({consensus_motion})"
        )

    pair_metrics = [p for p in rgb["pairs"] if p.get("decode_ok")]
    ssim_values = [float(p["ssim"]) for p in pair_metrics]
    motion_values = [float(p["global_motion_px"]) for p in pair_metrics]
    dynamic_values = [float(p["dynamic_fraction"]) for p in pair_metrics]
    overlap_values = [float(x) for x in lidar["overlap"] if x is not None]
    lidar_median_values = [float(x) for x in lidar.get("median_nn_m", []) if x is not None]
    lidar_tail_values = [float(x) for x in lidar.get("tail_fraction", []) if x is not None]

    return {
        "sequence": seq,
        "status": "warning" if issues else "ok",
        "counts": counts,
        "frames_audited": n,
        "recorded_pose_motion": recorded_pose_motion,
        "sensor_consensus_motion": consensus_motion,
        # Backward compatibility with V15/V16 callers.
        "ego_motion": recorded_pose_motion,
        "pose": {
            "moving_step_fraction": pose_moving_fraction,
            "total_translation_m": float(np.sum(translation_steps)) if translation_steps else 0.0,
            "net_translation_m": (
                float(np.linalg.norm(pose_positions[-1] - pose_positions[0]))
                if len(pose_positions) >= 2
                else 0.0
            ),
            "max_translation_step_m": max(translation_steps, default=0.0),
            "max_rotation_step_deg": max(rotation_steps, default=0.0),
            "moving_ranges": pose_moving_ranges,
            "stationary_ranges": pose_stationary_ranges,
        },
        "sensor_consensus": {
            "moving_step_fraction": consensus_fraction,
            "moving_ranges": consensus_moving_ranges,
            "stationary_ranges": consensus_stationary_ranges,
        },
        "rgb_motion": {
            "ego_motion_ranges": rgb_ego_motion_ranges,
            "scene_dynamic_ranges": rgb_scene_dynamic_ranges,
            "visual_freeze_runs": rgb_visual_freeze_runs,
            "mean_ssim": float(np.mean(ssim_values)) if ssim_values else None,
            "median_global_motion_px": float(np.median(motion_values)) if motion_values else None,
            "max_global_motion_px": float(np.max(motion_values)) if motion_values else None,
            "mean_dynamic_feature_fraction": float(np.mean(dynamic_values)) if dynamic_values else None,
            "decode_failures": rgb["decode_failures"],
        },
        "lidar_motion": {
            "ego_motion_ranges": lidar_ego_motion_ranges,
            "scene_dynamic_ranges": lidar_scene_dynamic_ranges,
            "mean_voxel_overlap": float(np.mean(overlap_values)) if overlap_values else None,
            "min_voxel_overlap": float(np.min(overlap_values)) if overlap_values else None,
            "median_symmetric_nn_m": float(np.median(lidar_median_values)) if lidar_median_values else None,
            "mean_scene_tail_fraction": float(np.mean(lidar_tail_values)) if lidar_tail_values else None,
        },
        "object_motion_audit_only": {
            "moving_ranges": object_motion_ranges,
            "max_moving_actor_count": max(labels["moving_actor_counts"], default=0),
        },
        "timestamps": {
            "nonmonotonic_steps": timestamp_nonmonotonic,
            "median_dt_s": float(np.median(ts_diff)) if len(ts_diff) else None,
        },
        "imu": {
            "accel_mean": accel_arr.mean(axis=0).tolist() if len(accel_arr) else None,
            "accel_std": accel_arr.std(axis=0).tolist() if len(accel_arr) else None,
            "gyro_mean": gyro_arr.mean(axis=0).tolist() if len(gyro_arr) else None,
            "gyro_std": gyro_arr.std(axis=0).tolist() if len(gyro_arr) else None,
            "max_gyro_norm": float(np.linalg.norm(gyro_arr, axis=1).max()) if len(gyro_arr) else None,
        },
        "duplicates": {
            # V15/V16 compatibility: rgb_runs means exact decoded pixels.
            "rgb_runs": rgb_pixel_runs,
            "rgb_file_byte_runs": rgb_file_runs,
            "rgb_pixel_runs": rgb_pixel_runs,
            "rgb_near_runs": rgb_near_runs,
            "rgb_visual_freeze_runs": rgb_visual_freeze_runs,
            "lidar_exact_runs": lidar_runs,
            "suspect_rgb_runs": suspect_rgb_runs,
            "suspect_lidar_runs": suspect_lidar_runs,
        },
        "cross_sensor": {
            "pose_moves_rgb_lidar_still_frames": pose_says_move_sensors_still,
            "rgb_lidar_move_pose_still_frames": sensors_say_move_pose_still,
            "rgb_repeat_while_pose_moves_frames": rgb_repeat_while_pose_moves,
            "lidar_repeat_while_pose_moves_frames": lidar_repeat_while_pose_moves,
        },
        "mot_active_candidate_ranges": mot_active_ranges,
        "usable_fusion_ranges": usable_ranges,
        "usable_moving_localization_ranges": usable_moving_ranges,
        "usable_stationary_localization_ranges": usable_stationary_ranges,
        "issues": issues,
        "observations": observations,
    }


def _sequence_ids(carla_root: str) -> List[str]:
    seq_root = os.path.join(os.path.expanduser(carla_root), "sequences")
    if not os.path.isdir(seq_root):
        return []
    output: List[str] = []
    for name in sorted(os.listdir(seq_root)):
        path = os.path.join(seq_root, name)
        if os.path.isdir(path):
            try:
                output.append(f"{int(name):04d}")
            except ValueError:
                continue
    return output


def _print_report(result: Dict):
    seq = result.get("sequence", "????")
    print("\n" + "=" * 96)
    print(
        f"SEQUENCE {seq} | status={result.get('status')} | "
        f"pose={result.get('recorded_pose_motion', 'unknown')} | "
        f"sensor-consensus={result.get('sensor_consensus_motion', 'unknown')}"
    )
    print("=" * 96)
    if result.get("status") in {"missing", "empty"}:
        print(result)
        return

    print(f"Frames audited: {result['frames_audited']} | counts={result['counts']}")
    pose = result["pose"]
    print(
        "Recorded pose: "
        f"net={pose['net_translation_m']:.3f} m | total={pose['total_translation_m']:.3f} m | "
        f"max step={pose['max_translation_step_m']:.4f} m | "
        f"max rot={pose['max_rotation_step_deg']:.3f} deg | "
        f"moving={100.0*pose['moving_step_fraction']:.1f}%"
    )
    print(f"Pose moving ranges:              {_format_runs(pose['moving_ranges'])}")
    print(f"Pose stationary ranges:          {_format_runs(pose['stationary_ranges'])}")

    consensus = result["sensor_consensus"]
    print(
        f"Sensor-consensus moving ranges:  {_format_runs(consensus['moving_ranges'])}"
    )
    print(
        f"Sensor-consensus stationary:     {_format_runs(consensus['stationary_ranges'])}"
    )

    rgb = result["rgb_motion"]
    print(f"RGB dominant ego-motion ranges:  {_format_runs(rgb['ego_motion_ranges'])}")
    print(f"RGB non-rigid scene motion:      {_format_runs(rgb['scene_dynamic_ranges'])}")
    print(f"RGB perceptually-static runs:    {_format_runs(rgb['visual_freeze_runs'])}")

    lidar = result["lidar_motion"]
    print(f"LiDAR global-motion ranges:      {_format_runs(lidar['ego_motion_ranges'])}")
    print(f"LiDAR local scene-change ranges: {_format_runs(lidar['scene_dynamic_ranges'])}")

    obj = result["object_motion_audit_only"]
    print(
        "GT object-motion ranges*:       "
        f"{_format_runs(obj['moving_ranges'])}"
    )
    print("  * audit/evaluation only; never estimator input")
    print(f"MOT-active candidate ranges:     {_format_runs(result['mot_active_candidate_ranges'])}")

    dup = result["duplicates"]
    print(f"RGB exact PIXEL repeats:         {_format_runs(dup['rgb_pixel_runs'])}")
    print(f"SUSPECT frozen RGB:              {_format_runs(dup['suspect_rgb_runs'])}")
    print(f"LiDAR exact-repeat runs:         {_format_runs(dup['lidar_exact_runs'])}")
    print(f"SUSPECT frozen LiDAR:            {_format_runs(dup['suspect_lidar_runs'])}")

    print(f"Timestamp non-monotonic:         {result['timestamps']['nonmonotonic_steps']}")
    print(f"Usable fusion ranges:            {_format_runs(result['usable_fusion_ranges'])}")
    print(
        "Usable moving-localization:     "
        f"{_format_runs(result['usable_moving_localization_ranges'])}"
    )
    print(
        "Usable stationary-localization: "
        f"{_format_runs(result['usable_stationary_localization_ranges'])}"
    )

    if result["issues"]:
        print("Issues: " + "; ".join(result["issues"]))
    else:
        print("Issues: no cross-sensor integrity conflict detected")
    if result.get("observations"):
        print("Observations: " + "; ".join(result["observations"]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit CARLA-LVIMOT dataset motion/integrity before expensive processing"
    )
    parser.add_argument("--carla-root", required=True)
    parser.add_argument("--sequences", nargs="*", default=None, help="Sequence IDs; omit for all")
    parser.add_argument("--output", default="dataset_audit_v18.json")
    parser.add_argument("--frozen-run-min", type=int, default=3)
    parser.add_argument("--rgb-analysis-width", type=int, default=320)
    parser.add_argument("--lidar-sample-stride", type=int, default=8)
    args = parser.parse_args()

    sequences = args.sequences or _sequence_ids(args.carla_root)
    if not sequences:
        raise SystemExit("No CARLA sequences found")

    thresholds = AuditThresholds(
        frozen_run_min=max(2, int(args.frozen_run_min)),
        rgb_analysis_width=max(160, int(args.rgb_analysis_width)),
        lidar_sample_stride=max(1, int(args.lidar_sample_stride)),
    )

    results = []
    for seq in sequences:
        result = audit_sequence(args.carla_root, seq, thresholds)
        results.append(result)
        _print_report(result)

    summary = {
        "carla_root": os.path.abspath(os.path.expanduser(args.carla_root)),
        "version": 18,
        "method": {
            "rgb": "pHash + SSIM + sparse optical flow + dominant affine/RANSAC",
            "lidar": "bounded robust symmetric nearest-neighbour background consistency",
            "imu": "gyro + acceleration-change supporting evidence",
            "pose": "recorded pose audit only",
            "objects": "CARLA labels audit/evaluation only",
        },
        "thresholds": thresholds.__dict__,
        "sequences": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 96)
    print(f"AUDIT COMPLETE -> {args.output}")
    print("No estimator run was performed.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
