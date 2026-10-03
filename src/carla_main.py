import os
import sys
import json
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from carla_reader import CarlaReader
from carla_calibration import CarlaCalibration
from carla_detection import CarlaDetector
from carla_pipeline_core import CarlaLVIMOTCore


def _json_safe_worker(value):
    """Convert NumPy-heavy frame results in the writer process, not the main loop."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe_worker(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_worker(v) for v in value]
    return value


def _write_json_frame_worker(frame_file, result):
    """Serialize a complete per-frame record outside the main process/GIL."""
    safe = _json_safe_worker(result)
    with open(frame_file, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)
    return frame_file


class CarlaLVIMOTPipeline:
    """Offline CARLA dataset wrapper around the same GT-free core used live."""

    def __init__(
        self,
        carla_root,
        sequence,
        output_root="outputs_carla",
        brief_bytes=32,
        voxel_size=0.20,
        save_lidar_points=True,
        detector_mode="camera_lidar",
        yolo_model="yolov8n.pt",
        yolo_confidence=0.12,
        yolo_image_size=1280,
        yolo_device="auto",
        async_output=True,
    ):
        self.carla_root = os.path.expanduser(carla_root)
        self.sequence = str(sequence).zfill(4)
        self.output_root = os.path.expanduser(output_root)
        self.reader = CarlaReader(self.carla_root, self.sequence)
        self.calibration = CarlaCalibration(
            os.path.join(self.carla_root, "calibration", "carla_calibration.json")
        )
        # CARLA labels are retained only for evaluation.
        self.gt_detector = CarlaDetector(self.reader, self.calibration)
        self.core = CarlaLVIMOTCore(
            self.calibration,
            brief_bytes=brief_bytes,
            voxel_size=voxel_size,
            detector_mode=detector_mode,
            yolo_model=yolo_model,
            yolo_confidence=yolo_confidence,
            yolo_image_size=yolo_image_size,
            yolo_device=yolo_device,
        )
        self.save_lidar_points = bool(save_lidar_points)
        self.output_sequence = os.path.join(self.output_root, f"sequence_{self.sequence}")
        os.makedirs(self.output_sequence, exist_ok=True)
        self.output_file = os.path.join(self.output_sequence, "results.json")
        self.compact_results = []
        self.async_output = bool(async_output)
        # JSON encoding of ~70k LiDAR points is Python/GIL heavy.  A thread
        # caused occasional 100-300 ms stalls in temporal/factor stages.  Use
        # one spawned writer process so CUDA, NumPy/SciPy and JSON serialization
        # do not contend for the main interpreter lock.
        self._output_executor = (
            ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            if self.async_output else None
        )
        self._pending_writes = []
        self._max_pending_writes = 2

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, dict):
            return {str(k): CarlaLVIMOTPipeline._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [CarlaLVIMOTPipeline._json_safe(v) for v in value]
        return value

    def process_frame(self, frame):
        image = self.reader.load_image(frame)
        lidar = self.reader.load_lidar(frame)
        imu_data = self.reader.load_imu(frame)
        timestamp = float(self.reader.load_timestamp(frame))

        # Evaluation-only data. These values are not used by the estimator.
        gt_pose = self.reader.load_pose(frame)
        gt_objects = self.gt_detector.load_detections(frame)

        result = self.core.process(
            frame_id=int(frame),
            timestamp=timestamp,
            image=image,
            lidar=lidar,
            imu_data=imu_data,
            ground_truth_ego_pose=gt_pose,
            ground_truth_objects=gt_objects,
            include_lidar_points=self.save_lidar_points,
        )
        result["sequence"] = self.sequence
        result["evaluation_ground_truth_available"] = True
        return result

    def _write_frame_file(self, frame_file, result):
        safe = self._json_safe(result)
        with open(frame_file, "w", encoding="utf-8") as f:
            # Preserve the complete human-readable per-frame record.  Writing
            # happens on one bounded background worker so disk serialization
            # can overlap the next sensor frame without unbounded RAM growth.
            json.dump(safe, f, indent=2)
        return frame_file

    def _compact_result(self, result):
        tracking = result.get("tracking", {}).get("active_tracks", [])
        compact = {
            "sequence": result.get("sequence", self.sequence),
            "frame": result.get("frame"),
            "timestamp": result.get("timestamp"),
            "estimator": result.get("estimator", {}),
            "lidar": {k: v for k, v in result.get("lidar", {}).items() if k != "points"},
            "camera": result.get("camera", {}),
            "objects": result.get("objects", {}),
            "tracking": {
                "active_tracks": [
                    {
                        "track_id": t.get("track_id"),
                        "position": t.get("position"),
                        "velocity": t.get("velocity"),
                        "speed": t.get("speed"),
                        "motion_state": t.get("motion_state"),
                        "hits": t.get("hits"),
                        "missed": t.get("missed"),
                    }
                    for t in tracking
                ]
            },
            "state_estimation": {
                "frame_id": result.get("state_estimation", {}).get("frame_id"),
                "pose": result.get("state_estimation", {}).get("pose"),
                "velocity": result.get("state_estimation", {}).get("velocity"),
            },
            "graph": result.get("graph", {}),
            "mapping_4d": result.get("mapping_4d", {}),
            "sensor_integrity": result.get("sensor_integrity", {}),
            "performance": result.get("performance", {}),
        }
        return self._json_safe(compact)

    def _reap_output_writes(self, force=False):
        while self._pending_writes and (force or len(self._pending_writes) >= self._max_pending_writes):
            future = self._pending_writes.pop(0)
            future.result()

    def save_frame(self, result):
        frame = int(result["frame"])
        frame_file = os.path.join(self.output_sequence, f"frame_{frame:06d}.json")
        self.compact_results.append(self._compact_result(result))
        if self._output_executor is None:
            self._write_frame_file(frame_file, result)
        else:
            self._reap_output_writes(force=False)
            self._pending_writes.append(
                self._output_executor.submit(_write_json_frame_worker, frame_file, result)
            )
        return frame_file

    def flush_output(self):
        self._reap_output_writes(force=True)
        if self._output_executor is not None:
            self._output_executor.shutdown(wait=True)
            self._output_executor = None

    def run(self, start_frame=0, end_frame=None):
        total_frames = self.reader.num_frames()
        if end_frame is None:
            end_frame = total_frames - 1
        start_frame = max(0, int(start_frame))
        end_frame = min(int(end_frame), total_frames - 1)
        if end_frame < start_frame:
            raise ValueError("end_frame must be >= start_frame")

        print("\n" + "=" * 70)
        print("CARLA LVIMOT OFFLINE VALIDATION")
        print("=" * 70)
        print(f"Sequence: {self.sequence}")
        print(f"Frames:   {start_frame} -> {end_frame}")
        print("Estimator input: Camera + LiDAR + IMU only")
        print("CARLA pose/labels: evaluation only")
        print(
            f"LiDAR object range: {self.core.object_sensor_range:.1f} m "
            f"(dataset calibration: {self.calibration.lidar_range:.1f} m)"
        )
        print("=" * 70)

        run_start = time.perf_counter()
        for frame in range(start_frame, end_frame + 1):
            print(f"\nProcessing frame {frame} ...")
            result = self.process_frame(frame)
            save_t0 = time.perf_counter()
            frame_file = self.save_frame(result)
            save_queue_ms = (time.perf_counter() - save_t0) * 1000.0
            perf = result.get("performance", {})
            print(
                "  "
                f"LiDAR {perf.get('lidar_features_ms', 0):.0f} ms | "
                f"Visual {perf.get('camera_features_ms', 0):.0f} ms | "
                f"DetWait {perf.get('detector_wait_ms', 0):.0f} ms | "
                f"TempV {perf.get('temporal_visual_ms', 0):.0f} ms | "
                f"TempL {perf.get('temporal_lidar_ms', 0):.0f} ms | "
                f"Factor {perf.get('factor_build_ms', 0):.0f} ms | "
                f"Graph {perf.get('graph_opt_ms', 0):.0f} ms | "
                f"Map {perf.get('mapping_ms', 0):.0f} ms | "
                f"SaveQ {save_queue_ms:.0f} ms | "
                f"Total {perf.get('total_ms', 0):.0f} ms | "
                f"RSS {perf.get('rss_mb', 0):.0f} MB"
            )
            print(
                "  "
                f"FAST {result['camera']['fast_keypoints_raw']} -> {result['camera']['fast_keypoints_used']} | "
                f"BRIEF {result['camera']['brief_keypoints']} | "
                f"Planes {result['lidar']['planar_features']} -> {result['lidar']['temporal_features_used']} | "
                f"YOLO {result['objects'].get('detector_diagnostics', {}).get('camera_detections', 0)} "
                f"[{result['objects'].get('detector_diagnostics', {}).get('inference_device', 'n/a')}, "
                f"FP16={result['objects'].get('detector_diagnostics', {}).get('fp16', False)}] | "
                f"Proposals {result['objects']['sensor_proposals']} "
                f"(Cam {result['objects'].get('camera_supported_proposals', 0)}) | "
                f"Fb {int(perf.get('tracking_fallback_candidates', 0))} | "
                f"Tracks {result['objects']['confirmed_tracks']} | "
                f"EvalGT {result['objects'].get('evaluation', {}).get('eligible_gt_objects', 0)}/"
                f"{result['objects'].get('evaluation', {}).get('raw_gt_objects', 0)} | "
                f"EvalTracks {result['objects'].get('evaluation', {}).get('scored_tracks', 0)} | "
                f"GTBoxes {result['objects'].get('evaluation', {}).get('projected_gt_boxes', 0)} | "
                f"GraphObj {result.get('graph', {}).get('object_states', 0)} | "
                f"StaleRGB {result.get('sensor_integrity', {}).get('stale_rgb', False)} | "
                f"StaleLiDAR {result.get('sensor_integrity', {}).get('stale_lidar', False)} | "
                f"Stationary {result.get('sensor_integrity', {}).get('stationary_active', False)} "
                f"(S{result.get('sensor_integrity', {}).get('stationary_streak', 0)}, "
                f"G={result.get('sensor_integrity', {}).get('gyro_norm_rad_s', 0.0):.4f}, "
                f"Aerr={result.get('sensor_integrity', {}).get('accel_gravity_error_m_s2', 0.0):.3f}, "
                f"L={result.get('sensor_integrity', {}).get('lidar_stationary_metric_m') if result.get('sensor_integrity', {}).get('lidar_stationary_metric_m') is not None else float('nan'):.3f}, "
                f"Ov={result.get('sensor_integrity', {}).get('lidar_stationary_overlap', 0.0):.2f}, "
                f"Vflow={result.get('sensor_integrity', {}).get('visual_motion_px') if result.get('sensor_integrity', {}).get('visual_motion_px') is not None else float('nan'):.2f}px, "
                f"Vfrac={result.get('sensor_integrity', {}).get('visual_motion_fraction', 0.0):.2f}, "
                f"Ltrans={result.get('sensor_integrity', {}).get('lidar_translation_m') if result.get('sensor_integrity', {}).get('lidar_translation_m') is not None else float('nan'):.3f}m, "
                f"Veto={result.get('sensor_integrity', {}).get('translation_motion_veto', False)}) | "
                f"SE3={'Y' if result.get('lidar_odometry', {}).get('accepted', False) else 'N'} "
                f"(M={result.get('lidar_odometry', {}).get('matches', 0)}, "
                f"I={result.get('lidar_odometry', {}).get('inliers', 0)}, "
                f"E={result.get('lidar_odometry', {}).get('rmse_m') if result.get('lidar_odometry', {}).get('rmse_m') is not None else float('nan'):.3f}m, "
                f"d=[{result.get('lidar_odometry', {}).get('delta_t', [0.0, 0.0, 0.0])[0]:.3f},"
                f"{result.get('lidar_odometry', {}).get('delta_t', [0.0, 0.0, 0.0])[1]:.3f},"
                f"{result.get('lidar_odometry', {}).get('delta_t', [0.0, 0.0, 0.0])[2]:.3f}], "
                f"R={np.rad2deg(result.get('lidar_odometry', {}).get('rotation_rad') or 0.0):.2f}deg) | "
                f"Saved {frame_file}"
            )

        self.flush_output()
        evaluation = self.core.evaluation_summary()
        summary = {
            "sequence": self.sequence,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "processed_frames": end_frame - start_frame + 1,
            "estimator_ground_truth_used": False,
            "runtime_seconds": float(time.perf_counter() - run_start),
            "evaluation": evaluation,
            "mapping_4d": self.core.mapper_4d.export_summary(),
            "frames": self.compact_results,
        }
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(summary), f, indent=2)

        tracking = evaluation.get("tracking") or {}
        localization = evaluation.get("localization") or {}
        print("\n" + "=" * 70)
        print("OFFLINE VALIDATION COMPLETE")
        print(f"Processed frames: {summary['processed_frames']}")
        print(f"Results: {self.output_file}")
        mota = tracking.get("mota")
        if mota is None:
            print("MOTA: N/A (no eligible GT objects; evaluation invalid)")
        else:
            print(f"MOTA: {mota:.4f}")
        print(
            "Tracking counts: "
            f"GT {tracking.get('total_gt_objects', 0)} | "
            f"Outputs {tracking.get('total_tracker_outputs', 0)} | "
            f"Matches {tracking.get('total_matches', 0)} | "
            f"FP {tracking.get('false_positives', 0)} | "
            f"FN {tracking.get('false_negatives', 0)} | "
            f"IDSW {tracking.get('id_switches', 0)}"
        )
        print(
            "Match modes: "
            f"CameraIoU {tracking.get('camera_iou_matches', 0)} | "
            f"BEV {tracking.get('bev_matches', 0)}"
        )
        matched_iou = tracking.get("mean_matched_iou")
        if matched_iou is not None:
            print(f"Mean matched camera IoU: {matched_iou:.4f}")
        pos_rmse = tracking.get("position_rmse")
        vel_rmse = tracking.get("velocity_rmse")
        nearest = tracking.get("mean_nearest_bev_distance")
        print(
            "Position RMSE: "
            + (f"{pos_rmse:.4f} m" if pos_rmse is not None else "N/A (no matched tracks)")
        )
        print(
            "Velocity RMSE: "
            + (f"{vel_rmse:.4f} m/s" if vel_rmse is not None else "N/A (no matched tracks)")
        )
        if nearest is not None:
            print(f"Mean nearest GT-track BEV distance: {nearest:.4f} m")
        print(f"Raw ATE RMSE: {localization.get('raw_ate_rmse', 0.0):.4f} m")
        print(f"Aligned ATE RMSE: {localization.get('aligned_ate_rmse', 0.0):.4f} m")
        print(f"RPE RMSE: {localization.get('rpe_rmse', 0.0):.4f} m")
        print("=" * 70)
        return summary


def main():
    parser = argparse.ArgumentParser(description="CARLA LVIMOT offline/live runner")
    parser.add_argument("--live", action="store_true", help="Run the live CARLA sensor pipeline")
    parser.add_argument("--carla-root", default=None, help="Offline CARLA-LVIMOT dataset root")
    parser.add_argument("--sequence", default="0006")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--output-root", default="outputs_carla")
    parser.add_argument("--no-save-lidar-points", action="store_true")
    parser.add_argument("--detector-mode", choices=["camera_lidar", "lidar"], default="camera_lidar")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-confidence", type=float, default=0.06)
    parser.add_argument("--yolo-image-size", type=int, default=1280)
    parser.add_argument("--yolo-device", default="auto")
    parser.add_argument("--sync-output", action="store_true", help="Disable bounded asynchronous frame JSON writing")

    # Live-mode arguments are accepted here and delegated unchanged.
    parser.add_argument("--carla-install", default=os.environ.get("CARLA_LAUNCHER", r"C:\CARLA\CarlaUE4.exe" if os.name == "nt" else os.path.expanduser("~/CARLA/CarlaUE4.sh")))
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--autopilot", action="store_true")
    parser.add_argument("--evaluate-ground-truth", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--traffic-vehicles", type=int, default=12, help="NPC autopilot vehicles to spawn in live mode")
    parser.add_argument("--fixed-delta", type=float, default=0.1, help="Live synchronous CARLA step in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic live spawn/traffic seed")
    parser.add_argument("--sensor-timeout", type=float, default=3.0, help="Seconds to wait for each synchronized live sensor frame")
    args = parser.parse_args()

    if args.live:
        from carla_live_main import run_from_namespace
        return run_from_namespace(args)

    if not args.carla_root:
        parser.error("--carla-root is required in offline mode")
    pipeline = CarlaLVIMOTPipeline(
        carla_root=args.carla_root,
        sequence=args.sequence,
        output_root=args.output_root,
        save_lidar_points=not args.no_save_lidar_points,
        detector_mode=args.detector_mode,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        yolo_image_size=args.yolo_image_size,
        yolo_device=args.yolo_device,
        async_output=not args.sync_output,
    )
    pipeline.run(start_frame=args.start_frame, end_frame=args.end_frame)


if __name__ == "__main__":
    main()
