import json
import os
import random
import traceback
from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from carla_api_loader import load_carla
from carla_calibration import CarlaCalibration
from carla_pipeline_core import CarlaLVIMOTCore
from carla_live_adapter import CarlaLiveSensorAdapter
from carla_live_visualizer import CarlaLiveVisualizer


def _locrot(transform_dict):
    loc = transform_dict.get("location", transform_dict)
    rot = transform_dict.get("rotation", {})
    return loc, rot


def _to_carla_transform(carla, transform_dict):
    loc, rot = _locrot(transform_dict)
    return carla.Transform(
        carla.Location(
            x=float(loc.get("x", 0.0)),
            y=float(loc.get("y", 0.0)),
            z=float(loc.get("z", 0.0)),
        ),
        carla.Rotation(
            pitch=float(rot.get("pitch", 0.0)),
            yaw=float(rot.get("yaw", 0.0)),
            roll=float(rot.get("roll", 0.0)),
        ),
    )


def _snapshot_actor(snapshot, actor_id):
    if snapshot is None:
        return None
    try:
        return snapshot.find(int(actor_id))
    except Exception:
        return None


def _gt_pose(actor, snapshot=None):
    actor_snapshot = _snapshot_actor(snapshot, actor.id)
    if actor_snapshot is not None:
        t = actor_snapshot.get_transform()
    else:
        t = actor.get_transform()
    return {
        "location": {
            "x": float(t.location.x),
            "y": float(t.location.y),
            "z": float(t.location.z),
        },
        "rotation": {
            "pitch": float(t.rotation.pitch),
            "yaw": float(t.rotation.yaw),
            "roll": float(t.rotation.roll),
        },
    }


def _world_vertices(actor, transform):
    """Evaluation-only CARLA 3-D bounding-box vertices in world coordinates."""
    try:
        vertices = actor.bounding_box.get_world_vertices(transform)
    except Exception:
        return []
    return [
        [float(v.x), float(v.y), float(v.z)]
        for v in vertices
    ]


def _gt_objects(world, ego_id, snapshot=None):
    """Read CARLA actor ground truth for post-estimation evaluation only.

    When the live synchronizer returns a delayed GPU-camera frame, ``snapshot``
    is the buffered world snapshot for that same CARLA frame.  This keeps
    optional evaluation aligned without exposing simulator state to the
    estimator.
    """
    output = []
    actors = list(world.get_actors().filter("vehicle.*")) + list(
        world.get_actors().filter("walker.pedestrian.*")
    )
    for actor in actors:
        if int(actor.id) == int(ego_id):
            continue

        actor_snapshot = _snapshot_actor(snapshot, actor.id)
        if snapshot is not None and actor_snapshot is None:
            # The actor did not exist in the synchronized evaluation frame.
            continue

        if actor_snapshot is not None:
            t = actor_snapshot.get_transform()
            v = actor_snapshot.get_velocity()
        else:
            t = actor.get_transform()
            v = actor.get_velocity()

        ext = actor.bounding_box.extent
        world_corners = _world_vertices(actor, t)
        output.append(
            {
                "actor_id": int(actor.id),
                "type": actor.type_id,
                "location": {
                    "x": float(t.location.x),
                    "y": float(t.location.y),
                    "z": float(t.location.z),
                },
                "velocity": {
                    "x": float(v.x),
                    "y": float(v.y),
                    "z": float(v.z),
                },
                "dimensions": [
                    float(2.0 * ext.x),
                    float(2.0 * ext.y),
                    float(2.0 * ext.z),
                ],
                "rotation": {
                    "pitch": float(t.rotation.pitch),
                    "yaw": float(t.rotation.yaw),
                    "roll": float(t.rotation.roll),
                },
                "bounding_box": {
                    "world_corners": world_corners,
                },
            }
        )
    return output


def _choose_blueprint(rng, blueprints):
    candidates = list(blueprints)
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]


def _configure_vehicle_blueprint(bp, rng, role_name=None):
    if role_name and bp.has_attribute("role_name"):
        bp.set_attribute("role_name", str(role_name))
    if bp.has_attribute("color"):
        values = list(bp.get_attribute("color").recommended_values)
        if values:
            bp.set_attribute("color", values[rng.randrange(len(values))])
    if bp.has_attribute("driver_id"):
        values = list(bp.get_attribute("driver_id").recommended_values)
        if values:
            bp.set_attribute("driver_id", values[rng.randrange(len(values))])
    return bp


def _spawn_ego(carla, world, rng):
    library = world.get_blueprint_library()
    blueprints = list(library.filter("vehicle.*"))
    if not blueprints:
        raise RuntimeError("No vehicle blueprint found in CARLA world")
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("No vehicle spawn points available in CARLA map")
    rng.shuffle(spawn_points)

    for spawn in spawn_points:
        bp = _configure_vehicle_blueprint(_choose_blueprint(rng, blueprints), rng, "hero")
        vehicle = world.try_spawn_actor(bp, spawn)
        if vehicle is not None:
            return vehicle, spawn_points
    raise RuntimeError("Could not spawn ego vehicle at any CARLA map spawn point")


def _spawn_traffic(world, traffic_manager, rng, spawn_points, count):
    count = max(0, int(count))
    if count == 0:
        return []

    library = world.get_blueprint_library()
    blueprints = list(library.filter("vehicle.*"))
    points = list(spawn_points)
    rng.shuffle(points)
    actors = []
    tm_port = int(traffic_manager.get_port())

    for spawn in points:
        if len(actors) >= count:
            break
        bp = _configure_vehicle_blueprint(_choose_blueprint(rng, blueprints), rng, "autopilot")
        actor = world.try_spawn_actor(bp, spawn)
        if actor is None:
            continue
        actor.set_autopilot(True, tm_port)
        actors.append(actor)
    return actors


def _compact_live_record(result: Dict) -> Dict:
    tracks = result.get("tracking", {}).get("active_tracks", [])
    return {
        "frame": result.get("frame"),
        "carla_frame": result.get("carla_frame"),
        "timestamp": result.get("timestamp"),
        "state_estimation": result.get("state_estimation", {}),
        "tracks": [
            {
                "track_id": t.get("track_id"),
                "class": t.get("class"),
                "position": t.get("position"),
                "velocity": t.get("velocity"),
                "speed": t.get("speed"),
                "motion_state": t.get("motion_state"),
                "hits": t.get("hits"),
                "missed": t.get("missed"),
                "bbox": t.get("bbox"),
            }
            for t in tracks
        ],
        "mapping_4d": result.get("mapping_4d", {}),
        "sensor_integrity": result.get("sensor_integrity", {}),
        "evaluation_frame": result.get("objects", {}).get("evaluation", {}),
        "performance": result.get("performance", {}),
        "live_sync": result.get("live_sync", {}),
    }


def _print_preflight(client, world, calibration, fixed_delta, traffic_vehicles):
    try:
        server_version = client.get_server_version()
    except Exception:
        server_version = "unknown"
    try:
        client_version = client.get_client_version()
    except Exception:
        client_version = "unknown"
    map_name = getattr(world.get_map(), "name", "unknown")

    print("\n" + "=" * 70)
    print("LVIMOT LIVE V1 - CARLA PREFLIGHT")
    print("=" * 70)
    print(f"CARLA client: {client_version}")
    print(f"CARLA server: {server_version}")
    print(f"Map:          {map_name}")
    print(f"Fixed dt:     {fixed_delta:.3f} s ({1.0 / fixed_delta:.1f} Hz)")
    print(f"Camera:       {calibration.width} x {calibration.height}, FOV {calibration.fov:.1f} deg")
    print(f"LiDAR:        {calibration.lidar_channels} ch, {calibration.lidar_range:.1f} m")
    print(f"NPC vehicles: {int(traffic_vehicles)} requested")
    print("Estimator:    RGB + LiDAR + IMU only")
    print("CARLA GT:     post-estimation evaluation only")
    print("=" * 70)


def run_live(
    carla_install: str,
    calibration_file: str,
    host: str = "127.0.0.1",
    port: int = 2000,
    seconds: float = 120.0,
    autopilot: bool = False,
    evaluate_ground_truth: bool = False,
    no_display: bool = False,
    output_root: str = "outputs_carla",
    detector_mode: str = "camera_lidar",
    yolo_model: str = "yolov8n.pt",
    yolo_confidence: float = 0.06,
    yolo_image_size: int = 1280,
    yolo_device: str = "auto",
    traffic_vehicles: int = 12,
    fixed_delta: float = 0.1,
    seed: int = 42,
    sensor_timeout: float = 3.0,
):
    fixed_delta = float(fixed_delta)
    if not np.isfinite(fixed_delta) or fixed_delta <= 0.0:
        raise ValueError("--fixed-delta must be a positive number of seconds")

    carla = load_carla(carla_install)
    calibration = CarlaCalibration(calibration_file)
    core = CarlaLVIMOTCore(
        calibration,
        detector_mode=detector_mode,
        yolo_model=yolo_model,
        yolo_confidence=yolo_confidence,
        yolo_image_size=yolo_image_size,
        yolo_device=yolo_device,
    )
    visualizer = CarlaLiveVisualizer()
    rng = random.Random(int(seed))

    client = carla.Client(host, int(port))
    client.set_timeout(10.0)
    world = client.get_world()
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager()
    all_actors: List = []
    sensor_adapter: Optional[CarlaLiveSensorAdapter] = None

    output_dir = os.path.join(output_root, "live")
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "live_frames.jsonl")
    summary_path = os.path.join(output_dir, "live_results.json")

    _print_preflight(client, world, calibration, fixed_delta, traffic_vehicles)

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        try:
            traffic_manager.set_random_device_seed(int(seed))
        except Exception:
            pass

        vehicle, spawn_points = _spawn_ego(carla, world, rng)
        all_actors.append(vehicle)

        if autopilot:
            vehicle.set_autopilot(True, traffic_manager.get_port())

        traffic = _spawn_traffic(
            world,
            traffic_manager,
            rng,
            spawn_points,
            traffic_vehicles,
        )
        all_actors.extend(traffic)

        bp = world.get_blueprint_library()

        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(calibration.width))
        cam_bp.set_attribute("image_size_y", str(calibration.height))
        cam_bp.set_attribute("fov", str(calibration.fov))
        cam_bp.set_attribute("sensor_tick", str(fixed_delta))
        camera = world.spawn_actor(
            cam_bp,
            _to_carla_transform(carla, calibration.camera_transform),
            attach_to=vehicle,
        )
        all_actors.append(camera)

        lidar_bp = bp.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(calibration.lidar_channels))
        lidar_bp.set_attribute("range", str(calibration.lidar_range))
        lidar_bp.set_attribute("rotation_frequency", str(1.0 / fixed_delta))
        lidar_cfg = calibration.data.get("lidar", {})
        if "points_per_second" in lidar_cfg:
            lidar_bp.set_attribute("points_per_second", str(int(lidar_cfg["points_per_second"])))
        if "upper_fov" in lidar_cfg:
            lidar_bp.set_attribute("upper_fov", str(float(lidar_cfg["upper_fov"])))
        if "lower_fov" in lidar_cfg:
            lidar_bp.set_attribute("lower_fov", str(float(lidar_cfg["lower_fov"])))
        lidar_sensor = world.spawn_actor(
            lidar_bp,
            _to_carla_transform(carla, calibration.lidar_transform),
            attach_to=vehicle,
        )
        all_actors.append(lidar_sensor)

        imu_bp = bp.find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", str(fixed_delta))
        imu_sensor = world.spawn_actor(
            imu_bp,
            _to_carla_transform(carla, calibration.imu_transform),
            attach_to=vehicle,
        )
        all_actors.append(imu_sensor)

        sensor_adapter = CarlaLiveSensorAdapter(
            world,
            camera,
            lidar_sensor,
            imu_sensor,
            timeout_s=sensor_timeout,
            max_timestamp_skew_s=max(1e-4, fixed_delta * 0.05),
        )

        max_frames = max(1, int(np.ceil(float(seconds) / fixed_delta)))
        print("\n" + "=" * 70)
        print("LVIMOT LIVE V1")
        print("RGB + LiDAR + IMU -> localization + MOT + factor graph + 4D map")
        print(f"Spawned traffic vehicles: {len(traffic)}")
        print("CARLA ground truth is never an estimator input")
        print("Press Q or ESC in the visualization window to stop")
        print("=" * 70)

        processed_frames = 0
        with open(jsonl_path, "w", encoding="utf-8") as log_file:
            for live_idx in range(max_frames):
                live_frame = sensor_adapter.tick()

                # Evaluation-only values are intentionally read after the live
                # RGB/LiDAR/IMU packet has already been assembled.  The core
                # consumes them only after estimator outputs are finalized.
                gt_pose = (
                    _gt_pose(vehicle, live_frame.world_snapshot)
                    if evaluate_ground_truth
                    else None
                )
                gt_objects = (
                    _gt_objects(world, vehicle.id, live_frame.world_snapshot)
                    if evaluate_ground_truth
                    else None
                )

                result = core.process(
                    frame_id=live_idx,
                    timestamp=live_frame.timestamp,
                    image=live_frame.image,
                    lidar=live_frame.lidar,
                    imu_data=live_frame.imu,
                    ground_truth_ego_pose=gt_pose,
                    ground_truth_objects=gt_objects,
                    include_lidar_points=False,
                )
                result["carla_frame"] = int(live_frame.carla_frame)
                result["live_sync"] = {
                    "timestamp_skew_s": float(live_frame.timestamp_skew_s),
                    "world_frame_at_return": live_frame.world_frame_at_return,
                    "simulation_lag_frames": int(live_frame.simulation_lag_frames),
                    **sensor_adapter.diagnostics(),
                }
                log_file.write(json.dumps(_json_safe(_compact_live_record(result))) + "\n")
                log_file.flush()
                processed_frames += 1

                perf = result.get("performance", {})
                integrity = result.get("sensor_integrity", {})
                stationary = bool(integrity.get("stationary_active", False))
                print(
                    f"frame {live_idx:05d} / CARLA {live_frame.carla_frame:06d} | "
                    f"{perf.get('total_ms', 0):7.0f} ms | "
                    f"RSS {perf.get('rss_mb', 0):6.0f} MB | "
                    f"tracks {len(result.get('tracking', {}).get('active_tracks', [])):2d} | "
                    f"map {result.get('mapping_4d', {}).get('total_static_voxels', 0):6d} | "
                    f"stationary={stationary} | "
                    f"sync={live_frame.timestamp_skew_s * 1000.0:.2f} ms | "
                    f"camlag={live_frame.simulation_lag_frames}f | "
                    f"pump={sensor_adapter.diagnostics().get('world_ticks_last_call', 0)}"
                )

                if not no_display and cv2 is not None:
                    display = visualizer.render(
                        live_frame.image,
                        result,
                        core.mapper_4d.get_static_map_points(),
                    )
                    cv2.imshow("LVIMOT LIVE V1: CARLA | 4D MAP", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

        evaluation = core.evaluation_summary() if evaluate_ground_truth else None
        summary = {
            "mode": "live_carla",
            "version": "LVIMOT_LIVE_V1",
            "processed_frames": int(processed_frames),
            "estimator_inputs": ["RGB", "LiDAR", "IMU"],
            "estimator_ground_truth_used": False,
            "ground_truth_evaluation_enabled": bool(evaluate_ground_truth),
            "evaluation": evaluation,
            "mapping_4d": core.mapper_4d.export_summary(),
            "live_sync": sensor_adapter.diagnostics(),
            "spawned_traffic_vehicles": int(len(traffic)),
            "fixed_delta_seconds": float(fixed_delta),
            "frame_log": jsonl_path,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2)

        print("\n" + "=" * 70)
        print("LVIMOT LIVE V1 COMPLETE")
        print(f"Processed frames: {processed_frames}")
        print(f"Results: {summary_path}")
        if evaluation is not None:
            localization = evaluation.get("localization", {})
            tracking = evaluation.get("tracking", {})
            print(
                f"Live GT evaluation: ATE {localization.get('raw_ate_rmse', 0.0):.4f} m | "
                f"RPE {localization.get('rpe_rmse', 0.0):.4f} m | "
                f"MOTA {tracking.get('mota')}"
            )
        print("=" * 70)
        return summary
    except BaseException:
        # Print the real Python-side failure *before* touching CARLA actors.
        # CARLA 0.9.13 can terminate the interpreter during sensor teardown,
        # which otherwise masks the original exception and leaves only
        # Windows fast-fail 0xC0000409.
        print("\n[LVIMOT LIVE ERROR BEFORE CLEANUP]", flush=True)
        traceback.print_exc()
        raise
    finally:
        # Disable Python callbacks first, but deliberately do not call the
        # native CARLA sensor.stop() method here.  CARLA 0.9.13 on Windows
        # can fast-fail during native unsubscribe/teardown.  The sensor actors
        # are destroyed together with the other actors in the batch below.
        if sensor_adapter is not None:
            try:
                sensor_adapter.close()
            except Exception:
                traceback.print_exc()

        # Destroy actors through one client batch instead of repeated native
        # Actor.destroy()/Sensor.stop() calls from Python.
        try:
            actor_ids = [int(a.id) for a in reversed(all_actors)]
            if actor_ids:
                client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in actor_ids])
        except Exception:
            traceback.print_exc()

        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        if (not no_display) and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    # Handles np.bool_, np.integer, np.floating and other NumPy scalar
    # subclasses that can appear only after temporal state exists.
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def run_from_namespace(args):
    calibration_file = args.calibration_file
    if calibration_file is None:
        if args.carla_root:
            calibration_file = os.path.join(
                args.carla_root,
                "calibration",
                "carla_calibration.json",
            )
        else:
            raise ValueError(
                "Live mode requires --calibration-file or --carla-root pointing "
                "to the dataset calibration root"
            )
    return run_live(
        carla_install=args.carla_install,
        calibration_file=calibration_file,
        host=args.host,
        port=args.port,
        seconds=args.seconds,
        autopilot=args.autopilot,
        evaluate_ground_truth=args.evaluate_ground_truth,
        no_display=args.no_display,
        output_root=args.output_root,
        detector_mode=args.detector_mode,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        yolo_image_size=args.yolo_image_size,
        yolo_device=args.yolo_device,
        traffic_vehicles=args.traffic_vehicles,
        fixed_delta=args.fixed_delta,
        seed=args.seed,
        sensor_timeout=args.sensor_timeout,
    )
