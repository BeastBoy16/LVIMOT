from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineLimits:
    # Front-end bounds. These apply identically in offline and live mode.
    # V12 keeps the same FAST/BRIEF + planar-feature algorithms, but only the
    # strongest bounded working sets enter temporal matching and optimization.
    max_visual_features: int = 700
    max_visual_tracks: int = 800
    visual_history_size: int = 12

    max_planar_features_for_temporal: int = 600
    max_planar_landmarks: int = 800
    lidar_landmark_history_size: int = 8
    # The full LiDAR cloud is still deskewed/saved. Planar extraction only
    # evaluates a spatially uniform bounded candidate set after voxelization.
    max_lidar_planar_candidates: int = 16000
    # V13: preserve a dense voxel support cloud for neighborhood geometry,
    # but evaluate planarity at a bounded subset of query points.
    max_lidar_planar_queries: int = 6000

    # Pose-graph work is deliberately smaller than the front-end feature count.
    max_lidar_factors_per_frame: int = 8
    max_visual_factors_per_frame: int = 4
    max_object_factors_per_frame: int = 2
    max_graph_objects: int = 2

    sliding_window_size: int = 4
    optimizer_max_nfev: int = 2

    # 4D map bounds. Full-resolution LiDAR remains in per-frame output, while
    # the rolling voxel map consumes a deterministic subset to avoid spending
    # hundreds of milliseconds reinserting redundant neighboring returns.
    max_map_planar_landmarks: int = 8000
    max_map_static_voxels: int = 120000
    max_dynamic_tube_history: int = 240
    map_point_stride: int = 8
    # Static geometry changes much more slowly than dynamic tracks.  V13
    # updates dynamic tubes every frame but only reinserts static LiDAR /
    # planar landmarks at this cadence.
    map_static_update_interval: int = 2
    map_planar_update_interval: int = 2
    map_evict_interval: int = 30

    # GT-free LiDAR object proposals.
    object_voxel_size: float = 0.45
    object_min_points: int = 18
    object_max_points: int = 3500
    # 0.0 means: use the LiDAR range recorded in carla_calibration.json.
    object_max_range: float = 0.0
    object_min_height: float = -3.0
    object_max_height: float = 4.0
    max_object_proposals: int = 8
    object_min_vehicle_score: float = 0.52

    # MOT bounds / confirmation. Weak hypotheses may remain internal for
    # reassociation; publication stays sensor-only and requires temporal support.
    tracker_distance_threshold: float = 2.75
    tracker_max_missed: int = 3
    tracker_min_hits: int = 3
    tracker_history_size: int = 24
    # Camera+LiDAR V22 MOT policy. Distant vehicles may have weak camera
    # confidence but still strong LiDAR support. Such candidates may enter the
    # tentative tracker at a lower fused score and confirm only after repeated
    # camera+LiDAR support. This follows the existing LVIMOT temporal-fusion
    # architecture rather than replacing the detector/tracker.
    tracker_new_track_min_score: float = 0.20
    tracker_publish_min_score: float = 0.42
    tracker_publish_max_missed: int = 0
    tracker_gap_publish_min_hits: int = 5
    tracker_min_camera_support: int = 6
    tracker_fast_confirm_hits: int = 2
    tracker_fast_confirm_min_score: float = 0.42
    tracker_fast_confirm_min_support: int = 8
    tracker_weak_confirm_min_score: float = 0.22
    tracker_weak_confirm_min_camera_confidence: float = 0.06
    tracker_weak_confirm_min_support: int = 10
    tracker_weak_confirm_hits: int = 2
    # A camera+LiDAR hypothesis that remains static in world coordinates while
    # the ego is sensor-confirmed stationary belongs to the static map, not the
    # dynamic MOT output. Keep it internally so it can be promoted if it moves.
    tracker_static_suppress_min_hits: int = 5
    tracker_static_suppress_min_confidence: float = 0.85
    tracker_fallback_min_hits: int = 3
    tracker_fallback_min_support: int = 18
    tracker_lidar_fallback_max_proposals: int = 6

    # Sensor-only integrity / stationary-mode thresholds. These never use GT.
    stale_hash_digest_bytes: int = 8
    stationary_gyro_threshold_rad_s: float = 0.04
    # Support both common CARLA IMU encodings: specific-force magnitude near g,
    # and gravity-removed acceleration magnitude near zero.
    stationary_accel_tolerance_m_s2: float = 0.90
    stationary_accel_zero_tolerance_m_s2: float = 0.60
    # Raw-LiDAR angular range signature.  Dynamic actors affect only a minority
    # of beams; ego motion changes the static background across many beams.
    stationary_lidar_range_delta_m: float = 0.18
    stationary_lidar_min_overlap: float = 0.30
    stationary_lidar_azimuth_bins: int = 360
    stationary_lidar_elevation_bins: int = 32
    stationary_min_planar_points: int = 80

    # V24 translation-motion veto.  Constant-speed driving has low gyro and
    # low linear acceleration, so IMU quietness is NOT sufficient evidence of
    # zero velocity.  Stationary mode may engage only when image-background
    # flow and robust scan-to-scan LiDAR registration also agree on zero
    # translation.  These are sensor-only checks and never use CARLA GT.
    stationary_visual_flow_threshold_px: float = 0.35
    stationary_visual_min_tracks: int = 24
    stationary_visual_motion_fraction_threshold: float = 0.30
    stationary_visual_track_motion_px: float = 0.45
    stationary_lidar_translation_threshold_m: float = 0.08
    stationary_lidar_rotation_threshold_rad: float = 0.012
    stationary_lidar_icp_max_points: int = 3500
    stationary_lidar_icp_max_correspondence_m: float = 1.50
    stationary_lidar_icp_trim_fraction: float = 0.70
    stationary_lidar_icp_iterations: int = 4

    # V25 moving-ego metric LiDAR odometry. The same static planar geometry
    # already extracted by LVIMOT is registered frame-to-frame and enters the
    # existing factor graph as a binary SE(3) relative-pose measurement.
    lidar_se3_enabled: bool = True
    lidar_se3_min_matches: int = 24
    lidar_se3_max_iterations: int = 5
    lidar_se3_trim_fraction: float = 0.75
    lidar_se3_huber_m: float = 0.20
    lidar_se3_accept_rmse_m: float = 0.20
    lidar_se3_max_translation_m: float = 1.50
    lidar_se3_max_rotation_rad: float = 0.20
    lidar_se3_translation_weight: float = 8.0
    lidar_se3_rotation_weight: float = 12.0
    stationary_confirm_frames: int = 5
    stationary_release_frames: int = 1
    stationary_pose_prior_weight: float = 120.0
    stationary_velocity_prior_weight: float = 150.0


DEFAULT_LIMITS = PipelineLimits()
