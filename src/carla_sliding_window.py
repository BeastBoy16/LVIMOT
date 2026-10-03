import numpy as np
from typing import Dict, List, Optional

from carla_factors import MarginalizationFactor
from carla_factor_graph import CarlaFactorGraph


class CarlaSlidingWindowEstimator:
    def __init__(
        self,
        window_size: int = 8,
        robust_loss: str = "cauchy",
        loss_scale: float = 1.0,
        max_lidar_factors_per_frame: int = 48,
        max_visual_factors_per_frame: int = 40,
        max_object_factors_per_frame: int = 24,
        max_objects: int = 40,
        optimizer_max_nfev: int = 10,
    ):
        self.window_size = int(window_size)
        self.robust_loss = robust_loss
        self.loss_scale = loss_scale
        self.max_lidar_factors_per_frame = int(max_lidar_factors_per_frame)
        self.max_visual_factors_per_frame = int(max_visual_factors_per_frame)
        self.max_object_factors_per_frame = int(max_object_factors_per_frame)
        self.max_objects = int(max_objects)
        self.optimizer_max_nfev = max(2, int(optimizer_max_nfev))

        self.factor_graph = CarlaFactorGraph(
            robust_loss=robust_loss,
            loss_scale=loss_scale,
        )

        self.estimated_poses: Dict[
            int,
            np.ndarray,
        ] = {}

        self.estimated_velocities: Dict[
            int,
            np.ndarray,
        ] = {}

        self.estimated_objects: Dict[
            int,
            np.ndarray,
        ] = {}

        self.current_prior: Optional[
            MarginalizationFactor
        ] = None

    def add_keyframe_data(
        self,
        frame_id: int,
        timestamp: float,
        init_pose: np.ndarray,
        init_vel: np.ndarray,
        imu_preintegration: Optional[Dict] = None,
        planar_features: Optional[List[Dict]] = None,
        lidar_relative_pose: Optional[Dict] = None,
        visual_observations: Optional[List[Dict]] = None,
        object_observations: Optional[List[Dict]] = None,
        prev_frame_id: Optional[int] = None,
        stationary_pose_prior: Optional[np.ndarray] = None,
        stationary_velocity_prior: Optional[np.ndarray] = None,
        stationary_pose_weight: float = 20.0,
        stationary_velocity_weight: float = 20.0,
    ):
        self.factor_graph.add_keyframe(
            frame_id,
            timestamp,
        )

        self.estimated_poses[
            frame_id
        ] = np.asarray(
            init_pose,
            dtype=np.float64,
        ).copy()

        self.estimated_velocities[
            frame_id
        ] = np.asarray(
            init_vel,
            dtype=np.float64,
        ).copy()

        if len(self.factor_graph.keyframe_ids) == 1 and not self.factor_graph.pose_prior_factors:
            self.factor_graph.add_pose_prior(frame_id, np.asarray(init_pose, dtype=np.float64))

        if stationary_pose_prior is not None:
            self.factor_graph.add_pose_prior(
                frame_id,
                np.asarray(stationary_pose_prior, dtype=np.float64),
                weight_rotation=float(stationary_pose_weight),
                weight_position=float(stationary_pose_weight),
            )

        if stationary_velocity_prior is not None:
            self.factor_graph.add_velocity_prior(
                frame_id,
                np.asarray(stationary_velocity_prior, dtype=np.float64),
                weight=float(stationary_velocity_weight),
            )

        if (
            prev_frame_id is not None
            and imu_preintegration is not None
        ):
            dt = imu_preintegration.get(
                "dt",
                0.1,
            )

            dp = imu_preintegration.get(
                "delta_p",
                np.zeros(3),
            )

            dv = imu_preintegration.get(
                "delta_v",
                np.zeros(3),
            )

            dR = imu_preintegration.get(
                "delta_R",
                np.eye(3),
            )

            self.factor_graph.add_imu_factor(
                prev_frame_id,
                frame_id,
                dp,
                dv,
                dR,
                dt,
            )

        if (
            prev_frame_id is not None
            and lidar_relative_pose is not None
            and bool(lidar_relative_pose.get("accepted", False))
        ):
            self.factor_graph.add_lidar_relative_factor(
                prev_frame_id,
                frame_id,
                np.asarray(lidar_relative_pose.get("delta_R", np.eye(3)), dtype=np.float64),
                np.asarray(lidar_relative_pose.get("delta_t", np.zeros(3)), dtype=np.float64),
                weight_translation=float(lidar_relative_pose.get("weight_translation", 8.0)),
                weight_rotation=float(lidar_relative_pose.get("weight_rotation", 12.0)),
            )

        if planar_features:
            for feat in planar_features[: self.max_lidar_factors_per_frame]:
                self.factor_graph.add_lidar_factor(
                    frame_id=frame_id,
                    point_lidar=feat.get(
                        "point",
                        feat.get(
                            "center",
                            np.zeros(3),
                        ),
                    ),
                    plane_normal_w=feat.get(
                        "normal",
                        np.array(
                            [0.0, 0.0, 1.0]
                        ),
                    ),
                    plane_center_w=feat.get(
                        "center",
                        np.zeros(3),
                    ),
                    weight=3.0,
                )

        if visual_observations:
            for obs in visual_observations[: self.max_visual_factors_per_frame]:
                measurement_uv = obs.get(
                    "measurement_uv",
                    obs.get(
                        "uv",
                        np.zeros(2),
                    ),
                )

                K = obs.get(
                    "K",
                    np.eye(3),
                )

                landmark_3d_w = obs.get(
                    "landmark_3d_w",
                    obs.get(
                        "landmark",
                        np.zeros(3),
                    ),
                )

                weight = obs.get(
                    "weight",
                    1.0,
                )

                self.factor_graph.add_visual_factor(
                    frame_id=frame_id,
                    measurement_uv=measurement_uv,
                    K=K,
                    landmark_3d_w=landmark_3d_w,
                    weight=weight,
                    camera_translation_ego=obs.get("camera_translation_ego"),
                    camera_rotation_ego=obs.get("camera_rotation_ego"),
                    use_carla_camera_axes=bool(obs.get("use_carla_camera_axes", False)),
                )

        if object_observations:
            object_observations = list(object_observations)[: self.max_object_factors_per_frame]
            origin_frame = (
                self.factor_graph.keyframe_ids[0]
                if self.factor_graph.keyframe_ids
                else frame_id
            )

            origin_timestamp = (
                self.factor_graph.timestamps.get(
                    origin_frame,
                    timestamp,
                )
            )

            for obj in object_observations:
                tid = obj.get(
                    "track_id",
                    0,
                )

                if tid not in self.estimated_objects and len(self.estimated_objects) < self.max_objects:
                    p0 = np.asarray(
                        obj.get(
                            "location",
                            np.zeros(3),
                        ),
                        dtype=np.float64,
                    )

                    v0 = np.asarray(
                        obj.get(
                            "velocity",
                            np.zeros(3),
                        ),
                        dtype=np.float64,
                    )

                    self.estimated_objects[
                        tid
                    ] = np.hstack(
                        [p0, v0]
                    )

                if tid not in self.estimated_objects:
                    continue

                loc = obj.get(
                    "location",
                    np.zeros(3),
                )

                dt_obj = (
                    float(timestamp)
                    - float(origin_timestamp)
                )

                self.factor_graph.add_dynamic_object_factor(
                    frame_id=frame_id,
                    track_id=tid,
                    obs_local=loc,
                    dt_from_origin=dt_obj,
                    weight_pos=4.0,
                )

        if (
            len(
                self.factor_graph.keyframe_ids
            )
            >= self.window_size
        ):
            self.optimize_and_marginalize()

    def marginalize_oldest_state(
        self,
        optimized_vector: np.ndarray,
        mapping: Dict,
        optimized_residuals: Optional[np.ndarray] = None,
        optimized_jacobian: Optional[np.ndarray] = None,
    ):
        if len(
            self.factor_graph.keyframe_ids
        ) < 2:
            return

        oldest_fid = (
            self.factor_graph.keyframe_ids[0]
        )

        remaining_fids = (
            self.factor_graph.keyframe_ids[1:]
        )

        pose_start, pose_end = mapping[
            "ego_poses"
        ][oldest_fid]

        vel_start, vel_end = mapping[
            "ego_velocities"
        ][oldest_fid]

        eliminated_indices = np.concatenate(
            [
                np.arange(
                    pose_start,
                    pose_end,
                    dtype=int,
                ),
                np.arange(
                    vel_start,
                    vel_end,
                    dtype=int,
                ),
            ]
        )

        (
            x_remaining,
            H_prior,
            b_prior,
        ) = self.factor_graph.build_marginalization_system(
            optimized_vector,
            mapping,
            eliminated_indices,
            residuals=optimized_residuals,
            jacobian=optimized_jacobian,
        )

        self.current_prior = MarginalizationFactor(
            linearization_point=x_remaining,
            H_prior=H_prior,
            b_prior=b_prior,
        )

        remaining_keys = []

        for fid in remaining_fids:
            remaining_keys.append(("pose", fid))
            remaining_keys.append(("velocity", fid))

        for tid in sorted(
            self.estimated_objects.keys()
        ):
            remaining_keys.append(("object", tid))

        self.current_prior.state_keys = (
            remaining_keys
        )

        self.factor_graph.retain_keyframes(
            remaining_fids
        )

        self.factor_graph.marginalization_factors.clear()

        self.factor_graph.add_marginalization_factor(
            self.current_prior
        )

        self.estimated_poses = {
            fid: self.estimated_poses[fid]
            for fid in remaining_fids
            if fid in self.estimated_poses
        }

        self.estimated_velocities = {
            fid: self.estimated_velocities[fid]
            for fid in remaining_fids
            if fid in self.estimated_velocities
        }

        active_tids = {int(item["track_id"]) for item in self.factor_graph.dynamic_factors}
        self.estimated_objects = {
            tid: state for tid, state in self.estimated_objects.items()
            if tid in active_tids
        }

    def optimize_and_marginalize(self) -> Dict:
        res = self.factor_graph.optimize(
            initial_poses=self.estimated_poses,
            initial_velocities=self.estimated_velocities,
            initial_objects=self.estimated_objects,
            max_nfev=self.optimizer_max_nfev,
        )

        if "optimized_vector" not in res or not np.isfinite(float(res.get("cost", np.inf))):
            return res

        self.estimated_poses.update(
            res["optimized_poses"]
        )

        self.estimated_velocities.update(
            res["optimized_velocities"]
        )

        self.estimated_objects.update(
            res["optimized_objects"]
        )

        if (
            len(
                self.factor_graph.keyframe_ids
            )
            >= self.window_size
        ):
            self.marginalize_oldest_state(
                res["optimized_vector"],
                res["mapping"],
                res.get("optimized_residuals"),
                res.get("optimized_jacobian"),
            )

        return res

    def diagnostics(self) -> Dict:
        fg = self.factor_graph
        return {
            "window_keyframes": int(len(fg.keyframe_ids)),
            "pose_prior_factors": int(len(fg.pose_prior_factors)),
            "velocity_prior_factors": int(len(fg.velocity_prior_factors)),
            "imu_factors": int(len(fg.imu_factors)),
            "lidar_factors": int(len(fg.lidar_factors)),
            "lidar_relative_factors": int(len(fg.lidar_relative_factors)),
            "visual_factors": int(len(fg.visual_factors)),
            "dynamic_factors": int(len(fg.dynamic_factors)),
            "marginalization_factors": int(len(fg.marginalization_factors)),
            "object_states": int(len(self.estimated_objects)),
            "optimizer_max_nfev": int(self.optimizer_max_nfev),
        }

    def get_current_estimate(
        self,
        frame_id: int,
    ) -> Optional[Dict]:
        if frame_id not in self.estimated_poses:
            return None

        return {
            "frame_id": frame_id,
            "pose": self.estimated_poses[
                frame_id
            ].tolist(),
            "velocity": self.estimated_velocities.get(
                frame_id,
                np.zeros(3),
            ).tolist(),
            "objects": {
                k: v.tolist()
                for k, v in self.estimated_objects.items()
            },
        }