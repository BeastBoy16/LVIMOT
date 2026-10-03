import numpy as np
from typing import Dict, List, Optional, Tuple

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from scipy.optimize._numdiff import group_columns

from carla_factors import (
    PosePriorFactor,
    VelocityPriorFactor,
    IMUPreintegrationFactor,
    LiDARPlanarFactor,
    LiDARRelativePoseFactor,
    VisualReprojectionFactor,
    DynamicObjectFactor,
    MarginalizationFactor,
)


class CarlaFactorGraph:
    def __init__(
        self,
        robust_loss: str = "cauchy",
        loss_scale: float = 1.0,
    ):
        self.robust_loss = robust_loss
        self.loss_scale = loss_scale

        self.pose_prior_factors: List[Dict] = []
        self.velocity_prior_factors: List[Dict] = []
        self.imu_factors: List[Dict] = []
        self.lidar_factors: List[Dict] = []
        self.lidar_relative_factors: List[Dict] = []
        self.visual_factors: List[Dict] = []
        self.dynamic_factors: List[Dict] = []
        self.marginalization_factors: List[
            MarginalizationFactor
        ] = []

        self.keyframe_ids: List[int] = []
        self.timestamps: Dict[int, float] = {}

    def clear(self):
        self.pose_prior_factors.clear()
        self.velocity_prior_factors.clear()
        self.imu_factors.clear()
        self.lidar_factors.clear()
        self.lidar_relative_factors.clear()
        self.visual_factors.clear()
        self.dynamic_factors.clear()
        self.marginalization_factors.clear()
        self.keyframe_ids.clear()
        self.timestamps.clear()

    def add_keyframe(
        self,
        frame_id: int,
        timestamp: float,
    ):
        if frame_id not in self.keyframe_ids:
            self.keyframe_ids.append(frame_id)
            self.timestamps[frame_id] = float(timestamp)

    def add_pose_prior(
        self,
        frame_id: int,
        pose: np.ndarray,
        weight_rotation: float = 50.0,
        weight_position: float = 50.0,
    ):
        self.pose_prior_factors.append({
            "frame_id": int(frame_id),
            "factor": PosePriorFactor(pose, weight_rotation, weight_position),
        })

    def add_velocity_prior(
        self,
        frame_id: int,
        velocity: np.ndarray,
        weight: float = 20.0,
    ):
        self.velocity_prior_factors.append({
            "frame_id": int(frame_id),
            "factor": VelocityPriorFactor(velocity, weight),
        })

    def add_imu_factor(
        self,
        frame_i: int,
        frame_j: int,
        delta_p: np.ndarray,
        delta_v: np.ndarray,
        delta_R: np.ndarray,
        dt: float,
    ):
        factor = IMUPreintegrationFactor(
            delta_p,
            delta_v,
            delta_R,
            dt,
        )

        self.imu_factors.append(
            {
                "frame_i": frame_i,
                "frame_j": frame_j,
                "factor": factor,
            }
        )

    def add_lidar_factor(
        self,
        frame_id: int,
        point_lidar: np.ndarray,
        plane_normal_w: np.ndarray,
        plane_center_w: np.ndarray,
        weight: float = 5.0,
    ):
        factor = LiDARPlanarFactor(
            point_lidar,
            plane_normal_w,
            plane_center_w,
            weight=weight,
        )

        self.lidar_factors.append(
            {
                "frame_id": frame_id,
                "factor": factor,
            }
        )

    def add_lidar_relative_factor(
        self,
        frame_i: int,
        frame_j: int,
        delta_R: np.ndarray,
        delta_t: np.ndarray,
        weight_translation: float = 8.0,
        weight_rotation: float = 12.0,
    ):
        factor = LiDARRelativePoseFactor(
            delta_R,
            delta_t,
            weight_translation=weight_translation,
            weight_rotation=weight_rotation,
        )
        self.lidar_relative_factors.append(
            {
                "frame_i": int(frame_i),
                "frame_j": int(frame_j),
                "factor": factor,
            }
        )

    def add_visual_factor(
        self,
        frame_id: int,
        measurement_uv: np.ndarray,
        K: np.ndarray,
        landmark_3d_w: np.ndarray,
        weight: float = 1.0,
        camera_translation_ego: Optional[np.ndarray] = None,
        camera_rotation_ego: Optional[np.ndarray] = None,
        use_carla_camera_axes: bool = False,
    ):
        factor = VisualReprojectionFactor(
            measurement_uv,
            K,
            landmark_3d_w,
            weight=weight,
            camera_translation_ego=camera_translation_ego,
            camera_rotation_ego=camera_rotation_ego,
            use_carla_camera_axes=use_carla_camera_axes,
        )

        self.visual_factors.append(
            {
                "frame_id": frame_id,
                "factor": factor,
            }
        )

    def add_dynamic_object_factor(
        self,
        frame_id: int,
        track_id: int,
        obs_local: np.ndarray,
        dt_from_origin: float,
        weight_pos: float = 5.0,
    ):
        factor = DynamicObjectFactor(
            obs_local,
            dt_from_origin,
            weight_pos=weight_pos,
        )

        self.dynamic_factors.append(
            {
                "frame_id": frame_id,
                "track_id": track_id,
                "factor": factor,
            }
        )

    def add_marginalization_factor(
        self,
        factor: MarginalizationFactor,
    ):
        self.marginalization_factors.append(factor)

    @staticmethod
    def euler_to_rot(
        rpy: np.ndarray,
    ) -> np.ndarray:
        roll, pitch, yaw = rpy

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        return np.array(
            [
                [
                    cp * cy,
                    cy * sp * sr - sy * cr,
                    -cy * sp * cr - sy * sr,
                ],
                [
                    cp * sy,
                    sy * sp * sr + cy * cr,
                    -sy * sp * cr + cy * sr,
                ],
                [
                    sp,
                    -cp * sr,
                    cp * cr,
                ],
            ],
            dtype=np.float64,
        )

    def _pack_states(
        self,
        ego_poses: Dict[int, np.ndarray],
        ego_velocities: Dict[int, np.ndarray],
        object_states: Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, Dict]:
        vec_list = []

        mapping = {
            "ego_poses": {},
            "ego_velocities": {},
            "object_states": {},
        }

        offset = 0

        for fid in self.keyframe_ids:
            p = np.asarray(
                ego_poses.get(
                    fid,
                    np.zeros(6, dtype=np.float64),
                ),
                dtype=np.float64,
            )

            mapping["ego_poses"][fid] = (
                offset,
                offset + 6,
            )

            vec_list.append(p)
            offset += 6

            v = np.asarray(
                ego_velocities.get(
                    fid,
                    np.zeros(3, dtype=np.float64),
                ),
                dtype=np.float64,
            )

            mapping["ego_velocities"][fid] = (
                offset,
                offset + 3,
            )

            vec_list.append(v)
            offset += 3

        for tid in sorted(object_states.keys()):
            ostate = np.asarray(
                object_states[tid],
                dtype=np.float64,
            )

            if len(ostate) != 6:
                raise ValueError(
                    "Dynamic object state must have 6 elements."
                )

            mapping["object_states"][tid] = (
                offset,
                offset + 6,
            )

            vec_list.append(ostate)
            offset += 6

        if vec_list:
            flat_vector = np.concatenate(vec_list)
        else:
            flat_vector = np.empty(
                0,
                dtype=np.float64,
            )

        return flat_vector, mapping

    @staticmethod
    def _state_keys(
        mapping: Dict,
    ) -> List[Tuple]:
        keys = []

        pose_items = sorted(
            mapping["ego_poses"].items(),
            key=lambda item: item[1][0],
        )

        velocity_items = sorted(
            mapping["ego_velocities"].items(),
            key=lambda item: item[1][0],
        )

        object_items = sorted(
            mapping["object_states"].items(),
            key=lambda item: item[1][0],
        )

        for fid, (start, end) in pose_items:
            for _ in range(end - start):
                keys.append(("pose", fid))

        for fid, (start, end) in velocity_items:
            for _ in range(end - start):
                keys.append(("velocity", fid))

        for tid, (start, end) in object_items:
            for _ in range(end - start):
                keys.append(("object", tid))

        return keys

    @staticmethod
    def _indices_for_key(
        mapping: Dict,
        key: Tuple,
    ) -> np.ndarray:
        kind, identifier = key

        if kind == "pose":
            start, end = mapping["ego_poses"][identifier]
        elif kind == "velocity":
            start, end = mapping["ego_velocities"][identifier]
        elif kind == "object":
            start, end = mapping["object_states"][identifier]
        else:
            raise KeyError(f"Unknown state key: {kind}")

        return np.arange(
            start,
            end,
            dtype=int,
        )

    def _unpack_states(
        self,
        flat_vector: np.ndarray,
        mapping: Dict,
    ) -> Tuple[Dict, Dict, Dict]:
        ego_poses = {}
        ego_velocities = {}
        object_states = {}

        for fid, (s, e) in mapping["ego_poses"].items():
            ego_poses[fid] = flat_vector[s:e].copy()

        for fid, (s, e) in mapping["ego_velocities"].items():
            ego_velocities[fid] = flat_vector[s:e].copy()

        for tid, (s, e) in mapping["object_states"].items():
            object_states[tid] = flat_vector[s:e].copy()

        return (
            ego_poses,
            ego_velocities,
            object_states,
        )

    def _prior_indices(
        self,
        factor: MarginalizationFactor,
        mapping: Dict,
    ) -> Optional[np.ndarray]:
        keys = getattr(
            factor,
            "state_keys",
            None,
        )

        if keys is None:
            prior_dim = len(factor.x_lin)

            if prior_dim > sum(
                end - start
                for group in (
                    mapping["ego_poses"],
                    mapping["ego_velocities"],
                    mapping["object_states"],
                )
                for start, end in group.values()
            ):
                return None

            return np.arange(
                prior_dim,
                dtype=int,
            )

        indices = []

        for key in keys:
            if key[0] == "pose":
                if key[1] not in mapping["ego_poses"]:
                    return None
            elif key[0] == "velocity":
                if key[1] not in mapping["ego_velocities"]:
                    return None
            elif key[0] == "object":
                if key[1] not in mapping["object_states"]:
                    return None
            else:
                return None

            indices.extend(
                self._indices_for_key(
                    mapping,
                    key,
                ).tolist()
            )

        indices = np.asarray(
            indices,
            dtype=int,
        )

        if len(indices) != len(factor.x_lin):
            return None

        return indices

    def compute_all_residuals(
        self,
        flat_vector: np.ndarray,
        mapping: Dict,
    ) -> np.ndarray:
        (
            ego_poses,
            ego_velocities,
            object_states,
        ) = self._unpack_states(
            flat_vector,
            mapping,
        )

        residuals = []

        for prior_item in self.pose_prior_factors:
            fid = prior_item["frame_id"]
            if fid in ego_poses:
                r = prior_item["factor"].compute_residual(ego_poses[fid])
                residuals.extend(np.asarray(r).reshape(-1).tolist())

        for velocity_item in self.velocity_prior_factors:
            fid = velocity_item["frame_id"]
            if fid in ego_velocities:
                r = velocity_item["factor"].compute_residual(ego_velocities[fid])
                residuals.extend(np.asarray(r).reshape(-1).tolist())

        for imu_item in self.imu_factors:
            fi = imu_item["frame_i"]
            fj = imu_item["frame_j"]

            if fi in ego_poses and fj in ego_poses:
                pose_i = ego_poses[fi]
                pose_j = ego_poses[fj]

                v_i = ego_velocities[fi]
                v_j = ego_velocities[fj]

                R_i, p_i = (
                    self.euler_to_rot(pose_i[:3]),
                    pose_i[3:],
                )

                R_j, p_j = (
                    self.euler_to_rot(pose_j[:3]),
                    pose_j[3:],
                )

                r = imu_item["factor"].compute_residual(
                    R_i,
                    p_i,
                    v_i,
                    R_j,
                    p_j,
                    v_j,
                )

                residuals.extend(
                    np.asarray(r).reshape(-1).tolist()
                )

        for lidar_rel_item in self.lidar_relative_factors:
            fi = lidar_rel_item["frame_i"]
            fj = lidar_rel_item["frame_j"]
            if fi in ego_poses and fj in ego_poses:
                pose_i = ego_poses[fi]
                pose_j = ego_poses[fj]
                R_i = self.euler_to_rot(pose_i[:3])
                R_j = self.euler_to_rot(pose_j[:3])
                r = lidar_rel_item["factor"].compute_residual(
                    R_i, pose_i[3:], R_j, pose_j[3:]
                )
                residuals.extend(np.asarray(r).reshape(-1).tolist())

        for lidar_item in self.lidar_factors:
            fid = lidar_item["frame_id"]

            if fid in ego_poses:
                pose = ego_poses[fid]

                R, p = (
                    self.euler_to_rot(pose[:3]),
                    pose[3:],
                )

                r = lidar_item["factor"].compute_residual(
                    R,
                    p,
                )

                residuals.append(float(r))

        for vis_item in self.visual_factors:
            fid = vis_item["frame_id"]

            if fid in ego_poses:
                pose = ego_poses[fid]

                R_e, p_e = (
                    self.euler_to_rot(pose[:3]),
                    pose[3:],
                )

                R_c = R_e.T
                t_c = -R_c @ p_e

                r = vis_item["factor"].compute_residual(
                    R_c,
                    t_c,
                )

                residuals.extend(
                    np.asarray(r).reshape(-1).tolist()
                )

        for dyn_item in self.dynamic_factors:
            fid = dyn_item["frame_id"]
            tid = dyn_item["track_id"]

            if fid in ego_poses and tid in object_states:
                pose = ego_poses[fid]
                ostate = object_states[tid]

                R, p = (
                    self.euler_to_rot(pose[:3]),
                    pose[3:],
                )

                r = dyn_item["factor"].compute_residual(
                    R,
                    p,
                    ostate[:3],
                    ostate[3:],
                )

                residuals.extend(
                    np.asarray(r).reshape(-1).tolist()
                )

        for marg_factor in self.marginalization_factors:
            indices = self._prior_indices(
                marg_factor,
                mapping,
            )

            if indices is None:
                continue

            current_x = flat_vector[indices]

            r = marg_factor.compute_residual(
                current_x
            )

            residuals.extend(
                np.asarray(r).reshape(-1).tolist()
            )

        if not residuals:
            return np.zeros(
                1,
                dtype=np.float64,
            )

        return np.asarray(
            residuals,
            dtype=np.float64,
        )

    def _residual_structure(
        self,
        mapping: Dict,
    ) -> Tuple[int, List[np.ndarray]]:
        dependencies = []

        def pose_indices(fid):
            if fid not in mapping["ego_poses"]:
                return np.empty(0, dtype=int)
            return self._indices_for_key(
                mapping,
                ("pose", fid),
            )

        def velocity_indices(fid):
            if fid not in mapping["ego_velocities"]:
                return np.empty(0, dtype=int)
            return self._indices_for_key(
                mapping,
                ("velocity", fid),
            )

        def object_indices(tid):
            if tid not in mapping["object_states"]:
                return np.empty(0, dtype=int)
            return self._indices_for_key(
                mapping,
                ("object", tid),
            )

        row_count = 0

        for item in self.pose_prior_factors:
            fid = item["frame_id"]
            if fid in mapping["ego_poses"]:
                dependencies.append((row_count, row_count + 6, pose_indices(fid)))
                row_count += 6

        for item in self.velocity_prior_factors:
            fid = item["frame_id"]
            if fid in mapping["ego_velocities"]:
                dependencies.append((row_count, row_count + 3, velocity_indices(fid)))
                row_count += 3

        for item in self.imu_factors:
            fi = item["frame_i"]
            fj = item["frame_j"]

            if (
                fi in mapping["ego_poses"]
                and fj in mapping["ego_poses"]
            ):
                cols = np.unique(
                    np.concatenate(
                        [
                            pose_indices(fi),
                            velocity_indices(fi),
                            pose_indices(fj),
                            velocity_indices(fj),
                        ]
                    )
                )
                dependencies.append(
                    (row_count, row_count + 9, cols)
                )
                row_count += 9

        for item in self.lidar_relative_factors:
            fi = item["frame_i"]
            fj = item["frame_j"]
            if fi in mapping["ego_poses"] and fj in mapping["ego_poses"]:
                cols = np.unique(np.concatenate([pose_indices(fi), pose_indices(fj)]))
                dependencies.append((row_count, row_count + 6, cols))
                row_count += 6

        for item in self.lidar_factors:
            fid = item["frame_id"]

            if fid in mapping["ego_poses"]:
                dependencies.append(
                    (
                        row_count,
                        row_count + 1,
                        pose_indices(fid),
                    )
                )
                row_count += 1

        for item in self.visual_factors:
            fid = item["frame_id"]

            if fid in mapping["ego_poses"]:
                dependencies.append(
                    (
                        row_count,
                        row_count + 2,
                        pose_indices(fid),
                    )
                )
                row_count += 2

        for item in self.dynamic_factors:
            fid = item["frame_id"]
            tid = item["track_id"]

            if (
                fid in mapping["ego_poses"]
                and tid in mapping["object_states"]
            ):
                cols = np.unique(
                    np.concatenate(
                        [
                            pose_indices(fid),
                            object_indices(tid),
                        ]
                    )
                )

                dependencies.append(
                    (
                        row_count,
                        row_count + 3,
                        cols,
                    )
                )
                row_count += 3

        for factor in self.marginalization_factors:
            indices = self._prior_indices(
                factor,
                mapping,
            )

            if indices is not None:
                prior_dim = len(
                    factor.compute_residual(
                        factor.x_lin
                    )
                )

                dependencies.append(
                    (
                        row_count,
                        row_count + prior_dim,
                        indices,
                    )
                )

                row_count += prior_dim

        return row_count, dependencies

    def build_jacobian_sparsity(
        self,
        mapping: Dict,
        state_size: int,
    ) -> csr_matrix:
        row_count, dependencies = (
            self._residual_structure(mapping)
        )

        sparsity = lil_matrix(
            (row_count, state_size),
            dtype=np.int8,
        )

        for row_start, row_end, cols in dependencies:
            if len(cols) == 0:
                continue

            sparsity[
                row_start:row_end,
                cols,
            ] = 1

        return sparsity.tocsr()

    def numerical_jacobian(
        self,
        flat_vector: np.ndarray,
        mapping: Dict,
        relative_step: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(
            flat_vector,
            dtype=np.float64,
        )

        r0 = self.compute_all_residuals(
            x,
            mapping,
        )

        m = len(r0)
        n = len(x)

        sparsity = self.build_jacobian_sparsity(
            mapping,
            n,
        )

        groups = group_columns(
            sparsity,
        )

        J = np.zeros(
            (m, n),
            dtype=np.float64,
        )

        for group_id in range(
            int(np.max(groups)) + 1
            if len(groups)
            else 0
        ):
            columns = np.flatnonzero(
                groups == group_id
            )

            if len(columns) == 0:
                continue

            xp = x.copy()
            xm = x.copy()
            steps = {}

            for index in columns:
                step = relative_step * max(
                    1.0,
                    abs(x[index]),
                )

                xp[index] += step
                xm[index] -= step
                steps[index] = step

            rp = self.compute_all_residuals(
                xp,
                mapping,
            )

            rm = self.compute_all_residuals(
                xm,
                mapping,
            )

            for index in columns:
                step = steps[index]

                row_indices = (
                    sparsity[:, index]
                    .nonzero()[0]
                )

                if len(row_indices):
                    J[
                        row_indices,
                        index,
                    ] = (
                        rp[row_indices]
                        - rm[row_indices]
                    ) / (2.0 * step)

        return r0, J

    def build_marginalization_system(
        self,
        flat_vector: np.ndarray,
        mapping: Dict,
        eliminated_indices: np.ndarray,
        residuals: Optional[np.ndarray] = None,
        jacobian: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        # Reuse the optimizer's final residual/Jacobian when available.
        if residuals is None or jacobian is None:
            r, J = self.numerical_jacobian(
                flat_vector,
                mapping,
            )
        else:
            r = np.asarray(
                residuals,
                dtype=np.float64,
            ).reshape(-1)

            if hasattr(jacobian, "toarray"):
                J = jacobian.toarray()
            else:
                J = np.asarray(
                    jacobian,
                    dtype=np.float64,
                )

            if J.ndim != 2 or J.shape[1] != len(flat_vector):
                raise ValueError(
                    "Cached Jacobian does not match the current state vector."
                )

            if J.shape[0] != len(r):
                raise ValueError(
                    "Cached Jacobian and residual dimensions do not match."
                )

        all_indices = np.arange(
            len(flat_vector)
        )

        keep_mask = np.ones(
            len(flat_vector),
            dtype=bool,
        )

        keep_mask[
            np.asarray(
                eliminated_indices,
                dtype=int,
            )
        ] = False

        remaining_indices = all_indices[
            keep_mask
        ]

        m = np.asarray(
            eliminated_indices,
            dtype=int,
        )

        r_idx = remaining_indices

        # Form only the blocks required by the Schur complement.
        Jm = J[:, m]
        Jr = J[:, r_idx]

        Hmm = Jm.T @ Jm
        Hmr = Jm.T @ Jr
        Hrm = Hmr.T
        Hrr = Jr.T @ Jr

        gm = Jm.T @ r
        gr = Jr.T @ r

        diagonal = np.diag(Hmm)

        if len(diagonal):
            scale = max(
                1.0,
                float(
                    np.max(
                        np.abs(
                            diagonal
                        )
                    )
                ),
            )
        else:
            scale = 1.0

        regularization = (
            1e-8 * scale
        )

        Hmm_reg = Hmm + (
            regularization
            * np.eye(
                len(m),
                dtype=np.float64,
            )
        )

        try:
            Hmm_inv_Hmr = np.linalg.solve(
                Hmm_reg,
                Hmr,
            )

            Hmm_inv_gm = np.linalg.solve(
                Hmm_reg,
                gm,
            )
        except np.linalg.LinAlgError:
            Hmm_inv_Hmr = (
                np.linalg.pinv(
                    Hmm_reg
                )
                @ Hmr
            )

            Hmm_inv_gm = (
                np.linalg.pinv(
                    Hmm_reg
                )
                @ gm
            )

        H_prior = (
            Hrr
            - Hrm @ Hmm_inv_Hmr
        )

        b_prior = (
            gr
            - Hrm @ Hmm_inv_gm
        )

        H_prior = 0.5 * (
            H_prior + H_prior.T
        )

        x_remaining = flat_vector[
            remaining_indices
        ].copy()

        return (
            x_remaining,
            H_prior,
            b_prior,
        )

    def retain_keyframes(
        self,
        remaining_fids: List[int],
    ):
        remaining_set = set(
            remaining_fids
        )

        self.pose_prior_factors = [
            item for item in self.pose_prior_factors
            if item["frame_id"] in remaining_set
        ]

        self.velocity_prior_factors = [
            item for item in self.velocity_prior_factors
            if item["frame_id"] in remaining_set
        ]

        self.imu_factors = [
            item
            for item in self.imu_factors
            if (
                item["frame_i"] in remaining_set
                and item["frame_j"] in remaining_set
            )
        ]

        self.lidar_relative_factors = [
            item
            for item in self.lidar_relative_factors
            if (
                item["frame_i"] in remaining_set
                and item["frame_j"] in remaining_set
            )
        ]

        self.lidar_factors = [
            item
            for item in self.lidar_factors
            if item["frame_id"] in remaining_set
        ]

        self.visual_factors = [
            item
            for item in self.visual_factors
            if item["frame_id"] in remaining_set
        ]

        self.dynamic_factors = [
            item
            for item in self.dynamic_factors
            if item["frame_id"] in remaining_set
        ]

        self.keyframe_ids = [
            fid
            for fid in self.keyframe_ids
            if fid in remaining_set
        ]

        self.timestamps = {
            fid: self.timestamps[fid]
            for fid in self.keyframe_ids
            if fid in self.timestamps
        }

    def optimize(
        self,
        initial_poses: Dict[int, np.ndarray],
        initial_velocities: Dict[int, np.ndarray],
        initial_objects: Optional[
            Dict[int, np.ndarray]
        ] = None,
        max_nfev: int = 25,
    ) -> Dict:
        initial_objects = (
            initial_objects or {}
        )

        x0, mapping = self._pack_states(
            initial_poses,
            initial_velocities,
            initial_objects,
        )

        if len(x0) == 0:
            return {
                "success": False,
                "cost": 0.0,
                "optimized_poses": {},
                "optimized_velocities": {},
                "optimized_objects": {},
            }

        def loss_func(x):
            return self.compute_all_residuals(
                x,
                mapping,
            )

        jac_sparsity = (
            self.build_jacobian_sparsity(
                mapping,
                len(x0),
            )
        )

        res = least_squares(
            loss_func,
            x0,
            jac="2-point",
            jac_sparsity=jac_sparsity,
            method="trf",
            loss=self.robust_loss,
            f_scale=self.loss_scale,
            ftol=1e-3,
            xtol=1e-3,
            gtol=1e-3,
            max_nfev=max_nfev,
        )

        (
            opt_poses,
            opt_vels,
            opt_objs,
        ) = self._unpack_states(
            res.x,
            mapping,
        )

        return {
            "success": bool(res.success),
            "status": int(res.status),
            "cost": float(res.cost),
            "optimized_poses": opt_poses,
            "optimized_velocities": opt_vels,
            "optimized_objects": opt_objs,
            "optimized_vector": res.x,
            "mapping": mapping,
            "optimized_residuals": np.asarray(
                res.fun, dtype=np.float64
            ).copy(),
            "optimized_jacobian": res.jac.copy()
            if hasattr(res.jac, "copy")
            else np.asarray(
                res.jac, dtype=np.float64
            ).copy(),
        }