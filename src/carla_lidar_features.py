import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

from scipy.spatial import cKDTree


class CarlaLiDARPlanarFeatureExtractor:
    """Fast planar feature extractor.

    The original implementation performed one Python covariance/eigendecomposition
    per downsampled LiDAR point.  This version keeps the same geometric test but
    computes fixed-size local neighborhoods and their 3x3 covariance matrices in
    NumPy batches.  Raw LiDAR data and saved per-frame point clouds are unchanged.
    """

    def __init__(
        self,
        voxel_size=0.20,
        search_radius=0.50,
        min_neighbors=10,
        planarity_threshold=0.15,
        query_batch_size=4096,
        max_neighbors=24,
        max_candidate_points=None,
        max_query_points=None,
    ):
        self.voxel_size = float(voxel_size)
        self.search_radius = float(search_radius)
        self.min_neighbors = int(min_neighbors)
        self.planarity_threshold = float(planarity_threshold)
        self.query_batch_size = max(256, int(query_batch_size))
        self.max_neighbors = max(self.min_neighbors, int(max_neighbors))
        self.max_candidate_points = (
            None if max_candidate_points in (None, 0) else max(256, int(max_candidate_points))
        )
        self.max_query_points = (
            self.max_candidate_points
            if max_query_points in (None, 0)
            else max(256, int(max_query_points))
        )

    def voxel_downsample(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("Expected point cloud with shape (N, >=3)")
        xyz = points[:, :3]
        if o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            return np.asarray(pcd.voxel_down_sample(voxel_size=self.voxel_size).points)
        coords = np.floor(xyz / self.voxel_size).astype(np.int64)
        _, unique_indices = np.unique(coords, axis=0, return_index=True)
        return xyz[np.sort(unique_indices)]

    def _bounded_candidates(self, xyz):
        limit = self.max_query_points
        if limit is None or len(xyz) <= limit:
            return xyz
        # Query only a deterministic, FoV-wide subset, but keep the complete
        # voxelized cloud as KD-tree support.  V12 thinned the support cloud
        # itself, which reduced neighborhood density and wasted work rebuilding
        # geometry at points that never enter the temporal/factor back-end.
        ids = np.linspace(0, len(xyz) - 1, limit, dtype=np.int64)
        return xyz[ids]

    def extract(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("Expected point cloud with shape (N, >=3)")
        xyz = points[:, :3]
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz) == 0:
            return []
        support_xyz = self.voxel_downsample(xyz)
        if len(support_xyz) == 0:
            return []
        query_xyz = self._bounded_candidates(support_xyz)
        if len(query_xyz) == 0:
            return []

        # Build neighborhoods from the dense voxelized support cloud while
        # evaluating covariance only at bounded query points.
        tree = cKDTree(support_xyz)
        features = []
        k = min(self.max_neighbors, len(support_xyz))

        for start in range(0, len(query_xyz), self.query_batch_size):
            stop = min(start + self.query_batch_size, len(query_xyz))
            query_pts = query_xyz[start:stop]
            distances, indices = tree.query(
                query_pts,
                k=k,
                distance_upper_bound=self.search_radius,
                workers=-1,
            )
            if k == 1:
                distances = distances[:, None]
                indices = indices[:, None]

            valid = np.isfinite(distances) & (indices < len(support_xyz))
            counts = valid.sum(axis=1)
            keep_rows = counts >= self.min_neighbors
            if not np.any(keep_rows):
                continue

            row_ids = np.flatnonzero(keep_rows)
            idx = indices[keep_rows].copy()
            mask = valid[keep_rows]
            idx[~mask] = 0
            neigh = support_xyz[idx]
            weights = mask[..., None].astype(np.float64)
            counts_f = counts[keep_rows].astype(np.float64)

            centroid = (neigh * weights).sum(axis=1) / counts_f[:, None]
            centered = (neigh - centroid[:, None, :]) * weights
            covariance = np.einsum("nki,nkj->nij", centered, centered)
            covariance /= np.maximum(counts_f - 1.0, 1.0)[:, None, None]

            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            total = eigenvalues.sum(axis=1)
            usable = total > 1e-8
            if not np.any(usable):
                continue
            l1 = eigenvalues[:, 0]
            l2 = eigenvalues[:, 1]
            planarity = np.zeros_like(total)
            planarity[usable] = (l2[usable] - l1[usable]) / total[usable]
            selected = usable & (planarity > self.planarity_threshold)
            if not np.any(selected):
                continue

            selected_ids = np.flatnonzero(selected)
            source_points = query_pts[row_ids[selected_ids]]
            selected_centroids = centroid[selected_ids]
            selected_evals = eigenvalues[selected_ids]
            selected_planarity = planarity[selected_ids]
            selected_normals = eigenvectors[selected_ids, :, 0]
            normal_norms = np.linalg.norm(selected_normals, axis=1)
            selected_normals = selected_normals / np.maximum(normal_norms[:, None], 1e-9)
            surface_variation = selected_evals[:, 0] / np.maximum(selected_evals.sum(axis=1), 1e-12)

            for i in range(len(selected_ids)):
                features.append(
                    {
                        "point": source_points[i].copy(),
                        "center": selected_centroids[i].copy(),
                        "normal": selected_normals[i].copy(),
                        "planarity": float(selected_planarity[i]),
                        "surface_variation": float(surface_variation[i]),
                        "eigenvalues": selected_evals[i].copy(),
                    }
                )
        return features
