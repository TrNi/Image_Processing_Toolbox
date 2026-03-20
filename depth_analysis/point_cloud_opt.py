"""
point_cloud_opt.py
==================
Point-cloud consistency analysis for multi-model depth evaluation.

The key entry point for typical use is :func:`get_point_cloud_errors`, which
accepts a stack of depth maps, converts them to 3-D point clouds and measures
pairwise ICP alignment error w.r.t. a median-depth reference cloud.

For more advanced use, instantiate :class:`PointCloudConsistencyAnalyzer`
directly to access pairwise ICP, Generalised-ICP or global multi-way alignment.

Usage
-----
::

    from depth_analysis.point_cloud_opt import get_point_cloud_errors
    err_maps = get_point_cloud_errors(depth_stack, K_inv)
    # err_maps: np.ndarray, shape (K, H, W)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from typing import List, Dict, Union, Tuple

try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    _OPEN3D_AVAILABLE = False

from depth_analysis.depth_reproj_eval import px_to_camera


class PointCloudConsistencyAnalyzer:
    """Algorithms for 3-D point-cloud consistency analysis.

    Parameters
    ----------
    point_clouds : list of np.ndarray
        Each element is an ``(N, 3)`` array representing a point cloud.
        All clouds must be expressed in the same reference frame (e.g. left
        camera coordinates).
    verbose : bool
        Print progress messages during computation.
    """

    def __init__(self, point_clouds: List[np.ndarray], verbose: bool = True):
        if not _OPEN3D_AVAILABLE:
            raise ImportError(
                "open3d is required for PointCloudConsistencyAnalyzer. "
                "Install it with: pip install open3d"
            )
        if not isinstance(point_clouds, list) or len(point_clouds) < 2:
            raise ValueError("Input must be a list of at least two point clouds.")

        self.point_clouds_np = point_clouds
        self.point_clouds_o3d = [self._to_o3d_pcd(pc) for pc in point_clouds]
        self.num_clouds = len(point_clouds)
        self.verbose = verbose

    @staticmethod
    def _to_o3d_pcd(pcd_np: np.ndarray) -> "o3d.geometry.PointCloud":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        return pcd

    def compute_pairwise_gicp_error(
        self,
        reference_idx: int = 0,
        voxel_size: float = 0.05,
        max_correspondence_distance: float = 0.07,
        icp_criteria=None,
    ) -> Dict[str, Union[Dict, np.ndarray]]:
        """Pairwise Generalised-ICP (GICP) alignment error.

        ICP is run on down-sampled clouds for speed; the fitted transformation
        is then applied to the full-resolution cloud to compute per-point
        distances against the full-resolution target.

        Parameters
        ----------
        reference_idx : int
            Index of the reference (target) point cloud.
        voxel_size : float
            Voxel size for downsampling.
        max_correspondence_distance : float
            Maximum ICP correspondence distance.
        icp_criteria : optional
            Open3D ICPConvergenceCriteria.  Defaults to a sensible setting.

        Returns
        -------
        dict
            ``'aggr_errors'`` — per-cloud fitness and inlier RMSE.
            ``'error_maps'`` — per-point distance arrays.
        """
        if icp_criteria is None:
            icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
            )

        if not 0 <= reference_idx < self.num_clouds:
            raise ValueError(f"reference_idx must be between 0 and {self.num_clouds - 1}.")

        results = {"aggr_errors": {}, "error_maps": {}}
        pcd_target_full = self.point_clouds_o3d[reference_idx]
        pcd_target_down = pcd_target_full.voxel_down_sample(voxel_size)

        if self.verbose:
            print(f"GICP: using cloud {reference_idx} as reference "
                  f"(voxel_size={voxel_size})")

        for i in range(self.num_clouds):
            if i == reference_idx:
                continue
            pcd_source_full = self.point_clouds_o3d[i]
            pcd_source_down = pcd_source_full.voxel_down_sample(voxel_size)

            search_param = o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2, max_nn=30
            )
            try:
                pcd_source_down.estimate_covariances(search_param)
                pcd_target_down.estimate_covariances(search_param)
            except RuntimeError as err:
                print(f"  Warning: covariance estimation failed for pair "
                      f"({i}, {reference_idx}): {err}")
                continue

            estimation_method = (
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
            )
            reg_result = o3d.pipelines.registration.registration_icp(
                pcd_source_down, pcd_target_down,
                max_correspondence_distance, np.identity(4),
                estimation_method, icp_criteria,
            )

            results["aggr_errors"][i] = {
                "fitness": reg_result.fitness,
                "inlier_rmse": reg_result.inlier_rmse,
            }

            pcd_source_transformed = o3d.geometry.PointCloud(pcd_source_full)
            pcd_source_transformed.transform(reg_result.transformation)
            distances = pcd_source_transformed.compute_point_cloud_distance(pcd_target_full)
            results["error_maps"][i] = np.asarray(distances)

        return results

    def compute_pairwise_icp_error(
        self,
        reference_idx: int = 0,
        voxel_size: float = 0.05,
        max_correspondence_distance: float = 0.05,
        estimation_method=None,
        icp_criteria=None,
    ) -> Dict[str, Union[Dict, np.ndarray]]:
        """Pairwise point-to-point ICP alignment error.

        Parameters
        ----------
        reference_idx : int
            Index of the reference point cloud.
        voxel_size : float
            Voxel size for downsampling.
        max_correspondence_distance : float
            Maximum ICP correspondence distance.
        estimation_method : optional
            Open3D TransformationEstimation.  Defaults to PointToPoint.
        icp_criteria : optional
            Open3D ICPConvergenceCriteria.

        Returns
        -------
        dict
            ``'aggr_errors'`` and ``'error_maps'``.
        """
        if estimation_method is None:
            estimation_method = (
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
            )
        if icp_criteria is None:
            icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30
            )

        if not 0 <= reference_idx < self.num_clouds:
            raise ValueError(f"reference_idx must be between 0 and {self.num_clouds - 1}.")

        results = {"aggr_errors": {}, "error_maps": {}}
        pcd_target = self.point_clouds_o3d[reference_idx]
        pcd_target_down = pcd_target.voxel_down_sample(voxel_size)

        if self.verbose:
            print(f"ICP: using cloud {reference_idx} as reference.")

        for i in range(self.num_clouds):
            if i == reference_idx:
                continue
            pcd_source_full = self.point_clouds_o3d[i]
            pcd_source_down = pcd_source_full.voxel_down_sample(voxel_size)

            reg_result = o3d.pipelines.registration.registration_icp(
                pcd_source_down, pcd_target_down,
                max_correspondence_distance, np.identity(4),
                estimation_method, icp_criteria,
            )
            results["aggr_errors"][i] = {
                "fitness": reg_result.fitness,
                "inlier_rmse": reg_result.inlier_rmse,
            }

            pcd_source_transformed = o3d.geometry.PointCloud(pcd_source_full)
            pcd_source_transformed.transform(reg_result.transformation)
            distances = pcd_source_transformed.compute_point_cloud_distance(pcd_target)
            results["error_maps"][i] = np.asarray(distances)

        return results

    def compute_global_alignment_error(
        self,
        voxel_size: float = 0.05,
        max_correspondence_distance_pairwise: float = 0.07,
        max_correspondence_distance_refine: float = 0.05,
    ) -> Tuple[List[np.ndarray], "o3d.geometry.PointCloud", Dict[str, np.ndarray]]:
        """Global multi-way point-cloud alignment and consensus error.

        Builds a pose graph from pairwise ICP registrations, optimises it
        globally, merges all aligned clouds into a consensus model, then
        computes per-point deviations of each cloud from that model.

        Parameters
        ----------
        voxel_size : float
            Voxel size for downsampling.
        max_correspondence_distance_pairwise : float
            ICP correspondence distance for the pairwise initialisation.
        max_correspondence_distance_refine : float
            Correspondence distance for final refinement.

        Returns
        -------
        aligned_pcds_np : list of np.ndarray
            Globally aligned point clouds as NumPy arrays.
        pcd_consensus : o3d.geometry.PointCloud
            Merged consensus model.
        error_maps : dict
            Per-cloud distance arrays from the consensus model.
        """
        if self.verbose:
            print("Starting global alignment ...")

        pcds_down = [pcd.voxel_down_sample(voxel_size) for pcd in self.point_clouds_o3d]

        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

        for i in range(self.num_clouds):
            for j in range(i + 1, self.num_clouds):
                if self.verbose:
                    print(f"  Pairwise registration: cloud {i} ↔ cloud {j}")
                T_icp, info_icp = self._pairwise_register(
                    pcds_down[i], pcds_down[j],
                    voxel_size, max_correspondence_distance_pairwise,
                )
                if j == i + 1:
                    odometry = np.dot(T_icp, odometry)
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                    )
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            i, j, T_icp, info_icp, uncertain=False
                        )
                    )
                else:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            i, j, T_icp, info_icp, uncertain=True
                        )
                    )

        if self.verbose:
            print("Optimising pose graph ...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance_refine,
            edge_prune_threshold=0.25,
            reference_node=0,
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

        if self.verbose:
            print("Building consensus model ...")
        pcd_consensus = o3d.geometry.PointCloud()
        aligned_pcds = []
        for i in range(self.num_clouds):
            pcd_aligned = o3d.geometry.PointCloud(self.point_clouds_o3d[i])
            pcd_aligned.transform(pose_graph.nodes[i].pose)
            aligned_pcds.append(pcd_aligned)
            pcd_consensus += pcd_aligned

        pcd_consensus_down = pcd_consensus.voxel_down_sample(voxel_size)
        error_maps = {}
        for i in range(self.num_clouds):
            distances = aligned_pcds[i].compute_point_cloud_distance(pcd_consensus_down)
            error_maps[i] = np.asarray(distances)

        aligned_pcds_np = [np.asarray(pcd.points) for pcd in aligned_pcds]
        return aligned_pcds_np, pcd_consensus, error_maps

    def _pairwise_register(
        self,
        source: "o3d.geometry.PointCloud",
        target: "o3d.geometry.PointCloud",
        voxel_size: float,
        max_correspondence_distance: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not target.has_normals():
            target.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
            )
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance, icp_result.transformation
        )
        o3d.utility.set_verbosity_level(
            o3d.utility.VerbosityLevel.Info if self.verbose else o3d.utility.VerbosityLevel.Error
        )
        return icp_result.transformation, info


def get_point_cloud_errors(depth_data: np.ndarray, K_inv: np.ndarray,
                           voxel_size: float = 0.025) -> np.ndarray:
    """Compute pairwise ICP error maps for a stack of depth maps.

    A median reference cloud is prepended to the stack and used as the ICP
    target, ensuring the error is relative to the ensemble consensus rather
    than a single model.

    Parameters
    ----------
    depth_data : np.ndarray
        Depth maps, shape ``(K, H, W)``.  At least 2 maps required.
    K_inv : np.ndarray
        3×3 inverse intrinsic matrix.
    voxel_size : float
        Voxel size for ICP downsampling.

    Returns
    -------
    np.ndarray
        Per-pixel ICP error maps, shape ``(K, H, W)``.
    """
    if not _OPEN3D_AVAILABLE:
        raise ImportError(
            "open3d is required for get_point_cloud_errors. "
            "Install it with: pip install open3d"
        )

    assert depth_data.shape[0] >= 2, (
        f"At least 2 depth maps are required, got {depth_data.shape[0]}."
    )

    median_depth = np.median(depth_data, axis=0)
    data_with_ref = np.concatenate((median_depth[None, ...], depth_data), axis=0)
    num_maps = data_with_ref.shape[0]
    H, W = depth_data.shape[1], depth_data.shape[2]

    pc_list = [
        px_to_camera(data_with_ref[i], K_inv).reshape(-1, 3)
        for i in range(num_maps)
    ]

    analyzer = PointCloudConsistencyAnalyzer(pc_list, verbose=True)
    icp_results = analyzer.compute_pairwise_icp_error(
        reference_idx=0, voxel_size=voxel_size
    )

    print("\nICP Errors (Fitness / Inlier RMSE) w.r.t. median depth:")
    for i, errs in icp_results["aggr_errors"].items():
        print(f"  Cloud {i}: Fitness={errs['fitness']:.4f}, RMSE={errs['inlier_rmse']:.4f}")

    err_maps = np.zeros((num_maps - 1, H, W), dtype=np.float32)
    for i, err_map in enumerate(icp_results["error_maps"].values()):
        err_maps[i] = err_map.reshape(H, W)

    return err_maps


if __name__ == "__main__":
    print("point_cloud_opt module loaded successfully.")
    if _OPEN3D_AVAILABLE:
        print("  open3d is available — full functionality enabled.")
    else:
        print("  open3d not found — install with: pip install open3d")
