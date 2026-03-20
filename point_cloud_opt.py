import open3d as o3d
import numpy as np
from typing import List, Dict, Union, Tuple
from depth_reproj_eval import px_to_camera

class PointCloudConsistencyAnalyzer:
    """
    Implements algorithms for 3D point cloud consistency analysis.    
    """
    def __init__(self, point_clouds: List[np.ndarray], verbose: bool = True):
        """
        Initializes the analyzer with a list of point clouds.

        Args:
            point_clouds (List[np.ndarray]): A list where each element is an (N, 3)
                                            numpy array representing a point cloud PC_k.
                                            Different list elements can have different sizes.
                                            All point clouds are assumed to be in the
                                            same reference frame (e.g., the left camera).
            verbose (bool): If True, prints status updates during computation.
        """
        if not isinstance(point_clouds, list) or len(point_clouds) < 2:
            raise ValueError("Input must be a list of at least two point clouds.")
        
        self.point_clouds_np = point_clouds
        self.point_clouds_o3d = [self._to_o3d_pcd(pc) for pc in point_clouds]
        self.num_clouds = len(point_clouds)
        self.verbose = verbose

    def _to_o3d_pcd(self, pcd_np: np.ndarray) -> o3d.geometry.PointCloud:
        """Converts a numpy array to an Open3D PointCloud object."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        return pcd

    def compute_pairwise_gicp_error(
        self,
        reference_idx: int = 0,
        voxel_size: float = 0.05,
        max_correspondence_distance: float = 0.07,
        icp_criteria: o3d.pipelines.registration.ICPConvergenceCriteria = \
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=50)
    ) -> Dict[str, Union[Dict, np.ndarray]]:
        """
        Implements a robust pairwise point cloud registration using Generalized-ICP (GICP).

        ICP is performed on downsampled point clouds for efficiency. The final
        transformation is then applied to the original full-resolution point
        cloud to compute a full-resolution error map.

        Args:
            reference_idx (int): The index of the point cloud to be used as the reference.
            voxel_size (float): The voxel size for downsampling.
            max_correspondence_distance (float): Max ICP correspondence distance.
            icp_criteria: Convergence criteria for the ICP algorithm.

        Returns:
            Dict: A dictionary containing the global errors ('aggr_errors') and
                the full-resolution per-point error maps ('error_maps').
        """
        if not 0 <= reference_idx < self.num_clouds:
            raise ValueError(f"reference_idx must be between 0 and {self.num_clouds-1}.")

        results = {
            "aggr_errors": {},
            "error_maps": {}
        }
        
        # Keep the original full-resolution target for final error calculation
        pcd_target_full = self.point_clouds_o3d[reference_idx]
        
        # Downsample the target for fast registration
        pcd_target_down = pcd_target_full.voxel_down_sample(voxel_size)

        if self.verbose:
            print(f"Using point cloud {reference_idx} as reference. Downsampling with voxel_size={voxel_size}")

        for i in range(self.num_clouds):
            if i == reference_idx:
                continue

            if self.verbose:
                print(f"  Registering point cloud {i} to reference {reference_idx} using GICP...")

            # Keep the original full-resolution source
            pcd_source_full = self.point_clouds_o3d[i]
            # Downsample the source for fast registration
            pcd_source_down = pcd_source_full.voxel_down_sample(voxel_size)

            # --- Use Generalized-ICP ---
            # GICP requires point covariance matrices.
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
            try:
                pcd_source_down.estimate_covariances(search_param)
                pcd_target_down.estimate_covariances(search_param)
            except RuntimeError as e:
                print(f"Warning: Could not estimate covariances for pair ({i}, {reference_idx}). Skipping. Error: {e}")
                continue

            estimation_method = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
            
            # Perform ICP registration on DOWN-SAMPLED clouds
            reg_result = o3d.pipelines.registration.registration_icp(
                pcd_source_down, pcd_target_down, max_correspondence_distance,
                np.identity(4), estimation_method, icp_criteria
            )            

            # Store global error metrics (from the downsampled registration)
            results["aggr_errors"][i] = {
                "fitness": reg_result.fitness,
                "inlier_rmse": reg_result.inlier_rmse
            }
            
            # Apply transform to FULL resolution cloud 
            # Create a copy to avoid modifying the original object
            pcd_source_transformed_full = o3d.geometry.PointCloud(pcd_source_full)
            pcd_source_transformed_full.transform(reg_result.transformation)
            
            # Compute per-point distances against the FULL resolution target
            distances = pcd_source_transformed_full.compute_point_cloud_distance(pcd_target_full)
            results["error_maps"][i] = np.asarray(distances)            

        return results


    def compute_pairwise_icp_error( 
        self,
        reference_idx: int = 0,        
        voxel_size: float = 0.05,
        max_correspondence_distance: float = 0.05,
        estimation_method: o3d.pipelines.registration.TransformationEstimation = \
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        icp_criteria: o3d.pipelines.registration.ICPConvergenceCriteria = \
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=30)
    ) -> Dict[str, Union[Dict, np.ndarray]]:
        """
        Implements pairwise point cloud registration error analysis.

        Each point cloud PC_k is registered to a reference point cloud PC_ref using ICP.
        It computes the global registration fitness and RMSE, and calculates a
        per-point error map by finding the distance from each transformed point in
        PC_k to its nearest neighbor in PC_ref.

        Args:
            reference_idx (int): The index of the point cloud in the input list
                                 to be used as the reference (target)
            voxel_size (float): The voxel size for downsampling.
            max_correspondence_distance (float): Maximum distance to search for
                                                 correspondences in ICP.
            estimation_method: The ICP estimation method (PointToPoint or PointToPlane).
            icp_criteria: Convergence criteria for the ICP algorithm.

        Returns:
            Dict: A dictionary containing:
                  'aggr_errors': A dictionary mapping each source cloud index to its
                                   ICP fitness and inlier RMSE.
                  'error_maps': A dictionary mapping each source cloud index
                                          to its per-point distance error map.
        """
        if not 0 <= reference_idx < self.num_clouds:
            raise ValueError(f"reference_idx must be between 0 and {self.num_clouds-1}.")

        results = {
            "aggr_errors": {},
            "error_maps": {}
        }
        
        pcd_target = self.point_clouds_o3d[reference_idx]
        pcd_target_down = pcd_target.voxel_down_sample(voxel_size)
        if self.verbose:
            print(f"Using point cloud {reference_idx} as the reference for pairwise ICP.")

        for i in range(self.num_clouds):
            if i == reference_idx:
                continue

            if self.verbose:
                print(f"  Registering point cloud {i} to reference {reference_idx}...")

            # pcd_source = self.point_clouds_o3d[i]
            # Keep the original full-resolution source
            pcd_source_full = self.point_clouds_o3d[i]
            # Downsample the source for fast registration
            pcd_source_down = pcd_source_full.voxel_down_sample(voxel_size)

            # Perform ICP registration
            reg_result = o3d.pipelines.registration.registration_icp(
                pcd_source_down, pcd_target_down, max_correspondence_distance,
                np.identity(4), estimation_method, icp_criteria
            )

            # Store global error metrics
            results["aggr_errors"][i] = {
                "fitness": reg_result.fitness,
                "inlier_rmse": reg_result.inlier_rmse
            }
            
            # Transform source point cloud with the estimated transformation
            # pcd_source_transformed = o3d.geometry.PointCloud(pcd_source)
            # pcd_source_transformed.transform(reg_result.transformation)
            pcd_source_transformed_full = o3d.geometry.PointCloud(pcd_source_full)
            pcd_source_transformed_full.transform(reg_result.transformation)
            
            # Compute per-point distances to the target
            distances = pcd_source_transformed_full.compute_point_cloud_distance(pcd_target)
            results["error_maps"][i] = np.asarray(distances)

        return results

    def compute_global_alignment_error(
        self,
        voxel_size: float = 0.05,
        max_correspondence_distance_pairwise: float = 0.07,
        max_correspondence_distance_refine: float = 0.05,
    ) -> Tuple[List[np.ndarray], o3d.geometry.PointCloud, Dict[str, np.ndarray]]:
        """
        Implements global point cloud alignment and outlier detection (Section 2.3.2).

        This method performs a multiway registration to align all point clouds
        into a globally consistent coordinate frame. It then merges them to create a
        consensus point cloud and calculates the per-point deviation of each
        aligned cloud from this consensus model.

        Args:
            voxel_size (float): Voxel size for downsampling before registration.
                                A larger value increases speed but reduces accuracy.
            max_correspondence_distance_pairwise (float): ICP correspondence distance for
                                                          initial pairwise alignment.
            max_correspondence_distance_refine (float): ICP correspondence distance for
                                                        the final refinement step.

        Returns:
            Tuple: A tuple containing:
                   - List[np.ndarray]: A list of the globally aligned point clouds.
                   - o3d.geometry.PointCloud: The final merged consensus point cloud.
                   - Dict[str, np.ndarray]: A dictionary mapping each cloud index to its
                                            per-point distance error map from the
                                            consensus cloud.
        """
        if self.verbose:
            print("Starting global alignment of all point clouds...")
            print(f"Downsampling with voxel size: {voxel_size}")
            
        pcds_down = [pcd.voxel_down_sample(voxel_size) for pcd in self.point_clouds_o3d]

        # 1. Pairwise registration to build the pose graph
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        
        for i in range(self.num_clouds):
            for j in range(i + 1, self.num_clouds):
                if self.verbose:
                    print(f"  Performing pairwise registration between cloud {i} and {j}...")
                
                transformation_icp, information_icp = self._pairwise_register(
                    pcds_down[i], pcds_down[j], voxel_size, max_correspondence_distance_pairwise
                )
                
                if j == i + 1: # Odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            i, j, transformation_icp, information_icp, uncertain=False
                        )
                    )
                else: # Loop closure
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            i, j, transformation_icp, information_icp, uncertain=True
                        )
                    )

        # 2. Global optimization
        if self.verbose:
            print("Optimizing the pose graph globally...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance_refine,
            edge_prune_threshold=0.25,
            reference_node=0)
        
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

        # 3. Create consensus model and calculate errors
        if self.verbose:
            print("Creating consensus model and calculating per-point errors...")
        pcd_consensus = o3d.geometry.PointCloud()
        aligned_pcds = []
        for i in range(self.num_clouds):
            pcd_aligned = o3d.geometry.PointCloud(self.point_clouds_o3d[i])
            pcd_aligned.transform(pose_graph.nodes[i].pose)
            aligned_pcds.append(pcd_aligned)
            pcd_consensus += pcd_aligned
        
        # Downsample the final consensus cloud to speed up nearest neighbor search
        pcd_consensus_down = pcd_consensus.voxel_down_sample(voxel_size)

        error_maps = {}
        for i in range(self.num_clouds):
            distances = aligned_pcds[i].compute_point_cloud_distance(pcd_consensus_down)
            error_maps[i] = np.asarray(distances)
            
        aligned_pcds_np = [np.asarray(pcd.points) for pcd in aligned_pcds]

        return aligned_pcds_np, pcd_consensus, error_maps

    def _pairwise_register(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        voxel_size: float,
        max_correspondence_distance: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Helper for pairwise registration, returning transformation and information matrix."""

        if not target.has_normals():
            if self.verbose:
                print("    -> Estimating normals for target cloud...")
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
            target.estimate_normals(search_param)
        
        
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        # A more refined ICP on original clouds can be done here if needed, but for
        # pose graph this is usually sufficient.
        
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance, icp_coarse.transformation
        )
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info if self.verbose else o3d.utility.VerbosityLevel.Error)
        return icp_coarse.transformation, information_icp


def get_point_cloud_errors(depth_data, K_inv):
    assert depth_data.shape[0]>=2, f"At least 2 depth maps are required for point cloud errors, currently supplied only {depth_data.shape[0]}"
    median_depth = np.median(depth_data, axis=0)    

    depth_data = np.concatenate((median_depth[None,...], depth_data), axis=0) # first map is a median depth map.
    num_maps = depth_data.shape[0] 

    pc_list = [px_to_camera(depth_data[i,...], K_inv) for i in range(num_maps)]

    analyzer = PointCloudConsistencyAnalyzer(pc_list)

    print("\n--- Running Pairwise ICP Analysis ---")
    icp_errors = analyzer.compute_pairwise_icp_error(reference_idx=0, voxel_size=0.025)

    print("\n ICP Errors (Fitness and Inlier RMSE) w.r.t. median depth point cloud:")
    for i, errors in icp_errors["aggr_errors"].items():
        print(f"  Cloud {i}: Fitness={errors['fitness']:.4f}, RMSE={errors['inlier_rmse']:.4f}")
    
    # g_analyzer = PointCloudConsistencyAnalyzer(pc_list[1:])
    # print("\n--- Running Global Alignment Analysis ---")
    # aligned_pc_np, pc_consensus, global_error_maps = g_analyzer.compute_global_alignment_error(
    #     voxel_size=0.01
    # )

    # for i, error_map in global_error_maps.items():
    #     print(f"  Cloud {i}: Mean Error = {np.mean(error_map):.4f}, Std Dev = {np.std(error_map):.4f}")

    '''
    The `aligned_pc_np` contains the globally consistent point clouds,
    and `global_error_maps` contains the per-pixel error signal relative to
    the fused model, which can be directly used as a confidence map.
    
    # # Optional: Visualize the results
    o3d.visualization.draw_geometries(analyzer.point_clouds_o3d) # Original
    o3d.visualization.draw_geometries(aligned_pc_np) # Aligned
    o3d.visualization.draw_geometries([pcd_consensus]) # Consensus model
    '''
    err_maps = np.zeros((num_maps-1, depth_data.shape[1], depth_data.shape[2]))
    for i, err_map in enumerate(list(icp_errors['error_maps'].values())):
        err_maps[i, ...] = err_map.reshape(depth_data.shape[1], depth_data.shape[2])

    return err_maps