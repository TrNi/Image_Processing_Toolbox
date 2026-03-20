"""
depth_analysis — core depth estimation quality metrics
======================================================

Modules
-------
depth_reproj_eval          Camera geometry, reprojection, photometric error maps.
geometric_structure_errors Gradient-consistency and local planarity error metrics.
uncertainty_and_weights    IQR / MAD ensemble uncertainty and weighted depth fusion.
point_cloud_opt            Point-cloud consistency analysis via ICP / GICP.
get_errors                 High-level class that orchestrates error computation and saving.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from depth_analysis.depth_reproj_eval import (
    load_h5_images,
    load_camera_params,
    get_Kinv_uv1,
    px_to_camera,
    project_to_view,
    photometric_errors,
    get_errors,
)
from depth_analysis.geometric_structure_errors import (
    compute_grad,
    compute_grad_error,
    get_planarity_error,
)
from depth_analysis.uncertainty_and_weights import (
    calculate_individual_mad_uncertainty,
    get_iqr_uncertainty,
    simple_weighted_fusion,
)
from depth_analysis.point_cloud_opt import (
    PointCloudConsistencyAnalyzer,
    get_point_cloud_errors,
)

__all__ = [
    "load_h5_images", "load_camera_params", "get_Kinv_uv1",
    "px_to_camera", "project_to_view", "photometric_errors", "get_errors",
    "compute_grad", "compute_grad_error", "get_planarity_error",
    "calculate_individual_mad_uncertainty", "get_iqr_uncertainty", "simple_weighted_fusion",
    "PointCloudConsistencyAnalyzer", "get_point_cloud_errors",
]
