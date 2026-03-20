"""
visualization — depth map and error plotting utilities
======================================================

Modules
-------
visualize_depth            Interactive per-image depth-map viewer with fused depth output.
visualize_error_analysis   CDF plots, error-map figures and depth-comparison figures.
plots_from_csvs            Error-statistic trend plots vs. focal length / aperture.
plot_one_row               Single-row publication figure (ECCV / CVPR template).
generate_comparison_figure Multi-scene 3-column comparison figure with bounding-box crops.
vis_blur_rois              Interactive ROI picker for bokeh / blur comparison figures.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
