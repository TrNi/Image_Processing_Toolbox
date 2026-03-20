"""
Image Processing Toolbox
========================
A modular Python toolbox for computational photography and depth-estimation research.

Sub-packages
------------
depth_analysis    No-reference depth quality metrics (gradient, planarity, IQR, ICP).
visualization     Depth-map and error-distribution plotting utilities.
image_processing  Image alignment, cropping, and geometric transformation tools.
file_tools        HDF5 / NumPy / pickle file format conversion and dataset management.
calibration       Camera calibration board generation (ChArUco).
pipelines         End-to-end depth analysis pipeline runners.
visuals           Publication-quality figure utilities (PDF, CVPR/ECCV layouts).

Quick start
-----------
    from depth_analysis import get_iqr_uncertainty, simple_weighted_fusion
    from visualization.visualize_error_analysis import main as visualise
    from image_processing.align_images import align_images
    from file_tools.jpg_to_h5 import jpg_to_h5
    from calibration.charuco_board import generate_charuco_board
"""
