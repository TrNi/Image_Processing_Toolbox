"""
image_processing — image alignment, cropping, and transformation utilities
===========================================================================

Modules
-------
align_images        Phase-correlation (+ ORB fallback) image alignment.
align_whitebal      Focus-masked SIFT alignment with white-balance matching.
apply_crop_from_json Apply crop regions stored in alignment JSON files to image directories.
interactive_crop    Interactive batch crop tool with rectangle selector GUI.
visualize_and_crop  Display an image at a folder index, then interactively crop it.
crop_images         Multi-region crop with normalised coordinate array export.
crop_jpg            Simple width-based JPG crop (centre-aligned).
resize_images       Batch resize images with bilinear interpolation.
split_quadrants     Split each image into four equal quadrants.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
