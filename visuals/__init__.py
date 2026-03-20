"""
visuals — publication-quality figure utilities
===============================================

Modules
-------
jpg_to_pdf          Trim images / HDF5 arrays and save as high-DPI PDF + PNG.
merge_imgs          Combine 4 images into a 1×4 CVPR/ECCV publication figure.

Sub-packages
------------
mono_stereo_depths  Scene-specific depth map visualisation helpers.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
