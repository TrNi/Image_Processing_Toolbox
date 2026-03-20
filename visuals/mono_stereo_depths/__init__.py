"""
mono_stereo_depths — scene-specific depth visualisation helpers
===============================================================

Modules
-------
depth_map_visualization   Utility functions for visualising depth maps.
illusion_crop_visualization Depth-of-field illusion crop visualisation.
prepare_jpg_h5            Prepare JPEG / HDF5 data for depth experiments.
sanity_plots              Sanity-check plots for stereo and mono depth data.
visualise_data            General data visualisation helpers.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
