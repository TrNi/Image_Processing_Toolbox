"""
pipelines — end-to-end depth analysis pipelines
================================================

Modules
-------
run_depth_analysis          Full pipeline: compute per-image error maps → save → visualise.
run_depth_analysis_folders  Folder-based variant: point at pre-computed error_data.pkl dirs.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
