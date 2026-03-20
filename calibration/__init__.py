"""
calibration — camera calibration board utilities
================================================

Modules
-------
charuco_board   Generate a ChArUco calibration board PDF at print resolution and
                save its parameters as a pickle file.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
