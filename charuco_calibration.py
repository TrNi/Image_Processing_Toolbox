"""
charuco_calibration.py
-------
This script has been reorganised.
New location: calibration.charuco_board

Update your imports:
    from calibration.charuco_board import ...

Or run directly:
    python calibration/charuco_board.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration.charuco_board import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from calibration.charuco_board import main
        main()
    except ImportError:
        pass
