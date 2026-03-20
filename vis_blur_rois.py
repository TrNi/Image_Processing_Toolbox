"""
vis_blur_rois.py
-------
This script has been reorganised.
New location: visualization.vis_blur_rois

Update your imports:
    from visualization.vis_blur_rois import ...

Or run directly:
    python visualization/vis_blur_rois.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualization.vis_blur_rois import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from visualization.vis_blur_rois import main
        main()
    except ImportError:
        pass
