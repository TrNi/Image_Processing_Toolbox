"""
point_cloud_opt.py
-------
This script has been reorganised.
New location: depth_analysis.point_cloud_opt

Update your imports:
    from depth_analysis.point_cloud_opt import ...

Or run directly:
    python depth_analysis/point_cloud_opt.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from depth_analysis.point_cloud_opt import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from depth_analysis.point_cloud_opt import main
        main()
    except ImportError:
        pass
