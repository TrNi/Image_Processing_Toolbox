"""
run_error_analysis_folders.py
-------
This script has been reorganised.
New location: pipelines.run_depth_analysis_folders

Update your imports:
    from pipelines.run_depth_analysis_folders import ...

Or run directly:
    python pipelines/run_depth_analysis_folders.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipelines.run_depth_analysis_folders import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from pipelines.run_depth_analysis_folders import main
        main()
    except ImportError:
        pass
