"""
geometric_structure_errors.py
-------
This script has been reorganised.
New location: depth_analysis.geometric_structure_errors

Update your imports:
    from depth_analysis.geometric_structure_errors import ...

Or run directly:
    python depth_analysis/geometric_structure_errors.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from depth_analysis.geometric_structure_errors import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from depth_analysis.geometric_structure_errors import main
        main()
    except ImportError:
        pass
