"""
visualize.py
-------
This script has been reorganised.
New location: visualization.visualize_depth

Update your imports:
    from visualization.visualize_depth import ...

Or run directly:
    python visualization/visualize_depth.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualization.visualize_depth import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from visualization.visualize_depth import main
        main()
    except ImportError:
        pass
