"""
visualize_error_analysis.py
-------
This script has been reorganised.
New location: visualization.visualize_error_analysis

Update your imports:
    from visualization.visualize_error_analysis import ...

Or run directly:
    python visualization/visualize_error_analysis.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualization.visualize_error_analysis import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from visualization.visualize_error_analysis import main
        main()
    except ImportError:
        pass
