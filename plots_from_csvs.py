"""
plots_from_csvs.py
-------
This script has been reorganised.
New location: visualization.plots_from_csvs

Update your imports:
    from visualization.plots_from_csvs import ...

Or run directly:
    python visualization/plots_from_csvs.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualization.plots_from_csvs import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from visualization.plots_from_csvs import main
        main()
    except ImportError:
        pass
