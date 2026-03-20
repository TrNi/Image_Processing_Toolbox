"""
plot_one_row.py
-------
This script has been reorganised.
New location: visualization.plot_one_row

Update your imports:
    from visualization.plot_one_row import ...

Or run directly:
    python visualization/plot_one_row.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualization.plot_one_row import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from visualization.plot_one_row import main
        main()
    except ImportError:
        pass
