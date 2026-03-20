"""
generate_3fig6col_comparison_figure.py
-------
This script has been reorganised.
New location: visualization.generate_comparison_figure

Update your imports:
    from visualization.generate_comparison_figure import ...

Or run directly:
    python visualization/generate_comparison_figure.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualization.generate_comparison_figure import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from visualization.generate_comparison_figure import main
        main()
    except ImportError:
        pass
