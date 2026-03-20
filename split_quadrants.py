"""
split_quadrants.py
-------
This script has been reorganised.
New location: image_processing.split_quadrants

Update your imports:
    from image_processing.split_quadrants import ...

Or run directly:
    python image_processing/split_quadrants.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_processing.split_quadrants import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from image_processing.split_quadrants import main
        main()
    except ImportError:
        pass
