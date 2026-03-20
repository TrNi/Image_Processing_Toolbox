"""
visualize_and_crop_index.py
-------
This script has been reorganised.
New location: image_processing.visualize_and_crop

Update your imports:
    from image_processing.visualize_and_crop import ...

Or run directly:
    python image_processing/visualize_and_crop.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_processing.visualize_and_crop import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from image_processing.visualize_and_crop import main
        main()
    except ImportError:
        pass
