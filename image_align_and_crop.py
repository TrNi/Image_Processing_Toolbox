"""
image_align_and_crop.py
-------
This script has been reorganised.
New location: image_processing.align_images

Update your imports:
    from image_processing.align_images import ...

Or run directly:
    python image_processing/align_images.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_processing.align_images import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from image_processing.align_images import main
        main()
    except ImportError:
        pass
