"""
crop_jpg_images.py
-------
This script has been reorganised.
New location: image_processing.crop_jpg

Update your imports:
    from image_processing.crop_jpg import ...

Or run directly:
    python image_processing/crop_jpg.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_processing.crop_jpg import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from image_processing.crop_jpg import main
        main()
    except ImportError:
        pass
