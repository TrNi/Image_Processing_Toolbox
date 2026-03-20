"""
image_align_crop_whitebal.py
-------
This script has been reorganised.
New location: image_processing.align_whitebal

Update your imports:
    from image_processing.align_whitebal import ...

Or run directly:
    python image_processing/align_whitebal.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_processing.align_whitebal import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from image_processing.align_whitebal import main
        main()
    except ImportError:
        pass
