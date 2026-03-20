"""
apply_crop_from_json.py
-------
This script has been reorganised.
New location: image_processing.apply_crop_from_json

Update your imports:
    from image_processing.apply_crop_from_json import ...

Or run directly:
    python image_processing/apply_crop_from_json.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_processing.apply_crop_from_json import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from image_processing.apply_crop_from_json import main
        main()
    except ImportError:
        pass
