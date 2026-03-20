"""
move_indexed_images.py
-------
This script has been reorganised.
New location: file_tools.move_images

Update your imports:
    from file_tools.move_images import ...

Or run directly:
    python file_tools/move_images.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from file_tools.move_images import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from file_tools.move_images import main
        main()
    except ImportError:
        pass
