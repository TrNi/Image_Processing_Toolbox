"""
get_file_ids.py
-------
This script has been reorganised.
New location: file_tools.get_gdrive_ids

Update your imports:
    from file_tools.get_gdrive_ids import ...

Or run directly:
    python file_tools/get_gdrive_ids.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from file_tools.get_gdrive_ids import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from file_tools.get_gdrive_ids import main
        main()
    except ImportError:
        pass
