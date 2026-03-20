"""
organise_source_and_destination.py
-------
This script has been reorganised.
New location: file_tools.organise_files

Update your imports:
    from file_tools.organise_files import ...

Or run directly:
    python file_tools/organise_files.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from file_tools.organise_files import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from file_tools.organise_files import main
        main()
    except ImportError:
        pass
