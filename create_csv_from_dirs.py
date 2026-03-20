"""
create_csv_from_dirs.py
-------
This script has been reorganised.
New location: file_tools.create_csv

Update your imports:
    from file_tools.create_csv import ...

Or run directly:
    python file_tools/create_csv.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from file_tools.create_csv import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from file_tools.create_csv import main
        main()
    except ImportError:
        pass
