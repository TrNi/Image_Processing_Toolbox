"""
pkl_to_h5.py
-------
This script has been reorganised.
New location: file_tools.pkl_to_h5

Update your imports:
    from file_tools.pkl_to_h5 import ...

Or run directly:
    python file_tools/pkl_to_h5.py --help
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from file_tools.pkl_to_h5 import *  # noqa: F401,F403 (re-export for backward compat)

if __name__ == '__main__':
    try:
        from file_tools.pkl_to_h5 import main
        main()
    except ImportError:
        pass
