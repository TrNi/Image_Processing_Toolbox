"""
get_gdrive_ids.py
=================
Resolve Google Drive file IDs from local mirrored paths.

When Google Drive is mirrored locally, each file is accompanied by a
``<filename>.gdoc`` / ``<filename>.gsheet`` / ``<filename>.json`` sidecar
that contains the Drive ID.  This script finds those IDs so you can
programmatically reference files in Drive without hard-coding IDs.

Usage
-----
::

    python file_tools/get_gdrive_ids.py \\
        --search_dir /path/to/googledrive_mirror \\
        --glob "*.json" --out ids.csv

    # Find ID for a single file by its Drive-URL sidecar
    python file_tools/get_gdrive_ids.py \\
        --sidecar "/path/to/MyFile.gdoc"
"""

import sys
import json
import csv
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Sidecar extensions used by Google Drive desktop client
SIDECAR_EXTS = {'.gdoc', '.gsheet', '.gslides', '.gdraw', '.gmap',
                '.gsite', '.gjam', '.gform', '.json'}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def extract_id_from_sidecar(sidecar_path: Path) -> str | None:
    """Extract a Drive file ID from a sidecar JSON file.

    Parameters
    ----------
    sidecar_path : Path
        Path to a ``.gdoc``, ``.gsheet`` or similar sidecar file.

    Returns
    -------
    str or None
        The Drive ID if found, otherwise ``None``.
    """
    try:
        data = json.loads(sidecar_path.read_text(encoding='utf-8', errors='ignore'))
        # Common key names used by Drive client
        for key in ('id', 'doc_id', 'drive_id', 'resource_id'):
            if key in data:
                val = str(data[key])
                # resource_id is often "document:<id>" or just the ID
                if ':' in val:
                    val = val.split(':', 1)[1]
                return val
        # Try URL field — extract ID from Drive URL
        url = data.get('url', '')
        if '/d/' in url:
            return url.split('/d/')[1].split('/')[0]
        if 'id=' in url:
            return url.split('id=')[1].split('&')[0]
    except Exception:
        pass
    return None


def find_drive_ids(search_dir: str, glob: str = "*.json",
                   recursive: bool = False) -> list:
    """Scan a directory for sidecar files and extract Drive IDs.

    Parameters
    ----------
    search_dir : str
        Directory to search.
    glob : str
        Glob pattern for sidecar files.
    recursive : bool
        If ``True``, search recursively.

    Returns
    -------
    list of dict
        Each entry has keys ``'name'``, ``'path'``, ``'id'``.
    """
    search_path = Path(search_dir)
    if recursive:
        sidecar_files = list(search_path.rglob(glob))
    else:
        sidecar_files = list(search_path.glob(glob))

    results = []
    for f in sorted(sidecar_files):
        drive_id = extract_id_from_sidecar(f)
        if drive_id:
            results.append({
                'name': f.stem,
                'path': str(f),
                'id':   drive_id,
            })
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Resolve Google Drive file IDs from local sidecar files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
    # Search a mirrored Drive directory
    python file_tools/get_gdrive_ids.py \\
        --search_dir /mnt/gdrive/MyProject --glob "*.gdoc" --out ids.csv

    # Single sidecar file
    python file_tools/get_gdrive_ids.py --sidecar /mnt/gdrive/Report.gdoc
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--search_dir', help="Directory to scan for sidecar files.")
    group.add_argument('--sidecar',    help="Path to a single sidecar file.")

    parser.add_argument('--glob',      default="*.json",
                        help="Glob pattern for sidecar files (default: *.json).")
    parser.add_argument('--recursive', action='store_true',
                        help="Search recursively.")
    parser.add_argument('--out',       default=None,
                        help="Output CSV path (default: print to stdout).")
    args = parser.parse_args()

    if args.sidecar:
        drive_id = extract_id_from_sidecar(Path(args.sidecar))
        if drive_id:
            print(f"Drive ID: {drive_id}")
        else:
            print(f"Could not extract Drive ID from: {args.sidecar}")
        return

    results = find_drive_ids(args.search_dir, args.glob, args.recursive)
    if not results:
        print("No Drive IDs found.")
        return

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=['name', 'path', 'id'])
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {len(results)} IDs → {out_path}")
    else:
        print(f"{'Name':<40}  {'Drive ID'}")
        print("-" * 72)
        for r in results:
            print(f"{r['name']:<40}  {r['id']}")
        print(f"\n{len(results)} ID(s) found.")


if __name__ == "__main__":
    main()
