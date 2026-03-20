"""
create_csv.py
=============
Build a CSV index from one or more image directories with optional URL prefixes.

Each output row: ``<url_prefix>/<filename>``.  Useful for generating
dataset manifests for training pipelines or remote-storage datasets.

Usage
-----
::

    python file_tools/create_csv.py \\
        --dirs /data/scene1/images /data/scene2/images \\
        --prefixes https://storage.example.com/scene1 \\
                   https://storage.example.com/scene2 \\
        --out dataset.csv

    # Without URL prefixes — just lists absolute paths
    python file_tools/create_csv.py \\
        --dirs /data/images --out paths.csv
"""

import sys
import argparse
import csv
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def create_csv(directories: list, out_path: str,
               prefixes: list = None, header: str = "image_path",
               extensions: set = None):
    """Create a single-column CSV of image file paths/URLs.

    Parameters
    ----------
    directories : list of str
        Source directories to scan.
    out_path : str
        Output CSV path.
    prefixes : list of str, optional
        URL or path prefix for each directory.  Must have the same length as
        *directories* when provided.  If ``None``, absolute file paths are
        used.
    header : str
        Column header string (default: ``'image_path'``).
    extensions : set, optional
        Set of lowercase extensions to include (default: common image types).
    """
    if extensions is None:
        extensions = IMAGE_EXTS
    if prefixes is None:
        prefixes = [None] * len(directories)
    if len(prefixes) != len(directories):
        raise ValueError(
            f"Number of prefixes ({len(prefixes)}) must match "
            f"directories ({len(directories)})."
        )

    rows  = []
    total = 0

    for dir_path, prefix in zip(directories, prefixes):
        dir_p = Path(dir_path)
        if not dir_p.exists():
            print(f"WARNING: Directory not found: {dir_path}")
            continue
        files = sorted(
            f for f in dir_p.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        if not files:
            print(f"WARNING: No matching files in: {dir_path}")
            continue

        for f in files:
            if prefix:
                entry = f"{prefix.rstrip('/')}/{f.name}"
            else:
                entry = str(f.resolve())
            rows.append([entry])
            total += 1

        print(f"  {dir_path}: {len(files)} image(s) added")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([header])
        writer.writerows(rows)

    print(f"\nSaved {total} entries → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a CSV index from image directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
    # With URL prefixes
    python file_tools/create_csv.py \\
        --dirs /data/scene1 /data/scene2 \\
        --prefixes https://cdn.example.com/s1 https://cdn.example.com/s2 \\
        --out dataset.csv

    # Absolute local paths
    python file_tools/create_csv.py --dirs /data/images --out paths.csv
""",
    )
    parser.add_argument("--dirs",     nargs="+", required=True,
                        help="Source directories to scan.")
    parser.add_argument("--prefixes", nargs="+", default=None,
                        help="URL/path prefix per directory (same count as --dirs).")
    parser.add_argument("--out",      required=True,
                        help="Output CSV path.")
    parser.add_argument("--header",   default="image_path",
                        help="CSV column header (default: image_path).")
    args = parser.parse_args()

    create_csv(
        directories=args.dirs,
        out_path=args.out,
        prefixes=args.prefixes,
        header=args.header,
    )


if __name__ == "__main__":
    main()
