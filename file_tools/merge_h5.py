"""
merge_h5.py
===========
Concatenate multiple HDF5 files along the batch (first) dimension.

Usage
-----
::

    # Merge all H5 files in a directory
    python file_tools/merge_h5.py --input_dir /path/to/h5s --output merged.h5

    # Merge specific files
    python file_tools/merge_h5.py --files a.h5 b.h5 c.h5 --output merged.h5

    # Optional: limit total frames kept
    python file_tools/merge_h5.py --input_dir /path/to/h5s --output merged.h5 --max 500
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def merge_h5_files(h5_files: list, output_path: str,
                   key: str = None, max_frames: int = None):
    """Concatenate HDF5 datasets along axis 0.

    Parameters
    ----------
    h5_files : list of str
        Input HDF5 file paths, in merge order.
    output_path : str
        Output HDF5 path.
    key : str, optional
        Dataset key to merge.  If ``None``, auto-detect from the first file.
    max_frames : int, optional
        Stop after this many total frames.
    """
    if not h5_files:
        print("No input files specified.")
        return

    # Discover key if not given
    with h5py.File(h5_files[0], 'r') as fh:
        if key is None:
            for candidate in ('data', 'images', 'rgb', 'depth', 'depths', 'frames'):
                if candidate in fh:
                    key = candidate
                    break
            if key is None:
                key = list(fh.keys())[0]
        first_shape = fh[key].shape
        first_dtype = fh[key].dtype

    print(f"Merging key='{key}'  base shape={first_shape}  dtype={first_dtype}")
    print(f"Input files: {len(h5_files)}")

    # First pass: count total frames
    total = 0
    for p in h5_files:
        with h5py.File(p, 'r') as fh:
            if key not in fh:
                print(f"  WARNING: key '{key}' not found in {Path(p).name}, skipping.")
                continue
            n = fh[key].shape[0]
            total += n
            print(f"  {Path(p).name}: {n} frames")

    if max_frames is not None:
        total = min(total, max_frames)
    print(f"Total frames to write: {total}")

    # Create output dataset
    item_shape = first_shape[1:]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as fout:
        ds = fout.create_dataset(
            key,
            shape=(total, *item_shape),
            dtype=first_dtype,
            chunks=(1, *item_shape),
            compression='gzip', compression_opts=4,
        )
        write_idx = 0
        for p in h5_files:
            if write_idx >= total:
                break
            with h5py.File(p, 'r') as fh:
                if key not in fh:
                    continue
                src  = fh[key]
                n    = min(src.shape[0], total - write_idx)
                data = src[:n]
                ds[write_idx:write_idx + n] = data
                write_idx += n
                print(f"  Copied {n} frames from {Path(p).name}  "
                      f"(total so far: {write_idx}/{total})")

    size_mb = Path(output_path).stat().st_size / 1024**2
    print(f"\nDone.  {output_path}  shape=({total},{','.join(map(str, item_shape))})  "
          f"{size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Concatenate HDF5 files along the batch dimension.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
    # Merge all *.h5 files in a directory
    python file_tools/merge_h5.py \\
        --input_dir /data/h5s --output merged.h5

    # Merge specific files
    python file_tools/merge_h5.py \\
        --files a.h5 b.h5 c.h5 --output merged.h5

    # Keep only the first 100 frames
    python file_tools/merge_h5.py \\
        --input_dir /data/h5s --output merged.h5 --max 100
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", help="Directory of *.h5 files to merge.")
    group.add_argument("--files", nargs="+", help="Explicit list of H5 files.")

    parser.add_argument("--output",  required=True, help="Output H5 path.")
    parser.add_argument("--key",     default=None,
                        help="Dataset key (auto-detected from first file).")
    parser.add_argument("--max",     type=int, default=None, dest="max_frames",
                        help="Maximum total frames to include.")
    args = parser.parse_args()

    if args.input_dir:
        h5_dir   = Path(args.input_dir)
        h5_files = sorted(h5_dir.glob("*.h5")) + sorted(h5_dir.glob("*.hdf5"))
        h5_files = [str(f) for f in h5_files]
        if not h5_files:
            print(f"No H5 files found in: {args.input_dir}")
            return
    else:
        h5_files = args.files

    merge_h5_files(h5_files, args.output, key=args.key, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
