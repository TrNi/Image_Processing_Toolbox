"""
pkl_to_h5.py
============
Convert gzip-compressed pickle error files to HDF5.

Each pickle is expected to contain a dict with nested numpy arrays.
The hierarchy is preserved in the HDF5 file as groups and datasets.

Usage
-----
::

    python file_tools/pkl_to_h5.py /path/to/error_data.pkl --out /path/to/output.h5

    # Convert all pkl files in a directory
    python file_tools/pkl_to_h5.py /path/to/dir --out_dir /path/to/h5s
"""

import sys
import argparse
import gzip
import pickle
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import h5py


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _write_to_h5_group(group: h5py.Group, data: dict, path: str = ""):
    """Recursively write a nested dict of arrays to an H5 group."""
    for key, value in data.items():
        full_key = f"{path}/{key}" if path else key
        if isinstance(value, dict):
            sub = group.require_group(key)
            _write_to_h5_group(sub, value, full_key)
        else:
            arr = np.asarray(value)
            try:
                group.create_dataset(
                    key, data=arr,
                    compression='gzip', compression_opts=4,
                )
                print(f"  Dataset: {full_key}  shape={arr.shape}  dtype={arr.dtype}")
            except Exception as exc:
                print(f"  WARNING: could not write '{full_key}': {exc}")


def pkl_to_h5(pkl_path: str, out_path: str):
    """Convert a (gzip-pickled) dict file to HDF5.

    Parameters
    ----------
    pkl_path : str
        Path to the input pickle file (plain or gzip-compressed).
    out_path : str
        Output HDF5 path.
    """
    pkl_path = Path(pkl_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {pkl_path.name}")
    try:
        with gzip.open(str(pkl_path), 'rb') as fh:
            data = pickle.load(fh)
    except (OSError, gzip.BadGzipFile):
        with open(str(pkl_path), 'rb') as fh:
            data = pickle.load(fh)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a dict in {pkl_path.name}, got {type(data).__name__}"
        )

    print(f"Converting to HDF5: {out_path}")
    with h5py.File(str(out_path), 'w') as fh:
        _write_to_h5_group(fh, data)

    size_mb = out_path.stat().st_size / 1024**2
    print(f"Done.  {out_path}  {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert gzip-pickled dict files to HDF5.",
    )
    parser.add_argument("input",
                        help="Input pickle file or directory.")
    parser.add_argument("--out",     dest="out_path",  default=None,
                        help="Output HDF5 path (single file mode).")
    parser.add_argument("--out_dir", dest="out_dir",   default=None,
                        help="Output directory (directory mode).")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        out_dir = Path(args.out_dir or str(input_path) + "_h5")
        out_dir.mkdir(parents=True, exist_ok=True)
        pkl_files = list(input_path.glob("*.pkl")) + \
                    list(input_path.glob("*.pickle"))
        if not pkl_files:
            print(f"No .pkl / .pickle files found in: {input_path}")
            return
        for p in pkl_files:
            pkl_to_h5(str(p), str(out_dir / p.with_suffix('.h5').name))
    else:
        out = args.out_path or str(input_path.with_suffix('.h5'))
        pkl_to_h5(str(input_path), out)


if __name__ == "__main__":
    main()
