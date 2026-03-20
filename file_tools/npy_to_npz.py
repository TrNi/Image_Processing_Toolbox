"""
npy_to_npz.py
=============
Convert a ``.npy`` file containing a dict-like object (pickled with NumPy)
to a compressed ``.npz`` archive.

Usage
-----
::

    python file_tools/npy_to_npz.py /path/to/data.npy --out /path/to/data.npz
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np


def npy_to_npz(npy_path: str, out_path: str = None):
    """Convert a ``.npy`` file to a compressed ``.npz`` archive.

    If the loaded object is a ``numpy.ndarray``, it is stored under the key
    ``'arr_0'``.  If it is a dict, each key-value pair is stored as a named
    array.

    Parameters
    ----------
    npy_path : str
        Input ``.npy`` file path.
    out_path : str, optional
        Output ``.npz`` path.  Defaults to the same stem with ``.npz``
        extension in the same directory.
    """
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"File not found: {npy_path}")

    if out_path is None:
        out_path = npy_path.with_suffix('.npz')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {npy_path}")
    data = np.load(str(npy_path), allow_pickle=True)

    if isinstance(data, np.ndarray) and data.dtype == object:
        # Likely a pickled dict stored via np.save
        item = data.item()
        if isinstance(item, dict):
            print(f"  Detected pickled dict with {len(item)} keys: {list(item.keys())}")
            arrays = {k: np.asarray(v) for k, v in item.items()}
        else:
            print(f"  Detected pickled object of type {type(item).__name__}; "
                  f"storing as 'arr_0'.")
            arrays = {'arr_0': np.asarray(item)}
    elif isinstance(data, np.ndarray):
        print(f"  Detected array: shape={data.shape}  dtype={data.dtype}")
        arrays = {'arr_0': data}
    else:
        raise TypeError(f"Unexpected type loaded from .npy: {type(data).__name__}")

    np.savez_compressed(str(out_path), **arrays)
    size_mb = out_path.stat().st_size / 1024**2
    print(f"Saved → {out_path}  ({size_mb:.2f} MB)")
    for k, v in arrays.items():
        print(f"  key='{k}'  shape={np.asarray(v).shape}  dtype={np.asarray(v).dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a .npy file to a compressed .npz archive.",
    )
    parser.add_argument("npy_path",   help="Input .npy file path.")
    parser.add_argument("--out",      dest="out_path", default=None,
                        help="Output .npz path (default: same stem + .npz).")
    args = parser.parse_args()
    npy_to_npz(args.npy_path, args.out_path)


if __name__ == "__main__":
    main()
