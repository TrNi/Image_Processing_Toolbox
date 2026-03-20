"""
illustrate_h5.py
================
Display multiple images from an HDF5 dataset in an interactive grid.

Usage
-----
::

    python file_tools/illustrate_h5.py /path/to/data.h5 \\
        --key data --rows 3 --cols 4 --start 0
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np
import matplotlib.pyplot as plt


def illustrate_h5(h5_path: str, key: str = None,
                  rows: int = 3, cols: int = 4,
                  start: int = 0, cmap: str = None):
    """Display images from an HDF5 dataset in a grid.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    key : str, optional
        Dataset key.  Auto-detected when ``None``.
    rows, cols : int
        Grid dimensions.
    start : int
        First frame index to display.
    cmap : str, optional
        Matplotlib colormap for single-channel images (default: ``'gray'``
        for 1-channel, ``None`` for RGB).
    """
    with h5py.File(h5_path, 'r') as fh:
        if key is None:
            for candidate in ('data', 'images', 'rgb', 'frames'):
                if candidate in fh:
                    key = candidate
                    break
            if key is None:
                key = list(fh.keys())[0]

        dataset = fh[key]
        n       = dataset.shape[0]
        total   = rows * cols
        end     = min(start + total, n)
        frames  = dataset[start:end][()]

    print(f"HDF5: {Path(h5_path).name}  key='{key}'  "
          f"shape={frames.shape}  frames [{start},{end})")

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for r in range(rows):
        for c in range(cols):
            ax  = axes[r, c]
            idx = r * cols + c
            if idx >= len(frames):
                ax.axis('off')
                continue

            arr = frames[idx]
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(-1)

            if arr.dtype in (np.float32, np.float64, np.float16):
                lo, hi = arr.min(), arr.max()
                arr = (arr - lo) / (hi - lo + 1e-8)

            use_cmap = cmap if cmap else ('gray' if arr.ndim == 2 else None)
            ax.imshow(arr, cmap=use_cmap)
            ax.set_title(f"[{start + idx}]", fontsize=8)
            ax.axis('off')

    plt.suptitle(f"{Path(h5_path).name}  |  key='{key}'",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Display images from an HDF5 dataset in a grid.",
    )
    parser.add_argument("h5_path", help="Path to the HDF5 file.")
    parser.add_argument("--key",   default=None,
                        help="Dataset key (auto-detected by default).")
    parser.add_argument("--rows",  type=int, default=3)
    parser.add_argument("--cols",  type=int, default=4)
    parser.add_argument("--start", type=int, default=0,
                        help="First frame index (default: 0).")
    parser.add_argument("--cmap",  default=None,
                        help="Matplotlib colormap for single-channel data.")
    args = parser.parse_args()
    illustrate_h5(args.h5_path, args.key,
                  args.rows, args.cols, args.start, args.cmap)


if __name__ == "__main__":
    main()
