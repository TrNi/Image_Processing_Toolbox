"""
extract_from_h5.py
==================
Extract individual images from an HDF5 dataset and save them as JPEG or PNG.

The script handles both ``(N, H, W, C)`` (channel-last) and ``(N, C, H, W)``
(channel-first) array layouts, automatically detecting and transposing as
needed.

Usage
-----
::

    python file_tools/extract_from_h5.py /path/to/data.h5 \\
        --key images --out_dir extracted/ --format jpg --quality 95

    # Only save frames 0-9
    python file_tools/extract_from_h5.py data.h5 --start 0 --end 10
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def extract_images_from_h5(h5_path: str, key: str = None,
                            out_dir: str = "extracted",
                            fmt: str = "jpg", quality: int = 95,
                            start: int = 0, end: int = None):
    """Extract frames from an HDF5 dataset as image files.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    key : str, optional
        Dataset key.  If ``None``, the first dataset found is used.
    out_dir : str
        Output directory.
    fmt : str
        Output format: ``'jpg'`` or ``'png'``.
    quality : int
        JPEG quality 1–100 (ignored for PNG).
    start : int
        First frame index to extract (inclusive).
    end : int, optional
        Last frame index to extract (exclusive).  ``None`` = extract all.
    """
    h5_path  = Path(h5_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as fh:
        if key is None:
            # Try common key names, fall back to first available
            for candidate in ('data', 'images', 'rgb', 'frames'):
                if candidate in fh:
                    key = candidate
                    break
            if key is None:
                key = list(fh.keys())[0]
            print(f"Using dataset key: '{key}'")

        if key not in fh:
            print(f"Available keys: {list(fh.keys())}")
            raise KeyError(f"Key '{key}' not found in {h5_path.name}")

        dataset   = fh[key]
        n_frames  = dataset.shape[0]
        end       = n_frames if end is None else min(end, n_frames)
        n_extract = end - start

        print(f"Dataset: {key}  shape={dataset.shape}  dtype={dataset.dtype}")
        print(f"Extracting frames [{start}, {end})  → {n_extract} images")

        for i in range(start, end):
            arr = dataset[i]

            # Detect channel-first layout: (C, H, W) where C is 1, 3, or 4
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
                arr = np.transpose(arr, (1, 2, 0))

            # Squeeze single-channel
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(-1)

            # Normalise float arrays to uint8
            if arr.dtype in (np.float32, np.float64, np.float16):
                lo, hi = arr.min(), arr.max()
                if hi > lo:
                    arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)

            img      = Image.fromarray(arr)
            ext      = 'jpg' if fmt.lower() in ('jpg', 'jpeg') else 'png'
            filename = f"{h5_path.stem}_{key}_{i:05d}.{ext}"
            save_path = out_path / filename

            if ext == 'jpg':
                img.save(str(save_path), 'JPEG', quality=quality)
            else:
                img.save(str(save_path), 'PNG')

            if (i - start) % max(1, n_extract // 10) == 0:
                print(f"  [{i - start + 1}/{n_extract}] {filename}")

    print(f"\nDone.  {n_extract} images saved to: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract individual images from an HDF5 dataset.",
    )
    parser.add_argument("h5_path",  help="Path to the HDF5 file.")
    parser.add_argument("--key",    default=None,
                        help="Dataset key (default: auto-detect).")
    parser.add_argument("--out_dir", default="extracted",
                        help="Output directory (default: extracted).")
    parser.add_argument("--format",  default="jpg", choices=["jpg", "png"],
                        help="Output image format (default: jpg).")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality 1–100 (default: 95).")
    parser.add_argument("--start",   type=int, default=0,
                        help="First frame index (default: 0).")
    parser.add_argument("--end",     type=int, default=None,
                        help="Last frame index exclusive (default: all).")
    args = parser.parse_args()

    extract_images_from_h5(
        h5_path=args.h5_path,
        key=args.key,
        out_dir=args.out_dir,
        fmt=args.format,
        quality=args.quality,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
