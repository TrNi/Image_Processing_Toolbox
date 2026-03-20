"""
npy_to_h5.py
============
Combine a sequence of ``.npy`` array files into a single HDF5 file.

Each ``.npy`` file contributes one frame (or several if it contains a
batch dimension).  All arrays must share the same spatial shape after
the batch dimension.

Usage
-----
::

    python file_tools/npy_to_h5.py /path/to/npy_dir output.h5 \\
        --key depth --glob "depth_*.npy"
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import h5py


def npy_to_h5(npy_dir: str, out_path: str, key: str = 'data',
              glob: str = "*.npy"):
    """Pack ``.npy`` files from *npy_dir* into a single HDF5 dataset.

    Parameters
    ----------
    npy_dir : str
        Directory containing ``.npy`` files.
    out_path : str
        Output HDF5 path.
    key : str
        Dataset key in the output file (default: ``'data'``).
    glob : str
        Glob pattern for matching files (default: ``'*.npy'``).
    """
    dir_path  = Path(npy_dir)
    npy_files = sorted(dir_path.glob(glob))

    if not npy_files:
        print(f"No .npy files matching '{glob}' found in: {npy_dir}")
        return
    print(f"Found {len(npy_files)} .npy file(s).")

    # Load first file to check shape and dtype
    first = np.load(str(npy_files[0]))
    print(f"First file: {npy_files[0].name}  shape={first.shape}  dtype={first.dtype}")

    # Determine per-file frame count
    if first.ndim == 1:
        raise ValueError("1-D arrays are not supported.  Expected ≥ 2-D arrays.")

    frames_per_file = first.shape[0]
    item_shape = first.shape[1:] if first.ndim > 1 else ()
    total_frames = frames_per_file * len(npy_files)

    print(f"Frames per file: {frames_per_file}")
    print(f"Item shape:      {item_shape}")
    print(f"Total frames:    {total_frames}")
    print(f"Writing → {out_path}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, 'w') as fh:
        ds = fh.create_dataset(
            key,
            shape=(total_frames, *item_shape),
            dtype=first.dtype,
            chunks=(1, *item_shape) if item_shape else (1,),
            compression='gzip', compression_opts=4,
        )
        write_idx = 0
        for i, npy_file in enumerate(npy_files):
            arr = np.load(str(npy_file))
            if arr.shape[0] != frames_per_file or arr.shape[1:] != item_shape:
                raise ValueError(
                    f"Shape mismatch in {npy_file.name}: "
                    f"expected ({frames_per_file},{','.join(map(str, item_shape))}), "
                    f"got {arr.shape}"
                )
            ds[write_idx:write_idx + frames_per_file] = arr
            write_idx += frames_per_file
            print(f"  [{i+1}/{len(npy_files)}] {npy_file.name}")

    size_mb = Path(out_path).stat().st_size / 1024**2
    print(f"\nDone.  {out_path}  shape=({total_frames},{','.join(map(str, item_shape))})  "
          f"{size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Combine a sequence of .npy files into a single HDF5 file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
    python file_tools/npy_to_h5.py /data/npy_frames output.h5 \\
        --key depth --glob "depth_*.npy"
""",
    )
    parser.add_argument("npy_dir",  help="Directory containing .npy files.")
    parser.add_argument("out_path", help="Output HDF5 path.")
    parser.add_argument("--key",    default="data",
                        help="HDF5 dataset key (default: data).")
    parser.add_argument("--glob",   default="*.npy",
                        help="Glob pattern for .npy files (default: *.npy).")
    args = parser.parse_args()
    npy_to_h5(args.npy_dir, args.out_path, key=args.key, glob=args.glob)


if __name__ == "__main__":
    main()
