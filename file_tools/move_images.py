"""
move_images.py
==============
Move images at specified indices from a source directory to a destination.

Images are sorted lexicographically; indices are zero-based.

Usage
-----
::

    python file_tools/move_images.py \\
        --src /path/to/source \\
        --dst /path/to/destination \\
        --indices 3 5 7 8 10 13 15 17 18

    # Move every 3rd image
    python file_tools/move_images.py \\
        --src /path/to/source \\
        --dst /path/to/destination \\
        --step 3
"""

import sys
import shutil
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def move_images(src_dir: str, dst_dir: str,
                indices: list = None, step: int = None,
                copy: bool = False):
    """Move (or copy) images at the specified indices.

    Parameters
    ----------
    src_dir : str
        Source directory.
    dst_dir : str
        Destination directory.
    indices : list of int, optional
        Explicit zero-based indices.  Mutually exclusive with *step*.
    step : int, optional
        Move every *step*-th image (0, step, 2*step, …).
    copy : bool
        If ``True``, copy files instead of moving them.
    """
    if (indices is None) == (step is None):
        raise ValueError("Provide exactly one of --indices or --step.")

    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    all_images = sorted(
        f for f in src_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    if not all_images:
        print(f"No images found in: {src_dir}")
        return

    print(f"Source  : {src_dir}  ({len(all_images)} images total)")

    if step is not None:
        indices = list(range(0, len(all_images), step))

    # Validate indices
    invalid = [i for i in indices if not (0 <= i < len(all_images))]
    if invalid:
        print(f"WARNING: Out-of-range indices ignored: {invalid}")
    indices = [i for i in indices if 0 <= i < len(all_images)]

    if not indices:
        print("No valid indices to process.")
        return

    dst_path.mkdir(parents=True, exist_ok=True)
    verb = "Copying" if copy else "Moving"
    print(f"{verb} {len(indices)} image(s) → {dst_dir}")

    for idx in sorted(indices):
        src_file = all_images[idx]
        dst_file = dst_path / src_file.name
        if copy:
            shutil.copy2(str(src_file), str(dst_file))
        else:
            shutil.move(str(src_file), str(dst_file))
        print(f"  [{idx:>4}] {src_file.name}")

    print(f"\nDone.  {len(indices)} file(s) {verb.lower()[:-3]}d.")


def main():
    parser = argparse.ArgumentParser(
        description="Move (or copy) images at specified indices between directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
    # Move specific indices
    python file_tools/move_images.py \\
        --src /data/raw --dst /data/selected --indices 3 5 7 8

    # Copy every 5th image
    python file_tools/move_images.py \\
        --src /data/raw --dst /data/sampled --step 5 --copy
""",
    )
    parser.add_argument("--src",     required=True, help="Source directory.")
    parser.add_argument("--dst",     required=True, help="Destination directory.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--indices", nargs="+", type=int,
                       help="Zero-based image indices to move.")
    group.add_argument("--step",    type=int,
                       help="Move every N-th image (0, N, 2N, …).")

    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of moving them.")
    args = parser.parse_args()

    move_images(
        src_dir=args.src,
        dst_dir=args.dst,
        indices=args.indices,
        step=args.step,
        copy=args.copy,
    )


if __name__ == "__main__":
    main()
