"""
split_quadrants.py
==================
Split images into four equal quadrants (top-left, top-right, bottom-left, bottom-right).

Output filenames have ``_0``, ``_1``, ``_2``, ``_3`` suffixes corresponding
to the four quadrants.  Handles lower-case and upper-case extensions.

Usage
-----
::

    python image_processing/split_quadrants.py /path/to/input /path/to/output
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def split_into_quadrants(input_dir: str, save_dir: str):
    """Split every image in *input_dir* into four quadrant sub-images.

    Parameters
    ----------
    input_dir : str
        Source directory.
    save_dir : str
        Output directory (created if absent).
    """
    input_path = Path(input_dir)
    save_path  = Path(save_dir)

    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        return

    save_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )

    if not img_files:
        print(f"No image files found in: {input_dir}")
        return

    print(f"Found {len(img_files)} image(s) to split.")

    processed = skipped = 0
    for img_file in img_files:
        try:
            img = Image.open(img_file)
            w, h = img.size
            hw, hh = w // 2, h // 2

            quadrants = [
                (0,  0,  hw, hh),  # _0 top-left
                (hw, 0,  w,  hh),  # _1 top-right
                (0,  hh, hw, h),   # _2 bottom-left
                (hw, hh, w,  h),   # _3 bottom-right
            ]

            for idx, coords in enumerate(quadrants):
                q = img.crop(coords)
                q.save(str(save_path / f"{img_file.stem}_{idx}{img_file.suffix}"),
                       quality=100)

            print(f"  {img_file.name}  {w}×{h} → 4 × {hw}×{hh}")
            processed += 1

        except Exception as exc:
            print(f"  ERROR {img_file.name}: {exc}")
            skipped += 1

    print(f"\nDone.  Processed: {processed} ({processed * 4} quadrants), "
          f"Skipped: {skipped}")
    print(f"Output: {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split images into four equal quadrants.",
    )
    parser.add_argument("input_dir", help="Source directory.")
    parser.add_argument("save_dir",  help="Output directory.")
    args = parser.parse_args()
    split_into_quadrants(args.input_dir, args.save_dir)


if __name__ == '__main__':
    main()
