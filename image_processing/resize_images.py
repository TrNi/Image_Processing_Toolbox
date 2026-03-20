"""
resize_images.py
================
Batch resize images with bilinear interpolation.

Handles both lower-case and upper-case JPEG extensions.

Usage
-----
::

    python image_processing/resize_images.py /path/to/input /path/to/output \\
        --width 1920 --height 1080
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def resize_images(input_dir: str, output_dir: str,
                  target_width: int, target_height: int):
    """Resize all images in *input_dir* to *target_width* × *target_height*.

    Parameters
    ----------
    input_dir : str
        Source directory.
    output_dir : str
        Output directory (created if absent).
    target_width, target_height : int
        Target dimensions in pixels.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )

    if not img_files:
        print(f"No image files found in: {input_dir}")
        return

    print(f"Found {len(img_files)} image(s).  Target: {target_width}×{target_height}")

    processed = skipped = 0
    for img_file in img_files:
        try:
            img = Image.open(img_file)
            orig_w, orig_h = img.size
            resized = img.resize((target_width, target_height), Image.BILINEAR)
            resized.save(str(output_path / img_file.name), quality=100)
            print(f"  {img_file.name}  {orig_w}×{orig_h} → {target_width}×{target_height}")
            processed += 1
        except Exception as exc:
            print(f"  ERROR {img_file.name}: {exc}")
            skipped += 1

    print(f"\nDone.  Processed: {processed}, Skipped: {skipped}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch resize images with bilinear interpolation.",
    )
    parser.add_argument("input_dir",  help="Source directory.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--width",  type=int, required=True,
                        help="Target width in pixels.")
    parser.add_argument("--height", type=int, required=True,
                        help="Target height in pixels.")
    args = parser.parse_args()
    resize_images(args.input_dir, args.output_dir, args.width, args.height)


if __name__ == '__main__':
    main()
