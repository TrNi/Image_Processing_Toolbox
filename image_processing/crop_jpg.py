"""
crop_jpg.py
===========
Simple width-based crop for JPEG images (keeps full height, crops from left).

Useful for removing lens-distortion or calibration artefacts from the right
edge of rectified images.  Handles both lower-case and upper-case ``.jpg`` /
``.JPG`` extensions.

Usage
-----
::

    python image_processing/crop_jpg.py /path/to/input /path/to/output --width 5472
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image


IMAGE_EXTS = {'.jpg', '.jpeg'}


def crop_images(input_dir: str, save_dir: str, crop_width: int = 5472):
    """Crop all JPEG images in *input_dir* to *crop_width* pixels wide.

    Parameters
    ----------
    input_dir : str
        Source directory.
    save_dir : str
        Output directory (created if it does not exist).
    crop_width : int
        Target width.  Images narrower than this are skipped.
    """
    input_path = Path(input_dir)
    save_path  = Path(save_dir)

    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        return

    save_path.mkdir(parents=True, exist_ok=True)

    jpg_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )

    if not jpg_files:
        print(f"No JPEG files found in: {input_dir}")
        return

    print(f"Found {len(jpg_files)} JPEG file(s).  Target width: {crop_width}")

    processed = skipped = 0
    for img_file in jpg_files:
        try:
            img = Image.open(img_file)
            w, h = img.size

            if w < crop_width:
                print(f"  SKIP: {img_file.name} is only {w} px wide "
                      f"(need {crop_width}).")
                skipped += 1
                continue

            cropped = img.crop((0, 0, crop_width, h))

            # Strip any 'vs..._' prefix from filenames (legacy naming)
            stem = img_file.name
            idx  = stem.find("vs")
            stem = stem[idx + 3:] if idx != -1 else stem

            cropped.save(str(save_path / stem), quality=100)
            print(f"  {img_file.name}  {w}×{h} → {crop_width}×{h}")
            processed += 1

        except Exception as exc:
            print(f"  ERROR {img_file.name}: {exc}")
            skipped += 1

    print(f"\nDone.  Processed: {processed}, Skipped: {skipped}")
    print(f"Output: {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Crop JPEG images to a fixed width (keeps full height).",
    )
    parser.add_argument("input_dir", help="Source directory.")
    parser.add_argument("save_dir",  help="Output directory.")
    parser.add_argument("--width", type=int, default=5472,
                        help="Target width in pixels (default: 5472).")
    args = parser.parse_args()
    crop_images(args.input_dir, args.save_dir, args.width)


if __name__ == '__main__':
    main()
