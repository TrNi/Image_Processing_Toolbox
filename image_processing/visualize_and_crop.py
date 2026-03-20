"""
visualize_and_crop.py
=====================
View an image at a specific folder index, then enter interactive crop mode.

Workflow
--------
1. ``--view`` mode: Opens *index*-th image (lexicographic order) in a folder
   and displays it with coordinate grid helpers for identifying crop corners.
2. ``--crop`` mode: Applies a specified crop rectangle and saves the result.

Usage
-----
::

    # First, preview image at index 7
    python image_processing/visualize_and_crop.py /path/to/folder --index 7 --view

    # Then, crop it
    python image_processing/visualize_and_crop.py /path/to/folder --index 7 \\
        --crop --x0 200 --y0 100 --x1 5400 --y1 3600 --out_dir cropped/
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sorted_images(folder: Path) -> list:
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def view_image(folder: str, index: int):
    """Display the image at *index* with a coordinate grid.

    Parameters
    ----------
    folder : str
        Image directory.
    index : int
        Zero-based index into the sorted file list.
    """
    folder_path = Path(folder)
    images      = _get_sorted_images(folder_path)
    if not images:
        print(f"No images found in: {folder}")
        return
    if not (0 <= index < len(images)):
        print(f"Index {index} out of range.  Folder contains {len(images)} images.")
        return

    img_path = images[index]
    img      = Image.open(img_path)
    arr      = np.array(img)
    h, w     = arr.shape[:2]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(arr)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.set_xlabel("x (column)  →", fontsize=9)
    ax.set_ylabel("y (row)  ↓", fontsize=9)
    ax.set_title(
        f"{img_path.name}  [{index}/{len(images)-1}]  {w}×{h} px\n"
        f"Use coordinates displayed here with the --crop flag.",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()


def crop_image(folder: str, index: int,
               x0: int, y0: int, x1: int, y1: int,
               out_dir: str):
    """Crop the image at *index* to the specified rectangle and save it.

    Parameters
    ----------
    folder : str
        Image directory.
    index : int
        Zero-based index.
    x0, y0 : int
        Top-left corner (column, row) of the crop rectangle.
    x1, y1 : int
        Bottom-right corner (column, row) — exclusive.
    out_dir : str
        Output directory.
    """
    folder_path = Path(folder)
    images      = _get_sorted_images(folder_path)
    if not images:
        print(f"No images found in: {folder}")
        return
    if not (0 <= index < len(images)):
        print(f"Index {index} out of range.  Folder contains {len(images)} images.")
        return

    img_path = images[index]
    img      = Image.open(img_path)
    w, h     = img.size
    print(f"Source: {img_path.name}  {w}×{h} px")
    print(f"Crop  : x=[{x0},{x1}), y=[{y0},{y1})")

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x0 >= x1 or y0 >= y1:
        raise ValueError(
            f"Invalid crop region: x=[{x0},{x1}), y=[{y0},{y1}) "
            f"for image {w}×{h}"
        )

    cropped  = img.crop((x0, y0, x1, y1))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / img_path.name
    cropped.save(str(save_path), quality=100)
    cw, ch = x1 - x0, y1 - y0
    print(f"Saved: {save_path}  {cw}×{ch} px")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="View a folder image by index, then optionally crop it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
    # View image at index 7 (shows coordinate grid for choosing crop corners)
    python image_processing/visualize_and_crop.py /data/my_folder --index 7 --view

    # Crop it once you've noted the coordinates
    python image_processing/visualize_and_crop.py /data/my_folder --index 7 \\
        --crop --x0 200 --y0 100 --x1 5400 --y1 3600 --out_dir /data/cropped
""",
    )
    parser.add_argument("folder", help="Image directory.")
    parser.add_argument("--index", type=int, default=0,
                        help="Zero-based image index (default: 0).")
    parser.add_argument("--view", action="store_true",
                        help="Display the image at the given index.")
    parser.add_argument("--crop", action="store_true",
                        help="Crop the image and save it.")
    parser.add_argument("--x0", type=int, default=0)
    parser.add_argument("--y0", type=int, default=0)
    parser.add_argument("--x1", type=int, default=None)
    parser.add_argument("--y1", type=int, default=None)
    parser.add_argument("--out_dir", default="cropped",
                        help="Output directory for cropped image (default: cropped).")
    args = parser.parse_args()

    if not args.view and not args.crop:
        parser.error("Specify at least one of --view or --crop.")

    if args.view:
        view_image(args.folder, args.index)

    if args.crop:
        if args.x1 is None or args.y1 is None:
            parser.error("--crop requires --x1 and --y1.")
        crop_image(args.folder, args.index,
                   args.x0, args.y0, args.x1, args.y1,
                   args.out_dir)


if __name__ == "__main__":
    main()
