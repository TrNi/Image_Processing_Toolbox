"""
jpg_to_h5.py
============
Pack a directory of JPEG / PNG images into a single HDF5 file.

Images are stored under the key ``'data'`` as an ``(N, C, H, W)`` uint8
array.  Handles both lower-case and upper-case extensions.

Usage
-----
::

    python file_tools/jpg_to_h5.py /path/to/images output.h5 \\
        --key data --max 200 --resize 1080 720
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import h5py
from PIL import Image


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def jpg_to_h5(image_dir: str, out_path: str, key: str = 'data',
              max_images: int = None, resize: tuple = None):
    """Pack images from *image_dir* into a single HDF5 file.

    Parameters
    ----------
    image_dir : str
        Source directory.
    out_path : str
        Output ``*.h5`` path.
    key : str
        HDF5 dataset key (default: ``'data'``).
    max_images : int, optional
        Maximum number of images to pack (default: all).
    resize : tuple of (width, height), optional
        Resize all images to this size before packing.
    """
    dir_path = Path(image_dir)
    img_files = sorted(
        f for f in dir_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )
    if not img_files:
        print(f"No images found in: {image_dir}")
        return

    if max_images is not None:
        img_files = img_files[:max_images]
    print(f"Packing {len(img_files)} images → {out_path}")

    # Load first image to determine shape
    first = np.array(Image.open(img_files[0]).convert('RGB'))
    if resize is not None:
        w, h = resize
        first = np.array(Image.fromarray(first).resize((w, h), Image.BILINEAR))
    H, W, C = first.shape

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as fh:
        ds = fh.create_dataset(
            key,
            shape=(len(img_files), C, H, W),
            dtype=np.uint8,
            chunks=(1, C, H, W),
            compression='gzip', compression_opts=4,
        )
        for i, img_file in enumerate(img_files):
            arr = np.array(Image.open(img_file).convert('RGB'))
            if resize is not None:
                arr = np.array(Image.fromarray(arr).resize((w, h), Image.BILINEAR))
            if arr.shape[:2] != (H, W):
                raise ValueError(
                    f"Image {img_file.name} has shape {arr.shape[:2]}, "
                    f"expected ({H}, {W}).  "
                    "Use --resize to enforce a common size."
                )
            ds[i] = np.transpose(arr, (2, 0, 1))
            if (i + 1) % max(1, len(img_files) // 10) == 0:
                print(f"  [{i+1}/{len(img_files)}] {img_file.name}")

    size_mb = Path(out_path).stat().st_size / 1024**2
    print(f"\nDone.  {out_path}  shape=({len(img_files)},{C},{H},{W})  {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Pack a directory of images into a single HDF5 file.",
    )
    parser.add_argument("image_dir", help="Source directory.")
    parser.add_argument("out_path",  help="Output HDF5 path.")
    parser.add_argument("--key",     default="data",
                        help="HDF5 dataset key (default: data).")
    parser.add_argument("--max",     type=int, default=None,
                        dest="max_images",
                        help="Maximum number of images to pack.")
    parser.add_argument("--resize",  nargs=2, type=int, default=None,
                        metavar=('W', 'H'),
                        help="Resize images to W×H pixels before packing.")
    args = parser.parse_args()
    jpg_to_h5(
        args.image_dir, args.out_path,
        key=args.key,
        max_images=args.max_images,
        resize=tuple(args.resize) if args.resize else None,
    )


if __name__ == "__main__":
    main()
