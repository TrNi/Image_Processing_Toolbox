"""
crop_images.py
==============
Multi-region batch crop with normalised coordinate array export.

For each image in the input list and each specified crop region, this script:

- Saves a per-image batch file ``<stem>_crops.npy`` of shape
  ``(P, C, H_crop, W_crop)`` where P = number of regions.
- Saves per-region normalised coordinate arrays ``coords_region<i>.npy``
  of shape ``(1, 1, H_crop, W_crop, 2)`` — useful for coordinate-based
  neural network conditioning.
- Saves a combined ``coords_all_regions.npy`` of shape ``(P, H_crop, W_crop, 2)``.
- Saves individual JPEG crops for visual inspection.
- Writes a ``crop_info.txt`` summary.

Usage
-----
::

    python image_processing/crop_images.py \\
        --images /path/to/img1.jpg /path/to/img2.jpg \\
        --regions 100,200,300,400  500,600,700,800 \\
        --out_dir /path/to/output \\
        --color rgb
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def crop_images_multiple_regions(
    image_paths: list,
    crop_regions: list,
    save_dir: str,
    color_mode: str = 'rgb',
):
    """Crop multiple images using multiple regions and save results.

    Parameters
    ----------
    image_paths : list of str
        Paths to source images (must all have the same dimensions).
    crop_regions : list of tuple
        Each tuple is ``(h_start, h_end, w_start, w_end)``.
    save_dir : str
        Output directory.
    color_mode : str
        ``'rgb'`` (3-channel) or ``'gray'`` (1-channel).

    Returns
    -------
    list of str
        Crop information records written to ``crop_info.txt``.
    """
    if color_mode not in ('rgb', 'gray'):
        raise ValueError(f"color_mode must be 'rgb' or 'gray', got {color_mode!r}")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Validate images and check consistent dimensions
    print("Validating input images …")
    valid_paths    = []
    image_dims     = None

    for p in image_paths:
        p = Path(p)
        if not p.exists():
            print(f"  WARNING: not found: {p}")
            continue
        arr = np.array(Image.open(p))
        h, w = arr.shape[:2]
        if image_dims is None:
            image_dims = (h, w)
            print(f"  Reference dimensions: {h}×{w}")
        elif (h, w) != image_dims:
            raise ValueError(
                f"Dimension mismatch!\n"
                f"  Expected {image_dims[0]}×{image_dims[1]}\n"
                f"  Got      {h}×{w} for {p.name}"
            )
        valid_paths.append(p)
        print(f"  {p.name}: {h}×{w}")

    if not valid_paths:
        raise ValueError("No valid images found.")
    print(f"\n{len(valid_paths)} images at {image_dims[0]}×{image_dims[1]}\n")

    img_h, img_w = image_dims

    # Validate crop coordinates
    for ri, (hs, he, ws, we) in enumerate(crop_regions):
        if hs < 0 or ws < 0 or he > img_h or we > img_w:
            raise ValueError(
                f"Region {ri} out of bounds: H[{hs}:{he}], W[{ws}:{we}] "
                f"for image {img_h}×{img_w}"
            )

    crop_info_list = []
    total_crops    = 0

    print(f"Generating {len(valid_paths)} × {len(crop_regions)} crops …")
    print(f"Color mode: {color_mode.upper()}\n")

    for img_idx, img_path in enumerate(valid_paths):
        img     = Image.open(img_path)
        arr     = np.array(img)

        if color_mode == 'gray' and arr.ndim == 3:
            arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        crops = []
        for hs, he, ws, we in crop_regions:
            crop = arr[hs:he, ws:we]
            if color_mode == 'gray':
                if crop.ndim == 2:
                    crop = crop[:, :, np.newaxis]
            if crop.ndim == 3:
                crop = np.transpose(crop, (2, 0, 1))
            else:
                crop = crop[np.newaxis]
            crops.append(crop)

        batch = np.stack(crops, axis=0)   # (P, C, H_crop, W_crop)
        batch_file = save_path / f"{img_path.stem}_crops.npy"
        np.save(str(batch_file), batch)
        print(f"[{img_idx+1}/{len(valid_paths)}] {batch_file.name}  "
              f"shape={batch.shape}")

    # Coordinate arrays
    cx = (img_w - 1) / 2.0
    cy = (img_h - 1) / 2.0
    coord_arrays = []

    for ri, (hs, he, ws, we) in enumerate(crop_regions):
        y_c = np.arange(hs, he)
        x_c = np.arange(ws, we)
        yy, xx = np.meshgrid(y_c, x_c, indexing='ij')
        yy_n = (yy - cy) / cy
        xx_n = (xx - cx) / cx
        coords = np.stack([yy_n, xx_n], axis=-1)   # (H_crop, W_crop, 2)
        coords_5d = coords[np.newaxis, np.newaxis]  # (1,1,H_crop,W_crop,2)
        np.save(str(save_path / f"coords_region{ri}.npy"), coords_5d)
        coord_arrays.append(coords)
        print(f"Saved coords_region{ri}.npy  "
              f"Y=[{yy_n.min():.4f},{yy_n.max():.4f}]  "
              f"X=[{xx_n.min():.4f},{xx_n.max():.4f}]")

        # Individual JPEG crops
        for img_path in valid_paths:
            img  = Image.open(img_path)
            arr  = np.array(img)
            crop = arr[hs:he, ws:we]
            Image.fromarray(crop).save(
                str(save_path / f"{img_path.stem}_crop_region{ri}.jpg"),
                'JPEG', quality=95,
            )
            crop_info_list.append(f"{img_path.name},{hs},{he},{ws},{we},{ri}")
            total_crops += 1

    # Combined coordinate array
    overall = np.stack(coord_arrays, axis=0)  # (P, H_crop, W_crop, 2)
    np.save(str(save_path / "coords_all_regions.npy"), overall)
    print(f"\nSaved coords_all_regions.npy  shape={overall.shape}")

    # Crop info text
    info_path = save_path / "crop_info.txt"
    with open(info_path, 'w') as fh:
        fh.write("image_name,h_start,h_end,w_start,w_end,region_index\n")
        for line in crop_info_list:
            fh.write(line + "\n")

    print(f"\n{'='*60}")
    print(f"Total crops:      {total_crops}")
    print(f"Coord arrays:     {len(crop_regions)} per-region + 1 combined")
    print(f"Crop info:        {info_path}")
    print(f"{'='*60}")
    return crop_info_list


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_region(s: str) -> tuple:
    """Parse ``'h_start,h_end,w_start,w_end'`` into a 4-tuple of ints."""
    parts = s.split(',')
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"Region must be 'h_start,h_end,w_start,w_end', got: {s!r}"
        )
    return tuple(map(int, parts))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-region batch crop with coordinate array export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
    python image_processing/crop_images.py \\
        --images img1.jpg img2.jpg \\
        --regions 100,500,200,600  600,1000,200,600 \\
        --out_dir cropped/ --color rgb
""",
    )
    parser.add_argument('--images',  nargs='+', required=True,
                        help="Source image paths.")
    parser.add_argument('--regions', nargs='+', required=True,
                        metavar='H0,H1,W0,W1',
                        help="Crop regions as 'h_start,h_end,w_start,w_end'.")
    parser.add_argument('--out_dir', required=True,
                        help="Output directory.")
    parser.add_argument('--color',   default='rgb', choices=['rgb', 'gray'],
                        help="Color mode (default: rgb).")
    args = parser.parse_args()

    regions = [_parse_region(r) for r in args.regions]
    crop_images_multiple_regions(
        image_paths=args.images,
        crop_regions=regions,
        save_dir=args.out_dir,
        color_mode=args.color,
    )


if __name__ == '__main__':
    main()
