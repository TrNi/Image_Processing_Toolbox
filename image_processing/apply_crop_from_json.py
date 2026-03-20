"""
apply_crop_from_json.py
=======================
Apply crop regions stored in alignment JSON files to one or more image directories.

JSON files (produced by :mod:`image_processing.align_whitebal`) must contain
the keys ``crop_x0``, ``crop_y0``, ``crop_x1``, ``crop_y1``.

Images and JSON files are matched by lexicographic sort order.

Usage
-----
::

    python image_processing/apply_crop_from_json.py \\
        /path/to/json_dir \\
        /path/to/images_dir1 /path/to/images_dir2 \\
        --output_suffix _cropped
"""

import sys
import json
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def apply_crop_from_json(json_dir: str, source_dirs: list,
                         output_suffix: str = "_cropped"):
    """Crop images in each *source_dir* using the JSON files in *json_dir*.

    Parameters
    ----------
    json_dir : str
        Directory containing ``*.json`` alignment files.
    source_dirs : list of str
        One or more directories of images to crop.
    output_suffix : str
        Suffix appended to each source directory name for the output.
    """
    json_path  = Path(json_dir)
    json_files = sorted(json_path.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {json_dir}")
    print(f"Found {len(json_files)} JSON file(s) in {json_dir}")

    for src_dir in source_dirs:
        src_path = Path(src_dir)
        if not src_path.exists():
            print(f"WARNING: Source directory not found: {src_dir}")
            continue

        exts  = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        imgs  = sorted(f for f in src_path.glob("*") if f.suffix.lower() in exts)
        if not imgs:
            print(f"WARNING: No images in: {src_dir}")
            continue

        n_pairs = min(len(imgs), len(json_files))
        if len(imgs) != len(json_files):
            print(f"WARNING: {src_dir} — {len(imgs)} images vs {len(json_files)} JSONs; "
                  f"processing {n_pairs} pairs.")

        out_dir = src_path.parent / f"{src_path.name}{output_suffix}"
        out_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing: {src_dir}")
        print(f"Output  → : {out_dir}")
        print(f"{'='*60}")

        for i in range(n_pairs):
            img_path  = imgs[i]
            json_file = json_files[i]

            try:
                with open(json_file) as fh:
                    data = json.load(fh)
                x0 = data["crop_x0"]
                y0 = data["crop_y0"]
                x1 = data["crop_x1"]
                y1 = data["crop_y1"]
            except (KeyError, json.JSONDecodeError) as exc:
                print(f"  ERROR loading {json_file.name}: {exc}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ERROR loading image: {img_path.name}")
                continue

            cropped    = img[y0:y1, x0:x1]
            out_path   = out_dir / f"{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"  [{i+1}/{n_pairs}] {img_path.name} → {out_path.name} "
                  f"({x1-x0}×{y1-y0})")

        print(f"Completed: {out_dir}")

    print(f"\n{'='*60}\nAll directories processed.\n{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply crop regions from alignment JSONs to image directories.",
    )
    parser.add_argument("json_dir",
                        help="Directory containing alignment JSON files.")
    parser.add_argument("source_dirs", nargs="+",
                        help="One or more source image directories to crop.")
    parser.add_argument("--output_suffix", default="_cropped",
                        help="Suffix for output directory names (default: _cropped).")
    args = parser.parse_args()

    apply_crop_from_json(args.json_dir, args.source_dirs, args.output_suffix)


if __name__ == "__main__":
    main()
