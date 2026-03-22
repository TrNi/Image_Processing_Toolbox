"""Generate a seven-tile collage that matches the ICCP single-column width.

The layout mirrors the provided reference:
    ┌───────┬─────────────────────────┬───────┬
    │ Img 1 │           Img 2                │ Img 7 │
    │       ├──────────┬──────────────┬──────┤       │
    │       │ Img 3    │  Img 4       │ Img5 │       │
    │       ├──────────┴──────────────┴──────┤       │
    │       │           Img 6                │       │
    └───────┴─────────────────────────────────┴───────┴

Usage (PIL backend):
    python collage_maker/make_collage.py \
        image1.jpg image2.jpg image3.jpg image4.jpg image5.jpg image6.jpg image7.jpg \
        --backend pil --output collage

Usage (OpenCV backend):
    python collage_maker/make_collage.py \
        img1.png img2.png img3.png img4.png img5.png img6.png img7.png \
        --backend cv2 --output collage_cv

Use "--backend both" to render each version in a single invocation.
"""
import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

try:  # Optional dependency for OpenCV backend
    import cv2
except ImportError:  # cv2 may not be installed
    cv2 = None


COL_FRACTIONS = (0.20, 0.20, 0.20, 0.20, 0.20)
ROW_FRACTIONS = (0.40, 0.35, 0.25)


class Panel:
    def __init__(self, name, grid_col, grid_row):
        self.name = name
        self.grid_col = grid_col
        self.grid_row = grid_row


PANELS = (
    Panel("left", (0, 1), (0, 3)),
    Panel("top", (1, 4), (0, 1)),
    Panel("mid_left", (1, 2), (1, 2)),
    Panel("mid_center", (2, 3), (1, 2)),
    Panel("mid_right", (3, 4), (1, 2)),
    Panel("bottom", (1, 4), (2, 3)),
    Panel("right", (4, 5), (0, 3)),
)


def _px_sizes(total, fractions, gap):
    """Convert fractional splits to pixel sizes while respecting gaps."""
    fracs = list(fractions)
    available = total - gap * (len(fracs) - 1)
    raw_sizes = [available * f for f in fracs]
    rounded = [int(round(size)) for size in raw_sizes]
    diff = available - sum(rounded)
    # Correct rounding drift on the largest remainder
    if diff:
        order = np.argsort([abs(r - size) for r, size in zip(rounded, raw_sizes)])
        for idx in order:
            if diff == 0:
                break
            rounded[idx] += int(np.sign(diff))
            diff -= int(np.sign(diff))
    rounded[-1] += available - sum(rounded)
    return rounded


def _positions(sizes, gap):
    pos = [0]
    for size in sizes[:-1]:
        pos.append(pos[-1] + size + gap)
    return pos


def _compute_boxes(width_px, height_px, gap):
    col_sizes = _px_sizes(width_px, COL_FRACTIONS, gap)
    row_sizes = _px_sizes(height_px, ROW_FRACTIONS, gap)
    col_positions = _positions(col_sizes, gap)
    row_positions = _positions(row_sizes, gap)

    boxes = []
    for panel in PANELS:
        c0, c1 = panel.grid_col
        r0, r1 = panel.grid_row
        x = col_positions[c0]
        y = row_positions[r0]
        w = sum(col_sizes[c0:c1]) + gap * (c1 - c0 - 1)
        h = sum(row_sizes[r0:r1]) + gap * (r1 - r0 - 1)
        boxes.append((x, y, w, h))
    return boxes


def _resize_cover_pil(img, target_w, target_h):
    src_w, src_h = img.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_h = target_h
        new_w = int(round(new_h * src_ratio))
    else:
        new_w = target_w
        new_h = int(round(new_w / src_ratio))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _resize_cover_cv(img, target_w, target_h):
    src_h, src_w = img.shape[:2]
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_h = target_h
        new_w = int(round(new_h * src_ratio))
    else:
        new_w = target_w
        new_h = int(round(new_w / src_ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (new_w - target_w) // 2
    y0 = (new_h - target_h) // 2
    return resized[y0 : y0 + target_h, x0 : x0 + target_w]


def _build_pil(images, width_px, height_px, gap, dest, dpi):
    boxes = _compute_boxes(width_px, height_px, gap)
    canvas = Image.new("RGB", (width_px, height_px), color=(255, 255, 255))
    for img_path, box in zip(images, boxes):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            x, y, w, h = box
            canvas.paste(_resize_cover_pil(img, w, h), (x, y))
    base = dest.with_suffix("")
    out_png = base.with_suffix(f".{dpi}dpi.png")
    out_pdf = base.with_suffix(f".{dpi}dpi.pdf")
    canvas.save(out_png, format="PNG", dpi=(dpi, dpi))
    canvas.save(out_pdf, format="PDF", dpi=(dpi, dpi))
    print(f"Saved PIL collage -> {out_png}")
    print(f"Saved PIL collage -> {out_pdf}")


def _build_cv2(images, width_px, height_px, gap, dest, dpi=600):
    if cv2 is None:  # runtime guard
        raise ImportError("OpenCV (cv2) is not installed; install opencv-python to use this backend.")
    boxes = _compute_boxes(width_px, height_px, gap)
    canvas = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    for img_path, box in zip(images, boxes):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        x, y, w, h = box
        patch = _resize_cover_cv(img, w, h)
        canvas[y : y + h, x : x + w] = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    base = dest.with_suffix("")
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    
    out_png = base.with_suffix(f".{dpi}dpi.png")
    out_pdf = base.with_suffix(f".{dpi}dpi.pdf")
    pil_img.save(out_png, format="PNG", dpi=(dpi, dpi))
    pil_img.save(out_pdf, format="PDF", dpi=(dpi, dpi))
    print(f"Saved OpenCV collage -> {out_png}")
    print(f"Saved OpenCV collage -> {out_pdf}")


def make_collage(images, backend, width_in, dpi, gap, height_ratio, output):
    if len(images) != 7:
        raise ValueError("Exactly seven images are required to build the collage.")

    width_px = int(round(width_in * dpi))
    height_px = int(round(width_px * height_ratio))

    if backend in {"pil", "both"}:
        pil_path = output.with_suffix(".pil.jpg") if backend == "both" else output
        _build_pil(images, width_px, height_px, gap, pil_path, dpi)
    if backend in {"cv2", "both"}:
        cv_path = output.with_suffix(".cv2.jpg") if backend == "both" else output
        _build_cv2(images, width_px, height_px, gap, cv_path, dpi)


def parse_args():
    parser = argparse.ArgumentParser(description="Assemble seven images into a tight ICCP-width collage.")
    parser.add_argument("images", nargs=7, type=Path, help="Paths to the seven source images.")
    parser.add_argument("--backend", choices=["pil", "cv2", "both"], default="pil", help="Rendering backend.")
    parser.add_argument("--width-in", type=float, default=6.5, help="Target paper width in inches (default: 6.5).")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for sizing calculations (default: 300).")
    parser.add_argument("--height-ratio", type=float, default=3.5 / 6.5, help="Height as a fraction of width (default matches sample aspect).")
    parser.add_argument("--gap", type=int, default=4, help="Whitespace between tiles in pixels (default: 6).")
    parser.add_argument("--output", type=Path, default=Path("collage.jpg"), help="Destination file (extension inferred).")
    return parser.parse_args()


def main():
    args = parse_args()
    make_collage(
        images=args.images,
        backend=args.backend, 
        width_in=args.width_in,
        dpi=args.dpi,
        gap=args.gap,
        height_ratio=args.height_ratio,
        output=args.output,        
    )


if __name__ == "__main__":
    main()
