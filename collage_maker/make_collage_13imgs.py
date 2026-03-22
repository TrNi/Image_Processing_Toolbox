"""Build a 13-image collage that matches the ICCP single-column width.

Layout overview:
    ┌──────┬────────┬────────┬────────┬────────┐
    │Label │ Img 1  │ Img 2  │ Img 3  │ Img 4  │  ← Reflective row (row 1)
    │      ├────────┼────────┼────────┼───┬────┤
    │Label │ Img 5  │ Img 6  │ Img 7  │ 8 │    │  ← Semi-transparent (row 2)
    │      │ (Img5–7 span rows 2–3)     │  │    │
    │      ├────────┴────────┴────────┴───┼────┤
    │      │      Img 5–7 continue        │  9 │  ← Semi-transparent (row 3)
    │Label │  Img10 │  Img11 │  Img12 │  Img13 │  ← Fine details (row 4, no spans)
    └──────┴────────┴────────┴────────┴────────┘

Usage examples:
    python collage_maker/make_collage_13imgs.py \
        img01.jpg ... img13.jpg --backend pil --output collage13

    python collage_maker/make_collage_13imgs.py \
        img01.jpg ... img13.jpg --backend both --labels "Reflective" "Glass" "Fine"
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:  # Optional dependency for OpenCV backend
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

Backend = Literal["pil", "cv2", "both"]
COL_FRACTIONS_BY_ROW: dict[int, Tuple[float, float, float, float, float]] = {
    0: (0.0, 0.23+0.025, 0.22+0.025, 0.23+0.025, 0.22+0.025),
    1: (0.0, 0.15+0.025, 0.38+0.025, 0.15+0.025, 0.22+0.025),  # shared with row 2 for spanning panels
    2: (0.0, 0.15+0.025, 0.38+0.025, 0.15+0.025, 0.22+0.025),
    3: (0.0, 0.27+0.025, 0.26+0.025, 0.15+0.025, 0.22+0.025),
}
ROW_FRACTIONS: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
DEFAULT_LABELS = (
    "Reflective surfaces",
    "Semi-transparent surfaces",
    "Fine details",
)
LABEL_BG = (244, 245, 248)
LABEL_FG = (32, 34, 42)
CANVAS_BG = (255, 255, 255)


@dataclass(frozen=True)
class Slot:
    name: str
    grid_col: Tuple[int, int]
    grid_row: Tuple[int, int]


IMAGE_SLOTS: Sequence[Slot] = (
    Slot("img01", (1, 2), (0, 1)),
    Slot("img02", (2, 3), (0, 1)),
    Slot("img03", (3, 4), (0, 1)),
    Slot("img04", (4, 5), (0, 1)),
    Slot("img05", (1, 2), (1, 3)),
    Slot("img06", (2, 3), (1, 3)),
    Slot("img07", (3, 4), (1, 3)),
    Slot("img08", (4, 5), (1, 2)),
    Slot("img09", (4, 5), (2, 3)),
    Slot("img10", (1, 2), (3, 4)),
    Slot("img11", (2, 3), (3, 4)),
    Slot("img12", (3, 4), (3, 4)),
    Slot("img13", (4, 5), (3, 4)),
)

LABEL_SLOTS: Sequence[Slot] = (
    Slot("label_top", (0, 1), (0, 1)),
    Slot("label_mid", (0, 1), (1, 3)),
    Slot("label_bottom", (0, 1), (3, 4)),
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _px_sizes(total: int, fractions: Iterable[float], gap: int) -> List[int]:
    fracs = list(fractions)
    available = total - gap * (len(fracs) - 1)
    raw = [available * f for f in fracs]
    rounded = [int(round(val)) for val in raw]
    diff = available - sum(rounded)
    for idx in np.argsort([abs(r - val) for r, val in zip(rounded, raw)]):
        if diff == 0:
            break
        step = 1 if diff > 0 else -1
        rounded[idx] += step
        diff -= step
    rounded[-1] += available - sum(rounded)
    return rounded


def _positions(sizes: Sequence[int], gap: int) -> List[int]:
    coords = [0]
    for size in sizes[:-1]:
        coords.append(coords[-1] + size + gap)
    return coords


def _layout_boxes(width_px: int, height_px: int, gap: int) -> dict[str, Tuple[int, int, int, int]]:
    row_sizes = _px_sizes(height_px, ROW_FRACTIONS, gap)
    row_pos = _positions(row_sizes, gap)

    col_sizes_by_row = {}
    col_pos_by_row = {}
    for r in range(len(ROW_FRACTIONS)):
        if r not in COL_FRACTIONS_BY_ROW:
            raise ValueError(f"Missing column fractions for row {r}")
        col_sizes = _px_sizes(width_px, COL_FRACTIONS_BY_ROW[r], gap)
        col_sizes_by_row[r] = col_sizes
        col_pos_by_row[r] = _positions(col_sizes, gap)

    boxes: dict[str, Tuple[int, int, int, int]] = {}
    for slot in (*IMAGE_SLOTS, *LABEL_SLOTS):
        c0, c1 = slot.grid_col
        r0, r1 = slot.grid_row

        # Ensure spanning rows share identical column widths
        first_cols = col_sizes_by_row[r0]
        for r in range(r0 + 1, r1):
            if col_sizes_by_row[r] != first_cols:
                raise ValueError(f"Column fractions must match for rows {r0}-{r1-1} due to spanning slot '{slot.name}'")

        col_pos = col_pos_by_row[r0]
        x = col_pos[c0]
        y = row_pos[r0]
        w = sum(first_cols[c0:c1]) + gap * (c1 - c0 - 1)
        h = sum(row_sizes[r0:r1]) + gap * (r1 - r0 - 1)
        boxes[slot.name] = (x, y, w, h)
    return boxes


# ---------------------------------------------------------------------------
# Image resizing helpers
# ---------------------------------------------------------------------------

def _resize_cover_pil(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_h = target_h
        new_w = int(round(new_h * src_ratio))
    else:
        new_w = target_w
        new_h = int(round(new_w / src_ratio))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def _resize_cover_cv(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Label rendering
# ---------------------------------------------------------------------------

def _load_font(base_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ("SourceSansPro-SemiBold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(font_name, base_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_label_patch(width: int, height: int, text: str) -> Image.Image:
    label = Image.new("RGB", (width, height), LABEL_BG)
    text_canvas = Image.new("RGBA", (height, width), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_canvas)

    # Fit text so that after rotation it stays within the slot
    base_size = max(24, int(min(width, height) * 0.8))
    font = _load_font(base_size)
    for size in range(base_size, 4, -1):
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        # After 90° rotation, dimensions swap
        if bh <= width * 0.9 and bw <= height * 0.9:
            break

    tx = (height - bw) / 2
    ty = (width - bh) / 2
    draw.text((tx, ty), text, font=font, fill=LABEL_FG + (255,))

    text_rot = text_canvas.rotate(90, expand=True)
    label = label.convert("RGBA")
    label.alpha_composite(text_rot.crop((0, 0, width, height)))
    return label.convert("RGB")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _build_pil(images: Sequence[Path], labels: Sequence[str], 
    width_px: int, height_px: int, gap: int, dest: Path, dpi: int = 600) -> None:
    boxes = _layout_boxes(width_px, height_px, gap)
    canvas = Image.new("RGB", (width_px, height_px), color=CANVAS_BG)

    for img_path, slot in zip(images, IMAGE_SLOTS):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            x, y, w, h = boxes[slot.name]
            canvas.paste(_resize_cover_pil(img, w, h), (x, y))

    for text, slot in zip(labels, LABEL_SLOTS):
        label_img = _render_label_patch(*boxes[slot.name][2:], text)
        x, y, w, h = boxes[slot.name]
        canvas.paste(label_img, (x, y))

    base = dest.with_suffix("") 
    
    out_png = base.with_suffix(f".{dpi}dpi.png")
    out_pdf = base.with_suffix(f".{dpi}dpi.pdf")
    canvas.save(out_png, format="PNG", dpi=(dpi, dpi))
    canvas.save(out_pdf, format="PDF", dpi=(dpi, dpi))
    print(f"Saved PIL collage -> {out_png}")
    print(f"Saved PIL collage -> {out_pdf}")


def _build_cv2(images: Sequence[Path], labels: Sequence[str], 
    width_px: int, height_px: int, gap: int, dest: Path, dpi: int = 600) -> None:
    if cv2 is None:  # pragma: no cover
        raise ImportError("OpenCV (cv2) is not installed; install opencv-python to use this backend.")

    boxes = _layout_boxes(width_px, height_px, gap)
    canvas = np.full((height_px, width_px, 3), CANVAS_BG, dtype=np.uint8)

    for img_path, slot in zip(images, IMAGE_SLOTS):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x, y, w, h = boxes[slot.name]
        patch = _resize_cover_cv(rgb, w, h)
        canvas[y : y + h, x : x + w] = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

    for text, slot in zip(labels, LABEL_SLOTS):
        x, y, w, h = boxes[slot.name]
        label_img = _render_label_patch(w, h, text)
        canvas[y : y + h, x : x + w] = cv2.cvtColor(np.array(label_img), cv2.COLOR_RGB2BGR)

    base = dest.with_suffix("")
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    
    out_png = base.with_suffix(f".{dpi}dpi.png")
    out_pdf = base.with_suffix(f".{dpi}dpi.pdf")
    pil_img.save(out_png, format="PNG", dpi=(dpi, dpi))
    pil_img.save(out_pdf, format="PDF", dpi=(dpi, dpi))
    print(f"Saved OpenCV collage -> {out_png}")
    print(f"Saved OpenCV collage -> {out_pdf}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_collage(
    images: Sequence[Path],
    backend: Backend,
    width_in: float,
    height_ratio: float,
    dpi: int,
    gap: int,
    labels: Sequence[str],
    output: Path,
) -> None:
    if len(images) != len(IMAGE_SLOTS):
        raise ValueError(f"Expected {len(IMAGE_SLOTS)} images, received {len(images)}")
    if len(labels) != len(LABEL_SLOTS):
        raise ValueError(f"Expected {len(LABEL_SLOTS)} labels, received {len(labels)}")

    width_px = int(round(width_in * dpi))
    height_px = int(round(width_px * height_ratio))

    if backend in {"pil", "both"}:
        pil_path = output.with_suffix(".pil.jpg") if backend == "both" else output
        _build_pil(images, labels, width_px, height_px, gap, pil_path, dpi)
    if backend in {"cv2", "both"}:
        cv_path = output.with_suffix(".cv2.jpg") if backend == "both" else output
        _build_cv2(images, labels, width_px, height_px, gap, cv_path, dpi)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble 13 images into the ICCP-width collage layout.")
    parser.add_argument("images", nargs=len(IMAGE_SLOTS), type=Path, help="Paths to 13 source images in layout order.")
    parser.add_argument("--backend", choices=["pil", "cv2", "both"], default="pil", help="Rendering backend (default: PIL)")
    parser.add_argument("--width-in", type=float, default=6.5, help="Target paper width in inches (default: 6.5).")
    parser.add_argument("--height-ratio", type=float, default=4.2 / 6.5, help="Height as a fraction of width (default: 4.2 / 6.5).")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for sizing calculations (default: 300).")
    parser.add_argument("--gap", type=int, default=6, help="Whitespace between tiles in pixels (default: 6).")
    parser.add_argument(
        "--labels",
        nargs=len(LABEL_SLOTS),
        default=DEFAULT_LABELS,
        help="Three label strings for the category strips (top, middle, bottom).",
    )
    parser.add_argument("--output", type=Path, default=Path("collage13.jpg"), help="Destination file (extension inferred).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_collage(
        images=args.images,
        backend=args.backend,
        width_in=args.width_in,
        height_ratio=args.height_ratio,
        dpi=args.dpi,
        gap=args.gap,
        labels=args.labels,
        output=args.output,
    )


if __name__ == "__main__":
    main()
