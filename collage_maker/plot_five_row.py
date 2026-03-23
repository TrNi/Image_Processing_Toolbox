"""Plot 5 images in a single row at ICCP text width using a PIL canvas (no matplotlib).

Features
--------
- 5 panels across a single row, sized to ICCP full-page text width (~6.5 in by default).
- Vertical category label on the left; individual titles above each image.
- Publication-style fonts (Times New Roman if available).
- Saves PNG and PDF at configurable DPI (default 600).

Usage
-----
python collage_maker/plot_five_row.py \
    img1.jpg img2.jpg img3.jpg img4.jpg img5.jpg \
    --titles "A" "B" "C" "D" "E" \
    --side-label "Category" \
    --width-in 6.5 --gap 4 --dpi 600 \
    --output collage_five_row
"""

import argparse
import re
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

# Match a standalone 'f' surrounded on both sides by whitespace, slash, or string boundary
_F_ITALIC_RE = re.compile(r'(?<![^\s/])f(?![^\s/])')


def _load_font(size):
    for name in ("Times New Roman.ttf", "times.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_font_italic(size):
    for name in ("Times New Roman Italic.ttf", "timesi.ttf", "ariali.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return _load_font(size)


def _text_segments(text):
    """Split *text* into [(substr, is_italic)] where standalone 'f' chars are italic."""
    segments = []
    last = 0
    for m in _F_ITALIC_RE.finditer(text):
        if m.start() > last:
            segments.append((text[last:m.start()], False))
        segments.append((text[m.start():m.end()], True))
        last = m.end()
    if last < len(text):
        segments.append((text[last:], False))
    return segments or [(text, False)]


def _measure_mixed(segments, font_reg, font_ital, probe):
    """Return (total_width, max_height) for mixed-style segments."""
    tw, th = 0, 0
    for substr, italic in segments:
        font = font_ital if italic else font_reg
        tw += int(round(probe.textlength(substr, font=font)))
        bb = probe.textbbox((0, 0), substr, font=font)
        th = max(th, bb[3] - bb[1])
    return tw, th


def _draw_mixed(draw, x, y, segments, font_reg, font_ital, fill=(0, 0, 0)):
    """Draw mixed-style segments left-to-right starting at (x, y)."""
    cx = x
    for substr, italic in segments:
        font = font_ital if italic else font_reg
        draw.text((cx, y), substr, font=font, fill=fill)
        cx += int(round(draw.textlength(substr, font=font)))


def _resize_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
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


def plot_five_row(
    images: Sequence[Path],
    titles: Sequence[str],
    side_label: str,
    width_in: float,
    dpi: int,
    gap: int,
    output: Path,
):
    if len(images) != 5:
        raise ValueError("Exactly 5 images are required.")
    if len(titles) != 5:
        raise ValueError("Provide exactly 5 titles (one per image).")

    width_px = int(round(width_in * dpi))

    # Auto-detect aspect ratio from first image
    with Image.open(images[0]) as ref:
        ref_w, ref_h = ref.size
    img_aspect = ref_h / ref_w

    # Font sizes: ~7pt equivalent at target DPI
    font_size = max(8, int(width_px * 0.014)) # 0.018
    font_title = _load_font(font_size)
    font_title_ital = _load_font_italic(font_size)
    font_side = _load_font(font_size)
    font_side_ital = _load_font_italic(font_size)

    # Probe text heights using a dummy draw
    _probe_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    tb_ref = _probe_draw.textbbox((0, 0), "Ag", font=font_title)
    title_h = (tb_ref[3] - tb_ref[1]) + 3  # ascent/descent + 1px above + 2px below

    side_segs = _text_segments(side_label)
    side_text_w, _ = _measure_mixed(side_segs, font_side, font_side_ital, _probe_draw)
    # Use full em height (ascent+descent) so the column is never undersized
    try:
        _asc, _desc = font_side.getmetrics()
        side_line_h = _asc + _desc
    except AttributeError:
        side_line_h = max(8, int(font_size * 1.2))
    side_label_w = side_line_h + 2 * gap  # column width = full em + gap on each side

    # Panel dims
    # total width = side_label_w + gap + 5*panel_w + 4*gap (between panels)
    panel_w = (width_px - side_label_w - gap * 5) // 5
    panel_h = int(round(panel_w * img_aspect)) 

    # Canvas height = title row + gap + panels + gap
    height_px = title_h + 2*gap + panel_h + gap

    canvas = Image.new("RGB", (width_px, height_px), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Side label: draw mixed text on temp image, rotate 90°, paste snug to left
    # tmp is (text_length + pad, full_em + pad) so nothing clips after rotation
    tmp = Image.new("RGBA", (side_text_w + 4, side_line_h + 4), (255, 255, 255, 0))
    _draw_mixed(ImageDraw.Draw(tmp), 2, 2, side_segs, font_side, font_side_ital)
    tmp = tmp.rotate(90, expand=True)  # -> (side_line_h+4, side_text_w+4)
    rw, rh = tmp.size
    sx = (side_label_w - rw) // 2
    sy = title_h + gap + (panel_h - rh) // 2
    canvas.paste(tmp, (sx, sy), tmp)

    # Panels and per-image titles
    x0 = side_label_w + gap
    y_img = title_h + 2*gap
    for idx, (img_path, title) in enumerate(zip(images, titles)):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            panel = _resize_cover(img, panel_w, panel_h)
        x_panel = x0 + idx * (panel_w + gap)
        canvas.paste(panel, (x_panel, y_img))

        t_segs = _text_segments(title)
        tw, th = _measure_mixed(t_segs, font_title, font_title_ital, _probe_draw)
        tx = x_panel + (panel_w - tw) // 2
        ty = (title_h - th) // 2
        _draw_mixed(draw, tx, ty, t_segs, font_title, font_title_ital)

    out_png = output.with_suffix(".png")
    out_pdf = output.with_suffix(".pdf")
    canvas.save(out_png, format="PNG", dpi=(dpi, dpi))
    canvas.save(out_pdf, format="PDF", dpi=(dpi, dpi))
    print(f"Saved PNG -> {out_png}")
    print(f"Saved PDF -> {out_pdf}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 5 images in a row at ICCP text width with vertical side label and per-image titles (PIL)."
    )
    parser.add_argument("images", nargs=5, type=Path, help="Paths to 5 images in left-to-right order.")
    parser.add_argument(
        "--titles",
        nargs=5,
        default=["A", "B", "C", "D", "E"],
        help="Five titles for the images (left to right).",
    )
    parser.add_argument(
        "--side-label",
        default="Category",
        help="Vertical label placed on the left side of the row.",
    )
    parser.add_argument(
        "--width-in",
        type=float,
        default=6.5,
        help="Figure width in inches (ICCP full-page textwidth).",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=4,
        help="Gap in pixels between panels (default: 4).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output DPI (default: 600).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("collage_five_row"),
        help="Base output path (extensions added automatically).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_five_row(
        images=args.images,
        titles=args.titles,
        side_label=args.side_label,
        width_in=args.width_in,
        dpi=args.dpi,
        gap=args.gap,
        output=args.output,
    )


if __name__ == "__main__":
    main()
