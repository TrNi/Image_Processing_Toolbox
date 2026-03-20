"""
generate_comparison_figure.py
==============================
Generate a 3-column × N-row publication comparison figure.

Each column corresponds to one scene / condition.  Each row corresponds to
one method / model.  The left panel of every column shows the full image with
coloured bounding boxes; the right panel shows the stacked cropped regions.

The layout is driven by plain text configuration files — one per column —
making it easy to reproduce or update figures without editing Python code.

Text-file format
----------------
Lines 1–6:  Three bounding boxes, two lines each — ``x1,y1`` then ``x2,y2``.
Line  7:    ``_,<image_index>``  (image index within the folder).
Lines 8+:   ``<row_label>,<folder_path>``  (one per method/row).

Usage
-----
::

    python visualization/generate_comparison_figure.py \\
        config_scene1.txt config_scene2.txt config_scene3.txt \\
        --output comparison.png

"""

import sys
import os
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif']  = ['Times New Roman']

# ---------------------------------------------------------------------------
# Colour palette for bounding boxes
# ---------------------------------------------------------------------------

BBOX_COLORS = [
    (0.98, 0.95, 0.4),   # yellow
    (0.4,  0.7,  0.98),  # blue
    (0.85, 0.7,  0.95),  # violet
]

BBOX_COLORS_PIL = [
    (250, 242, 102),
    (102, 178, 250),
    (217, 178, 242),
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_config_file(filepath: str) -> tuple:
    """Parse a comparison-figure configuration text file.

    Parameters
    ----------
    filepath : str
        Path to the config file (see module docstring for format).

    Returns
    -------
    bboxes : list of (x1, y1, x2, y2)
        Three bounding boxes.
    index : int
        Image index within each folder.
    folders : list of (label, path)
        One entry per method/row.
    """
    with open(filepath, 'r') as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    bboxes = []
    for i in range(0, 6, 2):
        x1, y1 = map(int, lines[i].split(','))
        x2, y2 = map(int, lines[i + 1].split(','))
        bboxes.append((x1, y1, x2, y2))

    index = int(lines[6].split(',')[1])

    folders = []
    for line in lines[7:]:
        label, path = line.split(',', 1)
        folders.append((label.strip(), path.strip()))

    return bboxes, index, folders


def get_image_at_index(folder_path: str, index: int) -> Image.Image:
    """Load the image at *index* (sorted lexicographically) from *folder_path*."""
    folder = Path(folder_path)
    exts   = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    files  = sorted(f for f in folder.iterdir()
                    if f.is_file() and f.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No images in: {folder_path}")
    if not (0 <= index < len(files)):
        raise IndexError(f"Index {index} out of range ({len(files)} images).")
    return Image.open(files[index])


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_comparison_figure(config_files: list, output_path: str,
                                scale_factor: float = 1.5,
                                border_width: int = 20):
    """Generate the 3-column comparison figure.

    Parameters
    ----------
    config_files : list of str
        Exactly 3 text configuration files (one per scene column).
    output_path : str
        Path to save the output PNG.
    scale_factor : float
        Factor by which to upscale the source images before cropping.
    border_width : int
        Coloured border width in pixels around each crop.
    """
    if len(config_files) != 3:
        raise ValueError("Exactly 3 config files required.")

    all_data = [parse_config_file(f) for f in config_files]
    num_rows = len(all_data[0][2])

    # Width ratios: full image columns at 1.0, crop columns at 0.4
    width_ratios = [1.0, 0.4, 1.0, 0.4, 1.0, 0.4]
    fig_width    = 8.0
    fig_height   = fig_width * num_rows / 3.5 * 0.67

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = fig.add_gridspec(
        num_rows, 6,
        width_ratios=width_ratios,
        hspace=0.02, wspace=0.02,
        left=0.04, right=0.998, top=0.99, bottom=0.01,
    )

    for col_idx, (bboxes, index, folders) in enumerate(all_data):
        for row_idx, (label, folder_path) in enumerate(folders):
            try:
                img = get_image_at_index(folder_path, index)

                new_w = int(img.width  * scale_factor)
                new_h = int(img.height * scale_factor)
                img   = img.resize((new_w, new_h), Image.LANCZOS)

                scaled_bboxes = [
                    (int(x1 * scale_factor), int(y1 * scale_factor),
                     int(x2 * scale_factor), int(y2 * scale_factor))
                    for (x1, y1, x2, y2) in bboxes
                ]
                img_array = np.array(img)

                # --- Full image with bounding boxes ---
                ax_orig = fig.add_subplot(gs[row_idx, col_idx * 2])
                ax_orig.imshow(img_array, aspect='auto', interpolation='nearest')
                ax_orig.axis('off')
                ax_orig.set_xlim([0, img_array.shape[1]])
                ax_orig.set_ylim([img_array.shape[0], 0])
                plt.setp(ax_orig.spines.values(), visible=False)

                for bi, (x1, y1, x2, y2) in enumerate(scaled_bboxes):
                    ax_orig.add_patch(patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=1.2,
                        edgecolor=BBOX_COLORS[bi],
                        facecolor='none',
                    ))

                if col_idx == 0:
                    ax_orig.text(-0.015, 0.5, label,
                                 transform=ax_orig.transAxes,
                                 fontsize=7, va='center', ha='right',
                                 rotation=90, fontfamily='serif')

                # --- Stacked crops ---
                ax_crops = fig.add_subplot(gs[row_idx, col_idx * 2 + 1])
                ax_crops.axis('off')
                plt.setp(ax_crops.spines.values(), visible=False)

                max_crop_w = max(abs(x2 - x1) for (x1, y1, x2, y2) in scaled_bboxes)
                bordered_crops = []
                for ci, (x1, y1, x2, y2) in enumerate(scaled_bboxes):
                    crop = img.crop((x1, y1, x2, y2))
                    if crop.width < max_crop_w:
                        sf2 = max_crop_w / crop.width
                        crop = crop.resize(
                            (max_crop_w, int(crop.height * sf2)), Image.LANCZOS
                        )
                    brd = Image.new('RGB',
                                    (crop.width + 2 * border_width,
                                     crop.height + 2 * border_width),
                                    BBOX_COLORS_PIL[ci])
                    brd.paste(crop, (border_width, border_width))
                    bordered_crops.append(brd)

                total_h = sum(c.height for c in bordered_crops)
                max_w   = max(c.width  for c in bordered_crops)
                stacked = Image.new('RGB', (max_w, total_h), (255, 255, 255))
                y_off   = 0
                for c in bordered_crops:
                    stacked.paste(c, (0, y_off))
                    y_off += c.height

                stacked_arr = np.array(stacked)
                ax_crops.imshow(stacked_arr, aspect='auto', interpolation='nearest')
                ax_crops.set_xlim([0, stacked_arr.shape[1]])
                ax_crops.set_ylim([stacked_arr.shape[0], 0])

            except Exception as exc:
                import traceback
                print(f"Error at row={row_idx} col={col_idx}: {exc}")
                traceback.print_exc()
                for offset in range(2):
                    ax = fig.add_subplot(gs[row_idx, col_idx * 2 + offset])
                    ax.axis('off')

    plt.savefig(output_path, dpi=900, bbox_inches='tight', pad_inches=0.005)
    print(f"Saved → {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a 3-column × N-row comparison figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config file format
------------------
Line 1:  x1,y1   (bbox 0 top-left)
Line 2:  x2,y2   (bbox 0 bottom-right)
Lines 3-4: bbox 1
Lines 5-6: bbox 2
Line 7:  _,<image_index>
Lines 8+: <label>,<folder_path>

Example
-------
    python visualization/generate_comparison_figure.py \\
        scene1.txt scene2.txt scene3.txt --output comparison.png
""",
    )
    parser.add_argument('config_files', nargs=3,
                        help="Three config text files (one per column).")
    parser.add_argument('--output', default='comparison_figure.png',
                        help="Output figure path (default: comparison_figure.png).")
    parser.add_argument('--scale', type=float, default=1.5,
                        help="Image upscale factor before cropping (default: 1.5).")
    parser.add_argument('--border', type=int, default=20,
                        help="Coloured border width in pixels (default: 20).")
    args = parser.parse_args()

    for f in args.config_files:
        if not Path(f).exists():
            parser.error(f"Config file not found: {f}")

    generate_comparison_figure(
        config_files=args.config_files,
        output_path=args.output,
        scale_factor=args.scale,
        border_width=args.border,
    )


if __name__ == '__main__':
    main()
