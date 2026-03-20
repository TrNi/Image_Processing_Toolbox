"""
plot_one_row.py
===============
Single-row publication figure with two full-width panels and two 2×2 quadrant panels.

Layout (8 sub-columns)
-----------------------
| Col 1 (full) | Col 2 (full) | Col 3 (2×2 grid) | Col 4 (2×2 grid) |

Designed for ECCV / CVPR single-column width (≈7.16 inches).

Usage
-----
::

    python visualization/plot_one_row.py \\
        --images path/to/img1.png path/to/img2.png \\
                 path/to/tl.jpg path/to/tr.jpg path/to/bl.jpg path/to/br.jpg \\
                 path/to/tl2.jpg path/to/tr2.jpg path/to/bl2.jpg path/to/br2.jpg \\
        --titles "Column A" "Column B" "Column C" "Column D" \\
        --output output_figure.png
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family':                   'Times New Roman',
    'font.size':                     5,
    'axes.titlesize':                5,
    'axes.labelsize':                5,
    'xtick.labelsize':               5,
    'ytick.labelsize':               5,
    'legend.fontsize':               5,
    'figure.autolayout':             False,
    'figure.constrained_layout.use': False,
})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.titlepad']    = 0.2


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def create_one_row_plot(
    image_paths: list,
    column_titles: list,
    save_path: str,
    figsize: tuple = (7.16, 1.8),
    dpi: int = 900,
):
    """Create a one-row mixed-layout figure.

    Parameters
    ----------
    image_paths : list of str
        Exactly 10 image paths:

        - ``image_paths[0]``: single image for column 1.
        - ``image_paths[1]``: single image for column 2.
        - ``image_paths[2:6]``: 2×2 quadrants for column 3
          (top-left, top-right, bottom-left, bottom-right).
        - ``image_paths[6:10]``: 2×2 quadrants for column 4.

    column_titles : list of str
        Four column heading strings.
    save_path : str
        Output path (PNG).  A PDF with the same stem is also saved.
    figsize : tuple
        Figure dimensions in inches.
    dpi : int
        Output resolution.
    """
    if len(image_paths) != 10:
        raise ValueError(f"Expected 10 image paths, got {len(image_paths)}.")
    if len(column_titles) != 4:
        raise ValueError(f"Expected 4 column titles, got {len(column_titles)}.")

    images = [np.array(Image.open(p)) for p in image_paths]

    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(
        5, 8, figure=fig,
        height_ratios=[0.1, 1, 0.005, 1, 0.005],
        width_ratios=[1, 1, 0.8, 0.8, 1, 1.1, 1.1, 1.1],
        hspace=0.02, wspace=0.02,
    )

    def _title_ax(gs_slice, title):
        ax = fig.add_subplot(gs_slice)
        ax.text(0.5, 0.5, title, ha='center', va='center',
                fontsize=5, fontfamily='Times New Roman')
        ax.axis('off')

    def _img_ax(gs_slice, img):
        ax = fig.add_subplot(gs_slice)
        ax.imshow(img, aspect='auto')
        ax.axis('off')
        return ax

    # Column 1: single image
    _title_ax(gs[0, 0:2], column_titles[0])
    _img_ax(gs[1:, 0:2], images[0])

    # Column 2: single image
    _title_ax(gs[0, 2:4], column_titles[1])
    _img_ax(gs[1:, 2:4], images[1])

    # Column 3: 2×2 quadrants
    _title_ax(gs[0, 4:6], column_titles[2])
    _img_ax(gs[1, 4], images[2])
    _img_ax(gs[1, 5], images[3])
    _img_ax(gs[3, 4], images[4])
    _img_ax(gs[3, 5], images[5])

    # Column 4: 2×2 quadrants
    _title_ax(gs[0, 6:8], column_titles[3])
    _img_ax(gs[1, 6], images[6])
    _img_ax(gs[1, 7], images[7])
    _img_ax(gs[3, 6], images[8])
    _img_ax(gs[3, 7], images[9])

    save_path = str(save_path)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved PNG → {save_path}")

    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, format='pdf', dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved PDF → {pdf_path}")

    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a one-row publication figure (2 full + 2 quadrant panels).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
    python visualization/plot_one_row.py \\
        --images img1.png img2.png \\
                 tl.jpg tr.jpg bl.jpg br.jpg \\
                 tl2.jpg tr2.jpg bl2.jpg br2.jpg \\
        --titles "Setup" "Camera" "Reflective Surfaces" "Fine Details" \\
        --output figure_row.png
""",
    )
    parser.add_argument('--images', nargs=10, required=True, metavar='IMG',
                        help="10 image paths (see function docstring for layout).")
    parser.add_argument('--titles', nargs=4, required=True, metavar='TITLE',
                        help="4 column titles.")
    parser.add_argument('--output', required=True,
                        help="Output PNG path.")
    parser.add_argument('--figsize', nargs=2, type=float, default=[7.16, 1.8],
                        metavar=('W', 'H'),
                        help="Figure size in inches (default: 7.16 1.8).")
    parser.add_argument('--dpi', type=int, default=900,
                        help="Output DPI (default: 900).")
    args = parser.parse_args()

    create_one_row_plot(
        image_paths=args.images,
        column_titles=args.titles,
        save_path=args.output,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
