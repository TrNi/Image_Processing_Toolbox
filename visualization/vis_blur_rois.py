"""
vis_blur_rois.py
================
Interactive ROI-based comparison figure for bokeh / depth-of-field rendering.

Workflow
--------
1. Load an input (sharp) image, ground-truth (shallow-DoF) image, and any
   number of model prediction images.
2. An interactive GUI lets you drag a bounding-box on the ground-truth
   thumbnail to select the Region of Interest.
3. The saved ROI coordinates are used to build a two-row publication figure:
   top row = full images with green ROI box; bottom row = zoomed crops.
4. Output is saved as PNG, SVG and PDF.

Usage
-----
::

    python visualization/vis_blur_rois.py \\
        --input   /path/to/input_sharp.jpg \\
        --gt      /path/to/gt_shallow_dof.jpg \\
        --models  BokehMe:/path/to/bokehme_pred.jpg \\
                  Drbokeh:/path/to/drbokeh_pred.jpg \\
        --out_dir /path/to/output_dir
"""

import sys
import argparse
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector, Button


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family':    'DejaVu Serif',
    'font.size':      6,
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.titlepad']    = 2
plt.rcParams['pdf.fonttype']     = 42
plt.rcParams['ps.fonttype']      = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_rgb(path: str) -> np.ndarray:
    """Load an image as an RGB numpy array via OpenCV."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _make_thumbnail(img: np.ndarray, max_side: int = 5472):
    """Downscale *img* if its longest side exceeds *max_side*."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        nh, nw = int(h * scale), int(w * scale)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), scale
    return img.copy(), 1.0


def _clamp_roi(y, x, h, w, H, W):
    y = max(0, min(y, H - 1))
    x = max(0, min(x, W - 1))
    h = max(1, min(h, H - y))
    w = max(1, min(w, W - x))
    return y, x, h, w


# ---------------------------------------------------------------------------
# Interactive ROI picker
# ---------------------------------------------------------------------------

def interactive_roi_picker(
    images: list,
    titles: list,
    init_roi: tuple,
) -> tuple:
    """Show an interactive two-row GUI and return the final full-res ROI.

    Parameters
    ----------
    images : list of np.ndarray
        Full-resolution RGB images.
    titles : list of str
        Per-image titles.
    init_roi : tuple
        Initial ROI as ``(y, x, h, w)`` in full-resolution pixels.

    Returns
    -------
    tuple
        Final ROI ``(y, x, h, w)`` in full-resolution pixels.
    """
    target_h, target_w = images[-1].shape[:2]   # GT is last

    # Build thumbnails
    thumbs     = []
    thumb_scale = None
    for img in images:
        t, s = _make_thumbnail(img)
        thumbs.append(t)
        if thumb_scale is None:
            thumb_scale = s

    n = len(thumbs)
    fig_ui, axes_ui = plt.subplots(
        2, n, figsize=(min(5 * n, 28), 14)
    )
    if n == 1:
        axes_ui = np.array([[axes_ui[0]], [axes_ui[1]]])
    fig_ui.subplots_adjust(left=0.02, right=0.98, top=0.97,
                           bottom=0.08, wspace=0.08, hspace=0.15)

    # Maximise window
    try:
        mng = plt.get_current_fig_manager()
        try:    mng.window.showMaximized()
        except AttributeError:
            try: mng.window.state('zoomed')
            except AttributeError:
                pass
    except Exception:
        pass

    # Scale initial ROI to thumbnail space
    sy, sx = thumb_scale, thumb_scale
    roi_y0, roi_x0, roi_h0, roi_w0 = init_roi
    _roi_thumb = {
        "y": int(roi_y0 * sy),
        "x": int(roi_x0 * sx),
        "h": max(1, int(roi_h0 * sy)),
        "w": max(1, int(roi_w0 * sx)),
    }

    # Top-row: thumbnails with overlay rectangles
    rects_ui = []
    for col, (thumb, title) in enumerate(zip(thumbs, titles)):
        ax = axes_ui[0, col]
        ax.imshow(thumb, interpolation='bilinear')
        ax.set_title(title, fontsize=10, pad=3, fontweight='bold')
        ax.axis('off')
        r = Rectangle(
            (_roi_thumb["x"], _roi_thumb["y"]),
            _roi_thumb["w"], _roi_thumb["h"],
            linewidth=2.0, edgecolor='red', facecolor='none', zorder=5,
        )
        ax.add_patch(r)
        rects_ui.append(r)

    # Bottom-row: live crop previews
    crop_ims = []
    for col in range(n):
        ax = axes_ui[1, col]
        y, x, h, w = (_roi_thumb["y"], _roi_thumb["x"],
                       _roi_thumb["h"], _roi_thumb["w"])
        H_t, W_t = thumbs[col].shape[:2]
        init_crop = thumbs[col][
            max(0, y):min(H_t, y + h),
            max(0, x):min(W_t, x + w),
        ]
        im = ax.imshow(init_crop, interpolation='bilinear')
        ax.axis('off')
        crop_ims.append((ax, im))

    def _update_crops():
        y, x, h, w = (_roi_thumb["y"], _roi_thumb["x"],
                       _roi_thumb["h"], _roi_thumb["w"])
        for col in range(n):
            ax_c, im_c = crop_ims[col]
            H_t, W_t = thumbs[col].shape[:2]
            yc = max(0, min(y, H_t - 1))
            xc = max(0, min(x, W_t - 1))
            hc = max(1, min(h, H_t - yc))
            wc = max(1, min(w, W_t - xc))
            crop = thumbs[col][yc:yc + hc, xc:xc + wc]
            im_c.set_data(crop)
            ax_c.set_xlim(-0.5, crop.shape[1] - 0.5)
            ax_c.set_ylim(crop.shape[0] - 0.5, -0.5)

    def _update_rects():
        for r in rects_ui:
            r.set_xy((_roi_thumb["x"], _roi_thumb["y"]))
            r.set_width(_roi_thumb["w"])
            r.set_height(_roi_thumb["h"])
        _update_crops()
        fig_ui.canvas.draw_idle()

    def on_select(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, x2, y1, y2):
            return
        _roi_thumb["x"] = int(round(min(x1, x2)))
        _roi_thumb["y"] = int(round(min(y1, y2)))
        _roi_thumb["w"] = max(1, int(round(abs(x2 - x1))))
        _roi_thumb["h"] = max(1, int(round(abs(y2 - y1))))
        _update_rects()

    selector = RectangleSelector(
        axes_ui[0, -1], on_select,
        useblit=True, button=[1],
        minspanx=5, minspany=5,
        spancoords='data', interactive=True,
        props=dict(edgecolor='red', facecolor='none', linewidth=2.0),
    )
    selector.set_active(True)

    final_roi = [None]

    ax_save = fig_ui.add_axes([0.42, 0.02, 0.12, 0.045])
    btn = Button(ax_save, 'Save ROI ✔', color='lightgreen', hovercolor='lime')
    btn.label.set_fontsize(11)

    def on_save(event):
        ry = int(_roi_thumb["y"] / thumb_scale)
        rx = int(_roi_thumb["x"] / thumb_scale)
        rh = max(1, int(_roi_thumb["h"] / thumb_scale))
        rw = max(1, int(_roi_thumb["w"] / thumb_scale))
        ry, rx, rh, rw = _clamp_roi(ry, rx, rh, rw, target_h, target_w)
        final_roi[0] = (ry, rx, rh, rw)
        print(f"ROI saved: y={ry}, x={rx}, h={rh}, w={rw}")
        plt.close(fig_ui)

    btn.on_clicked(on_save)
    plt.show()

    if final_roi[0] is None:
        # User closed without saving — return initial
        return init_roi
    return final_roi[0]


# ---------------------------------------------------------------------------
# Publication figure builder
# ---------------------------------------------------------------------------

def build_publication_figure(
    images: list,
    titles: list,
    roi: tuple,
    out_dir: str,
    base_filename: str = "vis_blur_rois",
    zoom_factor: float = 2.0,
    roi_linewidth: float = 0.7,
    roi_color: tuple = (0.2, 1.0, 0.2),
    figsize: tuple = (7.7, 2.6),
):
    """Build and save the two-row publication figure.

    Parameters
    ----------
    images : list of np.ndarray
        Full-resolution RGB images.
    titles : list of str
    roi : tuple
        ``(y, x, h, w)`` in full-resolution pixels.
    out_dir : str
        Output directory.
    base_filename : str
        Base name for output files (no extension).
    zoom_factor : float
        Upscale factor for the zoomed crop row.
    roi_linewidth : float
        Line width of the ROI rectangle overlay.
    roi_color : tuple
        RGB colour (0–1) for the ROI rectangle.
    figsize : tuple
        Figure dimensions in inches.
    """
    roi_y, roi_x, roi_h, roi_w = roi
    n_images = len(images)

    fig, axes = plt.subplots(2, n_images, figsize=figsize)
    if n_images == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, (img, title) in enumerate(zip(images, titles)):
        target_h, target_w = images[-1].shape[:2]
        y, x, h, w = _clamp_roi(roi_y, roi_x, roi_h, roi_w,
                                 img.shape[0], img.shape[1])

        # Top row — full image + ROI box
        ax_full = axes[0, col]
        ax_full.imshow(img)
        ax_full.set_title(title)
        ax_full.axis('off')
        ax_full.add_patch(Rectangle(
            (x, y), w, h,
            linewidth=roi_linewidth,
            edgecolor=roi_color,
            facecolor='none',
        ))

        # Bottom row — zoomed crop
        ax_zoom = axes[1, col]
        crop = img[y:y + h, x:x + w]
        if zoom_factor != 1.0:
            new_h = int(crop.shape[0] * zoom_factor)
            new_w = int(crop.shape[1] * zoom_factor)
            crop  = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        ax_zoom.imshow(crop)
        ax_zoom.axis('off')

    fig.tight_layout(pad=0.03, w_pad=0.1, h_pad=0.01)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ext, dpi_val in [('png', 600), ('svg', None), ('pdf', 600)]:
        out_path = out_dir / f'{base_filename}.{ext}'
        kwargs = dict(bbox_inches='tight', pad_inches=0.01)
        if dpi_val:
            kwargs['dpi'] = dpi_val
        plt.savefig(str(out_path), format=ext, **kwargs)
        print(f"Saved {ext.upper()} → {out_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive ROI-based bokeh comparison figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
    python visualization/vis_blur_rois.py \\
        --input  /data/scene8/F22/IMG_0628.JPG \\
        --gt     /data/scene8/F2.8/IMG_0636.JPG \\
        --models BokehMe:/data/scene8/bokehme/frame_0000.JPG \\
                 Drbokeh:/data/scene8/drbokeh/IMG_0628.JPG \\
        --out_dir /data/output/figures \\
        --zoom 2.0
""",
    )
    parser.add_argument('--input', required=True,
                        help="Path to the sharp input image.")
    parser.add_argument('--gt', required=True,
                        help="Path to the ground-truth shallow-DoF image.")
    parser.add_argument('--models', nargs='+', required=True,
                        metavar='NAME:PATH',
                        help="Model predictions as 'model_name:image_path' pairs.")
    parser.add_argument('--out_dir', required=True,
                        help="Output directory.")
    parser.add_argument('--zoom', type=float, default=2.0,
                        help="Zoom factor for ROI crop row (default: 2.0).")
    parser.add_argument('--init_roi', nargs=4, type=int,
                        default=[600, 800, 256, 256],
                        metavar=('Y', 'X', 'H', 'W'),
                        help="Initial ROI (y x h w) for the interactive picker.")
    parser.add_argument('--no_interactive', action='store_true',
                        help="Skip the interactive picker and use --init_roi directly.")
    args = parser.parse_args()

    # Parse model entries
    model_names = []
    model_paths = []
    for entry in args.models:
        parts = entry.split(':', 1)
        if len(parts) != 2:
            parser.error(f"--models entries must be 'name:path', got: {entry!r}")
        model_names.append(parts[0])
        model_paths.append(parts[1])

    # Load images
    print("Loading images …")
    input_img = _load_rgb(args.input)
    gt_img    = _load_rgb(args.gt)
    pred_imgs = [_load_rgb(p) for p in model_paths]

    images = [input_img] + pred_imgs + [gt_img]
    titles = ["Input (Sharp)"] + model_names + ["Ground Truth"]

    roi = tuple(args.init_roi)

    # Interactive ROI selection
    if not args.no_interactive:
        roi = interactive_roi_picker(images, titles, roi)

    # Clamp to GT dimensions
    H_gt, W_gt = gt_img.shape[:2]
    roi = _clamp_roi(*roi, H_gt, W_gt)
    print(f"Final ROI: y={roi[0]}, x={roi[1]}, h={roi[2]}, w={roi[3]}")

    # Save ROI coordinates
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    roi_txt = Path(args.out_dir) / "roi_coords.txt"
    roi_txt.write_text(f"{roi[0]} {roi[1]} {roi[2]} {roi[3]}\n")
    print(f"ROI coords → {roi_txt}")

    # Build publication figure
    base_name = Path(args.gt).stem
    build_publication_figure(
        images=images,
        titles=titles,
        roi=roi,
        out_dir=args.out_dir,
        base_filename=base_name,
        zoom_factor=args.zoom,
    )


if __name__ == '__main__':
    main()
