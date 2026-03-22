"""
visualize_depth.py
==================
Interactive per-image depth-map viewer with ensemble fusion and error overlay.

Given a set of depth maps from multiple models (stereo and/or mono), this script:

1. Loads rectified stereo image pairs and all depth maps from HDF5 files.
2. Computes per-pixel error components (gradient, planarity, IQR uncertainty,
   ICP point-cloud consistency).
3. Fuses depth maps using inverse-error weighting.
4. Displays a three-row figure per image:
   - Row 1: individual depth maps.
   - Row 2: additional model depth maps (if > 4 models).
   - Row 3: error components or per-model total weight maps.
5. Saves fused results to ``<base_path>/ml_data/img_<idx>.h5``.

Usage
-----
::

    python visualization/visualize_depth.py \\
        --base /path/to/scene \\
        --left_rectified rectified_h5/rectified_lefts.h5 \\
        --params stereocal_params.npz \\
        --depths model_a:stereodepth/depth_model_a.h5 \\
                 model_b:stereodepth/depth_model_b.h5 \\
        --anonymous

Call :func:`visualize_depth_maps` from another script for programmatic use.
"""

import sys
import os
import argparse
import traceback
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use('TkAgg')
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
from scipy.ndimage import median_filter

from depth_analysis.depth_reproj_eval import (
    load_camera_params, get_Kinv_uv1, px_to_camera, project_to_view, get_errors,
)
from depth_analysis.uncertainty_and_weights import get_iqr_uncertainty
from depth_analysis.geometric_structure_errors import compute_grad
from depth_analysis.point_cloud_opt import get_point_cloud_errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_stats(arr: np.ndarray, maxval: float = 1000) -> dict:
    """Return descriptive statistics for a float array."""
    return {
        'min':     float(np.clip(np.nanmin(arr), 1e-6, maxval)),
        'max':     float(np.clip(np.nanmax(arr), 1e-6, maxval)),
        '5':       float(np.percentile(arr, 5)),
        '95':      float(np.percentile(arr, 95)),
        'num_nan': int(np.isnan(arr).sum()),
        'pct_nan': float(np.mean(np.isnan(arr)) * 100),
    }


def resize_image_hwc(img_hwc: np.ndarray, target_h: int, target_w: int,
                     interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(img_hwc, (target_w, target_h), interpolation=interpolation)


def _build_colormaps():
    """Return (cmap_depth, cmap_error) — trimmed turbo variants."""
    turbo_r = plt.get_cmap('turbo_r')
    cmap_depth = mcolors.ListedColormap(turbo_r(np.linspace(0, 0.8, 256)))
    cmap_depth = plt.get_cmap(cmap_depth, 50)

    turbo = plt.get_cmap('turbo')
    cmap_err = mcolors.ListedColormap(turbo(np.linspace(0.12, 1, 256)))
    cmap_err = plt.get_cmap(cmap_err, 50)

    return cmap_depth, cmap_err


# ---------------------------------------------------------------------------
# Main visualisation function
# ---------------------------------------------------------------------------

def visualize_depth_maps(
    base_path: str,
    left_rectified_path: str,
    depth_paths: dict,
    params_path: str = 'stereocal_params.npz',
    bottom_plot: str = "error_types",
    col_clip: int = 400,
    start_idx: int = 0,
):
    """Visualise depth maps from multiple models with error overlays.

    Parameters
    ----------
    base_path : str
        Root directory containing all data files.
    left_rectified_path : str
        Relative path (from *base_path*) to the left-rectified HDF5 file,
        e.g. ``'rectified_h5/rectified_lefts.h5'``.
    depth_paths : dict
        Mapping ``{model_name: relative_h5_path}``.
    params_path : str
        Relative path (from *base_path*) to the stereo calibration ``.npz``.
    bottom_plot : str
        ``'error_types'`` — show individual error components on row 3.
        ``'total_error'``  — show per-model weight maps on row 3.
    col_clip : int
        Number of columns to clip from the left edge (to remove rectification
        artefacts).
    start_idx : int
        First image index to visualise.
    """
    base_path = str(base_path)
    h5_files  = {}

    try:
        params  = load_camera_params(os.path.join(base_path, params_path))
        P2      = params['P2']
        K_inv   = params['K_inv']

        right_rectified_path = os.path.join(
            base_path, left_rectified_path.replace("left", "right")
        )
        h5_files['left_rectified']  = h5py.File(
            os.path.join(base_path, left_rectified_path), 'r')
        h5_files['right_rectified'] = h5py.File(right_rectified_path, 'r')

        for name, rel_path in depth_paths.items():
            h5_files[name] = h5py.File(os.path.join(base_path, rel_path), 'r')

        plot_cols   = 6
        bottom_cols = 6
        figsize     = (2.5 * plot_cols, 9)
        width_ratios = [1.5] + [1] * (plot_cols - 1)

        min_images = min(
            h5_files[f]['data'].shape[0] if 'data' in h5_files[f]
            else h5_files[f]['depth'].shape[0]
            for f in h5_files
        )

        model_names  = list(depth_paths.keys())
        cmap1, cmap2 = _build_colormaps()
        os.makedirs(os.path.join(base_path, 'ml_data'), exist_ok=True)

        reuse_K  = False
        min_h = min_w = np.inf

        current_idx = start_idx
        while current_idx < min_images:
            fig = plt.figure(figsize=figsize)
            gs  = fig.add_gridspec(
                3, plot_cols,
                height_ratios=[1, 1, 1],
                hspace=0.1, wspace=0.2,
                width_ratios=width_ratios,
            )
            axes = np.full((3, plot_cols), None, dtype=object)
            axes[0, 0] = fig.add_subplot(gs[:, 0])
            for i in range(1, plot_cols):
                if i < 5:
                    axes[0, i] = fig.add_subplot(gs[0, i])
                if 4 + i <= len(depth_paths) + 1:
                    axes[1, i] = fig.add_subplot(gs[1, i])
                if i <= bottom_cols:
                    axes[2, i] = fig.add_subplot(gs[2, i])
            axes_flat = axes.flatten()[1:]

            plt.subplots_adjust(left=0.015, bottom=0.01, top=0.95, right=0.98,
                                hspace=0.1, wspace=0.2)
            fig.suptitle(f'{os.path.basename(base_path)} : Image {current_idx}',
                         fontsize=10, y=0.98)

            # Load images
            rect_left  = h5_files['left_rectified']['data'][()][current_idx].transpose(1, 2, 0)
            rect_right = h5_files['right_rectified']['data'][()][current_idx].transpose(1, 2, 0)
            h_l, w_l, _ = rect_left.shape
            aspect_ratio = w_l / h_l

            if not reuse_K:
                min_h, min_w = h_l, w_l
                depth_names_valid = []
                for name in model_names:
                    h5 = h5_files[name]
                    key = 'depth' if 'depth' in h5 else ('depths' if 'depths' in h5 else 'data')
                    h_d, w_d = h5[key].shape[1:]
                    if abs(w_d / h_d - aspect_ratio) > 0.01:
                        print(f"Skipping {name}: aspect ratio mismatch")
                        continue
                    depth_names_valid.append(name)
                    min_h = min(min_h, h_d)
                    min_w = min(min_w, w_d)
                print(f"Common size: {min_h}×{min_w}")

            min_h, min_w = int(min_h), int(min_w)
            rect_left  = resize_image_hwc(rect_left,  min_h, min_w)
            rect_right = resize_image_hwc(rect_right, min_h, min_w)

            if not reuse_K:
                K_inv_uv1 = get_Kinv_uv1(K_inv, min_h, min_w)
                reuse_K   = True

            # Show combined left+right image
            separator = 255 * np.ones((200, min_w - col_clip, 3), dtype=np.uint8)
            combined = np.vstack((rect_left[:, col_clip:, :],
                                  separator,
                                  rect_right[:, :-col_clip, :]))
            axes[0, 0].imshow(combined)
            axes[0, 0].set_title('Left\n(Right below)', fontsize=9)
            axes[0, 0].axis('off')

            alpha  = 0.1
            kernel = 5
            g_i    = compute_grad(rect_left, k=kernel)
            g_i   /= g_i.max() + 1e-8

            depth_data = {}
            depth_stats = {}
            err_data   = {}
            err_stats  = {}

            for name in depth_names_valid:
                h5  = h5_files[name]
                key = 'depth' if 'depth' in h5 else ('depths' if 'depths' in h5 else 'data')
                d   = h5[key][()][current_idx].squeeze()
                d   = resize_image_hwc(d.astype(np.float32), min_h, min_w)
                depth_data[name] = d
                err_data[name]   = get_errors(
                    d, rect_left, rect_right, K_inv, K_inv_uv1, g_i, P2, alpha, kernel
                )

            depth_arr = np.stack(list(depth_data.values()), axis=0)
            try:
                iqr_errors = get_iqr_uncertainty(depth_arr)
                icp_errors = get_point_cloud_errors(depth_arr, K_inv)
            except Exception as exc:
                print(f"Uncertainty/ICP error at idx {current_idx}: {exc}")
                current_idx += 1
                plt.close('all')
                continue

            for i, name in enumerate(depth_names_valid):
                err_data[name]['icp_error'] = icp_errors[i].reshape(min_h, min_w)
                err_data[name]['iqr']       = iqr_errors[i]

            # Normalise errors globally
            error_min = defaultdict(lambda: np.inf)
            error_max = defaultdict(lambda: -np.inf)
            all_err   = err_data[depth_names_valid[0]]
            for name in err_data:
                for et in err_data[name]:
                    vals = err_data[name][et][:, col_clip:]
                    error_min[et] = min(error_min[et], float(np.percentile(vals, 1)))
                    error_max[et] = max(error_max[et], float(np.percentile(vals, 99)))

            fused_depth = np.zeros((min_h, min_w - col_clip), dtype=np.float32)
            weights_sum = np.zeros_like(fused_depth)
            weight_map  = {}

            for name in depth_names_valid:
                composite = (
                    0.3 * (err_data[name]['grad_error']      + 1e-8 - error_min['grad_error'])
                          / (error_max['grad_error']      - error_min['grad_error'] + 1e-8)
                    + 0.1 * (err_data[name]['planarity_error'] + 1e-8 - error_min['planarity_error'])
                          / (error_max['planarity_error'] - error_min['planarity_error'] + 1e-8)
                    + 5.5 * (err_data[name]['iqr']             + 1e-8 - error_min['iqr'])
                          / (error_max['iqr']             - error_min['iqr'] + 1e-8)
                    + 4.1 * (err_data[name]['icp_error']       + 1e-8 - error_min['icp_error'])
                          / (error_max['icp_error']       - error_min['icp_error'] + 1e-8)
                )
                composite = np.maximum(composite, 0) / 10
                err_data[name]  = composite[:, col_clip:]
                err_stats[name] = get_stats(err_data[name], maxval=500)
                depth_stats[name] = get_stats(depth_data[name][:, col_clip:], maxval=100)
                w = 1.0 / (err_data[name] + 1e-8)
                weight_map[name]  = w
                weights_sum      += w
                fused_depth      += depth_data[name][:, col_clip:] * w

            fused_depth /= weights_sum + 1e-8

            dmin = max(min(s['min'] for s in depth_stats.values()), 1e-3)
            dmax = max(s['max'] for s in depth_stats.values())
            emin = max(min(s['min'] for s in err_stats.values()),  1e-3)
            emax = max(s['max'] for s in err_stats.values())

            ax_iter = iter(axes_flat[axes_flat != None])

            for i, (name, _) in enumerate(depth_paths.items()):
                if name not in depth_data:
                    continue
                ax_d = next(ax_iter)
                im   = ax_d.imshow(
                    depth_data[name][:, col_clip:].round(3),
                    cmap=cmap1,
                    norm=mcolors.LogNorm(vmin=dmin, vmax=dmax),
                    interpolation='nearest',
                )
                ds = depth_stats[name]
                ax_d.set_title(
                    f'{name}\n{ds["5"]:.2f}–{ds["95"]:.2f}  NaN={ds["num_nan"]}({ds["pct_nan"]:.0f}%)',
                    fontsize=9,
                )
                cbar = plt.colorbar(im, ax=ax_d, fraction=0.035, pad=0.04)
                log_ticks = np.logspace(np.log10(dmin), np.log10(dmax), num=5)
                cbar.set_ticks(log_ticks)
                cbar.set_ticklabels([f'{t:.1f}' for t in log_ticks])
                cbar.ax.tick_params(labelsize=7)
                ax_d.axis('off')

                if bottom_plot == "error_types" and i == 0:
                    err_types = ["grad_error", "planarity_error", "iqr", "icp_error"]
                    for j, et in enumerate(err_types):
                        ax_e = axes[2, j + 1]
                        this_err = (all_err[et][:, col_clip:] + 1e-8 - error_min[et]) \
                                   / (error_max[et] - error_min[et] + 1e-8)
                        lo = np.percentile(this_err, 1)
                        hi = np.percentile(this_err, 99)
                        this_err = np.clip(this_err, lo, hi)
                        im_e = ax_e.imshow(this_err, cmap=cmap2, interpolation='nearest')
                        ax_e.set_title(f'Error component {j}', fontsize=9)
                        cbar_e = plt.colorbar(im_e, ax=ax_e, fraction=0.035, pad=0.04)
                        cbar_e.ax.tick_params(labelsize=7)
                        ax_e.axis('off')

                    ax_e_last = axes[2, plot_cols - 1]
                    this_w    = weight_map[name] / weights_sum
                    im_e = ax_e_last.imshow(this_w, cmap=cmap2, interpolation='nearest')
                    ax_e_last.set_title(f'Weight for {name}', fontsize=9)
                    cbar_e = plt.colorbar(im_e, ax=ax_e_last, fraction=0.035, pad=0.04)
                    cbar_e.ax.tick_params(labelsize=7)
                    ax_e_last.axis('off')

            # Fused depth panel
            ax_fused = next(ax_iter, None)
            if ax_fused is not None:
                d_fused_stats = {
                    'min': fused_depth.min(), 'max': fused_depth.max(),
                    '5':  np.percentile(fused_depth, 5),
                    '95': np.percentile(fused_depth, 95),
                    'num_nan': int(np.isnan(fused_depth).sum()),
                    'pct_nan': float(np.mean(np.isnan(fused_depth)) * 100),
                }
                im_f = ax_fused.imshow(
                    fused_depth.round(3),
                    cmap=cmap1,
                    norm=mcolors.LogNorm(vmin=dmin, vmax=dmax),
                    interpolation='nearest',
                )
                ax_fused.set_title(
                    f"Fused\n{d_fused_stats['5']:.2f}–{d_fused_stats['95']:.2f}"
                    f"  NaN={d_fused_stats['num_nan']}({d_fused_stats['pct_nan']:.0f}%)",
                    fontsize=9,
                )
                cbar_f = plt.colorbar(im_f, ax=ax_fused, fraction=0.035, pad=0.04)
                log_ticks = np.logspace(np.log10(dmin), np.log10(dmax), num=5)
                cbar_f.set_ticks(log_ticks)
                cbar_f.set_ticklabels([f'{t:.1f}' for t in log_ticks])
                cbar_f.ax.tick_params(labelsize=7)
                ax_fused.axis('off')

            # Share axes among depth/error panels
            ref_ax = axes[0, 1]
            if ref_ax is not None:
                for ax in axes_flat:
                    if ax is not None and ax is not ref_ax:
                        ax.sharex(ref_ax)
                        ax.sharey(ref_ax)

            # Save ML data to H5
            ml_path = os.path.join(base_path, 'ml_data', f'img_{current_idx}.h5')
            with h5py.File(ml_path, 'w') as fout:
                name0 = depth_names_valid[0]
                channels = [rect_left, rect_right, depth_data[name0][..., None]]
                for k, v in all_err.items():
                    channels.append(v[..., None])
                subarr = np.concatenate(channels, axis=2)
                fout.create_dataset('data',         data=subarr,       compression='gzip', compression_opts=9)
                fout.create_dataset('total_weight', data=weight_map[name0] / weights_sum, compression='gzip', compression_opts=9)
                fout.create_dataset('fused_depth',  data=fused_depth,  compression='gzip', compression_opts=9)

            plt.show(block=True)
            plt.close('all')
            current_idx += 1

    except Exception:
        print("Error occurred:\n" + traceback.format_exc())
    finally:
        for fh in h5_files.values():
            try:
                fh.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive depth-map viewer with multi-model error analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
Run with two stereo depth models:

    python visualization/visualize_depth.py \\
        --base /path/to/scene \\
        --left_rectified rectified_h5/rectified_lefts.h5 \\
        --params stereocal_params.npz \\
        --depths model_a:stereodepth/depth_model_a.h5 \\
                 model_b:stereodepth/depth_model_b.h5

Anonymise model names (stereo 0, stereo 1, …):

    python visualization/visualize_depth.py ... --anonymous
""",
    )
    parser.add_argument('--base', required=True,
                        help="Root directory containing all data files.")
    parser.add_argument('--left_rectified', required=True,
                        help="Relative path from --base to the left-rectified H5 file.")
    parser.add_argument('--params', default='stereocal_params.npz',
                        help="Relative path from --base to stereocal_params.npz.")
    parser.add_argument('--depths', nargs='+', required=True,
                        metavar='NAME:PATH',
                        help="Depth map entries as 'model_name:relative/path.h5'.")
    parser.add_argument('--bottom_plot', choices=['error_types', 'total_error'],
                        default='error_types',
                        help="What to show in the bottom figure row.")
    parser.add_argument('--col_clip', type=int, default=400,
                        help="Columns to clip from the left (rectification artefacts).")
    parser.add_argument('--start_idx', type=int, default=0,
                        help="First image index to visualise.")
    parser.add_argument('--anonymous', action='store_true',
                        help="Replace model names with 'stereo 0', 'mono 0', etc.")
    parser.add_argument('--stereo_kw', nargs='+', default=[],
                        metavar='KW',
                        help="Keywords that identify stereo models when using --anonymous "
                             "(case-insensitive substring match against model name).")
    args = parser.parse_args()

    depth_paths = {}
    for entry in args.depths:
        parts = entry.split(':', 1)
        if len(parts) != 2:
            parser.error(f"--depths entries must be 'name:path', got: {entry!r}")
        depth_paths[parts[0]] = parts[1]

    if args.anonymous:
        anon_paths = {}
        s_idx = m_idx = 0
        stereo_kw = [kw.lower() for kw in args.stereo_kw]
        for k, v in depth_paths.items():
            if any(kw in k.lower() for kw in stereo_kw):
                anon_paths[f'stereo {s_idx}'] = v; s_idx += 1
            else:
                anon_paths[f'mono {m_idx}'] = v;   m_idx += 1
        depth_paths = anon_paths

    visualize_depth_maps(
        base_path=args.base,
        left_rectified_path=args.left_rectified,
        depth_paths=depth_paths,
        params_path=args.params,
        bottom_plot=args.bottom_plot,
        col_clip=args.col_clip,
        start_idx=args.start_idx,
    )


if __name__ == '__main__':
    main()
