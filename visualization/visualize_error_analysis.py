"""
visualize_error_analysis.py
============================
CDF plots, depth-map comparison figures, and per-model error-map figures.

Functions
---------
get_pretty_name              Friendly display name from a raw model keyword.
plot_depth_maps              Side-by-side depth-map grid for mono and stereo models.
plot_error_maps              Publication-ready 4-panel error-component figure.
fuse_depth_maps              Error-weighted depth fusion.
plot_fused_depth             Visualise a fused depth map.
analyze_error_distributions  CDF and percentile-table analysis across all models.
main                         Entry point: loads error_data.pkl and runs analysis.

Usage
-----
::

    python visualization/visualize_error_analysis.py \\
        --error_data /path/to/err_GT/error_data.pkl

or call :func:`main` programmatically::

    from visualization.visualize_error_analysis import main
    main(specific_path=Path('/path/to/error_data.pkl'))
"""

import sys
import argparse
import csv
import gzip
import pickle
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family':                    'Times New Roman',
    'font.size':                      5,
    'axes.titlesize':                 5,
    'axes.labelsize':                 5,
    'xtick.labelsize':                5,
    'ytick.labelsize':                5,
    'legend.fontsize':                5,
    'figure.autolayout':              False,
    'figure.constrained_layout.use':  False,
})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.titlepad']    = 0.2

# ---------------------------------------------------------------------------
# Visual style constants
# ---------------------------------------------------------------------------

MARKERS = ['p', 'v', '<', '>', '^', 's', 'D', 'o']

COLORS = [
    '#e377c2',  # pink
    '#17becf',  # cyan
    '#1f77b4',  # blue
    '#7f7f7f',  # grey
    '#2ca02c',  # green
    '#d62728',  # red
    '#ff7f0e',  # orange
    '#9467bd',  # purple
]

LINESTYLES = ['--', '-.', ':', '--', '-', '-', '-', '-']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# User-extendable name map: add your model keywords → display names here.
# ---------------------------------------------------------------------------
_PRETTY_NAME_MAP: dict = {}


def get_pretty_name(name: str) -> str:
    """Return a display-friendly model name from a raw filename keyword.

    Looks up *name* (case-insensitive substring) in :data:`_PRETTY_NAME_MAP`.
    Add your own model names there before calling this function.
    Falls back to *name* unchanged if no match is found.
    """
    n = name.lower()
    for keyword, display in _PRETTY_NAME_MAP.items():
        if keyword.lower() in n:
            return display
    return name


def _trimmed_turbo(lo: float = 0.0, hi: float = 1.0, n: int = 256):
    """Return a trimmed turbo colormap."""
    colors = plt.cm.turbo(np.linspace(lo, hi, n))
    n_trim = int(n * 0.02)
    colors = colors[n_trim:-n_trim]
    return LinearSegmentedColormap.from_list('trimmed_turbo', colors)


# ---------------------------------------------------------------------------
# Depth map visualisation
# ---------------------------------------------------------------------------

def plot_depth_maps(depth_data: list, mono_models: list, stereo_models: list,
                    save_dir: Path, idx: int):
    """Plot all depth maps in a 2-row grid (mono on top, stereo on bottom).

    Parameters
    ----------
    depth_data : list of np.ndarray
        One array per model, mono models first.
    mono_models, stereo_models : list of str
        Model name keywords.
    save_dir : Path
        Output directory.
    idx : int
        Image index used for the output filename.
    """
    trimmed_turbo = _trimmed_turbo()
    n_mono   = len(mono_models)
    n_stereo = len(stereo_models)
    n_cols   = max(n_mono, n_stereo)

    fig = plt.figure(figsize=(7.6, 4))
    gs  = plt.GridSpec(2, n_cols + 1, width_ratios=[1] * n_cols + [0.05])

    all_valid = [d[~np.isnan(d)] for d in depth_data]
    vmin = np.mean([np.percentile(v, 5)  for v in all_valid])
    vmax = np.mean([np.percentile(v, 95) for v in all_valid])

    plt.suptitle(f'Depth Maps — Image {idx}')
    im = None

    for i, model in enumerate(mono_models):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(depth_data[i], cmap=trimmed_turbo, vmin=vmin, vmax=vmax)
        ax.set_title(f'Mono — {get_pretty_name(model)}')
        ax.axis('off')

    for i, model in enumerate(stereo_models):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(depth_data[n_mono + i], cmap=trimmed_turbo, vmin=vmin, vmax=vmax)
        ax.set_title(f'Stereo — {get_pretty_name(model)}')
        ax.axis('off')

    if im is not None:
        cbar_ax = fig.add_subplot(gs[:, -1])
        plt.colorbar(im, cax=cbar_ax, label='Depth (m)')

    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'depth_maps_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Error map visualisation (publication quality)
# ---------------------------------------------------------------------------

def plot_error_maps(error_maps: dict, model_names: list, save_dir, idx: int,
                    target_model: str = None):
    """Plot a 4-panel error-component figure for one model (publication-ready).

    Panels: Gradient | Planarity | ICP | IQR.
    Each panel gets its own full-height colorbar positioned manually to avoid
    axis distortion.

    Parameters
    ----------
    error_maps : dict
        ``error_maps[model_name][error_type][img_idx]`` → 2-D array.
    model_names : list of str
        List of model name strings as stored in *error_maps*.
    save_dir : str or Path
        Output directory.
    idx : int
        Image index to visualise.
    target_model : str, optional
        If given, only render the model whose pretty-name contains this string.
        If ``None``, the first model in *model_names* is rendered.
    """
    error_plotnames = {"grad": "Gradient", "plan": "Planarity", "icp": "ICP", "iqr": "IQR"}
    error_types = ['grad', 'plan', 'icp', 'iqr']
    trimmed_turbo = _trimmed_turbo()

    H_GAP   = 0.0005
    CB_WIDTH = 0.006

    for model in model_names:
        model_name = get_pretty_name(model)
        if target_model:
            if target_model.lower() not in model_name.lower():
                continue
        else:
            if model != model_names[0]:
                continue

        fig = plt.figure(figsize=(5.76, 1.2))
        gs  = plt.GridSpec(1, 7,
                           width_ratios=[1, 1, 0.001, 1, 0.001, 1, 0.001])

        mins, maxs = [], []
        for et in error_types:
            scale = 1e6 if et == 'plan' else 1
            d = error_maps[model_name][et][idx] * scale
            v = d[~np.isnan(d)]
            mins.append(float(np.percentile(v, 5)))
            maxs.append(float(np.percentile(v, 95)))

        vmin1, vmax1 = min(mins[:2]), max(maxs[:2])
        vmin2, vmax2 = mins[2], maxs[2]
        vmin3, vmax3 = mins[3], maxs[3]

        col_map = {0: (0,  vmin1, vmax1),
                   1: (1,  vmin1, vmax1),
                   2: (3,  vmin2, vmax2),
                   3: (5,  vmin3, vmax3)}
        cb_x_offsets = {1: -0.011, 2: 0.035, 3: 0.08}

        ims = {}
        axes_used = {}

        for j, et in enumerate(error_types):
            scale = 1e6 if et == 'plan' else 1
            d     = error_maps[model_name][et][idx] * scale
            gs_col, vmin_j, vmax_j = col_map[j]
            ax = fig.add_subplot(gs[0, gs_col])
            im = ax.imshow(d, cmap=trimmed_turbo, vmin=vmin_j, vmax=vmax_j)
            title_suf = " (×1e6)" if et == "plan" else ""
            ax.set_title(f'{error_plotnames[et]} Error{title_suf}')
            ax.axis('off')
            ims[j] = im
            axes_used[j] = ax

        # Place colorbars at precise positions
        for j in [1, 2, 3]:
            pos_img = axes_used[j].get_position()
            x_off   = cb_x_offsets[j]
            cax = fig.add_axes([
                pos_img.x1 + x_off,
                pos_img.y0,
                CB_WIDTH,
                pos_img.height,
            ])
            cbar = fig.colorbar(ims[j], cax=cax, orientation='vertical')
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            cbar.outline.set_visible(False)
            cax.set_frame_on(False)
            cbar.ax.tick_params(axis='y', direction='out', length=0.1,
                                pad=0.1, labelsize=4)

        plt.subplots_adjust(left=0.001, right=0.97, top=0.98, bottom=0.02,
                            hspace=0.01, wspace=0.09)

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f'error_maps_{idx}.png',
                    dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.savefig(save_dir / f'error_maps_{idx}.pdf',
                    bbox_inches='tight', pad_inches=0.01, dpi=900)
        plt.close(fig)
        print(f"Saved error maps → {save_dir}/error_maps_{idx}.[png|pdf]")


# ---------------------------------------------------------------------------
# Depth fusion
# ---------------------------------------------------------------------------

def fuse_depth_maps(depth_data: np.ndarray, error_maps: dict,
                    model_names: list) -> np.ndarray:
    """Error-weighted depth fusion.

    Parameters
    ----------
    depth_data : np.ndarray
        Stacked depth maps, shape ``(K, H, W)``.
    error_maps : dict
        ``error_maps[model_name][error_type]`` — 2-D error arrays.
    model_names : list of str

    Returns
    -------
    np.ndarray
        Fused depth map, shape ``(H, W)``.
    """
    weights = np.ones_like(depth_data)
    for i, model in enumerate(model_names):
        w = np.ones(depth_data.shape[1:], dtype=np.float32)
        for et in ['grad', 'plan', 'icp', 'iqr']:
            err  = error_maps[model][et]
            lo, hi = np.nanmin(err), np.nanmax(err)
            norm = 1 - (err - lo) / (hi - lo + 1e-8)
            w   *= norm
        weights[i] = w

    weights /= np.sum(weights, axis=0, keepdims=True) + 1e-8
    return np.sum(depth_data * weights, axis=0)


def plot_fused_depth(fused_depth: np.ndarray, save_dir: Path, idx: int):
    """Visualise a single fused depth map with a colorbar."""
    trimmed_turbo = _trimmed_turbo()
    valid = fused_depth[~np.isnan(fused_depth)]
    vmin  = float(np.percentile(valid, 5))
    vmax  = float(np.percentile(valid, 95))

    fig, (ax, cbar_ax) = plt.subplots(1, 2, figsize=(12, 8),
                                      gridspec_kw={'width_ratios': [1, 0.01]})
    im = ax.imshow(fused_depth, cmap=trimmed_turbo, vmin=vmin, vmax=vmax)
    ax.set_title(f'Fused Depth Map — Image {idx}')
    ax.axis('off')
    plt.colorbar(im, cax=cbar_ax, label='Depth (m)')
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'fused_depth_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# CDF / distribution analysis
# ---------------------------------------------------------------------------

def analyze_error_distributions(error_aggr: dict, save_dir: Path):
    """CDF plots and percentile CSV for all models and error types.

    Parameters
    ----------
    error_aggr : dict
        ``error_aggr[model_name][error_type]`` — shape ``(N, k)`` arrays
        of sorted error samples per image.
    save_dir : Path
        Directory to save CSV and PNG files.
    """
    error_types = ['grad', 'iqr', 'Prel', 'Pnorm']
    percentiles  = np.arange(0, 101, 10)
    plan_scale   = 1e6

    headers = [
        [f"# Planarity, RMS Orth, Preal, Pnorm errors scaled by {plan_scale}."],
        ['model', 'error_type'] + [f'p{p}' for p in percentiles],
    ]
    csv_data = []

    for model_name, model_errors in error_aggr.items():
        for et in error_types:
            vals = model_errors[et].flatten()
            if et in ["plan", "rms_orth", "Prel", "Pnorm"]:
                vals = vals * plan_scale
            pct_vals = np.percentile(vals, percentiles)
            csv_data.append([model_name, et] + [f'{v:.6f}' for v in pct_vals])

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / 'error_percentiles.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerows(headers)
        writer.writerows(csv_data)
    print(f"Saved percentile table → {csv_path}")

    # CDF plots
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    for et in error_types:
        plt.figure(figsize=(10, 6))
        scale = plan_scale if et in ['plan', 'rms_orth', 'Prel', 'Pnorm'] else 1

        for i, (model_name, model_errors) in enumerate(error_aggr.items()):
            vals   = model_errors[et].flatten() * scale
            sorted_v = np.sort(vals)
            n        = len(sorted_v)
            cdf      = np.arange(1, n + 1) / n
            plt.plot(sorted_v, cdf,
                     label=model_name, linewidth=1,
                     marker=MARKERS[i % len(MARKERS)],
                     color=COLORS[i % len(COLORS)],
                     linestyle=LINESTYLES[i % len(LINESTYLES)],
                     markersize=2.0)

        label_suf = ' (×1e6)' if et in ['plan', 'rms_orth', 'Prel', 'Pnorm'] else ''
        plt.grid(True, alpha=0.3)
        plt.xlabel(f'{et.upper()} Error{label_suf}')
        plt.ylabel('Cumulative Probability')
        plt.title(f'CDF of {et.upper()} Error')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        out_png = save_dir / f'error_cdf_{et}.png'
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved CDF → {out_png}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(datalist=None, specific_path=None):
    """Run error analysis visualisation.

    Parameters
    ----------
    datalist : list, optional
        Unused; kept for backward compatibility.
    specific_path : str or Path, optional
        Path to a ``error_data.pkl`` (possibly gzip-compressed) file.
    """
    if specific_path is None:
        return

    error_data_path = Path(specific_path)
    save_dir        = error_data_path.parent

    try:
        with gzip.open(error_data_path, 'rb') as fh:
            error_data = pickle.load(fh)
    except Exception:
        with open(error_data_path, 'rb') as fh:
            error_data = pickle.load(fh)

    analyze_error_distributions(error_data['error_aggr'], save_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="Visualise depth-estimation error distributions.",
    )
    parser.add_argument('--error_data', required=True,
                        help="Path to error_data.pkl (may be gzip-compressed).")
    args = parser.parse_args()
    main(specific_path=args.error_data)


if __name__ == '__main__':
    _cli()
