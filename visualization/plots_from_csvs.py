"""
plots_from_csvs.py
==================
Error-statistic trend plots from pre-computed CSV files.

Generates a combined two-row figure (focal-length sweep on top, aperture sweep
on bottom) for four error types: IQR, Gradient, P_rel, P_norm.

Usage
-----
::

    python visualization/plots_from_csvs.py \\
        --focal_csvs fl28:/data/output/config_fl28_F2.8/err_GT/error_percentiles.csv \\
                     fl70:/data/output/config_fl70_F2.8/err_GT/error_percentiles.csv \\
        --aperture_csvs F2.8:/data/output/config_fl70_F2.8/err_GT/error_percentiles.csv \\
                        F22:/data/output/config_fl70_F22.0/err_GT/error_percentiles.csv \\
        --out_dir /path/to/output \\
        --percentile p50
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family':    'Times New Roman',
    'font.size':      7,
    'axes.titlesize': 7,
    'axes.labelsize': 5.5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 6,
    'lines.linewidth': 0.7,
    'lines.markersize': 2.5,
})

# ---------------------------------------------------------------------------
# Visual style constants
# ---------------------------------------------------------------------------

MARKERS = ['p', 'v', '<', '>', '^', 's', 'D', 'o']
COLORS  = [
    '#e377c2', '#17becf', '#1f77b4', '#7f7f7f',
    '#2ca02c', '#d62728', '#ff7f0e', '#9467bd',
]
LINESTYLES = ['--', '-.', ':', '--', '-', '-', '-', '-']

ERROR_PLOTNAMES = {
    "grad":     "Gradient",
    "plan":     "Planarity",
    "icp":      "ICP",
    "iqr":      "IQR",
    "rms_orth": "RMS Orthogonality",
    "Prel":     "P_rel",
    "Pnorm":    "P_norm",
}

# Map model display names to short plot labels.
# Populate this with your own model names before running.
# e.g. MODEL_PLOTNAMES = {'My Model': 'm:MyModel', 'Baseline': 's:Baseline'}
MODEL_PLOTNAMES: dict = {}


# ---------------------------------------------------------------------------
# Core plotting
# ---------------------------------------------------------------------------

def plot_error_trends(
    dataframes_focal: dict,
    dataframes_aperture: dict,
    focal_xlabel: str,
    aperture_xlabel: str,
    plot_error_types: list,
    percentile: str,
    out_dir: Path,
    title_suffix_focal: str = "",
    title_suffix_aperture: str = "",
):
    """Generate the combined focal-length / aperture trend figure.

    Parameters
    ----------
    dataframes_focal : dict
        ``{label: pd.DataFrame}`` — one entry per focal length.
    dataframes_aperture : dict
        ``{label: pd.DataFrame}`` — one entry per aperture.
    focal_xlabel, aperture_xlabel : str
        X-axis labels for the two rows.
    plot_error_types : list of str
        Which error types to plot (columns of the figure).
    percentile : str
        Percentile column to read, e.g. ``'p50'``.
    out_dir : Path
        Output directory.
    title_suffix_focal, title_suffix_aperture : str
        Additional text appended to subplot titles.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cols  = len(plot_error_types)
    fig     = plt.figure(figsize=(7.6, 2.6))
    gs      = GridSpec(3, n_cols, figure=fig, height_ratios=[1.2, 0.6, 1.2])

    axes_focal    = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
    axes_aperture = [fig.add_subplot(gs[2, i], sharey=axes_focal[i]) for i in range(n_cols)]

    # Extract model list from first focal dataframe
    first_df = next(iter(dataframes_focal.values()))
    models   = first_df['model'].unique()

    legend_lines  = []
    legend_labels = []

    def _plot_row(axes_list, dataframes, x_labels, xlabel):
        x_pos = np.arange(len(x_labels))
        for idx_ax, et in enumerate(plot_error_types):
            ax = axes_list[idx_ax]
            for i, model in enumerate(models):
                y_values = []
                for key in x_labels:
                    df  = dataframes[key]
                    val = df[
                        (df['model'] == model) & (df['error_type'] == et)
                    ][percentile].values
                    if len(val) == 0:
                        y_values.append(np.nan)
                        continue
                    v = float(val[0])
                    if et == "Prel":
                        v /= 1e4
                    y_values.append(v)

                label = MODEL_PLOTNAMES.get(model, model)
                line, = ax.plot(x_pos, y_values,
                                marker=MARKERS[i % len(MARKERS)],
                                color=COLORS[i % len(COLORS)],
                                linestyle=LINESTYLES[i % len(LINESTYLES)],
                                label=label)
                if idx_ax == 0:
                    legend_lines.append(line)
                    legend_labels.append(label)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=0)
            ax.grid(True, alpha=0.3)

            pfx = "Median " if percentile == "p50" else ""
            if et == "Prel":
                ax.set_title(f"{pfx}Relative Planarity Magnitude (×1e4)")
            elif et == "Pnorm":
                ax.set_title(f"{pfx}Scale-normalised Planarity (×1e6)")
            else:
                ax.set_title(f"{pfx}{ERROR_PLOTNAMES.get(et, et)} Error")

            if idx_ax == 0:
                ax.set_ylabel("Magnitude")
            ax.set_xlabel(xlabel)
            ax.tick_params(pad=0.2)
            ax.xaxis.labelpad = 0.1
            ax.yaxis.labelpad = 0.16

    _plot_row(axes_focal,    dataframes_focal,    list(dataframes_focal.keys()),    focal_xlabel)
    _plot_row(axes_aperture, dataframes_aperture, list(dataframes_aperture.keys()), aperture_xlabel)

    # Shared legend between the two rows
    ncol = 4
    rows = (len(legend_labels) + ncol - 1) // ncol
    lines_rm, labels_rm = [], []
    for c in range(ncol):
        for r in range(rows):
            idx = r * ncol + c
            if idx < len(legend_labels):
                lines_rm.append(legend_lines[idx])
                labels_rm.append(legend_labels[idx])

    leg = fig.legend(lines_rm, labels_rm,
                     loc='upper center', bbox_to_anchor=(0.5, 0.53),
                     ncol=ncol, frameon=True,
                     columnspacing=0.6)
    leg.get_frame().set_linewidth(0.2)
    leg.get_frame().set_edgecolor('black')

    plt.subplots_adjust(top=0.999, bottom=0.001, left=0.0, right=1.0,
                        hspace=0.001, wspace=0.12)

    base = out_dir / f"error_trends_{percentile}"
    plt.savefig(str(base) + ".png",  dpi=600, bbox_inches='tight')
    plt.savefig(str(base) + ".svg",  format='svg', bbox_inches='tight')
    plt.savefig(str(base) + ".pdf",  format='pdf', dpi=900, bbox_inches='tight')
    plt.close()
    print(f"Saved → {base}.[png|svg|pdf]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_csv_dict(entries: list) -> dict:
    """Parse ``['label:path', ...]`` into ``{label: path}``."""
    out = {}
    for e in entries:
        parts = e.split(':', 1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"CSV entries must be 'label:path', got: {e!r}"
            )
        out[parts[0].strip()] = parts[1].strip()
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot error-metric trends vs. focal length and aperture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
    python visualization/plots_from_csvs.py \\
        --focal_csvs fl28:/data/output/config_fl28_F2.8/err_GT/error_percentiles.csv \\
                     fl70:/data/output/config_fl70_F2.8/err_GT/error_percentiles.csv \\
        --aperture_csvs F2.8:/data/output/config_fl70_F2.8/err_GT/error_percentiles.csv \\
                        F22.0:/data/output/config_fl70_F22.0/err_GT/error_percentiles.csv \\
        --out_dir /path/to/output \\
        --percentile p50
""",
    )
    parser.add_argument('--focal_csvs', nargs='+', required=True,
                        metavar='LABEL:PATH',
                        help="Focal-length CSVs: 'label:csv_path' pairs.")
    parser.add_argument('--aperture_csvs', nargs='+', required=True,
                        metavar='LABEL:PATH',
                        help="Aperture CSVs: 'label:csv_path' pairs.")
    parser.add_argument('--out_dir', required=True,
                        help="Output directory for figures.")
    parser.add_argument('--percentile', default='p50',
                        help="Percentile column name (default: p50 = median).")
    parser.add_argument('--error_types', nargs='+',
                        default=['iqr', 'grad', 'Prel', 'Pnorm'],
                        help="Error types to plot (default: iqr grad Prel Pnorm).")
    parser.add_argument('--focal_xlabel', default='Focal Length (mm)')
    parser.add_argument('--aperture_xlabel', default='Aperture (F)')
    args = parser.parse_args()

    focal_csv_dict    = _parse_csv_dict(args.focal_csvs)
    aperture_csv_dict = _parse_csv_dict(args.aperture_csvs)

    df_focal    = {k: pd.read_csv(v, skiprows=1) for k, v in focal_csv_dict.items()}
    df_aperture = {k: pd.read_csv(v, skiprows=1) for k, v in aperture_csv_dict.items()}

    plot_error_trends(
        dataframes_focal=df_focal,
        dataframes_aperture=df_aperture,
        focal_xlabel=args.focal_xlabel,
        aperture_xlabel=args.aperture_xlabel,
        plot_error_types=args.error_types,
        percentile=args.percentile,
        out_dir=Path(args.out_dir),
    )


if __name__ == '__main__':
    main()
