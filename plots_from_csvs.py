import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 5.5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 6,
    'lines.linewidth': 0.7,
    'lines.markersize': 2.5
})

plot_type = "focal_lengths"
# plot_type = "apertures"
outdir = "H:\\My Drive\\Research_collabs\\MODEST Research Collab\\CVPR_visuals"
# Example input: dict with dataset name -> csv file path
csv_dict = {
    "focal_lengths" : {
        "data": {
            "fl28": "E:\\pub_results\\scene9_fl28mm_F2.8\\err_GT\\error_percentiles.csv",
            "fl32": "E:\\pub_results\\scene9_fl32mm_F2.8\\err_GT\\error_percentiles.csv",
            "fl36": "E:\\pub_results\\scene9_fl36mm_F2.8\\err_GT\\error_percentiles.csv",
            "fl40": "E:\\pub_results\\scene9_fl40mm_F2.8\\err_GT\\error_percentiles.csv",
            "fl45": "E:\\pub_results\\scene9_fl45mm_F2.8\\err_GT\\error_percentiles.csv",
            "fl60": "E:\\pub_results\\scene9_fl60mm_F2.8\\err_GT\\error_percentiles.csv",
            "fl65": "E:\\pub_results\\scene9_fl65mm_F2.8\\err_GT\\error_percentiles.csv",
            "fl70": "E:\\pub_results\\scene9_fl70mm_F2.8\\err_GT\\error_percentiles.csv",
        },
        "titlesuf" : "Error vs Focal Lengths for constant aperture F2.8",
        "xlabel" : "Focal Length (mm)"
    },   

    "apertures" : {
        "data": {    
            "F2.8": "E:\\pub_results\\scene9_fl70mm_F2.8\\err_GT\\error_percentiles.csv",
            "F5.0": "E:\\pub_results\\scene9_fl70mm_F5.0\\err_GT\\error_percentiles.csv",
            "F9.0": "E:\\pub_results\\scene9_fl70mm_F9.0\\err_GT\\error_percentiles.csv",
            "F16.0": "E:\\pub_results\\scene9_fl70mm_F16.0\\err_GT\\error_percentiles.csv",    
            "F22.0": "E:\\pub_results\\scene9_fl70mm_F22.0\\err_GT\\error_percentiles.csv",    
        },
        "titlesuf" : "Error vs Apertures for constant Focal Length 70mm",
        "xlabel" : "Aperture (F)"
    }
}
# percentile string you want to plot
percentile = "p50"

# # define markers and line styles for 8 models
# markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']  # example markers
# colors = ['orange', 'gray', 'blue', 'green', 'red', 'purple', 'brown', 'pink']
# linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

markers = [ 'p', 'v', '<', '>','^','s','D','o']  # distinct and compact

# color scheme: cooler tones (category 1), warmer/brighter tones (category 2)
colors = [    
    '#e377c2',  # pink    
    '#17becf',  # cyan
    '#1f77b4',  # blue
    '#7f7f7f',  # gray    
    '#2ca02c',  # green (emphasized group starts)
    '#d62728',  # red
    '#ff7f0e',  # orange
    '#9467bd',  # purple
    
]

# linestyle pattern: simple alternation, but consistent within group
linestyles = [
    '--',   # baseline 1
    '-.',  # baseline 2
    ':',  # baseline 3
    '--',   # baseline 4
    '-',   # proposed 1 (same patterns reused to avoid clutter)
    '-',  # proposed 2
    '-',  # proposed 3
    '-',   # proposed 4
]

# error types
error_types = ["grad", "plan", "icp", "iqr", "rms_orth"]
plot_error_types = ["iqr", "grad", "Prel", "Pnorm"]
error_plotnames = {"grad": "Gradient", "plan": "Planarity", 
                    "icp": "ICP", "iqr": "IQR", 
                    "rms_orth": "RMS Orthogonality", 
                    "Prel": "P_rel", "Pnorm": "P_norm"}
model_plotnames = {    
    "Depth Pro": "m:DepthPro",
    "Metric3D V2": "m:Metric3D",
    "UniDepth V2": "m:UniDepth",
    "DAV2": "m:DAV2",
    "MonSter": "s:MonSter",
    "Foundation Stereo": "s:Foundation",
    "DEFOM Stereo": "s:DEFOM",
    "Selective IGEV": "s:Selective"
}
# first, load all CSVs into a dictionary of DataFrames
dataframes_focal = {k: pd.read_csv(v, skiprows=1) for k, v in csv_dict["focal_lengths"]["data"].items()}
dataframes_aperture = {k: pd.read_csv(v, skiprows=1) for k, v in csv_dict["apertures"]["data"].items()}
dataframes = {"focal_lengths": dataframes_focal, "apertures": dataframes_aperture}
# models (assuming the same models exist in all CSVs)
models = dataframes["focal_lengths"][next(iter(dataframes["focal_lengths"]))]['model'].unique()

# Create a single figure with GridSpec for custom subplot layout
fig = plt.figure(figsize=(7.6, 2.6))
# gs = GridSpec(4, 4, figure=fig)
gs = GridSpec(3, 4, figure=fig, height_ratios=[1.2, 0.6, 1.2])
# Create axes for subplots: top row for focal lengths, bottom row for apertures
axes_focal = []
axes_aperture = []
for i in range(4):
    if i == 0:
        ax_focal = fig.add_subplot(gs[0, i])
        ax_aperture = fig.add_subplot(gs[2, i], sharey=ax_focal)
    else:
        ax_focal = fig.add_subplot(gs[0, i])
        ax_aperture = fig.add_subplot(gs[2, i], sharey=ax_focal)
    axes_focal.append(ax_focal)
    axes_aperture.append(ax_aperture)

# Dictionary to store all lines for the legend
legend_lines = []
legend_labels = []

# Plot focal length data (top row)
for idx, error_type in enumerate(plot_error_types):
    ax = axes_focal[idx]
    x_labels = list(csv_dict["focal_lengths"]["data"].keys())
    x_pos = np.arange(len(x_labels))
    
    for i, model in enumerate(models):
        y_values = []
        for key in x_labels:
            df = dataframes["focal_lengths"][key]
            val = df[(df['model'] == model) & (df['error_type'] == error_type)][percentile].values[0]
            if error_type == "Prel":
                val = val/1e4
            y_values.append(val)
        
        line = ax.plot(x_pos, y_values, marker=markers[i], color=colors[i], 
                      linestyle=linestyles[i], label=model_plotnames[model])[0]
        if idx == 0:  # Only store legend info from first subplot
            legend_lines.append(line)
            legend_labels.append(model_plotnames[model])
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.grid(True, alpha=0.3)
    title_prefix = "Median " if percentile == "p50" else "" #else f"{percentile.replace('p', '')}%ile "
    if error_type == "Prel":
        title_suffix = " (x1e4)"
        ax.set_title(f"{title_prefix}{'Relative Planarity Magnitude'}{title_suffix}")
    elif error_type == "Pnorm":
        title_suffix = " (x1e6)"
        ax.set_title(f"{title_prefix}{'Scale-normalized Planarity Deviation'}{title_suffix}")
    else:
        title_suffix = " Error"
        ax.set_title(f"{title_prefix}{error_plotnames[error_type]}{title_suffix}")

    if idx == 0:  # Only add y-label to leftmost plots
        ax.set_ylabel("Magnitude")
    ax.set_xlabel(csv_dict["focal_lengths"]["xlabel"])
    ax.tick_params(pad=0.2)
    ax.xaxis.labelpad = 0.1
    ax.yaxis.labelpad = 0.16

# Plot aperture data (bottom row)
for idx, error_type in enumerate(plot_error_types):
    ax = axes_aperture[idx]
    x_labels = list(csv_dict["apertures"]["data"].keys())
    x_pos = np.arange(len(x_labels))
    
    for i, model in enumerate(models):
        y_values = []
        for key in x_labels:
            df = dataframes["apertures"][key]
            val = df[(df['model'] == model) & (df['error_type'] == error_type)][percentile].values[0]
            if error_type == "Prel":
                val = val/1e4
            y_values.append(val)
        
        ax.plot(x_pos, y_values, marker=markers[i], color=colors[i], 
               linestyle=linestyles[i], label=model_plotnames[model])
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.grid(True, alpha=0.3)
    if idx == 0:  # Only add y-label to leftmost plots
        ax.set_ylabel("Magnitude")
    ax.set_xlabel(csv_dict["apertures"]["xlabel"])
    ax.tick_params(pad=0.2)
    ax.xaxis.labelpad = 0.1
    ax.yaxis.labelpad = 0.16

# Add single legend between the two rows
# fig.legend(legend_lines, legend_labels, 
#           loc='center', bbox_to_anchor=(0.5, 0.5),
#           ncol=4, frameon=False)
# for ax in axes.flat:
#     ax.tick_params(pad=1.5)
# Reorder handles and labels for row-major filling
ncol = 4
lines_row_major = []
labels_row_major = []

rows = (len(legend_labels) + ncol - 1) // ncol
for c in range(ncol):
    for r in range(rows):    
        idx = r*ncol + c
        if idx < len(legend_labels):
            lines_row_major.append(legend_lines[idx])
            labels_row_major.append(legend_labels[idx])
leg = fig.legend(
    #legend_lines, legend_labels,
    lines_row_major, labels_row_major,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.53),   # move slightly above middle of figure
    ncol=4,
    frameon=True,
    # fontsize=,
    columnspacing=0.6,    
    # handlelength=1,
    # handletextpad=0.01
)
leg.get_frame().set_linewidth(0.2)   # set frame line width
leg.get_frame().set_edgecolor('black') 
plt.rcParams.update({
    'legend.borderaxespad': 0.01,   # space between legend and axes
    'legend.borderpad': 0.01,       # padding inside legend box
    'legend.handlelength': 1.0,
    'legend.handletextpad': 0.05,
    'legend.columnspacing': 0.6
})
# plt.tight_layout()
# Adjust layout to make room for legend
plt.subplots_adjust(top=0.999, bottom=0.001, left=0.0, right=1.0, hspace=0.001, wspace=0.12)

# Save plot data to CSV for both focal lengths and apertures
for plot_type in ["focal_lengths", "apertures"]:
    for error_type in plot_error_types:
        csv_data = {}
        for i, model in enumerate(models):
            y_values = []
            for key in csv_dict[plot_type]["data"].keys():
                df = dataframes[plot_type][key]
                val = df[(df['model'] == model) & (df['error_type'] == error_type)][percentile].values[0]
                if error_type == "plan":
                    val = val
                y_values.append(val)
            csv_data[model] = y_values
        
        # Create DataFrame with models as index (rows) and datasets as columns
        plot_df = pd.DataFrame(csv_data).T
        plot_df.columns = list(csv_dict[plot_type]["data"].keys())
        csv_base_path = outdir + "\\" + plot_type + "_" + error_type + "_" + percentile + "_data.csv"
        plot_df.to_csv(csv_base_path)

# Save in multiple formats with high DPI
plot_base_path = csv_base_path.replace("_data.csv", "_combined") 
plt.savefig(plot_base_path + ".png", dpi=600, bbox_inches='tight')
plt.savefig(plot_base_path + ".svg", format='svg', bbox_inches='tight')
plt.savefig(plot_base_path + ".pdf", format='pdf', dpi=900, bbox_inches='tight')
