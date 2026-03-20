import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
from pathlib import Path
import os
import csv
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import gzip

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 5,
    'axes.titlesize': 5,
    'axes.labelsize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
    'figure.autolayout': False,
    'figure.constrained_layout.use': False,
})
plt.rcParams['mathtext.fontset'] = 'stix'  # for math equations to also use serif fonts
plt.rcParams['axes.titlepad'] = 0.2

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

def get_pretty_name(name):
    """Converts a filename string to a display-friendly name."""
    name = name.lower() # Make matching case-insensitive
    if 'monster' in name: return 'MonSter'
    if 'foundation' in name: return 'Foundation Stereo'
    if 'defom' in name: return 'DEFOM Stereo'
    if 'selective' in name: return 'Selective IGEV'
    if 'depthpro' in name: return 'Depth Pro'
    if 'metric3d' in name: return 'Metric3D V2'
    if 'unidepth' in name: return 'UniDepth V2'
    if 'depth_anything' in name: return 'DAV2'
    return name # Return original name if no match

def plot_depth_maps(depth_data, mono_models, stereo_models, save_dir, idx):
    """Plot all depth maps (4 mono on top, 4 stereo on bottom) with single colorbar"""
    # Create figure with extra space for colorbar
    fig = plt.figure(figsize=(7.6, 4))
    gs = plt.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.05])
    
    # Calculate overall min/max from percentiles
    all_mins = []
    all_maxs = []
    for i in range(len(depth_data)):
        data = depth_data[i]
        valid_data = data[~np.isnan(data)]
        all_mins.append(np.percentile(valid_data, 5))
        all_maxs.append(np.percentile(valid_data, 95))
    
    vmin = np.mean(all_mins)
    vmax = np.mean(all_maxs)
    
    # Create trimmed turbo colormap (removing 2% from each end)
    turbo = plt.cm.turbo
    colors = turbo(np.linspace(0, 1, 256))
    n_trim = int(256 * 0.02)  # 2% trim
    colors = colors[n_trim:-n_trim]
    trimmed_turbo = LinearSegmentedColormap.from_list('trimmed_turbo', colors)
    
    plt.suptitle(f'Depth Maps - Image {idx}')
    
    # Plot mono models on top row
    for i, model in enumerate(mono_models):
        model_name = get_pretty_name(model)
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(depth_data[i], cmap=trimmed_turbo, vmin=vmin, vmax=vmax)
        ax.set_title(f'Mono - {model_name}')
        ax.axis('off')
    
    # Plot stereo models on bottom row
    for i, model in enumerate(stereo_models):
        model_name = get_pretty_name(model)
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(depth_data[i+len(mono_models)], cmap=trimmed_turbo, vmin=vmin, vmax=vmax)
        ax.set_title(f'Stereo - {model_name}')
        ax.axis('off')
    
    # Add single colorbar on the right
    cbar_ax = fig.add_subplot(gs[:, -1])
    plt.colorbar(im, cax=cbar_ax, label='Depth')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'depth_maps_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

# def plot_error_maps(error_maps, model_names, save_dir, idx):
#     """Plot error maps (8 rows for models, 4 columns for error types)"""
#     if idx != 7:
#         return
#     error_plotnames = {"grad": "Gradient", "plan": "Planarity", "icp": "ICP", "iqr": "IQR"}
#     error_types = ['grad', 'plan', 'icp', 'iqr']
    
#     # Create figure with extra space for colorbars
#     fig = plt.figure(figsize=(5.76,2.5))
#     gs = plt.GridSpec(1, 7, width_ratios=[1, 1, 0.005, 1, 0.005, 1, 0.005], height_ratios=[1])
#     #plt.suptitle(f'Error Maps - Image {idx}')
    
#     # Create trimmed turbo colormap (removing 2% from each end)
#     turbo = plt.cm.turbo
#     colors = turbo(np.linspace(0, 1, 256))
#     n_trim = int(256 * 0.02)  # 2% trim
#     colors = colors[n_trim:-n_trim]
#     trimmed_turbo = LinearSegmentedColormap.from_list('trimmed_turbo', colors)    
    
#     for i, model in enumerate(model_names):
#         model_name = get_pretty_name(model)
#         if "foundation" not in model_name.lower():
#             continue
#         print(model)
#         # Get min/max values for this model's error maps
#         error_mins = []
#         error_maxs = []
#         for error_type in error_types:
#             scale=1
#             if error_type == 'plan':
#                 scale = 1e6
#             data = error_maps[model_name][error_type][idx]*scale
#             error_mins.append(np.percentile(data[~np.isnan(data)], 5))
#             error_maxs.append(np.percentile(data[~np.isnan(data)], 95))
        
#         vmin1 = min(error_mins[:2])
#         vmax1 = max(error_maxs[:2])
#         vmin2 = error_mins[2]
#         vmax2 = error_maxs[2]
#         vmin3 = error_mins[3]
#         vmax3 = error_maxs[3]
        
#         # Plot each error type
#         for j, error_type in enumerate(error_types):
#             scale=1
#             if error_type == 'plan':
#                 scale = 1e6
#             data = error_maps[model_name][error_type][idx]*scale
            
#             if j in [0,1]:
#                 ax = fig.add_subplot(gs[0, j])
#                 im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin1, vmax=vmax1)
#                 if j==1:
#                     cbar_ax = fig.add_subplot(gs[0, 2])
#                     plt.colorbar(im, cax=cbar_ax)
#                     cbar_ax.tick_params(axis='y',   # 'y' for vertical colorbar, 'x' for horizontal
#                     direction='out',    # ticks pointing outwards
#                     length=2,           # tick stem length in points (shorter)
#                     pad=0.1,            # distance from tick to tick label in points
#                     labelsize=2)
#             elif j==2:
#                 ax = fig.add_subplot(gs[0, 3])
#                 im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin2, vmax=vmax2)
#                 cbar_ax = fig.add_subplot(gs[0, 4])
#                 plt.colorbar(im, cax=cbar_ax)
#                 ax.axis('off')
#                 pos = cbar_ax.get_position()
#                 cbar_ax.set_position([
#                     pos.x0,                      # same x
#                     pos.y0 + 0.15 * pos.height,  # shift upward a bit
#                     pos.width,                   # same width
#                     0.7 * pos.height             # 70% of original height
#                 ])
#                 cbar_ax.tick_params(axis='y',   # 'y' for vertical colorbar, 'x' for horizontal
#                     direction='out',    # ticks pointing outwards
#                     length=2,           # tick stem length in points (shorter)
#                     pad=0.1,            # distance from tick to tick label in points
#                     labelsize=2)
#             elif j==3:
#                 ax = fig.add_subplot(gs[0, 5])
#                 im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin3, vmax=vmax3)
#                 cbar_ax = fig.add_subplot(gs[0, 6])
#                 plt.colorbar(im, cax=cbar_ax)
#                 ax.axis('off')
#                 cbar_ax.tick_params(axis='y',   # 'y' for vertical colorbar, 'x' for horizontal
#                     direction='out',    # ticks pointing outwards
#                     length=2,           # tick stem length in points (shorter)
#                     pad=0.1,            # distance from tick to tick label in points
#                     labelsize=2)
#             title_suf = " (x1e6)" if error_type == "plan" else ""
#             ax.set_title(f'{error_plotnames[error_type]} Error{title_suf}')
#             ax.axis('off')            
        
#         # # Add colorbar for this row
#         # cbar_ax = fig.add_subplot(gs[i, -1])
#         # plt.colorbar(im, cax=cbar_ax)
#     plt.subplots_adjust(left=0.001, right=0.999, top=0.98, bottom=0.02, 
#                         hspace=0.01, wspace=0.05)
#     # plt.tight_layout()    
#     plt.savefig(save_dir / f'error_maps_{idx}.png', dpi=900, bbox_inches='tight', pad_inches=0.01)
#     print(f"✅ Saved PNG")    
    
#     # plt.savefig(save_dir / f'error_maps_{idx}.pdf', format='pdf', dpi=900, bbox_inches='tight', pad_inches=0.01)
#     # print(f"✅ Saved PDF")    
    
#     # plt.savefig(save_dir / f'error_maps_{idx}.svg', format='svg', bbox_inches='tight', pad_inches=0.01)
#     # print(f"✅ Saved SVG")
#     plt.close()

# def plot_error_maps(error_maps, model_names, save_dir, idx):
#     """Plot error maps (8 rows for models, 4 columns for error types)"""
#     if idx != 7:
#         return
#     error_plotnames = {"grad": "Gradient", "plan": "Planarity", "icp": "ICP", "iqr": "IQR"}
#     error_types = ['grad', 'plan', 'icp', 'iqr']

#     fig = plt.figure(figsize=(5.76, 1.2))
#     gs = plt.GridSpec(1, 7, width_ratios=[1, 1, 0.001, 1, 0.001, 1, 0.001], height_ratios=[1])

#     turbo = plt.cm.turbo
#     colors = turbo(np.linspace(0, 1, 256))
#     n_trim = int(256 * 0.02)
#     colors = colors[n_trim:-n_trim]
#     trimmed_turbo = LinearSegmentedColormap.from_list('trimmed_turbo', colors)

#     for i, model in enumerate(model_names):
#         model_name = get_pretty_name(model)
#         if "foundation" not in model_name.lower():
#             continue
#         print(model)

#         error_mins, error_maxs = [], []
#         for error_type in error_types:
#             scale = 1e6 if error_type == 'plan' else 1
#             data = error_maps[model_name][error_type][idx] * scale
#             valid = data[~np.isnan(data)]
#             error_mins.append(np.percentile(valid, 5))
#             error_maxs.append(np.percentile(valid, 95))

#         vmin1, vmax1 = min(error_mins[:2]), max(error_maxs[:2])
#         vmin2, vmax2 = error_mins[2], error_maxs[2]
#         vmin3, vmax3 = error_mins[3], error_maxs[3]

#         for j, error_type in enumerate(error_types):
#             scale = 1e6 if error_type == 'plan' else 1
#             data = error_maps[model_name][error_type][idx] * scale

#             if j in [0, 1]:
#                 ax = fig.add_subplot(gs[0, j])
#                 im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin1, vmax=vmax1)
#                 if j == 1:
#                     cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, use_gridspec=False)
#                     pos_img = ax.get_position()
#                     pos_cb  = cb.ax.get_position()
#                     print(pos_img)
#                     print(pos_cb)
#                     cb.ax.set_position([
#                         pos_img.x1 + 0.01,                  # shift right of image with small gap
#                         pos_img.y0, #+ 0.02 * pos_img.height, # small top/bottom margins
#                         pos_cb.width,                        # keep original width
#                         pos_img.height, #pos_img.height# pos_cb.height#                # match height of image
#                     ])
#                     cb.ax.tick_params(
#                         axis='y',
#                         direction='out',
#                         length=1,      # short stems
#                         pad=0.3,       # minimal gap between stem and label
#                         labelsize=3    # small font
#                     )
#             elif j == 2:
#                 ax = fig.add_subplot(gs[0, 3])
#                 im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin2, vmax=vmax2)
#                 cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, use_gridspec=False)
#                 pos_img = ax.get_position()
#                 pos_cb  = cb.ax.get_position()
#                 print(pos_img)
#                 print(pos_cb)
#                 cb.ax.set_position([
#                     pos_img.x1 + 0.05,                  # shift right of image with small gap
#                     pos_img.y0 - 0.1 * pos_img.height, # small top/bottom margins
#                     pos_cb.width,                        # keep original width
#                     1.5* pos_img.height                # match height of image
#                 ])
#                 cb.ax.tick_params(
#                     axis='y',
#                     direction='out',
#                     length=1,      # short stems
#                     pad=0.3,       # minimal gap between stem and label
#                     labelsize=3    # small font
#                 )

#             elif j == 3:
#                 ax = fig.add_subplot(gs[0, 5])
#                 im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin3, vmax=vmax3)
#                 cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, use_gridspec=False)
#                 pos_img = ax.get_position()
#                 pos_cb  = cb.ax.get_position()
#                 print(pos_img)
#                 print(pos_cb)
#                 cb.ax.set_position([
#                     pos_img.x1 + 0.1,                  # shift right of image with small gap
#                     pos_img.y0 + 0.02 * pos_img.height, # small top/bottom margins
#                     pos_cb.width,                        # keep original width
#                     pos_img.height                # match height of image
#                 ])
#                 cb.ax.tick_params(
#                     axis='y',
#                     direction='out',
#                     length=1,      # short stems
#                     pad=0.3,       # minimal gap between stem and label
#                     labelsize=3    # small font
#                 )
#             title_suf = " (x1e6)" if error_type == "plan" else ""
#             ax.set_title(f'{error_plotnames[error_type]} Error{title_suf}')
#             ax.axis('off')

#     plt.subplots_adjust(left=0.001, right=0.98, top=0.98, bottom=0.02,
#                         hspace=0.01, wspace=0.05)
#     plt.savefig(save_dir / f'error_maps_{idx}.png', dpi=200, bbox_inches='tight', pad_inches=0.01)
#     print("✅ Saved PNG")
#     plt.close()

def plot_error_maps(error_maps, model_names, save_dir, idx):
    """Plot error maps (1 row, 4 image columns for selected models).
    Preserves existing subplot geometry; places full-height colorbars by creating
    dedicated axes from each image Axes.get_position().
    """
    if idx != 7:
        return

    error_plotnames = {"grad": "Gradient", "plan": "Planarity", "icp": "ICP", "iqr": "IQR"}
    error_types = ['grad', 'plan', 'icp', 'iqr', 'rms_orth', 'Prel', 'Pnorm']

    fig = plt.figure(figsize=(5.76, 1.2))
    gs = plt.GridSpec(1, 7, width_ratios=[1, 1, 0.001, 1, 0.001, 1, 0.001], height_ratios=[1])

    # trimmed turbo as before
    turbo = plt.cm.turbo
    colors = turbo(np.linspace(0, 1, 256))
    n_trim = int(256 * 0.02)
    colors = colors[n_trim:-n_trim]
    trimmed_turbo = LinearSegmentedColormap.from_list('trimmed_turbo', colors)

    # horizontal gap (figure coordinates) between image right edge and cbar left edge
    H_GAP = 0.0005
    # colorbar width (figure coordinates); small consistent width
    CB_WIDTH = 0.006

    for i, model in enumerate(model_names):
        model_name = get_pretty_name(model)
        if "foundation" not in model_name.lower():
            continue

        # compute robust percentiles per-map for consistent vmin/vmax groups
        error_mins, error_maxs = [], []
        for error_type in error_types:
            scale = 1e6 if error_type == 'plan' else 1
            data = error_maps[model_name][error_type][idx] * scale
            valid = data[~np.isnan(data)]
            error_mins.append(np.percentile(valid, 5))
            error_maxs.append(np.percentile(valid, 95))

        vmin1, vmax1 = min(error_mins[:2]), max(error_maxs[:2])
        vmin2, vmax2 = error_mins[2], error_maxs[2]
        vmin3, vmax3 = error_mins[3], error_maxs[3]

        for j, error_type in enumerate(error_types):
            scale = 1e6 if error_type == 'plan' else 1
            data = error_maps[model_name][error_type][idx] * scale

            if j in [0, 1]:
                ax = fig.add_subplot(gs[0, j])
                im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin1, vmax=vmax1)
                # only place a colorbar for the second of the two images to match original
                if j == 1:
                    pos_img = ax.get_position()  # Bbox in figure coords
                    # create a new axes for the colorbar using exact image height
                    cax_pos = [
                        pos_img.x1 - 0.011,# + H_GAP,     # left
                        pos_img.y0,             # bottom aligned to image bottom
                        CB_WIDTH,               # width
                        pos_img.height          # full image height
                    ]
                    cax = fig.add_axes(cax_pos)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    # remove border/outline and frame
                    cbar.outline.set_visible(False)
                    cax.set_frame_on(False)
                    cbar.ax.tick_params(
                        axis='y', direction='out', length=0.1, pad=0.1, labelsize=4
                    )

            elif j == 2:
                ax = fig.add_subplot(gs[0, 3])
                im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin2, vmax=vmax2)
                pos_img = ax.get_position()
                cax_pos = [
                    pos_img.x1 + 0.035, # + 0.02,    # keep same small gap
                    pos_img.y0,            # align bottom
                    CB_WIDTH,              # consistent width
                    pos_img.height         # exact height
                ]
                cax = fig.add_axes(cax_pos)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                cbar.outline.set_visible(False)
                cax.set_frame_on(False)
                cbar.ax.tick_params(axis='y', direction='out', length=0.1, pad=0.1, labelsize=4)

            elif j == 3:
                ax = fig.add_subplot(gs[0, 5])
                im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin3, vmax=vmax3)
                pos_img = ax.get_position()
                cax_pos = [
                    pos_img.x1 + 0.08,    # keep same small gap
                    pos_img.y0,            # align bottom
                    CB_WIDTH,              # consistent width
                    pos_img.height         # exact height
                ]
                cax = fig.add_axes(cax_pos)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                cbar.outline.set_visible(False)
                cax.set_frame_on(False)
                cbar.ax.tick_params(axis='y', direction='out', length=0.1, pad=0.1, labelsize=4)   

            title_suf = " (x1e6)" if error_type == "plan" else ""
            ax.set_title(f'{error_plotnames[error_type]} Error{title_suf}')
            ax.axis('off')

    # preserve your original global subplot tuning
    plt.subplots_adjust(left=0.001, right=0.97, top=0.98, bottom=0.02,
                        hspace=0.01, wspace=0.09)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'error_maps_{idx}.png', dpi=600, bbox_inches='tight', pad_inches=0.01)    

    plt.savefig(save_dir / f'error_maps_{idx}.pdf', bbox_inches='tight', pad_inches=0.01, dpi=900)
    plt.close(fig)

#def plot_error_maps(error_maps, model_names, save_dir, idx):
    # """Plot error maps (8 rows for models, 4 columns for error types)"""
    # if idx != 7:
    #     return

    # import matplotlib.pyplot as plt
    # from matplotlib.colors import LinearSegmentedColormap
    # import numpy as np

    # error_plotnames = {"grad": "Gradient", "plan": "Planarity", "icp": "ICP", "iqr": "IQR"}
    # error_types = ['grad', 'plan', 'icp', 'iqr']

    # # --- Figure and grid setup ---
    # fig = plt.figure(figsize=(7.2, 2.8))  # taller and slightly narrower
    # gs = plt.GridSpec(len(model_names), 7, width_ratios=[1, 1, 0.03, 1, 0.03, 1, 0.03])

    # # --- Trimmed colormap ---
    # turbo = plt.cm.turbo
    # colors = turbo(np.linspace(0.03, 0.97, 256))
    # trimmed_turbo = LinearSegmentedColormap.from_list('trimmed_turbo', colors)

    # for i, model in enumerate(model_names):
    #     model_name = get_pretty_name(model)
    #     if "foundation" not in model_name.lower():
    #         continue

    #     # dynamic range estimation
    #     mins, maxs = [], []
    #     for e in error_types:
    #         scale = 1e6 if e == 'plan' else 1
    #         data = error_maps[model_name][e][idx] * scale
    #         mins.append(np.percentile(data[~np.isnan(data)], 5))
    #         maxs.append(np.percentile(data[~np.isnan(data)], 95))
    #     vmins = [min(mins[:2]), mins[2], mins[3]]
    #     vmaxs = [max(maxs[:2]), maxs[2], maxs[3]]

    #     for j, e in enumerate(error_types):
    #         scale = 1e6 if e == 'plan' else 1
    #         data = error_maps[model_name][e][idx] * scale

    #         col = [0, 1, 3, 5][j]
    #         vmin, vmax = vmins[min(j, 2)], vmaxs[min(j, 2)]

    #         ax = fig.add_subplot(gs[i, col])
    #         im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin, vmax=vmax)
    #         ax.set_title(
    #             f"{error_plotnames[e]} Error{' (×1e6)' if e == 'plan' else ''}",
    #             fontsize=7, pad=1.5,
    #         )
    #         ax.axis('off')

    #         cbar_col = col + 1
    #         cax = fig.add_subplot(gs[i, cbar_col])
    #         cbar = plt.colorbar(im, cax=cax)
    #         cbar.ax.tick_params(length=2, pad=0.6, labelsize=5)
    #         cbar.outline.set_linewidth(0.3)

    # # --- Compact layout ---
    # plt.subplots_adjust(left=0.001, right=0.995, top=0.995, bottom=0.005, wspace=0.02, hspace=0.001)

    # # --- Export high-quality vector/raster versions ---
    # for ext, dpi in [('png', 600), ('pdf', 900), ('svg', None)]:
    #     fig.savefig(save_dir / f"error_maps_{idx}.{ext}",
    #                 format=ext, dpi=dpi if dpi else None,
    #                 bbox_inches='tight', pad_inches=0.0)
    #     print(f"✅ Saved {ext.upper()}")

    # plt.close(fig)

def fuse_depth_maps(depth_data, error_maps, model_names):
    """Fuse depth maps using error-based weights"""
    weights = np.ones_like(depth_data)
    
    # Calculate weights based on normalized errors
    for i, model in enumerate(model_names):        
        model_weight = np.ones_like(depth_data[i])
        for error_type in ['grad', 'plan', 'icp', 'iqr']:
            error = error_maps[model][error_type]
            # Normalize error to [0,1] range and invert
            norm_error = 1 - (error - np.nanmin(error)) / (np.nanmax(error) - np.nanmin(error))
            model_weight *= norm_error
        weights[i] = model_weight
    
    # Normalize weights
    weights = weights / np.sum(weights, axis=0, keepdims=True)
    
    # Compute weighted average
    fused_depth = np.sum(depth_data * weights, axis=0)
    return fused_depth

def plot_fused_depth(fused_depth, save_dir, idx):
    """Plot fused depth map with trimmed turbo colormap"""
    # Create figure with extra space for colorbar
    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 0.01])
    
    # Calculate min/max from percentiles of valid data
    valid_data = fused_depth[~np.isnan(fused_depth)]
    vmin = np.percentile(valid_data, 5)
    vmax = np.percentile(valid_data, 95)
    
    # Create trimmed turbo colormap (removing 2% from each end)
    turbo = plt.cm.turbo
    colors = turbo(np.linspace(0, 1, 256))
    n_trim = int(256 * 0.02)  # 2% trim
    colors = colors[n_trim:-n_trim]
    trimmed_turbo = LinearSegmentedColormap.from_list('trimmed_turbo', colors)
    
    # Plot depth map
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(fused_depth, cmap=trimmed_turbo, vmin=vmin, vmax=vmax)
    ax.set_title(f'Fused Depth Map - Image {idx}')
    ax.axis('off')
    
    # Add colorbar on the right
    cbar_ax = fig.add_subplot(gs[1])
    plt.colorbar(im, cax=cbar_ax, label='Depth')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'fused_depth_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_error_distributions(error_aggr, save_dir):
    """
    Analyze error distributions across models and create CDF plots
    
    Args:
        error_aggr: Dictionary of error aggregates per model
        save_dir: Directory to save outputs
    """
    error_types = ['grad',  'iqr', 'Prel', 'Pnorm'] #'plan', 'icp','rms_orth', 
    percentiles = np.arange(0, 101, 10)  # 0, 10, 20, ..., 100
    
    # Prepare data for CSV
    csv_data = []
    plan_scale = 1e6
    headers = [[f"# Planarity, RMS Orth, Preal, Pnorm errors have been scaled by {plan_scale}."],
               ['model', 'error_type'] + [f'p{p}' for p in percentiles]]
    
    # Process each model and error type
    for model_name, model_errors in error_aggr.items():        
        for error_type in error_types:
            # Flatten the 1000-length arrays across all images
            error_values = model_errors[error_type].flatten()
            if error_type in ["plan", "rms_orth", "Prel", "Pnorm"]:
                error_values = error_values * plan_scale
            percentile_values = np.percentile(error_values, percentiles)
            
            # Add to CSV data
            row = [model_name, error_type] + [f'{v:.6f}' for v in percentile_values]
            csv_data.append(row)
    
    # Save CSV
    csv_path = save_dir / 'error_percentiles.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(headers)
        writer.writerows(csv_data)
    
    # Create CDF plots for each error type
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    for error_type in error_types:
        plt.figure(figsize=(10, 6))
        
        for i, (model_name, model_errors) in enumerate(error_aggr.items()):            
            # Get error values and sort them
            scale = plan_scale if error_type == 'plan' else 1
            error_values = model_errors[error_type].flatten()*scale            
            sorted_values = np.sort(error_values)
            
            # Calculate cumulative probabilities
            n = len(sorted_values)
            cumulative_prob = np.arange(1, n + 1) / n
            
            # Plot CDF
            plt.plot(sorted_values, cumulative_prob, label=model_name, linewidth=1, 
            marker=markers[i], color=colors[i], linestyle=linestyles[i], markersize=2.0)   # smaller markers
        
        plt.grid(True, alpha=0.3)
        plt.xlabel(f'{error_type.upper()} Error' + (' (×1e6)' if error_type in ['plan', 'rms_orth', 'Prel', 'Pnorm'] else ''))
        plt.ylabel('Cumulative Probability')
        plt.title(f'Cumulative Distribution of {error_type.upper()} Error Across Models')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir / f'error_cdf_{error_type}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main(datalist=None, specific_path=None):
    # Model lists
    MONO_MODELS = ['depthpro', 'metric3d', 'unidepth', 'depth_anything']
    STEREO_MODELS = ['monster', 'foundation', 'defom', 'selective']
    ALL_MODELS = MONO_MODELS + STEREO_MODELS
    
    if specific_path:
        # Use the specific path provided
        error_data_path = specific_path
        save_dir = error_data_path.parent
        try: 
            with gzip.open(error_data_path, 'rb') as f:
                error_data = pickle.load(f)
        except:
            with open(error_data_path, 'rb') as f:
                error_data = pickle.load(f)

        # depth_data_path = save_dir / "depth_data.pkl"
        # with open(depth_data_path, 'rb') as f:
        #     depth_data_arr = pickle.load(f)            
        
        # Process each image index
        # Plot error maps
        # for img_idx in range(error_data['error_maps']['MonSter']['grad'].shape[0]):
        #     plot_error_maps(error_data['error_maps'], ALL_MODELS, save_dir, img_idx)
        # for img_idx, depth_data in enumerate(depth_data_arr):            
        #     # Plot depth maps
        #     # plot_depth_maps(depth_data, MONO_MODELS, STEREO_MODELS, save_dir, img_idx)
            
            
            
        #     # Compute and plot fused depth
        #     # fused_depth = fuse_depth_maps(depth_data, error_data['error_maps'], ALL_MODELS)
        #     # plot_fused_depth(fused_depth, save_dir, img_idx)
        #     pass
            
        # Analyze error distributions
        analyze_error_distributions(error_data['error_aggr'], save_dir)
        
    else:
        pass
        # Use provided datalist
        # if not datalist:
        #     datalist = [
        #         {
        #             "base": r"I:\\My Drive\\Pubdata\\Scene9",
        #             "cameras": ['EOS6D_B_Left', 'EOS6D_A_Right'],
        #             "configs": [
        #                 {"fl":70, "F":2.8},
        #             ]
        #         },
        #     ]
        
        # for entry in datalist:
        #     base = Path(entry['base'])
        #     left_cam = entry['cameras'][0]
        #     for cfg in entry['configs']:
        #         fl_folder = f"fl_{int(cfg['fl'])}mm"
        #         F_folder = f"F{cfg['F']:.1f}"
        #         error_data_path = base / left_cam / fl_folder / "inference" / F_folder / "err_GT" / "error_data.pkl"
        #         save_dir = error_data_path.parent
                
        #         with open(error_data_path, 'rb') as f:
        #             error_data = pickle.load(f)
                
        #         # Process each image index
        #         for img_idx, depth_data in enumerate(error_data['depth_data_arr']):
        #             # Plot depth maps
        #             plot_depth_maps(depth_data, MONO_MODELS, STEREO_MODELS, save_dir, img_idx)
                    
        #             # Plot error maps
        #             plot_error_maps(error_data['error_maps'], ALL_MODELS, save_dir, img_idx)
                    
        #             # Compute and plot fused depth
        #             fused_depth = fuse_depth_maps(depth_data, error_data['error_maps'], ALL_MODELS)
        #             plot_fused_depth(fused_depth, save_dir, img_idx)
                    
        #         # Analyze error distributions
        #         analyze_error_distributions(error_data['error_aggr'], save_dir)

if __name__ == "__main__":
    main()
