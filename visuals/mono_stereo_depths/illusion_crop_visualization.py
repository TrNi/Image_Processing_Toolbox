# If requested frame index exceeds available frames in H5, 
# the code automatically loads the last valid frame instead, 
# ensuring no index errors or crashes occur. 
# example -> scene 6,7,8,9 stereo depth has only till fram index 2 , so it limits the mono to go to frame index 2import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
import os
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
from matplotlib.ticker import LogLocator, FormatStrFormatter
# Configure matplotlib styling
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 5,
    'axes.titlesize': 5,
    'axes.labelsize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,    
})
plt.rcParams['mathtext.fontset'] = 'stix'  # for math equations to also use serif fonts
plt.rcParams['axes.titlepad'] = 0.2
# Set Matplotlib to display plots inline in a notebook
# %matplotlib inline
crop_rows = 0

# === USER INPUT ===
# All paths are supplied via CLI -- see main() below.


# === HELPER FUNCTIONS ===

# User-extendable display-name map: keyword (lowercase) -> friendly label.
# e.g. _NAME_MAP = {'model_a': 'My Model', 'baseline': 'Baseline'}
_NAME_MAP: dict = {}


def get_pretty_name(name: str) -> str:
    """Return a display-friendly model name for a filename.

    Looks up each key in :data:`_NAME_MAP` as a case-insensitive substring.
    Falls back to *name* unchanged if no match is found.
    """
    n = name.lower()
    for keyword, display in _NAME_MAP.items():
        if keyword.lower() in n:
            return display
    return name

def load_h5_dataset(file_path, key_hint='disparity', index=0):
    """Loads a dataset from an H5 file with robust error handling."""
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return None
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            if not keys: return None
            key = next((k for k in keys if key_hint.lower() in k.lower()), keys[0])
            arr = np.array(f[key])
        if arr.ndim == 4: arr = arr[min(index, arr.shape[0] - 1)]
        elif arr.ndim == 3 and arr.shape[0] < 10: arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 3 and arr.shape[-1] != 3: arr = arr[min(index, arr.shape[0] - 1)]
        arr = np.nan_to_num(arr.astype(np.float32))
        print(f"✅ Loaded {os.path.basename(file_path)} | shape: {arr.shape}")
        return arr
    except Exception as e:
        print(f"⚠️ Error loading {file_path}: {e}")
        return None

def plot_rgb(ax, img, title):
    if img is not None:
        img_display = (img - img.min()) / (img.max() - img.min() + 1e-9)
        if img_display.ndim == 3 and img_display.shape[0] in [3, 4]:
            img_display = np.transpose(img_display, (1, 2, 0))
        img_display = img_display[..., ::-1] # RGB to BGR
        ax.imshow(img_display[crop_rows:, :])
        ax.set_title(title, pad=1, loc='center')
    ax.axis('off')

def plot_depth(ax, data, vmin, vmax, cmap):
    depth_map = data['map']
    # scale = 0.2
    # h, w = depth_map.shape[:2]
    # depth_small = cv2.resize(depth_map, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    depth_small = depth_map[crop_rows:, :]
    im = ax.imshow(depth_small, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    
    d_min, d_max = np.min(depth_small), np.max(depth_small)
    
    # NEW: Combine title and stats into a single line
    d_5, d_95 = np.percentile(depth_small, [5, 95])
    title_prefix = "m:" if any(c.isalpha() and c.islower() for c in data.get('is_mono', 'n')) else "s:"
    full_title = f"{title_prefix}{data['title']}  ({d_5:.2f} - {d_95:.2f}m)"
    ax.set_title(full_title, pad=1, loc='center')
    
    ax.set_xlabel("") # Remove bottom text
    ax.set_xticks([])
    ax.set_yticks([])
    return im

# === MAIN SCRIPT ===

def _process_entries(datalist, stereonames, mononames, frame_index):
    """Run the visualisation loop over a list of data entries."""
    for data_entry in datalist:
        left_h5_path = data_entry["left"]
        right_h5_path = data_entry["right"]
        stereodepth_folder = data_entry["stereodepth_path"]
        monodepth_folder = data_entry["monodepth_path"]
        outdir = data_entry["outdir"]

        os.makedirs(outdir, exist_ok=True)

        print(f"\n--- Processing Entry ---")
        print(f"Left: {left_h5_path}")
        print(f"Right: {right_h5_path}")
        print(f"Stereo Depth Folder: {stereodepth_folder}")
        print(f"Mono Depth Folder: {monodepth_folder}")
        print(f"Output Directory: {outdir}")

        left_rgb_img = load_h5_dataset(left_h5_path, key_hint='rectified', index=frame_index)
        mono_order   = [get_pretty_name(n) for n in mononames] if mononames else []
        stereo_order = [get_pretty_name(n) for n in stereonames] if stereonames else []

        unordered_mono_depths = []
        if os.path.exists(monodepth_folder):
            mono_files = [f for f in os.listdir(monodepth_folder) if f.endswith('.h5')]
            for mono_file in mono_files:
                if not mononames or any(name in mono_file.lower() for name in mononames):
                    depth = load_h5_dataset(os.path.join(monodepth_folder, mono_file),
                                            key_hint='depth', index=frame_index)
                    if depth is not None:
                        unordered_mono_depths.append(
                            {'map': np.squeeze(depth), 'title': get_pretty_name(mono_file)}
                        )

        right_rgb_img = load_h5_dataset(right_h5_path, key_hint='rectified', index=frame_index)

        unordered_stereo_depths = []
        if os.path.exists(stereodepth_folder):
            stereo_files = [f for f in os.listdir(stereodepth_folder) if f.endswith('.h5')]
            for stereo_file in stereo_files:
                if not stereonames or any(name in stereo_file.lower() for name in stereonames):
                    depth = load_h5_dataset(os.path.join(stereodepth_folder, stereo_file),
                                            key_hint='depth', index=frame_index)
                    if depth is not None:
                        if depth.ndim == 3 and depth.shape[-1] > 1:
                            depth = depth[..., 2]
                        unordered_stereo_depths.append(
                            {'map': np.squeeze(depth), 'title': get_pretty_name(stereo_file)}
                        )

        if not unordered_mono_depths and not unordered_stereo_depths:
            print("No depth maps could be loaded. Skipping this entry.")
            continue

        mono_depths = []
        for name in mono_order:
            for depth in unordered_mono_depths:
                if name.lower().replace(" ", "") in depth['title'].lower().replace(" ", ""):
                    mono_depths.append(depth)
                    break
        if not mono_depths:
            mono_depths = unordered_mono_depths

        stereo_depths = []
        for name in stereo_order:
            for depth in unordered_stereo_depths:
                if name.lower().replace(" ", "") in depth['title'].lower().replace(" ", ""):
                    stereo_depths.append(depth)
                    break
        if not stereo_depths:
            stereo_depths = unordered_stereo_depths

        mono_depths_flat   = np.concatenate([d['map'].flatten() for d in mono_depths])
        stereo_depths_flat = np.concatenate([d['map'].flatten() for d in stereo_depths])
        vmin_m, vmax_m = np.percentile(mono_depths_flat[mono_depths_flat > 0],   [5, 87])
        vmin_s, vmax_s = np.percentile(stereo_depths_flat[stereo_depths_flat > 0],[5, 87])

        combined_depths = mono_depths + stereo_depths
        all_depths_flat = np.concatenate([d['map'].flatten() for d in combined_depths])
        vmin, vmax = np.percentile(all_depths_flat[all_depths_flat > 0], [5, 95])
        vmin = max(vmin, 1e-3)
        print(f"Global 5-97 percentile depth range (log scale): [{vmin:.2f}, {vmax:.2f}]")

        original_map = plt.cm.turbo
        max_red = 0.82
        cmap1 = mcolors.LinearSegmentedColormap.from_list(
            'cmap1', original_map(np.linspace(0, max_red, 320))
        )
        cmap2 = mcolors.LinearSegmentedColormap.from_list(
            'cmap2', original_map(np.linspace(0, max_red, 320))
        )

        rows, cols = 2, 5
        fig, axes = plt.subplots(rows, cols, figsize=(7.6, 2.2), squeeze=False)

        plot_rgb(axes[0, 0], left_rgb_img, "Left Image (ref)")
        last_top_im = None
        for i, data in enumerate(mono_depths):
            last_top_im = plot_depth(axes[0, i + 1], data, vmin_m, vmax_m, cmap1)

        plot_rgb(axes[1, 0], right_rgb_img, "Right Image")
        last_bottom_im = None
        for i, data in enumerate(stereo_depths):
            last_bottom_im = plot_depth(axes[1, i + 1], data, vmin_s, vmax_s, cmap2)

        for i in range(len(mono_depths) + 1, cols):   axes[0, i].axis('off')
        for i in range(len(stereo_depths) + 1, cols): axes[1, i].axis('off')

        if last_top_im:
            fig.subplots_adjust(left=0.005, right=0.96, top=0.999, bottom=0.001,
                                hspace=0.003, wspace=0.02)
            cbar_ax = fig.add_axes([0.963, 0.53, 0.007, 0.43])
            cbar_ax.yaxis.set_major_formatter(mticker.NullFormatter())
            cbar = fig.colorbar(last_top_im, cax=cbar_ax)
            new_ticks = np.round(np.linspace(vmin_m, vmax_m, 5), 2)
            cbar.set_ticks(new_ticks)
            cbar.ax.minorticks_off()
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            cbar.ax.tick_params(axis='y', direction='out', length=2, pad=0.1, labelsize=5)
            cbar_ax.set_title("Depth (m)", pad=1.5)

        if last_bottom_im:
            fig.subplots_adjust(left=0.005, right=0.96, top=0.99, bottom=0.01,
                                hspace=0.003, wspace=0.02)
            cbar_ax = fig.add_axes([0.963, 0.04, 0.007, 0.43])
            cbar_ax.yaxis.set_major_formatter(mticker.NullFormatter())
            cbar = fig.colorbar(last_bottom_im, cax=cbar_ax)
            new_ticks = np.round(np.linspace(vmin_s, vmax_s, 5), 2)
            cbar.set_ticks(new_ticks)
            cbar.ax.minorticks_off()
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            cbar.ax.tick_params(axis='y', direction='out', length=2, pad=0.1, labelsize=5)
        else:
            plt.tight_layout()

        base_filename = (
            f"{os.path.basename(stereodepth_folder.rstrip('/\\'))}"
            f"_visualization_frame_{frame_index}"
        )
        pdf_path = os.path.join(outdir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved PDF: {pdf_path}")
        plt.close(fig)
        print(f"Completed processing for this entry.\n")


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(
        description="2-row depth-map figure for illusion crops: mono (top) vs stereo (bottom).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config JSON format
------------------
[
  {
    "left":             "/path/to/crop/_left.h5",
    "right":            "/path/to/crop/_right.h5",
    "stereodepth_path": "/path/to/crop",
    "monodepth_path":   "/path/to/crop/monodepth",
    "outdir":           "/path/to/output"
  }
]

Example
-------
    python visuals/mono_stereo_depths/illusion_crop_visualization.py \\
        --config /path/to/entries.json \\
        --stereo_names model_c model_d \\
        --mono_names model_a model_b \\
        --frame 0
""",
    )
    parser.add_argument('--config', required=True,
                        help='JSON file listing data entries (see format above).')
    parser.add_argument('--stereo_names', nargs='+', default=[],
                        help='Substring keywords identifying stereo depth HDF5 files.')
    parser.add_argument('--mono_names', nargs='+', default=[],
                        help='Substring keywords identifying monocular depth HDF5 files.')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to visualise (default: 0).')
    args = parser.parse_args()

    with open(args.config) as fh:
        datalist = json.load(fh)

    _process_entries(datalist, args.stereo_names, args.mono_names, args.frame)


if __name__ == "__main__":
    main()
