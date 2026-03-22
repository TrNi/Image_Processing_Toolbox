# If requested frame index exceeds available frames in H5,
# the code automatically loads the last valid frame instead,
# ensuring no index errors or crashes occur.
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # for math equations to also use serif fonts
# Set Matplotlib to display plots inline in a notebook
# %matplotlib inline

# === USER INPUT ===
# All paths and parameters are supplied via CLI -- see main() below.

# === HELPER FUNCTIONS ===

# User-extendable display-name map: keyword (lowercase) -> friendly label.
# Populate before calling get_pretty_name(), e.g.:
#   _NAME_MAP = {'model_a': 'My Model A', 'baseline': 'Baseline'}
_NAME_MAP: dict = {}


def get_pretty_name(name: str) -> str:
    """Return a display-friendly model name for a filename.

    Looks up each key in :data:`_NAME_MAP` as a case-insensitive substring.
    Falls back to the original name if no match is found.
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

def plot_rgb(ax, img, title, fontsize=10):
    if img is not None:
        img_display = (img - img.min()) / (img.max() - img.min() + 1e-9)
        if img_display.ndim == 3 and img_display.shape[0] in [3, 4]:
            img_display = np.transpose(img_display, (1, 2, 0))
        img_display = img_display[..., ::-1] # RGB to BGR
        ax.imshow(img_display)
        ax.set_title(title, fontsize=fontsize, fontname='Times New Roman', pad=4, loc='center')
    ax.axis('off')

def plot_depth(ax, data, vmin, vmax):
    depth_map = data['map']
    scale = 0.2
    h, w = depth_map.shape[:2]
    depth_small = cv2.resize(depth_map, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

    im = ax.imshow(depth_small, cmap='turbo', vmin=vmin, vmax=vmax)
    
    d_min, d_max = np.min(depth_small), np.max(depth_small)
    
    # NEW: Combine title and stats into a single line
    full_title = f"{data['title']}  ({d_min:.2f} - {d_max:.2f}m)"
    ax.set_title(full_title, fontsize=11, fontname='Times New Roman', pad=4, loc='center')
    
    ax.set_xlabel("") # Remove bottom text
    ax.set_xticks([])
    ax.set_yticks([])
    return im

# === MAIN SCRIPT ===

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="2-row depth-map figure: mono models (top) vs stereo models (bottom).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
    python visuals/mono_stereo_depths/depth_map_visualization.py \\
        --left_scene /path/to/scene/rectified/ \\
        --right_h5  /path/to/scene/rectified/rectified_rights.h5 \\
        --mono_files mono_a.h5 mono_b.h5 \\
        --frame 0
""",
    )
    parser.add_argument('--left_scene',  required=True,
                        help='Path to the left-camera rectified folder '
                             '(must contain rectified_lefts.h5 and stereo depth *.h5 files).')
    parser.add_argument('--right_h5',    required=True,
                        help='Path to rectified_rights.h5.')
    parser.add_argument('--mono_files',  nargs='+', default=[],
                        help='Relative paths (from --left_scene/../monodepth/) '
                             'to monocular depth HDF5 files.')
    parser.add_argument('--frame',       type=int, default=0,
                        help='Frame index to visualise (default: 0).')
    args = parser.parse_args()

    left_scene_path = args.left_scene
    right_rgb_path  = args.right_h5
    mono_model_files = args.mono_files
    frame_index      = args.frame

    # --- 1. Load all data first ---
    print("--- Loading Data ---")
    left_rgb_img = load_h5_dataset(
        os.path.join(left_scene_path, "rectified_lefts.h5"),
        key_hint='rectified', index=frame_index,
    )
    mono_folder_path = os.path.join(
        os.path.dirname(left_scene_path.rstrip('/\\')), "monodepth"
    )
    mono_depths = []
    for model_file in mono_model_files:
        depth = load_h5_dataset(
            os.path.join(mono_folder_path, model_file),
            key_hint='depth', index=frame_index,
        )
        if depth is not None:
            mono_depths.append({'map': np.squeeze(depth),
                                 'title': get_pretty_name(model_file)})

    right_rgb_img = load_h5_dataset(right_rgb_path, key_hint='rectified', index=frame_index)
    stereo_files  = sorted([
        f for f in os.listdir(left_scene_path)
        if f.endswith(".h5")
    ])
    stereo_depths = []
    for stereo_file in stereo_files:
        depth = load_h5_dataset(
            os.path.join(left_scene_path, stereo_file),
            key_hint='depth', index=frame_index,
        )
        if depth is not None:
            if depth.ndim == 3 and depth.shape[-1] > 1:
                depth = depth[..., 2]
            stereo_depths.append({'map': np.squeeze(depth),
                                   'title': get_pretty_name(stereo_file)})

    # --- 2. Prepare figure and normalization ---
    if not mono_depths and not stereo_depths:
        print("No depth maps could be loaded. Exiting.")
        return

    combined_depths  = mono_depths + stereo_depths
    all_depths_flat  = np.concatenate([d['map'].flatten() for d in combined_depths])
    vmin, vmax       = np.percentile(all_depths_flat, [0, 95])
    print(f"Global 5-95 percentile depth range: [{vmin:.2f}, {vmax:.2f}]")

    rows, cols = 2, 5
    fig, axes  = plt.subplots(rows, cols, figsize=(19, 6), squeeze=False)

    # --- 3. Plot ---
    plot_rgb(axes[0, 0], left_rgb_img, "Left Image (ref)")
    for i, data in enumerate(mono_depths):
        plot_depth(axes[0, i + 1], data, vmin, vmax)

    plot_rgb(axes[1, 0], right_rgb_img, "Right Image")
    last_im = None
    for i, data in enumerate(stereo_depths):
        last_im = plot_depth(axes[1, i + 1], data, vmin, vmax)

    for i in range(len(mono_depths) + 1, cols):   axes[0, i].axis('off')
    for i in range(len(stereo_depths) + 1, cols): axes[1, i].axis('off')

    if last_im:
        fig.subplots_adjust(left=0.005, right=0.96, top=0.99, bottom=0.01,
                            hspace=-0.4, wspace=0.02)
        cbar_ax = fig.add_axes([0.973, 0.155, 0.005, 0.69])
        fig.colorbar(last_im, cax=cbar_ax)
        cbar_ax.set_title("Depth (m)", fontsize=10, fontname='Times New Roman', pad=4)
    else:
        plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
