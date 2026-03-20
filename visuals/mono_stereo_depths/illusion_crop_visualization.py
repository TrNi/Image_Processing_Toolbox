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
# --- Provide a list of dictionaries, each containing paths for a scene ---
a = "I:\\My Drive\\Pubdata\\illusion_crops_new\\S9_f40_img_3901"
b = "I:\\My Drive\\Pubdata\\illusion_crops_new\\S9_f45_img_3962"
c = "I:\\My Drive\\Pubdata\\illusion_crops_new\\S9_f45_img_4023"
d = "I:\\My Drive\\Pubdata\\illusion_crops_new\\S9_f60_img_4378"
e = "I:\\My Drive\\Pubdata\\illusion_crops_new\\S9_f60_img_4380"
f = "I:\\My Drive\\Pubdata\\illusion_crops_new\\1"
g = "I:\\My Drive\\Pubdata\\illusion_crops_new\\5"
h = "I:\\My Drive\\Pubdata\\illusion_crops_new\\6"
i = "I:\\My Drive\\Pubdata\\illusion_crops_new\\7"
j = "I:\\My Drive\\Pubdata\\illusion_crops_new\\8"
k = "I:\\My Drive\\Pubdata\\illusion_crops_new\\9"
l = "I:\\My Drive\\Pubdata\\Scene6_illusions\\illusions_used_S6_3"
m = "I:\\My Drive\\Pubdata\\Scene6_illusions\\illusions_used_S6_16"
n = "I:\\My Drive\\Pubdata\\Scene6_illusions\\illusions_used_S6_15"
o = "I:\\My Drive\\Pubdata\\Scene6_illusions\\illusions_used_S6_8"
p = "I:\\My Drive\\Pubdata\\Scene6_illusions\\illusions_used_S6_10"
all_outdir = "H:\\My Drive\\Research_collabs\\N-V-D Research Collab\\CVPR_visuals\\illusion_depth_maps"
datalist = [
    # {
    #     "outdir": all_outdir,
    #     "left": a + "\\_left.h5",
    #     "right": a + "\\_right.h5",
    #     "stereodepth_path": a,
    #     "monodepth_path": a + "\\monodepth",        
    # },    
    # { 
    #     "outdir": all_outdir,
    #     "left": h + "\\_left.h5",
    #     "right": h + "\\_right.h5",
    #     "stereodepth_path": h,
    #     "monodepth_path": h + "\\monodepth",
    # },
    # { 
    #     "outdir": all_outdir,
    #     "left": j + "\\_left.h5",
    #     "right": j + "\\_right.h5",
    #     "stereodepth_path": j,
    #     "monodepth_path": j + "\\monodepth",
    # },  
    { 
        "outdir": all_outdir,
        "left": m + "\\_left.h5",
        "right": m + "\\_right.h5",
        "stereodepth_path": m,
        "monodepth_path": m + "\\monodepth",
    },   
    { 
        "outdir": all_outdir,
        "left": p + "\\_left.h5",
        "right": p + "\\_right.h5",
        "stereodepth_path": p,
        "monodepth_path": p + "\\monodepth",
    },    
]


# --- Define the model name patterns to identify files ---
stereonames = ["monster", "foundation", "defom", "selective"]
mononames = ["pro", "metric3d", "unidepth", "dav2"]
frame_index = 0

# === HELPER FUNCTIONS ===
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
    if 'depth_anything' in name or 'dav2' in name: return 'DAV2'
    return name # Return original name if no match

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
    title_prefix = "m:" if sum([1 if namek in data['title'].lower() else 0 for namek in mononames]) > 0 else "s:"
    full_title = f"{title_prefix}{data['title']}  ({d_5:.2f} - {d_95:.2f}m)"
    ax.set_title(full_title, pad=1, loc='center')
    
    ax.set_xlabel("") # Remove bottom text
    ax.set_xticks([])
    ax.set_yticks([])
    return im

# === MAIN SCRIPT ===

# Process each entry in the datalist
for data_entry in datalist:
    left_h5_path = data_entry["left"]
    right_h5_path = data_entry["right"]
    stereodepth_folder = data_entry["stereodepth_path"]
    monodepth_folder = data_entry["monodepth_path"]
    outdir = data_entry["outdir"]
    
    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    
    # --- 1. Load all data first ---
    print(f"\n--- Processing Entry ---")
    print(f"Left: {left_h5_path}")
    print(f"Right: {right_h5_path}")
    print(f"Stereo Depth Folder: {stereodepth_folder}")
    print(f"Mono Depth Folder: {monodepth_folder}")
    print(f"Output Directory: {outdir}")
    
    # a) Load Left RGB and Monocular Depths (Top Row)
    left_rgb_img = load_h5_dataset(left_h5_path, key_hint='rectified', index=frame_index)
    mono_order = ["DepthPro","Metric3D","UniDepth","DAV2"]
    stereo_order = ["Monster","Foundation","Defom","Selective"]

    # Find and load monocular depth files
    unordered_mono_depths = []
    if os.path.exists(monodepth_folder):
        mono_files = [f for f in os.listdir(monodepth_folder) if f.endswith('.h5')]
        for mono_file in mono_files:
            # Check if any of the mononames are in the filename
            if any(name in mono_file.lower() for name in mononames):
                depth = load_h5_dataset(os.path.join(monodepth_folder, mono_file), key_hint='depth', index=frame_index)
                if depth is not None:
                    title = get_pretty_name(mono_file)
                    unordered_mono_depths.append({'map': np.squeeze(depth), 'title': title})
    
    # b) Load Right RGB and Stereo Depths (Bottom Row)
    right_rgb_img = load_h5_dataset(right_h5_path, key_hint='rectified', index=frame_index)
    
    # Find and load stereo depth files
    unordered_stereo_depths = []
    if os.path.exists(stereodepth_folder):
        stereo_files = [f for f in os.listdir(stereodepth_folder) if f.endswith('.h5')]
        for stereo_file in stereo_files:
            # Check if any of the stereonames are in the filename
            if any(name in stereo_file.lower() for name in stereonames):
                depth = load_h5_dataset(os.path.join(stereodepth_folder, stereo_file), key_hint='depth', index=frame_index)
                if depth is not None:
                    if depth.ndim == 3 and depth.shape[-1] > 1:
                        depth = depth[..., 2] # Select 3rd channel for depth
                    title = get_pretty_name(stereo_file)
                    unordered_stereo_depths.append({'map': np.squeeze(depth), 'title': title})    

    # --- 2. Prepare figure and normalization ---
    if not unordered_mono_depths and not unordered_stereo_depths:
        print("❌ No depth maps could be loaded. Skipping this entry.")
        continue
    
    mono_depths = []
    for name in mono_order:
        for depth in unordered_mono_depths:
            print(name.lower(), depth['title'].lower())
            if name.lower().replace(" ", "") in depth['title'].lower().replace(" ", ""):
                mono_depths.append(depth)
                break

    stereo_depths = []
    for name in stereo_order:
        for depth in unordered_stereo_depths:
            print(name.lower(), depth['title'].lower())
            if name.lower().replace(" ", "") in depth['title'].lower().replace(" ", ""):
                stereo_depths.append(depth)
                break

    mono_depths_flat = np.concatenate([d['map'].flatten() for d in mono_depths])
    stereo_depths_flat = np.concatenate([d['map'].flatten() for d in stereo_depths])
    vmin_m, vmax_m = np.percentile(mono_depths_flat[mono_depths_flat > 0], [5, 87])
    vmin_s, vmax_s = np.percentile(stereo_depths_flat[stereo_depths_flat > 0], [5, 87])

    combined_depths = mono_depths + stereo_depths
    all_depths_flat = np.concatenate([d['map'].flatten() for d in combined_depths])
    # Ensure vmin is positive for logarithmic scale
    vmin, vmax = np.percentile(all_depths_flat[all_depths_flat > 0], [5, 95])
    vmin = max(vmin, 1e-3)  # Ensure vmin is at least 0.001 to avoid log(0)
    print(f"\n🎨 Global 5-97 percentile depth range (log scale): [{vmin:.2f}, {vmax:.2f}]")
    # Create truncated turbo colormap (remove 10% from both ends)
    original_map = plt.cm.turbo
    max_red = 0.82
    cmap1 = mcolors.LinearSegmentedColormap.from_list(
        'cmap1', original_map(np.linspace(0, max_red, 320))
    )
    # Create truncated turbo colormap (remove 10% from both ends)    
    cmap2 = mcolors.LinearSegmentedColormap.from_list(
        'cmap2', original_map(np.linspace(0, max_red, 320))
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'cmap', original_map(np.linspace(0, max_red, 320))
    )
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(7.6, 2.2), squeeze=False)
    
    # --- 3. Plot everything ---   
    
    # Plot Row 1 & 2
    plot_rgb(axes[0, 0], left_rgb_img, "Left Image (ref)")
    last_top_im = None
    for i, data in enumerate(mono_depths):
        last_top_im = plot_depth(axes[0, i + 1], data, vmin_m, vmax_m, cmap1)        
    
    plot_rgb(axes[1, 0], right_rgb_img, "Right Image")
    last_bottom_im = None
    for i, data in enumerate(stereo_depths):
        last_bottom_im = plot_depth(axes[1, i + 1], data, vmin_s, vmax_s, cmap2)
    
    # Clean up unused axes
    for i in range(len(mono_depths) + 1, cols): axes[0, i].axis('off')
    for i in range(len(stereo_depths) + 1, cols): axes[1, i].axis('off')
    
    # Add a single, shared color bar
    if last_top_im:
        # Adjusted hspace for a tighter layout
        fig.subplots_adjust(left=0.005, right=0.96, top=0.999, bottom=0.001, hspace=0.003, wspace=0.02)
        cbar_ax = fig.add_axes([0.963, 0.53, 0.007, 0.43])
        cbar_ax.yaxis.set_major_formatter(mticker.NullFormatter())
        # cbar_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        cbar = fig.colorbar(last_top_im, cax=cbar_ax) #, format='%.1f')          
        # Overwrite default log ticks with evenly-spaced linear ticks and remove minor log ticks
        new_ticks = np.round(np.linspace(vmin_m, vmax_m, 5), 2)
        cbar.set_ticks(new_ticks)               # set new major ticks
        cbar.ax.minorticks_off()                # disable residual minor log ticks
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # cbar.ax.tick_params(pad=0.1)
        cbar.ax.tick_params(axis='y',   # 'y' for vertical colorbar, 'x' for horizontal
                    direction='out',    # ticks pointing outwards
                    length=2,           # tick stem length in points (shorter)
                    pad=0.1,            # distance from tick to tick label in points
                    labelsize=5)
        cbar_ax.set_title("Depth (m)", pad=1.5)

    if last_bottom_im:
        # Adjusted hspace for a tighter layout
        fig.subplots_adjust(left=0.005, right=0.96, top=0.99, bottom=0.01, hspace=0.003, wspace=0.02)
        cbar_ax = fig.add_axes([0.963, 0.04, 0.007, 0.43])
        cbar_ax.yaxis.set_major_formatter(mticker.NullFormatter())
        # cbar_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        cbar = fig.colorbar(last_bottom_im, cax=cbar_ax) #, format='%.1f')          
        # Overwrite default log ticks with evenly-spaced linear ticks and remove minor log ticks
        new_ticks = np.round(np.linspace(vmin_s, vmax_s, 5), 2)
        cbar.set_ticks(new_ticks)               # set new major ticks
        cbar.ax.minorticks_off()                # disable residual minor log ticks
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # cbar.ax.tick_params(pad=0.1)
        cbar.ax.tick_params(axis='y',   # 'y' for vertical colorbar, 'x' for horizontal
                    direction='out',    # ticks pointing outwards
                    length=2,           # tick stem length in points (shorter)
                    pad=0.1,            # distance from tick to tick label in points
                    labelsize=5)
        
        # fmt = mticker.ScalarFormatter(useMathText=False)
        # fmt.set_powerlimits((0, 0))  # disables scientific notation
        # cbar.ax.yaxis.set_major_formatter(fmt)
        # cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

        # Place ticks at powers of 10 (optional) or linear-ish for your range
          # 1 decimal
        # cbar_ax.set_title("Depth (m)", pad=4)
    else:
        plt.tight_layout()
    
    # --- 4. Save the figure in multiple formats ---
    base_filename = f"{stereodepth_folder.split('\\')[-1]}_visualization_frame_{frame_index}"
    
    # # Save as PDF
    pdf_path = os.path.join(outdir, f"{base_filename}.pdf")    
    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"✅ Saved PDF: {pdf_path}")

    # Save as PNG
    # png_path = os.path.join(outdir, f"{base_filename}.png")
    # plt.savefig(png_path, dpi=900, bbox_inches='tight')
    # print(f"✅ Saved PNG: {png_path}")
    
    # # # Save as SVG
    # svg_path = os.path.join(outdir, f"{base_filename}.svg")
    # plt.savefig(svg_path, format='svg', bbox_inches='tight')
    # print(f"✅ Saved SVG: {svg_path}")
    
    plt.close(fig)
    print(f"✅ Completed processing for this entry.\n")
