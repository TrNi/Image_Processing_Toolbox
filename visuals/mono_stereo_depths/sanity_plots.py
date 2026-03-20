#!/usr/bin/env python3
"""
make_depth_figures.py

Requirements:
    pip install numpy h5py matplotlib tqdm
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm, colors, colormaps
from matplotlib.gridspec import GridSpec
import cv2
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------- USER DATASOURCE (as provided) ----------
datalist = [  
   {
       "base": r"I:\\My Drive\\Pubdata\\Scene6_illusions",
       "cameras": ['EOS6D_A_Left', 'EOS6D_B_Right'],
       "configs":[
           #{"fl":28, "F":22}, 
           {"fl":70, "F":16}, 
        ]
   },
#   {
#        "base": r"I:\\My Drive\\Pubdata\\Scene7",
#        "cameras": ['EOS6D_B_Left', 'EOS6D_A_Right'],
#        "configs":[
#            {"fl":28, "F":22}, 
#            {"fl":70, "F":2.8}, 
#         ]
#    },
#    {
#        "base": r"I:\\My Drive\\Pubdata\\Scene8",
#        "cameras": ['EOS6D_A_Left', 'EOS6D_B_Right'],
#        "configs":[
#            {"fl":28, "F":22}, 
#            {"fl":70, "F":2.8}, 
#         ]
#    },   
#    {
#        "base": r"I:\\My Drive\\Pubdata\\Scene9",
#        "cameras": ['EOS6D_B_Left', 'EOS6D_A_Right'],
#        "configs":[
#            {"fl":28, "F":22}, 
#            {"fl":70, "F":2.8}, 
#         ]
#    },
]

# ---------- PARAMETERS ----------
ROWS_PER_FIG = 1         # "Each figure has plots corresponding to the next 4 images"
FIGSIZE = (8, 5)         # typical figure size requested
DPI = 150
TEXT_COL_WIDTH = 0.6      # approximate width for the left text column (for layout heuristics)
DEPTH_KEY = "depth"
RECT_LEFT_KEY = "rectified_lefts"
RECT_RIGHT_KEY = "rectified_rights"

# stereo model keywords to locate depth files case-insensitively
MONO_MODELS = ['depthpro', 'metric3d', 'unidepth', 'depth_anything']
STEREO_MODELS = ['monster', 'foundation', 'defom', 'selective']
MODEL_KEYWORDS = MONO_MODELS + STEREO_MODELS
# mapping of found filename -> friendly title (derived from keyword or filename)
def friendly_title_from_name(fn):
    name = fn.lower()
    if 'monster' in name:
        return 'MonSter'
    if 'foundation' in name:
        return 'FoundationStereo'
    if 'defom' in name:
        return 'Defom'
    if 'selective' in name:
        return 'Selective-IGEV'
    if 'depthpro' in name:
        return 'Depth Pro'
    if 'metric3d' in name:
        return 'Metric3D'
    if 'unidepth' in name:
        return 'UniDepth V2'
    if 'depth_anything' in name:
        return 'DAV2'    
    # fallback: use file stem
    return Path(fn).stem

# ---------- helpers ----------
def find_h5_by_keywords(folder: Path, keywords):
    """Return dict keyword->Path of best-matching file in folder (case-insensitive)."""
    found = {}
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in ('.h5', '.hdf5')]
    lower_names = {f: f.name.lower() for f in files}
    for kw in keywords:
        candidate = None
        for f, lname in lower_names.items():
            if kw.lower() in lname:
                # prefer exact 'leftview' naming if present (not required, but helpful)
                candidate = f
                break
        if candidate is not None:
            found[kw] = candidate
    return found

def load_h5_dataset(h5path: Path, key: str):
    """Load dataset key from h5 file. Return numpy array or raise."""
    with h5py.File(h5path, 'r') as fh:
        if key not in fh:
            raise KeyError(f"Key '{key}' not found in {h5path}")
        data = fh[key][()]
    return data

def clamp_and_mask_depths(depth_arrays):
    """Concatenate valid finite>0 values across depth arrays and compute 5/95 percentiles."""
    vals = []
    for d in depth_arrays:
        if d is None:
            continue
        # Valid mask: finite and > 0
        m = np.isfinite(d) # & (d > 0)
        if np.any(m):
            vals.append(d[m].ravel())
    if len(vals) == 0:
        return None, None
    concat = np.concatenate(vals)
    p5 = float(np.nanpercentile(concat, 5.0))
    p95 = float(np.nanpercentile(concat, 95.0))
    if p5 == p95:
        # degenerate case: expand a bit
        p95 = p5 + 1e-6
    return p5, p95

def ensure_rgb_image(arr):
    """
    Input arr shape HxW or CxHxW or HxWxC (common variations). Return HxWx3 float image in [0,1].
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        img = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3:
        # guess whether channel-first
        if a.shape[0] in [1,3]:            
            img = np.moveaxis(a, 0, -1) # channel-first likely: C x H x W
        elif a.shape[2] in [1,3]:
            img = a # channel-last
        else:            
            img = a # ambiguous: treat as H x W x C

        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        elif img.shape[2] > 3:
            raise Exception(f"Unsupported image shape {a.shape}")
    else:
        raise Exception(f"Unsupported image shape {a.shape}")
    # normalize to [0,1] if needed
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        # if values outside [0,1], rescale by max
        # if img.max() > 1.0:
        #     img = img.astype(np.float32)
        #     img = img / max(1.0, img.max())
        # else:
        #     img = img.astype(np.float32)
        img = img.astype(np.float32)
        if img.max() > 255:
            img /= img.max()
        elif img.max() > 1.0:
            img /= 255.0
        else:
            img = img.astype(np.float32)
    
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    return img

# ---------- main processing ----------
def process_datalist(datalist):
    for entry in datalist:
        base = Path(entry['base'])
        left_cam = entry['cameras'][0]
        right_cam = entry['cameras'][1]

        for cfg in entry['configs']:
            # create path: e.g. base\\{left_cam}\\fl_{fl}mm\\inference\\F{F:.1f}\\rectified
            # try:
            
            fl = cfg['fl']
            F = float(cfg['F'])
            fl_folder = f"fl_{int(fl)}mm"
            F_folder = f"F{F:.1f}"
            # rectified dir paths
            left_rectified_dir = base / left_cam / fl_folder / "inference" / F_folder / "rectified"
            right_rectified_dir = base / right_cam / fl_folder / "inference" / F_folder / "rectified"

            logging.info(f"Processing config: base={base.name} fl={fl} F={F:.1f} left={left_cam} right={right_cam}")

            # rectified.h5 paths
            left_rect_h5 = left_rectified_dir / "rectified_lefts.h5"
            right_rect_h5 = right_rectified_dir / "rectified_rights.h5"

            if not left_rect_h5.exists():
                raise FileNotFoundError(f"{left_rect_h5} not found")
            if not right_rect_h5.exists():
                raise FileNotFoundError(f"{right_rect_h5} not found")

            # load rectified arrays
            left_rects = load_h5_dataset(left_rect_h5, RECT_LEFT_KEY)   # NxCxHxW
            right_rects = load_h5_dataset(right_rect_h5, RECT_RIGHT_KEY) # NxCxHxW

            # canonical shapes
            left_rects = np.asarray(left_rects)
            right_rects = np.asarray(right_rects)
            if left_rects.shape[0] != right_rects.shape[0]:
                logging.warning("Left and right have different N; using min(N)")
            N = min(left_rects.shape[0], right_rects.shape[0])
            left_rects = left_rects[:N]
            right_rects = right_rects[:N]

            # find depth files (search in left rectified dir)
            mono_found = find_h5_by_keywords(left_rectified_dir.parent / "monodepth", MONO_MODELS)
            stereo_found = find_h5_by_keywords(left_rectified_dir, STEREO_MODELS)
            # attempt to map keywords in desired order
            mono_depth_paths = []
            stereo_depth_paths = []
            mono_depth_titles = []
            stereo_depth_titles = []
            for kw in MONO_MODELS:
                p = mono_found.get(kw)
                if p is not None:
                    mono_depth_paths.append(p)
                    mono_depth_titles.append(friendly_title_from_name(p.name))
                else:
                    mono_depth_paths.append(None)
                    mono_depth_titles.append(kw)  # placeholder
            for kw in STEREO_MODELS:
                p = stereo_found.get(kw)
                if p is not None:
                    stereo_depth_paths.append(p)
                    stereo_depth_titles.append(friendly_title_from_name(p.name))
                else:
                    stereo_depth_paths.append(None)
                    stereo_depth_titles.append(kw)  # placeholder
            
            depth_titles = mono_depth_titles + stereo_depth_titles
            # load depth arrays for found; otherwise None
            depth_arrays = []
            for p in mono_depth_paths+stereo_depth_paths:
                if p is None:
                    depth_arrays.append(None)
                    continue
                try:
                    d = load_h5_dataset(p, DEPTH_KEY)  # expected NxHxW
                    d = np.asarray(d).astype(np.float32)# .transpose(1,2,0)
                    if d.shape[0] != N:
                        # truncate or pad if mismatch
                        if d.shape[0] > N:
                            d = d[:N]
                        else:
                            # pad with nan
                            # pad = np.full((N - d.shape[0],) + d.shape[1:], np.nan, dtype=d.dtype)
                            # d = np.concatenate([d, pad], axis=0)
                            N = d.shape[0]
                    print(f"Loaded depth from {p}: {d.shape}")
                    depth_arrays.append(d)
                except Exception as e:
                    logging.warning(f"Failed to load depth from {p}: {e}")
                    depth_arrays.append(None)

            # compute global vmin/vmax using 5th/95th percentile across depth_arrays
            valid_depths = [d for d in depth_arrays if d is not None]
            if len(valid_depths) == 0:
                logging.warning(f"No depth maps found for config {cfg} at {left_rectified_dir}. Skipping.")
                continue
            vmin, vmax = clamp_and_mask_depths(valid_depths)
            if vmin is None:
                logging.warning("Depth arrays contain no finite positive values. Skipping.")
                continue
            logging.info(f"Depth vmin/vmax (5th/95th percentile): {vmin:.3f} / {vmax:.3f}")

            # compute per-model min/max for subtitle (across N, only finite >0)
            per_model_range = []
            for d in depth_arrays:
                if d is None:
                    per_model_range.append((np.nan, np.nan))
                    continue
                mm = np.isfinite(d) & (d > 0)
                if np.any(mm):
                    #mn = float(np.nanmin(d[mm]))
                    mn = np.nanpercentile(d[mm], 5)
                    #mx = float(np.nanmax(d[mm]))
                    mx = np.nanpercentile(d[mm], 95)
                    per_model_range.append((mn, mx))
                else:
                    per_model_range.append((np.nan, np.nan))

            # plotting loop: produce paged figures, each with up to ROWS_PER_FIG image-rows
            # cmap = cm.get_cmap('turbo')  # matplotlib 'turbo' colormap
            cmap = colormaps.get_cmap('turbo').with_extremes(under='w', over='w', bad='w')
            cmin, cmax = 0.05, 0.95
            cmap = colors.LinearSegmentedColormap.from_list('turbo_trimmed', 
                                            [cmap(cmin), *[cmap(x) for x in np.linspace(cmin, cmax, 256)], cmap(cmax)])
            total_pages = int(np.ceil(N / ROWS_PER_FIG))
            for page in range(total_pages):
                start = page * ROWS_PER_FIG
                end = min(N, start + ROWS_PER_FIG)
                rows = end - start
                n_depth_cols = max(len(MONO_MODELS), len(STEREO_MODELS))  # number of depth columns
                
                # Create figure
                fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
                
                # Create GridSpec with custom width ratios (1 for text, 100 for each image)
                gs = GridSpec(rows*2, 2 + n_depth_cols, figure=fig, 
                            width_ratios=[0.1] + [100] * (1 + n_depth_cols))
                
                # Create axes using GridSpec
                axes = np.empty((rows*2, 2 + n_depth_cols), dtype=object)
                for r in range(rows*2):
                    for c in range(2 + n_depth_cols):
                        axes[r, c] = fig.add_subplot(gs[r, c])                    
                
                col_titles = ["", "Rectified Left"] + mono_depth_titles + ["", "Rectified Right"] + stereo_depth_titles
                # set global suptitle
                fig.suptitle(f"{base.name}, fl={fl}mm, F={F:.1f}", fontsize=12)

                # iterate rows
                for r_idx, i in enumerate(range(start, end)):
                    # text column: show image id
                    ax_text = axes[2*r_idx, 0]
                    ax_text.axis('off')
                    ax_text.text( 0.05, 0.5, f"Image {i}", rotation=90,
                                    va='center', ha='right', fontsize=7, family='monospace')
                    
                    ax_text2 = axes[2*r_idx+1, 0]
                    ax_text2.axis('off')

                    # left image
                    ax_left = axes[2*r_idx, 1]
                    try:
                        imgL = ensure_rgb_image(left_rects[i])
                    except Exception as e:
                        logging.warning(f"Failed to prepare left image index {i}: {e}")
                        imgL = np.zeros((10, 10, 3), dtype=np.float32)
                    ax_left.imshow(imgL)
                    ax_left.axis('off')

                    # right image
                    ax_right = axes[2*r_idx+1, 1]
                    try:
                        imgR = ensure_rgb_image(right_rects[i])
                    except Exception as e:
                        logging.warning(f"Failed to prepare right image index {i}: {e}")
                        imgR = np.zeros((10, 10, 3), dtype=np.float32)
                    ax_right.imshow(imgR)
                    ax_right.axis('off')

                    # depth columns
                    for c_idx, d in enumerate(depth_arrays):
                        ax = axes[2*r_idx + int(c_idx>=n_depth_cols), 2 + c_idx%n_depth_cols]
                        if d is None:
                            ax.text(0.98, 0.5, "missing", ha='center', va='center', fontsize=6)
                            ax.axis('off')
                            continue
                        depth_img = d[i]
                        # display with vmin/vmax; mask invalid areas as transparent using NaNs and set extent
                        depth_plot = np.copy(depth_img).astype(np.float32)
                        # create masked array for display
                        masked = np.ma.array(depth_plot, mask=~(np.isfinite(depth_plot))) # & (depth_plot > 0)))
                        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
                        ax.axis('off')

                        # small subtitle with min-max for this model
                        mn, mx = per_model_range[c_idx]
                        #if np.isfinite(mn) and np.isfinite(mx):
                        subtitle = f"{mn:.2f} - {mx:.2f} m"
                        # else:
                        #     subtitle = "n/a"
                        # place as small title above the first row only (requested "small font subtitle showing min-max depth")
                        # We'll put it as title for the column on the first row; for subsequent rows keep it empty to avoid repetition
                        if r_idx == 0:
                            ax.set_title(f"{depth_titles[c_idx]}\n{subtitle}", fontsize=7)
                        else:
                            ax.set_title(subtitle, fontsize=7)
                        # attach im to axis for colorbar later (but only once per figure)
                        axes[r_idx, 2 + c_idx%n_depth_cols]._my_im = im

                # set top-row labels for Rectified Left/Right (above the first row)
                # We have already set titles for depth columns; ensure Rectified Left/Right columns have titles
                # Place titles above the top row axes
                if rows > 0:
                    axes[0,1].set_title("Rectified Left", fontsize=7)
                    axes[1,1].set_title("Rectified Right", fontsize=7)

                # compact layout and colorbar: create one shared colorbar for depth columns on the right
                # find first depth image object
                im_for_cbar = None
                #for r in range(r_idx):
                for c_idx in range(len(MODEL_KEYWORDS)):
                    ax_c = axes[r_idx*2 + int(c_idx >= n_depth_cols), 2 + c_idx%n_depth_cols]
                    im = getattr(ax_c, "_my_im", None)
                    if im is not None:
                        im_for_cbar = im
                        break
                    ax_c = axes[r_idx*2 + 1 + int(c_idx >= n_depth_cols), 2 + c_idx%n_depth_cols]
                    im = getattr(ax_c, "_my_im", None)
                    if im is not None:
                        im_for_cbar = im
                        break
                # place colorbar if found
                if im_for_cbar is not None:
                    # make space on the right
                    fig.subplots_adjust(right=0.92, top=0.95, bottom=0.05, hspace=0.001, wspace=0.05)
                    cbar_ax = fig.add_axes([0.945, 0.15, 0.01, 0.4])  # [left, bottom, width, height]
                    cb = fig.colorbar(im_for_cbar, cax=cbar_ax)
                    #cb.set_label("Depth (m)", fontsize=9)
                else:
                    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.001, wspace=0.05)

                # Save figure into left_rectified_dir
                out_name = f"fig_{left_cam}_fl{int(fl)}_F{F:.1f}_page{page:03d}.png"
                out_path = left_rectified_dir / out_name
                #plt.show()
                fig.savefig(out_path, bbox_inches='tight', dpi=DPI)
                plt.close(fig)
                logging.info(f"Saved figure: {out_path}")

        # except Exception as exc:
        #     logging.error(f"Exception while processing config {cfg} for base {base}: {exc}", exc_info=False)
        #     # continue to next config (spec requested try-except for each config)
        #     continue

if __name__ == "__main__":
    process_datalist(datalist)
    logging.info("All done.")