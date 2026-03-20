import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.widgets import RectangleSelector, Button

# ==============================================================================
# === USER INPUT PATHS & CONFIGURATION (COPY/ADAPT FROM vis_blur.py) ===========
# ==============================================================================

# --- File Paths ---
# input_path = r"I:\My Drive\DOF_benchmarking\inference\fl_70\F22.0\IMG_4612.JPG"
# gt_path = r"I:\My Drive\DOF_benchmarking\inference\fl_70\F2.8\IMG_4620.JPG"

# model_jpg_paths = [
#     r"I:\My Drive\DOF_benchmarking\inference\fl_70\Bokehme_scale20_K40\frame_0001.JPG",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_70\Drbokeh_K25_fp0p25\IMG_4612.JPG",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_70\Bokehliciouslg_intp_fl70\net_large_f2.8_IMG_4612.JPG",
# ]
# for above, roi:
# top left: 730,2257
# bottom right: 3011,3557

input_path = r"I:\My Drive\DOF_benchmarking\inference\Scene8_6D_B_Right_fl70\F22.0\IMG_0628.JPG"
gt_path = r"I:\My Drive\DOF_benchmarking\inference\Scene8_6D_B_Right_fl70\F2.8\IMG_0636.JPG"

model_jpg_paths = [
    r"I:\My Drive\DOF_benchmarking\inference\Scene8_6D_B_Right_fl70\Bokehme_scale20_K40\frame_0000.JPG",
    r"I:\My Drive\DOF_benchmarking\inference\Scene8_6D_B_Right_fl70\Drbokeh_K25_fp0.25\IMG_0628.JPG",
    r"I:\My Drive\DOF_benchmarking\inference\Scene8_6D_B_Right_fl70\Bokehlicious_intp\net_large_f2.8_IMG_0628.JPG",
]


model_names = ["BokehMe", "Drbokeh", "Bokehlicious"]

outdir = r"H:\My Drive\Research_collabs\MODEST Research Collab\ECCV_Visuals\data_components\dof\roi"
os.makedirs(outdir, exist_ok=True)
base_filename = os.path.splitext(os.path.basename(gt_path))[0]
# ==============================================================================
# === ROI CONFIGURATION ========================================================
# ==============================================================================

# User specifies ROI by top-left and bottom-right corners
print("\n" + "="*70)
print("ROI CONFIGURATION")
print("="*70)
print("Please specify the Region of Interest (ROI) using corner coordinates.")
print("Format: y,x (row,column)")
print()

# Get top-left corner from user
top_left_input = input("Enter top-left corner (y,x): ").strip()
try:
    roi_top_left_y, roi_top_left_x = map(int, top_left_input.split(','))
except ValueError:
    print("❌ Invalid input format. Using default top-left: (600, 800)")
    roi_top_left_y, roi_top_left_x = 600, 800

# Get bottom-right corner from user
bottom_right_input = input("Enter bottom-right corner (y,x): ").strip()
try:
    roi_bottom_right_y, roi_bottom_right_x = map(int, bottom_right_input.split(','))
except ValueError:
    print("❌ Invalid input format. Using default bottom-right: (856, 1056)")
    roi_bottom_right_y, roi_bottom_right_x = 856, 1056

# Automatically calculate ROI parameters from corners
roi_y = roi_top_left_y
roi_x = roi_top_left_x
roi_h = roi_bottom_right_y - roi_top_left_y
roi_w = roi_bottom_right_x - roi_top_left_x

print()
print(f"📍 ROI Configuration:")
print(f"   Top-left corner: ({roi_top_left_y}, {roi_top_left_x})")
print(f"   Bottom-right corner: ({roi_bottom_right_y}, {roi_bottom_right_x})")
print(f"   Calculated ROI: y={roi_y}, x={roi_x}, h={roi_h}, w={roi_w}")
print("="*70)
print()

use_interactive_roi = True
draw_roi_rect = True
roi_rect_color = (0.2, 1.0, 0.2)
roi_rect_linewidth = 0.7
roi_zoom_factor = 2.0

# ==============================================================================
# === MATPLOTLIB GLOBAL STYLE ==================================================
# ==============================================================================

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 6,
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.titlepad'] = 2
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ==============================================================================
# === LOAD IMAGES ==============================================================
# ==============================================================================

print("📖 Loading Input and GT images...")
try:
    input_img_raw = cv2.imread(input_path)
    gt_img_raw    = cv2.imread(gt_path)
    if input_img_raw is None:
        raise FileNotFoundError(f"Input image not found at: {input_path}")
    if gt_img_raw is None:
        raise FileNotFoundError(f"GT image not found at: {gt_path}")
    input_img = cv2.cvtColor(input_img_raw, cv2.COLOR_BGR2RGB)
    gt_img    = cv2.cvtColor(gt_img_raw,    cv2.COLOR_BGR2RGB)
except Exception as e:
    print(f"❌ Error loading JPG images: {e}")
    exit()

# ==============================================================================
# === LOAD MODEL PREDICTIONS ===================================================
# ==============================================================================

preds = []
print(f"� Loading model prediction images...")

for jpg_path, model_name in zip(model_jpg_paths, model_names):
    print(f"📖 Loading prediction for '{model_name}' from: {jpg_path}")
    try:
        pred_img_raw = cv2.imread(jpg_path)
        if pred_img_raw is None:
            print(f"⚠️  Warning: Could not load image at '{jpg_path}'. Skipping.")
            continue
        pred_img = cv2.cvtColor(pred_img_raw, cv2.COLOR_BGR2RGB)
        preds.append(pred_img)
    except Exception as e:
        print(f"❌ Error reading JPG file {jpg_path}: {e}")

if len(preds) != len(model_jpg_paths):
    print("❌ Could not load all model predictions. Check paths.")
    exit()

# ==============================================================================
# === ALIGN IMAGE SIZES ========================================================
# ==============================================================================

print("📐 Resizing images to GT dimensions...")
target_h, target_w = gt_img.shape[:2]

def resize_image(im, size):
    ch, cw = im.shape[:2]
    th, tw = size
    interp = cv2.INTER_CUBIC if th * tw > ch * cw else cv2.INTER_AREA
    return cv2.resize(im, (tw, th), interpolation=interp)

input_img_resized = input_img #resize_image(input_img, (target_h, target_w))
preds_resized     = preds #[resize_image(p, (target_h, target_w)) for p in preds]

images_to_plot = [input_img_resized] + preds_resized + [gt_img]
titles         = ["Input (Sharp)"] + model_names + ["Ground Truth"]

# ==============================================================================
# === PRE-COMPUTE LOW-RES THUMBNAILS FOR FAST UI RENDERING ====================
# ==============================================================================
# Display thumbnails in the UI — keeps panning/zooming snappy.
# We cap the longest edge at UI_MAX_SIDE pixels so even 24-MP images feel fast.
UI_MAX_SIDE = 5472  # pixels — raise if you have a very large monitor

def make_thumbnail(im, max_side=UI_MAX_SIDE):
    h, w = im.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        nh, nw = int(h * scale), int(w * scale)
        return cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA), scale
    return im.copy(), 1.0

thumb_scale = None
thumbs = []
for img in images_to_plot:
    t, s = make_thumbnail(img)
    thumbs.append(t)
    if thumb_scale is None:
        thumb_scale = s   # all images same size → same scale

print(f"🖼  Thumbnail scale for UI: {thumb_scale:.3f}  "
      f"({thumbs[0].shape[1]}×{thumbs[0].shape[0]} px per image)")

# ==============================================================================
# === INTERACTIVE ROI PICKER ===================================================
# ==============================================================================

if use_interactive_roi:
    print("\n🖱  Interactive ROI mode is ON.")
    print("    • TOP ROW: Thumbnails with red ROI box overlay.")
    print("    • BOTTOM ROW: Large live crop previews for each image.")
    print("    • Drag/resize the red box on GT thumbnail to adjust ROI.")
    print("    • All crop previews update in real-time.")
    print("    • Click 'Save ROI ✔' when happy.\n")

    num_imgs_ui = len(thumbs)

    # ── Build 2-row figure: top=thumbnails, bottom=crops ──────────────────────
    fig_ui, axes_ui = plt.subplots(
        2, num_imgs_ui,
        figsize=(min(5 * num_imgs_ui, 28), 14),
    )
    
    # Handle single image case
    if num_imgs_ui == 1:
        axes_ui = np.array([[axes_ui[0]], [axes_ui[1]]])
    
    fig_ui.subplots_adjust(left=0.02, right=0.98, top=0.97,
                           bottom=0.08, wspace=0.08, hspace=0.15)

    # Maximise the window across common backends
    try:
        mng = plt.get_current_fig_manager()
        try:
            mng.window.showMaximized()          # Qt backends
        except AttributeError:
            try:
                mng.window.state('zoomed')      # Tk backend (Windows/macOS)
            except AttributeError:
                try:
                    mng.frame.Maximize(True)    # wx backend
                except AttributeError:
                    pass
    except Exception:
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # TOP ROW: Display thumbnails with ROI rectangles
    # ─────────────────────────────────────────────────────────────────────────
    for col_idx, (thumb, title) in enumerate(zip(thumbs, titles)):
        ax_top = axes_ui[0, col_idx]
        ax_top.imshow(thumb, interpolation='bilinear')
        ax_top.set_title(title, fontsize=10, pad=3, fontweight='bold')
        ax_top.axis('off')

    # Scale initial ROI to thumbnail space
    sy = thumb_scale
    sx = thumb_scale
    init_ty = int(roi_y * sy)
    init_tx = int(roi_x * sx)
    init_th = max(1, int(roi_h * sy))
    init_tw = max(1, int(roi_w * sx))

    # ── Overlay ROI rectangles on all TOP panels ───────────────────────────────
    from matplotlib.patches import Rectangle

    rects_ui = []
    for col_idx in range(num_imgs_ui):
        ax_top = axes_ui[0, col_idx]
        r = Rectangle(
            (init_tx, init_ty), init_tw, init_th,
            linewidth=2.0, edgecolor='red', facecolor='none', zorder=5,
        )
        ax_top.add_patch(r)
        rects_ui.append(r)

    # Track current ROI in thumbnail coords
    _roi_thumb = {"y": init_ty, "x": init_tx, "h": init_th, "w": init_tw}

    # ─────────────────────────────────────────────────────────────────────────
    # BOTTOM ROW: Initialize large crop previews
    # ─────────────────────────────────────────────────────────────────────────
    crop_ims = []
    for col_idx in range(num_imgs_ui):
        ax_crop = axes_ui[1, col_idx]
        
        # Extract initial crop from thumbnail
        y, x, h, w = _roi_thumb["y"], _roi_thumb["x"], _roi_thumb["h"], _roi_thumb["w"]
        H, W = thumbs[col_idx].shape[:2]
        y = max(0, min(y, H - 1))
        x = max(0, min(x, W - 1))
        h = max(1, min(h, H - y))
        w = max(1, min(w, W - x))
        init_crop = thumbs[col_idx][y:y + h, x:x + w]
        
        im = ax_crop.imshow(init_crop, interpolation='bilinear')
        ax_crop.axis('off')
        crop_ims.append((ax_crop, im))

    def _update_crops():
        """Update all large crop previews from their respective thumbnails."""
        y, x, h, w = (
            _roi_thumb["y"],
            _roi_thumb["x"],
            _roi_thumb["h"],
            _roi_thumb["w"],
        )
        
        for col_idx in range(num_imgs_ui):
            ax_crop, im = crop_ims[col_idx]
            thumb = thumbs[col_idx]
            
            # Clamp to thumbnail bounds
            H, W = thumb.shape[:2]
            yc = max(0, min(y, H - 1))
            xc = max(0, min(x, W - 1))
            hc = max(1, min(h, H - yc))
            wc = max(1, min(w, W - xc))
            
            crop = thumb[yc:yc + hc, xc:xc + wc]
            im.set_data(crop)
            ax_crop.set_xlim(-0.5, crop.shape[1] - 0.5)
            ax_crop.set_ylim(crop.shape[0] - 0.5, -0.5)

    def _update_rects():
        """Update rectangle positions and crop previews."""
        for r in rects_ui:
            r.set_xy((_roi_thumb["x"], _roi_thumb["y"]))
            r.set_width(_roi_thumb["w"])
            r.set_height(_roi_thumb["h"])
        _update_crops()
        fig_ui.canvas.draw_idle()

    # ── RectangleSelector (always active on GT thumbnail) ─────────────────────
    gt_ax_ui = axes_ui[0, -1]  # GT is the last column in top row

    def on_select(eclick, erelease):
        x1, y1 = eclick.xdata,   eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, x2, y1, y2):
            return

        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        # Update thumbnail-space ROI
        _roi_thumb["x"] = int(round(x_min))
        _roi_thumb["y"] = int(round(y_min))
        _roi_thumb["w"] = max(1, int(round(x_max - x_min)))
        _roi_thumb["h"] = max(1, int(round(y_max - y_min)))

        _update_rects()

    rect_selector = RectangleSelector(
        gt_ax_ui, on_select,
        useblit=True,
        button=[1],
        minspanx=5, minspany=5,
        spancoords='data',
        interactive=True,
        props=dict(edgecolor='red', facecolor='none', linewidth=2.0),
    )
    rect_selector.set_active(True)  # Always active from the start

    # ── Save Button ────────────────────────────────────────────────────────────
    ax_save = fig_ui.add_axes([0.42, 0.02, 0.12, 0.045])
    btn_save = Button(ax_save, 'Save ROI ✔', color='lightgreen', hovercolor='lime')
    btn_save.label.set_fontsize(11)

    def on_save_roi(event):
        global roi_y, roi_x, roi_h, roi_w

        # Convert thumbnail coords → full-resolution coords
        roi_x = int(_roi_thumb["x"] / thumb_scale)
        roi_y = int(_roi_thumb["y"] / thumb_scale)
        roi_w = max(1, int(_roi_thumb["w"] / thumb_scale))
        roi_h = max(1, int(_roi_thumb["h"] / thumb_scale))

        # Clamp
        roi_y = max(0, min(roi_y, target_h - 1))
        roi_x = max(0, min(roi_x, target_w - 1))
        roi_h = max(1, min(roi_h, target_h - roi_y))
        roi_w = max(1, min(roi_w, target_w - roi_x))

        roi_txt = os.path.join(outdir, "roi_coords.txt")
        with open(roi_txt, "w") as f:
            f.write(f"{roi_y} {roi_x} {roi_h} {roi_w}\n")

        print(f"💾 ROI saved → {roi_txt}")
        print(f"   roi_y={roi_y}, roi_x={roi_x}, roi_h={roi_h}, roi_w={roi_w}")
        plt.close(fig_ui)

    btn_save.on_clicked(on_save_roi)

    plt.show()

# ==============================================================================
# === CLAMP FINAL ROI ==========================================================
# ==============================================================================

def clamp_roi(y, x, h, w, H, W):
    y = max(0, min(y, H - 1))
    x = max(0, min(x, W - 1))
    h = max(1, min(h, H - y))
    w = max(1, min(w, W - x))
    return y, x, h, w

roi_y, roi_x, roi_h, roi_w = clamp_roi(roi_y, roi_x, roi_h, roi_w, target_h, target_w)
print(f"🔍 Final ROI — y={roi_y}, x={roi_x}, h={roi_h}, w={roi_w}  (full-res pixels)")

# ==============================================================================
# === BUILD FINAL 2-ROW PUBLICATION FIGURE =====================================
# ==============================================================================

print("🎨 Generating publication figure (full image + ROI zoom) …")

num_images = len(images_to_plot)
from matplotlib.patches import Rectangle

fig, axes = plt.subplots(
    2, num_images,
    figsize=(7.7, 2.6),
)
if num_images == 1:
    axes = np.array([[axes[0]], [axes[1]]])

for col_idx, (img, title) in enumerate(zip(images_to_plot, titles)):
    # TOP ROW — full image
    ax_full = axes[0, col_idx]
    ax_full.imshow(img)
    ax_full.set_title(title)
    ax_full.axis('off')

    if draw_roi_rect:
        ax_full.add_patch(Rectangle(
            (roi_x, roi_y), roi_w, roi_h,
            linewidth=roi_rect_linewidth,
            edgecolor=(0.2, 1.0, 0.2),
            facecolor='none',
        ))

    # BOTTOM ROW — zoomed crop
    ax_zoom = axes[1, col_idx]
    if col_idx==4:
        roi_x -= (30) #65
    crop = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    if roi_zoom_factor != 1.0:
        new_h = int(crop.shape[0] * roi_zoom_factor)
        new_w = int(crop.shape[1] * roi_zoom_factor)
        crop  = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    ax_zoom.imshow(crop)
    ax_zoom.axis('off')

fig.tight_layout(pad=0.03, w_pad=0.1, h_pad=0.01)

# ==============================================================================
# === SAVE OUTPUTS =============================================================
# ==============================================================================

print("💾 Saving outputs …")

png_path = os.path.join(outdir, f"{base_filename}.png")
plt.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
print(f"✅ PNG  (600 DPI): {png_path}")

svg_path = os.path.join(outdir, f"{base_filename}.svg")
plt.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.01)
print(f"✅ SVG (vector):  {svg_path}")

pdf_path = os.path.join(outdir, f"{base_filename}.pdf")
plt.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.01)
print(f"✅ PDF  (600 DPI): {pdf_path}")

plt.close(fig)
print("\n🎉 All done!")