#!/usr/bin/env python3
"""
interactive_visualize_crop.py

Usage:
    python interactive_visualize_crop.py <text_file> <savedir>

text_file : path to a text file where each line is a path to a JPG or PNG image.
savedir   : directory to save cropped images and crop_log.txt.

Controls:
    - Draw a rectangle on the image with left-click drag.
    - Drag the handles / edges to adjust the selection.
    - Live stats (W, H, H/W, W/H) update after each adjustment.
    - Click "Crop Now" to save the crop and move to the next image.
    - Click "Skip"     to skip the current image without saving.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Per-image interactive session
# ---------------------------------------------------------------------------

def process_image(img_path: Path, savedir: Path, log_path: Path) -> None:
    if not img_path.exists():
        print(f"  [SKIP] File not found: {img_path}")
        return

    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    H_img, W_img = img_array.shape[:2]

    # ---- Figure layout ------------------------------------------------
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#1e1e1e")

    # Title
    fig.suptitle(str(img_path), fontsize=8, color="white", y=0.99)

    # Main image axes (left=5%, bottom=20%, width=90%, height=76%)
    ax_img = fig.add_axes([0.05, 0.20, 0.90, 0.76])
    ax_img.imshow(img_array, aspect="auto")
    ax_img.set_xlim(0, W_img)
    ax_img.set_ylim(H_img, 0)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(
        "Draw a crop rectangle   |   drag handles / edges to adjust",
        color="white", fontsize=9, pad=4
    )
    for spine in ax_img.spines.values():
        spine.set_edgecolor("#555555")

    # Stats bar axes
    ax_stats = fig.add_axes([0.05, 0.11, 0.90, 0.07])
    ax_stats.set_facecolor("#2d2d2d")
    ax_stats.axis("off")
    stats_text = ax_stats.text(
        0.5, 0.5,
        "No selection yet — draw a rectangle on the image",
        transform=ax_stats.transAxes,
        fontsize=10, va="center", ha="center",
        family="monospace", color="#cccccc"
    )

    # ---- Shared state -------------------------------------------------
    state = {"coords": None, "crop_done": False}

    # ---- Callbacks ----------------------------------------------------
    def update_stats(x1: float, y1: float, x2: float, y2: float) -> None:
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(W_img, int(round(x2)))
        y2 = min(H_img, int(round(y2)))
        w = x2 - x1
        h = y2 - y1
        hw = h / w if w > 0 else float("inf")
        wh = w / h if h > 0 else float("inf")
        state["coords"] = (x1, y1, x2, y2)
        stats_text.set_text(
            f"Top-left: ({x1}, {y1})   |   Bottom-right: ({x2}, {y2})   |   "
            f"W: {w}   H: {h}   |   H/W: {hw:.4f}   W/H: {wh:.4f}"
        )
        stats_text.set_color("#00e676")
        fig.canvas.draw_idle()

    def onselect(eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        x1 = min(eclick.xdata, erelease.xdata)
        y1 = min(eclick.ydata, erelease.ydata)
        x2 = max(eclick.xdata, erelease.xdata)
        y2 = max(eclick.ydata, erelease.ydata)
        update_stats(x1, y1, x2, y2)

    selector = RectangleSelector(
        ax_img,
        onselect,
        useblit=True,
        button=[1],
        minspanx=2,
        minspany=2,
        spancoords="pixels",
        interactive=True,
        props=dict(facecolor="none", edgecolor="#ff5252", linewidth=1.5, alpha=0.9),
        handle_props=dict(markersize=6, markerfacecolor="white", markeredgecolor="#ff5252"),
    )

    # ---- Buttons ------------------------------------------------------
    ax_btn_crop = fig.add_axes([0.79, 0.025, 0.16, 0.07])
    btn_crop = Button(ax_btn_crop, "Crop Now", color="#00c853", hovercolor="#69f0ae")
    btn_crop.label.set_fontsize(11)
    btn_crop.label.set_fontweight("bold")

    ax_btn_skip = fig.add_axes([0.60, 0.025, 0.16, 0.07])
    btn_skip = Button(ax_btn_skip, "Skip", color="#455a64", hovercolor="#607d8b")
    btn_skip.label.set_fontsize(11)
    btn_skip.label.set_color("white")

    def on_crop(event) -> None:
        if state["coords"] is None:
            stats_text.set_text("⚠  Please draw a crop rectangle first!")
            stats_text.set_color("#ff5252")
            fig.canvas.draw_idle()
            return
        state["crop_done"] = True
        plt.close(fig)

    def on_skip(event) -> None:
        plt.close(fig)

    btn_crop.on_clicked(on_crop)
    btn_skip.on_clicked(on_skip)

    plt.show()

    # ---- Post-close: save if crop was confirmed -----------------------
    if state["crop_done"] and state["coords"] is not None:
        x1, y1, x2, y2 = state["coords"]
        w = x2 - x1
        h = y2 - y1
        hw = h / w if w > 0 else float("inf")
        wh = w / h if h > 0 else float("inf")

        cropped = img.crop((x1, y1, x2, y2))
        out_path = savedir / img_path.name
        cropped.save(str(out_path))
        print(f"  [SAVED] {out_path}   W={w}  H={h}  H/W={hw:.4f}  W/H={wh:.4f}")

        with open(log_path, "a") as f:
            f.write(f"image_path:    {img_path}\n")
            f.write(f"top_left:      ({x1}, {y1})\n")
            f.write(f"bottom_right:  ({x2}, {y2})\n")
            f.write(f"height:        {h}\n")
            f.write(f"width:         {w}\n")
            f.write(f"H_over_W:      {hw:.6f}\n")
            f.write(f"W_over_H:      {wh:.6f}\n")
            f.write("\n")
    else:
        print(f"  [SKIPPED] {img_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive batch image crop tool."
    )
    parser.add_argument(
        "text_file",
        help="Path to a text file with one image path per line (JPG / PNG).",
    )
    parser.add_argument(
        "savedir",
        help="Directory where cropped images and crop_log.txt will be saved.",
    )
    args = parser.parse_args()

    text_file = Path(args.text_file)
    if not text_file.exists():
        print(f"Error: text file not found: {text_file}")
        sys.exit(1)

    savedir = Path(args.savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    log_path = savedir / "crop_log.txt"

    with open(text_file, "r") as f:
        image_paths = [Path(line.strip()) for line in f if line.strip()]

    if not image_paths:
        print("No image paths found in the text file.")
        sys.exit(0)

    print(f"Found {len(image_paths)} image(s) to process.")
    print(f"Crops will be saved to: {savedir}")
    print(f"Crop log: {log_path}\n")

    for i, img_path in enumerate(image_paths):
        print(f"[{i + 1}/{len(image_paths)}] {img_path}")
        process_image(img_path, savedir, log_path)

    print("\nAll images processed.")
    if log_path.exists():
        print(f"Crop log saved: {log_path}")


if __name__ == "__main__":
    main()
