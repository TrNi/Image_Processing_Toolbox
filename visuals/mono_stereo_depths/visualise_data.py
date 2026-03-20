#!/usr/bin/env python3
"""
visualise_data.py

CVPR 2026 style plotting script for visualizing data across 10 scenes.
Creates a 2x5 grid with minimal padding and saves in multiple formats.

Requirements:
    pip install numpy matplotlib h5py opencv-python
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# ========== CVPR STYLE RCPARAMS ==========
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

# ========== USER INPUT ==========
# List of 10 paths to visualize
datalist = [

    r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene1\EOS6D_B_Left\fl_28mm\inference\F22.0\IMG_3607.JPG",
    r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene2\EOS6D_A_Left\fl_28mm\inference\F22.0\IMG_1690.JPG",
    r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene3\EOS6D_B_Left\fl_28mm\inference\F22.0\IMG_8388.JPG",    
    r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene4\EOS6D_B_Left\fl_32mm\inference\F22.0\IMG_1716.JPG",
    r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene5\EOS6D_A_Left\fl_32mm\inference\F22.0\IMG_6547.JPG",
    r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene6\EOS6D_B_Left\fl_32mm\inference\F22\IMG_6460.JPG",
    r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene7\EOS6D_A_Left\fl_28mm\inference\F22.0\IMG_9828.JPG",
    r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene8\EOS6D_B_Right\fl_50mm\inference\F22.0\IMG_0186.JPG",    
    r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene9\EOS6D_B_Right\fl_28mm\inference\F22.0\IMG_4954.JPG",
    r"I:\My Drive\DoF_Datasets\MODEST2\DOF_Scene1\EOS6D_A_Right\fl_65mm\inference\F22.0\IMG_5444.JPG"
    #r"I:\My Drive\DoF_Datasets\MODEST2\DOF_Scene1\EOS6D_A_Right\fl_45mm\inference\F22.0\IMG_4874.JPG",    
]

# Output directory for saving figures
output_dir = r"H:\My Drive\Research_collabs\MODEST Research Collab\ECCV_Visuals"

# ========== HELPER FUNCTIONS ==========
def load_data(file_path):
    """
    Loads a JPG image file.
    
    Args:
        file_path: Path to the JPG image file
    
    Returns:
        numpy array (RGB) or None if loading fails
    """
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return None
    
    try:
        # Load image using OpenCV
        img = cv2.imread(file_path)
        if img is not None:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"✅ Loaded {os.path.basename(file_path)} | shape: {img.shape}")
            return img
        else:
            print(f"⚠️ Failed to load image: {file_path}")
            return None
    except Exception as e:
        print(f"⚠️ Error loading {file_path}: {e}")
        return None


def plot_data(ax, data, title):
    """
    Plots data on the given axis with title.
    
    Args:
        ax: Matplotlib axis
        data: numpy array to plot
        title: Title for the subplot
    """
    if data is None:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=5)
        ax.set_title(title, fontsize=5, pad=0.2)
        ax.axis('off')
        return
    
    # Normalize data for display
    if data.ndim == 2:
        # Single channel data (e.g., depth map)
        im = ax.imshow(data, cmap='turbo')
    elif data.ndim == 3 and data.shape[-1] in [3, 4]:
        # RGB or RGBA image
        data_display = (data - data.min()) / (data.max() - data.min() + 1e-9)
        im = ax.imshow(data_display)
    else:
        # Fallback for other data types
        im = ax.imshow(data, cmap='turbo')
    
    ax.set_title(title, fontsize=5, pad=0.2)
    ax.axis('off')


def create_visualization(datalist, output_dir):
    """
    Creates a 2x5 grid visualization of the data.
    
    Args:
        datalist: List of 10 paths to data files
        output_dir: Directory to save output figures
    """
    if len(datalist) != 10:
        raise ValueError(f"Expected 10 data paths, got {len(datalist)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all data
    print("\n📂 Loading data...")
    data_arrays = []
    for i, path in enumerate(datalist):
        print(f"Loading {i+1}/10: {path}")
        data = load_data(path)
        data_arrays.append(data)
    
    # Create figure with 2 rows and 5 columns
    # figsize width = 7.6 inches as specified
    fig, axes = plt.subplots(2, 5, figsize=(7.6, 2.25), squeeze=False)
    
    # Adjust spacing - minimal padding, small hspace and wspace
    plt.subplots_adjust(left=0.001, right=0.999, top=0.995, bottom=0.005, 
                        hspace=0.000, wspace=0.01)
    
    # Plot data in 2 rows of 5
    for row in range(2):
        for col in range(5):
            idx = row * 5 + col
            ax = axes[row, col]
            
            # Determine title based on position
            if col == 0:
                # First plot in each row
                scene_num = idx + 1
                title = f"Scene {scene_num}"
            else:
                # Other plots
                title = f"{idx + 1}"
            
            # Plot the data
            plot_data(ax, data_arrays[idx], title)
    
    # Save in multiple formats for CVPR 2026
    base_filename = "data_visualization"    
    print("\n💾 Saving figures...")
    
    # Save as PNG with 900 DPI
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
    print(f"✅ Saved PNG: {png_path}")
    
    # Save as PDF with 900 DPI
    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    plt.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.01)
    print(f"✅ Saved PDF: {pdf_path}")
    
    # Save as SVG
    svg_path = os.path.join(output_dir, f"{base_filename}.svg")
    plt.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.01)
    print(f"✅ Saved SVG: {svg_path}")
    plt.close(fig)
    
    print("\n✅ Visualization complete!")


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("=" * 60)
    print("CVPR 2026 Data Visualization Script")
    print("=" * 60)
    
    create_visualization(datalist, output_dir)
    
    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
