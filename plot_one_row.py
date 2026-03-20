import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image

# Configure matplotlib to use Times New Roman formatting (same as visualize_error_analysis.py)
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


def create_one_row_plot(image_paths, column_titles, save_path, figsize=(7.16, 1.8), dpi=900):
    """
    Create a one-row plot layout for ECCV paper template.
    
    Args:
        image_paths: List of 10 image paths
                    - image_paths[0]: Column 1 (single image)
                    - image_paths[1]: Column 2 (single image)
                    - image_paths[2:6]: Column 3 (quadrants: top-left, top-right, bottom-left, bottom-right)
                    - image_paths[6:10]: Column 4 (quadrants: top-left, top-right, bottom-left, bottom-right)
        column_titles: List of 4 strings for column headings
        save_path: Path to save the output figure
        figsize: Figure size in inches (default: 7.16 for ECCV single column width)
        dpi: Resolution for saving (default: 300)
    """
    if len(image_paths) != 10:
        raise ValueError(f"Expected 10 image paths, got {len(image_paths)}")
    
    if len(column_titles) != 4:
        raise ValueError(f"Expected 4 column titles, got {len(column_titles)}")
    
    # Load all images
    images = []
    for path in image_paths:
        img = Image.open(path)
        images.append(np.array(img))
    
    # Create figure with GridSpec
    # Layout: 8 columns (2 for each major column), 5 rows (title + 2x2 quadrants with gaps)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(5, 8, figure=fig, 
                  height_ratios=[0.1, 1, 0.005, 1, 0.005], 
                  width_ratios=[1, 1, 0.8, 0.8, 1, 1.1, 1.1, 1.1], 
                  hspace=0.02, wspace=0.02)
    
    # Column 1: Single image with title
    ax_title1 = fig.add_subplot(gs[0, 0:2])
    ax_title1.text(0.5, 0.5, column_titles[0], ha='center', va='center',
                   fontsize=5, fontfamily='Times New Roman')
    ax_title1.axis('off')
    
    ax1 = fig.add_subplot(gs[1:, 0:2])
    ax1.imshow(images[0], aspect='auto')
    ax1.axis('off')
    
    # Column 2: Single image with title
    ax_title2 = fig.add_subplot(gs[0, 2:4])
    ax_title2.text(0.5, 0.5, column_titles[1], ha='center', va='center',
                   fontsize=5, fontfamily='Times New Roman')
    ax_title2.axis('off')
    
    ax2 = fig.add_subplot(gs[1:, 2:4])
    ax2.imshow(images[1], aspect='auto')
    ax2.axis('off')
    
    # Column 3: 2x2 quadrants with title
    ax_title3 = fig.add_subplot(gs[0, 4:6])
    ax_title3.text(0.5, 0.5, column_titles[2], ha='center', va='center',
                   fontsize=5, fontfamily='Times New Roman')
    ax_title3.axis('off')
    
    # Quadrants for column 3 (images 2-5)
    ax3_tl = fig.add_subplot(gs[1, 4])
    ax3_tl.imshow(images[2], aspect='auto')
    ax3_tl.axis('off')
    
    ax3_tr = fig.add_subplot(gs[1, 5])
    ax3_tr.imshow(images[3], aspect='auto')
    ax3_tr.axis('off')
    
    ax3_bl = fig.add_subplot(gs[3, 4])
    ax3_bl.imshow(images[4], aspect='auto')
    ax3_bl.axis('off')
    
    ax3_br = fig.add_subplot(gs[3, 5])
    ax3_br.imshow(images[5], aspect='auto')
    ax3_br.axis('off')
    
    # Column 4: 2x2 quadrants with title
    ax_title4 = fig.add_subplot(gs[0, 6:8])
    ax_title4.text(0.5, 0.5, column_titles[3], ha='center', va='center',
                   fontsize=5, fontfamily='Times New Roman')
    ax_title4.axis('off')
    
    # Quadrants for column 4 (images 6-9)
    ax4_tl = fig.add_subplot(gs[1, 6])
    ax4_tl.imshow(images[6], aspect='auto')
    ax4_tl.axis('off')
    
    ax4_tr = fig.add_subplot(gs[1, 7])
    ax4_tr.imshow(images[7], aspect='auto')
    ax4_tr.axis('off')
    
    ax4_bl = fig.add_subplot(gs[3, 6])
    ax4_bl.imshow(images[8], aspect='auto')
    ax4_bl.axis('off')
    
    ax4_br = fig.add_subplot(gs[3, 7])
    ax4_br.imshow(images[9], aspect='auto')
    ax4_br.axis('off')
    
    # Save figure
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    print(f"✅ Saved figure to {save_path}")

    # Save as PDF
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, format='pdf', dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    print(f"✅ Saved PDF to {pdf_path}")
    
    plt.close()


def main():
    """
    Example usage of the ECCV plot row generator.
    Replace the image paths and titles with your actual data.
    """

    rdir = r'H:\My Drive\Research_collabs\MODEST Research Collab\ECCV_Visuals\\'
    ref = rdir+r'data_components\quadrant_reflective\\'
    det = rdir+r'data_components\quadrant_details\\'
    dof = rdir+r'data_components\dof\\'

    # Example image paths (replace with your actual paths)
    image_paths = [
        rdir+"calibration-illustration.PNG",
        rdir+"camera_assembly.JPG",
        ref+r'IMG_0070_crop_2774_1453_3891_2694.jpg',
        ref+r'IMG_5939_crop_165_1895_2399_3643.jpg',
        ref+r'IMG_6963_crop_0_1827_1688_3645.jpg',
        ref+r'IMG_1626_crop_492_1938_1822_2795.jpg',        
        det+r'IMG_3736_crop_1059_1966_1934_2412.jpg',
        det+r'IMG_3609_crop_2521_2003_3271_2525.jpg',
        det+r'IMG_7455_crop_1571_28_3322_1366.jpg',        
        det+r'IMG_1062_crop_89_861_1524_1591.jpg',
        # dof+r'\dof_000000.png',
        # dof+r'\dof_000001.png',
        # dof+r'\dof_000002.png',
        # dof+r'\dof_000003.png',
    ]
    
    # Column titles
    column_titles = [
        "Calibration Viewpoints",
        "Stereo Camera Assembly",
        "Reflective Surfaces",
        "Fine Details, 3D Textures",
        #"DOF",
    ]
    
    # Output path
    save_path = rdir+"calib+cam+refl+detl.png"
    
    # Generate the plot
    create_one_row_plot(image_paths, column_titles, save_path)


if __name__ == "__main__":
    main()
