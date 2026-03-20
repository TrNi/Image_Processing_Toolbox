import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from typing import List, Tuple


def plot_four_images(
    image_data: List[Tuple[str, str]],
    savedir: str,
    dpi: int = 300,
    filename: str = "merged_images"
):
    """
    Plot 4 images in a 1x4 grid suitable for CVPR paper width.
    
    Parameters:
    -----------
    image_data : List[Tuple[str, str]]
        List of 4 tuples, each containing (image_path, title)
    savedir : str
        Directory to save the output files
    dpi : int
        DPI for output images (default: 300)
    filename : str
        Base filename for output (default: "merged_images")
    """
    if len(image_data) != 4:
        raise ValueError("Exactly 4 images are required")
    
    # CVPR paper full linewidth is approximately 6.5 inches
    fig_width = 6.5
    fig_height = fig_width * 0.28  # Adjust aspect ratio for 1x4 horizontal layout
    
    # Create figure with tight layout - 1 row, 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0, wspace=0.05, hspace=0)
    
    # Flatten axes for easier iteration (already flat for 1D, but keeps code consistent)
    if len(image_data) == 4:
        axes = axes.flatten()
    
    # Load and plot each image
    for idx, (img_path, title) in enumerate(image_data):
        img = mpimg.imread(img_path)
        
        # For the last image, adjust aspect ratio to match 2nd and 3rd images
        if idx == 3:
            # Get aspect ratio of the 2nd image (index 1)
            img_ref = mpimg.imread(image_data[1][0])
            ref_aspect = img_ref.shape[0] / img_ref.shape[1]
            
            axes[idx].imshow(img, aspect='auto')
            axes[idx].set_aspect(ref_aspect / (img.shape[0] / img.shape[1]))
        else:
            axes[idx].imshow(img)
        
        axes[idx].axis('off')
        
        # Set title with Times New Roman, size 12, pad 4
        axes[idx].set_title(
            title,
            fontfamily='Times New Roman',
            fontsize=8,
            pad=2
        )
    
    # Create output directory if it doesn't exist
    Path(savedir).mkdir(parents=True, exist_ok=True)
    
    # Save as PDF
    pdf_path = Path(savedir) / f"{filename}.pdf"
    plt.savefig(
        pdf_path,
        format='pdf',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0
    )
    print(f"Saved PDF: {pdf_path}")
    
    # Save as PNG
    png_path = Path(savedir) / f"{filename}.png"
    plt.savefig(
        png_path,
        format='png',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0
    )
    print(f"Saved PNG: {png_path}")
    
    plt.close(fig)


if __name__ == "__main__":
    # Example usage
    savedir=r"H:\My Drive\Research_collabs\MODEST Research Collab\CVPR_visuals\rebuttal_pdfs"
    image_data = [
        (savedir + r"\Charuco.PNG", "ChArUco Pattern"),
        (savedir + r"\rectified_lefts_rectified_lefts_idx7.PNG", "Rectified Left"),
        (savedir + r"\rectified_rights_rectified_rights_idx7.PNG", "Rectified Right"),
        (savedir + r"\Foundation.PNG", "Foundation Stereo"),
    ]
    
    plot_four_images(
        image_data=image_data,
        savedir=savedir,
        dpi=900,
        filename="merged_images"
    )