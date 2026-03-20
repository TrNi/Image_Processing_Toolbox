'''
 python .\generate_3fig6col_comparison_figure.py "I:\My Drive\DOF_benchmarking\inference\fl_70\crop_coords_4701.txt" "I:\My Drive\DOF_benchmarking\inference\Scene8_6D_B_Right_fl70\crop_coords_0.txt" "I:\My Drive\DOF_benchmarking\inference\fl_70\crop_coords_4620.txt" --output "H:\My Drive\Research_collabs\MODEST Research Collab\ECCV_Visuals\dof_new\3col.PNG"
'''

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import numpy as np

# Set Times New Roman as default font
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

# ECCV template page width (double column width in inches)
ECCV_PAGE_WIDTH = 6.875

# Bounding box colors (RGB normalized to 0-1 for matplotlib)
BBOX_COLORS = [
    (0.98, 0.95, 0.4),   # Whitish yellow/popping green
    (0.4, 0.7, 0.98),    # Royal whitish popping blue
    (0.85, 0.7, 0.95)    # Whitish violet
]

# PIL colors (RGB 0-255)
BBOX_COLORS_PIL = [
    (250, 242, 102),     # Whitish yellow/popping green
    (102, 178, 250),     # Royal whitish popping blue
    (217, 178, 242)      # Whitish violet
]


def parse_text_file(filepath):
    """
    Parse a text file containing bounding boxes, index, and folder paths.
    
    Returns:
        bboxes: List of 3 tuples [(x1, y1, x2, y2), ...]
        index: Image index
        folders: List of tuples [(label, path), ...]
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # First 6 lines are coordinates (3 bounding boxes)
    bboxes = []
    for i in range(0, 6, 2):
        x1, y1 = map(int, lines[i].split(','))
        x2, y2 = map(int, lines[i+1].split(','))
        bboxes.append((x1, y1, x2, y2))
    
    # Next line is index
    index_line = lines[6].split(',')
    index = int(index_line[1])
    
    # Remaining lines are folder paths
    folders = []
    for line in lines[7:]:
        parts = line.split(',', 1)  # Split on first comma only
        if len(parts) == 2:
            label = parts[0].strip()
            path = parts[1].strip()
            folders.append((label, path))
    
    return bboxes, index, folders


def get_image_at_index(folder_path, index):
    """
    Get the image at the specified index from a folder (sorted lexicographically).
    
    Returns:
        PIL Image object
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
    image_files = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        raise FileNotFoundError(f"No images found in: {folder_path}")
    
    if index < 0 or index >= len(image_files):
        raise IndexError(f"Index {index} out of range for folder: {folder_path}")
    
    return Image.open(image_files[index])


def crop_image(img, bbox):
    """
    Crop an image using bounding box coordinates.
    
    Args:
        img: PIL Image
        bbox: Tuple (x1, y1, x2, y2)
    
    Returns:
        Cropped PIL Image
    """
    return img.crop(bbox)


def generate_comparison_figure(text_files, output_path):
    """
    Generate a comparison figure from 3 text files.
    
    Args:
        text_files: List of 3 text file paths
        output_path: Path to save the output figure
    """
    if len(text_files) != 3:
        raise ValueError("Exactly 3 text files are required")
    
    # Parse all text files
    all_data = []
    for tf in text_files:
        bboxes, index, folders = parse_text_file(tf)
        all_data.append((bboxes, index, folders))
    
    # Determine number of rows (number of folders/methods)
    num_rows = len(all_data[0][2])
    
    # Create figure with proper width ratios
    # Even columns (crops) should be 75% of odd columns (originals)
    # Width ratios: [orig1=1.0, crops1=0.75, orig2=1.0, crops2=0.75, orig3=1.0, crops3=0.75]
    width_ratios = [1.0, 0.4, 1.0, 0.4, 1.0, 0.4]
    
    # Calculate figure dimensions
    # Assuming original images have aspect ratio ~1.5 (width/height)
    fig_width = 8.0  # Increased from ECCV_PAGE_WIDTH
    fig_height = fig_width * num_rows / 3.5 * 0.67
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create grid with custom width ratios
    gs = fig.add_gridspec(num_rows, 6, 
                          width_ratios=width_ratios,
                          hspace=0.02, wspace=0.02, 
                          left=0.04, right=0.998, top=0.99, bottom=0.01)
    
    # Process each column (text file)
    for col_idx, (bboxes, index, folders) in enumerate(all_data):
        # Process each row (method/folder)
        for row_idx, (label, folder_path) in enumerate(folders):
            try:
                # Load image
                img = get_image_at_index(folder_path, index)
                
                # Resize image by scale_factor
                scale_factor = 1.5
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Scale bounding boxes accordingly
                scaled_bboxes = []
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    scaled_bboxes.append((
                        int(x1 * scale_factor), int(y1 * scale_factor),
                        int(x2 * scale_factor), int(y2 * scale_factor)
                    ))
                
                img_array = np.array(img_resized)
                
                # Create subplot for original image with bounding boxes
                ax_orig = fig.add_subplot(gs[row_idx, col_idx * 2])
                ax_orig.imshow(img_array, aspect='auto', interpolation='nearest')
                ax_orig.axis('off')
                ax_orig.set_aspect('auto')
                ax_orig.margins(0)
                ax_orig.set_position(ax_orig.get_position())
                plt.setp(ax_orig.spines.values(), visible=False)
                ax_orig.set_xlim([0, img_array.shape[1]])
                ax_orig.set_ylim([img_array.shape[0], 0])
                
                # Draw all 3 bounding boxes on the original image
                for bbox_idx, scaled_bbox in enumerate(scaled_bboxes):
                    x1, y1, x2, y2 = scaled_bbox
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=1.2, edgecolor=BBOX_COLORS[bbox_idx],
                        facecolor='none'
                    )
                    ax_orig.add_patch(rect)
                
                # Add row label on the leftmost column (rotated vertically)
                if col_idx == 0:
                    ax_orig.text(-0.015, 0.5, label, transform=ax_orig.transAxes,
                               fontsize=7, va='center', ha='right', rotation=90,
                               fontfamily='serif', fontweight='normal')
                
                # Create subplot for stacked crops
                ax_crops = fig.add_subplot(gs[row_idx, col_idx * 2 + 1])
                ax_crops.axis('off')
                ax_crops.set_aspect('auto')
                ax_crops.margins(0)
                plt.setp(ax_crops.spines.values(), visible=False)
                
                # Create crops from resized image with scaled bboxes
                crop_images = []
                for scaled_bbox in scaled_bboxes:
                    cropped_img = crop_image(img_resized, scaled_bbox)
                    crop_images.append(cropped_img)
                
                # Determine target width for crops (to fill the column)
                # Get the maximum crop width
                max_crop_width = max(crop.width for crop in crop_images)
                
                # Stack crops vertically with colored borders
                bordered_crops = []
                border_width = 20
                
                for crop_idx, crop_img in enumerate(crop_images):
                    # Resize crop to fill width if needed (maintain aspect ratio)
                    if crop_img.width < max_crop_width:
                        scale_factor = max_crop_width / crop_img.width
                        new_w = max_crop_width
                        new_h = int(crop_img.height * scale_factor)
                        crop_img = crop_img.resize((new_w, new_h), Image.LANCZOS)
                    
                    # Create a new image with border
                    new_width = crop_img.width + 2 * border_width
                    new_height = crop_img.height + 2 * border_width
                    bordered = Image.new('RGB', (new_width, new_height), BBOX_COLORS_PIL[crop_idx], )
                    bordered.paste(crop_img, (border_width, border_width))
                    bordered_crops.append(bordered)
                
                # Stack the bordered crops vertically
                total_height = sum(crop.height for crop in bordered_crops)
                max_width = max(crop.width for crop in bordered_crops)
                
                stacked_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))
                y_offset = 0
                for crop in bordered_crops:
                    stacked_img.paste(crop, (0, y_offset))
                    y_offset += crop.height
                
                # Display the stacked crops
                stacked_array = np.array(stacked_img)
                ax_crops.imshow(stacked_array, aspect='auto', interpolation='nearest')
                ax_crops.set_xlim([0, stacked_array.shape[1]])
                ax_crops.set_ylim([stacked_array.shape[0], 0])
                
            except Exception as e:
                print(f"Error processing row {row_idx}, column {col_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Create empty subplot on error
                ax_orig = fig.add_subplot(gs[row_idx, col_idx * 2])
                ax_orig.axis('off')
                ax_crops = fig.add_subplot(gs[row_idx, col_idx * 2 + 1])
                ax_crops.axis('off')
    
    # Save figure with high quality
    plt.savefig(output_path, dpi=900, bbox_inches='tight', pad_inches=0.005)
    print(f"Figure saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate a comparison figure from 3 text files with bounding boxes and folder paths.'
    )
    parser.add_argument(
        'text_files',
        nargs=3,
        type=str,
        help='Paths to 3 text files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='comparison_figure.png',
        help='Output figure path (default: comparison_figure.png)'
    )
    
    args = parser.parse_args()
    
    # Validate text files exist
    for tf in args.text_files:
        if not Path(tf).exists():
            print(f"Error: Text file not found: {tf}")
            sys.exit(1)
    
    generate_comparison_figure(args.text_files, args.output)


if __name__ == '__main__':
    main()
