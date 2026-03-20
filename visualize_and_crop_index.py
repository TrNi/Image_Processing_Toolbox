'''
when prompted for cropping, just press enter, no need to crop. 
just save top-left x,y and bottom-right x,y for each of k crops in a text file for a single image.

second argument (1 in this example) is the index of the image to display.
python .\visualize_and_crop_index.py "I:\My Drive\DOF_benchmarking\inference\fl_70\F2.8_align_whitebal" 1 --savedir "H:\My Drive\Research_collabs\MODEST Research Collab\ECCV_Visuals\dof_new\scene8_new"
'''

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def visualize_image_at_index(folder_path, index, save_dir=None):
    """
    Display an image from a folder at the specified index after sorting lexicographically.
    After closing the plot, prompt user for crop coordinates and save the crop.
    
    Args:
        folder_path: Path to the folder containing images
        index: Index of the image to display (0-based)
        save_dir: Directory to save cropped images (optional)
    """
    # Convert to Path object
    folder = Path(folder_path)
    
    # Check if folder exists
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        sys.exit(1)
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
    
    # Get all image files and sort lexicographically
    image_files = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    
    # Check if folder has images
    if not image_files:
        print(f"Error: No image files found in '{folder_path}'.")
        sys.exit(1)
    
    # Check if index is valid
    if index < 0 or index >= len(image_files):
        print(f"Error: Index {index} is out of range. Folder contains {len(image_files)} images (valid indices: 0-{len(image_files)-1}).")
        sys.exit(1)
    
    # Get the image at the specified index
    image_path = image_files[index]
    
    print(f"Total images in folder: {len(image_files)}")
    print(f"Displaying image at index {index}: {image_path.name}")
    
    # Load and display the image
    try:
        img = Image.open(image_path)
        
        # Create figure and display image
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Index {index}: {image_path.name}\nSize: {img.width}x{img.height} pixels", fontsize=14, pad=10)
        plt.tight_layout()
        plt.show()
        
        # After plot is closed, ask for crop coordinates
        if save_dir:
            print("\n" + "="*60)
            print("Enter crop coordinates (or press Enter to skip cropping):")
            print("="*60)
            
            try:
                # Get top-left corner
                top_left = input("Enter top-left corner (x,y): ").strip()
                if not top_left:
                    print("Skipping crop.")
                    return
                
                # Get bottom-right corner
                bottom_right = input("Enter bottom-right corner (x,y): ").strip()
                if not bottom_right:
                    print("Skipping crop.")
                    return
                
                # Parse coordinates
                x1, y1 = map(int, top_left.split(','))
                x2, y2 = map(int, bottom_right.split(','))
                
                # Validate coordinates
                if x1 < 0 or y1 < 0 or x2 > img.width or y2 > img.height:
                    print(f"Error: Coordinates out of bounds. Image size is {img.width}x{img.height}")
                    return
                
                if x1 >= x2 or y1 >= y2:
                    print("Error: Invalid crop region. Top-left must be above and to the left of bottom-right.")
                    return
                
                # Crop the image
                cropped_img = img.crop((x1, y1, x2, y2))
                
                # Create save directory if it doesn't exist
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                
                # Generate output filename
                base_name = image_path.stem
                output_filename = f"{base_name}_crop_{x1}_{y1}_{x2}_{y2}.jpg"
                output_path = save_path / output_filename
                
                # Save as highest quality JPG
                cropped_img.save(output_path, 'JPEG', quality=100, subsampling=0)
                
                print(f"\nCropped image saved to: {output_path}")
                print(f"Crop size: {x2-x1}x{y2-y1} pixels")
                
            except ValueError as e:
                print(f"Error parsing coordinates: {e}")
                print("Please use format: x,y (e.g., 100,200)")
            except Exception as e:
                print(f"Error cropping/saving image: {e}")
        
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Display an image from a folder at the specified index (sorted lexicographically) and optionally crop it.'
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Path to the folder containing images'
    )
    parser.add_argument(
        'index',
        type=int,
        help='Index of the image to display (0-based)'
    )
    parser.add_argument(
        '--savedir',
        type=str,
        default=None,
        help='Directory to save cropped images (optional)'
    )
    
    args = parser.parse_args()
    
    visualize_image_at_index(args.folder, args.index, args.savedir)


if __name__ == '__main__':
    main()
