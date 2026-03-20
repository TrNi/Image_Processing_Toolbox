import os
import argparse
from pathlib import Path
from PIL import Image


def split_into_quadrants(input_dir, save_dir):
    """
    Split all JPG images from input directory into four quadrants.
    
    Args:
        input_dir (str): Path to directory containing input JPG images
        save_dir (str): Path to directory where quadrant images will be saved
    """
    # Convert to Path objects
    input_path = Path(input_dir)
    save_path = Path(save_dir)
    
    # Validate input directory
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        return
    
    # Create save directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JPG images (case-insensitive)
    jpg_files = list(input_path.glob('*.jpg')) 
    #+ list(input_path.glob('*.JPG')) + \
    #           list(input_path.glob('*.jpeg')) + list(input_path.glob('*.JPEG'))
    #
    if not jpg_files:
        print(f"No JPG images found in '{input_dir}'.")
        return
    
    print(f"Found {len(jpg_files)} JPG image(s) to process.")
    
    # Process each image
    processed = 0
    skipped = 0
    
    for img_file in jpg_files:
        try:
            # Open image
            img = Image.open(img_file)
            width, height = img.size
            
            # Calculate quadrant dimensions
            half_width = width // 2
            half_height = height // 2
            
            # Get base filename without extension
            base_name = img_file.stem
            extension = img_file.suffix
            
            # Define quadrants (left, top, right, bottom)
            quadrants = [
                (0, 0, half_width, half_height),              # Top-left: _0
                (half_width, 0, width, half_height),          # Top-right: _1
                (0, half_height, half_width, height),         # Bottom-left: _2
                (half_width, half_height, width, height)      # Bottom-right: _3
            ]
            
            # Crop and save each quadrant
            for idx, coords in enumerate(quadrants):
                quadrant_img = img.crop(coords)
                
                # Create filename with suffix
                save_file = save_path / f"{base_name}_{idx}{extension}"
                quadrant_img.save(save_file, quality=100)
            
            print(f"Processed: '{img_file.name}' ({width}x{height}) -> 4 quadrants ({half_width}x{half_height} each)")
            processed += 1
            
        except Exception as e:
            print(f"Error processing '{img_file.name}': {e}")
            skipped += 1
    
    print(f"\nComplete! Processed: {processed}, Skipped: {skipped}")
    print(f"Total quadrant images created: {processed * 4}")
    print(f"Quadrant images saved to: '{save_dir}'")


def main():
    parser = argparse.ArgumentParser(
        description='Split JPG images into four quadrants with _0, _1, _2, _3 suffixes.'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to directory containing input JPG images'
    )
    parser.add_argument(
        'save_dir',
        type=str,
        help='Path to directory where quadrant images will be saved'
    )
    
    args = parser.parse_args()
    
    split_into_quadrants(args.input_dir, args.save_dir)


if __name__ == '__main__':
    main()
