import os
import argparse
from pathlib import Path
from PIL import Image


def resize_images(input_dir, output_dir, target_width, target_height):
    """
    Resize all JPG images from input directory using bilinear interpolation.
    
    Args:
        input_dir (str): Path to directory containing input JPG images
        output_dir (str): Path to directory where resized images will be saved
        target_width (int): Target width for resized images
        target_height (int): Target height for resized images
    """
    # Convert to Path objects
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Validate input directory
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JPG images (case-insensitive)
    jpg_files = list(input_path.glob('*.JPG'))
    #list(input_path.glob('*.jpg')) + list(input_path.glob('*.JPG')) + \
    #list(input_path.glob('*.jpeg')) + list(input_path.glob('*.JPEG'))
    
    if not jpg_files:
        print(f"No JPG images found in '{input_dir}'.")
        return
    
    print(f"Found {len(jpg_files)} JPG image(s) to process.")
    print(f"Target size: {target_width}x{target_height}")
    
    # Process each image
    processed = 0
    skipped = 0
    
    for img_file in jpg_files:
        try:
            # Open image
            img = Image.open(img_file)
            original_width, original_height = img.size
            
            # Resize image using bilinear interpolation (BILINEAR)
            resized_img = img.resize((target_width, target_height), Image.BILINEAR)
            
            # Save with same name in output directory
            save_file = output_path / img_file.name
            resized_img.save(save_file, quality=100)
            
            print(f"Processed: '{img_file.name}' ({original_width}x{original_height} -> {target_width}x{target_height})")
            processed += 1
            
        except Exception as e:
            print(f"Error processing '{img_file.name}': {e}")
            skipped += 1
    
    print(f"\nComplete! Processed: {processed}, Skipped: {skipped}")
    print(f"Resized images saved to: '{output_dir}'")


def main():
    parser = argparse.ArgumentParser(
        description='Resize JPG images using bilinear interpolation.'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to directory containing input JPG images'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to directory where resized images will be saved'
    )
    parser.add_argument(
        '--width',
        type=int,
        required=True,
        help='Target width for resized images'
    )
    parser.add_argument(
        '--height',
        type=int,
        required=True,
        help='Target height for resized images'
    )
    
    args = parser.parse_args()
    
    resize_images(args.input_dir, args.output_dir, args.width, args.height)


if __name__ == '__main__':
    main()
