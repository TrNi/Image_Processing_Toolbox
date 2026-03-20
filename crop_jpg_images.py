import os
import argparse
from pathlib import Path
from PIL import Image


def crop_images(input_dir, save_dir, crop_width=5472):
    """
    Crop all JPG images from input directory to full height and specified width.
    
    Args:
        input_dir (str): Path to directory containing input JPG images
        save_dir (str): Path to directory where cropped images will be saved
        crop_width (int): Width to crop images to (default: 5472)
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
    #            list(input_path.glob('*.jpeg')) + list(input_path.glob('*.JPEG'))
    
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
            
            # Check if image is wide enough
            if width < crop_width:
                print(f"Warning: '{img_file.name}' is only {width}px wide, skipping (needs at least {crop_width}px).")
                skipped += 1
                continue
            
            # Crop image (left, top, right, bottom)
            # Crop from left edge, full height, to crop_width
            cropped_img = img.crop((0, 0, crop_width, height))
            
            # Save with same name in save directory
            imgname = img_file.name
            imgnameid = imgname.find("vs")
            if imgnameid==-1:
                imgnameid=0
            else:
                imgnameid += 3

            save_file = save_path / imgname[imgnameid:]
            cropped_img.save(save_file, quality=100)
            
            print(f"Processed: '{img_file.name}' ({width}x{height} -> {crop_width}x{height})")
            processed += 1
            
        except Exception as e:
            print(f"Error processing '{img_file.name}': {e}")
            skipped += 1
    
    print(f"\nComplete! Processed: {processed}, Skipped: {skipped}")
    print(f"Cropped images saved to: '{save_dir}'")


def main():
    parser = argparse.ArgumentParser(
        description='Crop JPG images to full height and specified width.'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to directory containing input JPG images'
    )
    parser.add_argument(
        'save_dir',
        type=str,
        help='Path to directory where cropped images will be saved'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=5472,
        help='Width to crop images to (default: 5472)'
    )
    
    args = parser.parse_args()
    
    crop_images(args.input_dir, args.save_dir, args.width)


if __name__ == '__main__':
    main()
