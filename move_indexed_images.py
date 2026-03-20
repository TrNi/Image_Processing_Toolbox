import os
import argparse
import shutil
from pathlib import Path


def move_indexed_images(input_dir, move_indices, output_dir):
    """
    Move images at specified indices from input directory to output directory.
    Images are sorted alphabetically before indexing.
    
    Args:
        input_dir (str): Path to directory containing input images
        move_indices (list): List of indices to move (0-based)
        output_dir (str): Path to directory where selected images will be moved
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
    
    # Find all image files (jpg, jpeg, png)
    image_files = []
    for ext in ['*.JPG']: #, '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(input_path.glob(ext)))
    
    if not image_files:
        print(f"No image files found in '{input_dir}'.")
        return
    
    # Sort alphabetically by filename
    image_files.sort(key=lambda x: x.name.lower())
    
    print(f"Found {len(image_files)} image(s) in '{input_dir}\n move indices = {move_indices}")
    print(f"Images sorted alphabetically.")
    for i in range(len(image_files)):
        print(f"{i}: {image_files[i].name}")
    
    # Validate indices
    invalid_indices = [idx for idx in move_indices if idx < 0 or idx >= len(image_files)]
    if invalid_indices:
        print(f"Warning: Invalid indices {invalid_indices} (valid range: 0-{len(image_files)-1})")
        move_indices = [idx for idx in move_indices if 0 <= idx < len(image_files)]
    
    if not move_indices:
        print("No valid indices to move.")
        return
    
    print(f"Moving {len(move_indices)} image(s) to '{output_dir}'...")
    
    # Move files at specified indices
    moved = 0
    failed = 0
    
    for idx in sorted(move_indices):
        img_file = image_files[idx]
        try:
            dest_file = output_path / img_file.name
            
            # Check if file already exists in destination
            if dest_file.exists():
                print(f"Warning: '{img_file.name}' already exists in output directory, skipping.")
                failed += 1
                continue
            
            # Move the file
            shutil.move(str(img_file), str(dest_file))
            print(f"Moved [Index {idx}]: '{img_file.name}'")
            moved += 1
            
        except Exception as e:
            print(f"Error moving '{img_file.name}': {e}")
            failed += 1
    
    print(f"\nComplete! Moved: {moved}, Failed/Skipped: {failed}")
    print(f"Images moved to: '{output_dir}'")


def parse_indices(indices_str):
    """
    Parse indices from string format.
    Supports comma-separated values and ranges (e.g., "0,1,2" or "0-5,10,15-20")
    
    Args:
        indices_str (str): String containing indices
        
    Returns:
        list: List of integer indices
    """
    indices = []
    parts = indices_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range (e.g., "0-5")
            try:
                start, end = map(int, part.split('-'))
                indices.extend(range(start, end + 1))
            except ValueError:
                print(f"Warning: Invalid range '{part}', skipping.")
        else:
            # Handle single index
            try:
                indices.append(int(part))
            except ValueError:
                print(f"Warning: Invalid index '{part}', skipping.")
    
    return indices


def main():
    parser = argparse.ArgumentParser(
        description='Move images at specified indices from input to output directory. '
                    'Images are sorted alphabetically before indexing.'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to directory containing input images'
    )
    parser.add_argument(
        'indices',
        type=str,
        help='Indices to move (comma-separated, supports ranges: e.g., "0,1,2" or "0-5,10")'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to directory where selected images will be moved'
    )    
    
    args = parser.parse_args()
    
    # Parse indices
    move_indices = [3,5,7,8,10,13,15,17,18]
    #parse_indices(args.indices)
    
    if not move_indices:
        print("Error: No valid indices provided.")
        return       
    
    move_indexed_images(args.input_dir, move_indices, args.output_dir)


if __name__ == '__main__':
    main()
