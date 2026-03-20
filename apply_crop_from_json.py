import cv2
import numpy as np
import json
from pathlib import Path
import argparse


def apply_crop_from_json(
    json_dir: str,
    source_dirs: list[str],
    output_suffix: str = "_cropped"
):
    """
    Apply cropping from alignment JSON files to multiple source directories.
    
    Args:
        json_dir: Directory containing alignment JSON files with crop parameters
        source_dirs: List of source image directories to process
        output_suffix: Suffix to append to output directory names (default: "_cropped")
    """
    json_path = Path(json_dir)
    
    # Get all JSON files and sort lexicographically
    json_files = sorted([f for f in json_path.glob("*.json")])
    
    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    print(f"Found {len(json_files)} JSON files in {json_dir}")
    
    # Process each source directory
    for src_dir in source_dirs:
        src_path = Path(src_dir)
        
        if not src_path.exists():
            print(f"WARNING: Source directory does not exist: {src_dir}")
            continue
        
        # Get image files and sort lexicographically
        image_files = sorted([f for f in src_path.glob("*") 
                            if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']])
        
        if len(image_files) == 0:
            print(f"WARNING: No images found in {src_dir}")
            continue
        
        # Check if number of images matches number of JSON files
        if len(image_files) != len(json_files):
            print(f"WARNING: Number mismatch in {src_dir}: {len(image_files)} images vs {len(json_files)} JSON files")
            print(f"  Will process min({len(image_files)}, {len(json_files)}) pairs")
        
        # Create output directory
        output_dir = src_path.parent / f"{src_path.name}{output_suffix}"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Processing: {src_dir}")
        print(f"Output to:  {output_dir}")
        print(f"{'='*60}")
        
        # Process each image-json pair
        num_pairs = min(len(image_files), len(json_files))
        
        for idx in range(num_pairs):
            img_path = image_files[idx]
            json_file = json_files[idx]
            
            # Load JSON
            try:
                with open(json_file, 'r') as f:
                    alignment_data = json.load(f)
            except Exception as e:
                print(f"  ERROR loading {json_file.name}: {e}")
                continue
            
            # Extract crop parameters
            try:
                x0 = alignment_data["crop_x0"]
                y0 = alignment_data["crop_y0"]
                x1 = alignment_data["crop_x1"]
                y1 = alignment_data["crop_y1"]
            except KeyError as e:
                print(f"  ERROR: Missing crop parameter in {json_file.name}: {e}")
                continue
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ERROR loading image: {img_path.name}")
                continue
            
            # Apply crop
            img_cropped = img[y0:y1, x0:x1]
            
            # Save cropped image as JPG with 100% quality
            output_path = output_dir / f"{img_path.stem}.jpg"
            cv2.imwrite(str(output_path), img_cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            print(f"  [{idx+1}/{num_pairs}] {img_path.name} -> {output_path.name} "
                  f"(crop: {x1-x0}x{y1-y0})")
        
        print(f"Completed: {output_dir}")
    
    print(f"\n{'='*60}")
    print("All directories processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply cropping from alignment JSON files to multiple source directories."
    )
    parser.add_argument("json_dir", 
                       help="Directory containing alignment JSON files with crop parameters")
    parser.add_argument("source_dirs", nargs='+',
                       help="One or more source image directories to process")
    parser.add_argument("--output_suffix", default="_cropped",
                       help="Suffix to append to output directory names (default: _cropped)")
    
    args = parser.parse_args()
    
    apply_crop_from_json(
        args.json_dir,
        args.source_dirs,
        output_suffix=args.output_suffix
    )
