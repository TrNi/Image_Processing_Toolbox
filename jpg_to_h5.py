import os
import argparse
from pathlib import Path
import numpy as np
import h5py
from PIL import Image


def jpg_to_h5(input_dir, output_file):
    """
    Convert all JPG images from input directory to a single HDF5 file.
    Images are stored with key "images" and preserve their original data type and scale.
    
    Args:
        input_dir (str): Path to directory containing input JPG images
        output_file (str): Path to output HDF5 file
    """
    # Convert to Path objects
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    # Validate input directory
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        return
    
    # Find all JPG images (case-insensitive)
    jpg_files = sorted(list(input_path.glob('*.jpg')))# + list(input_path.glob('*.JPG')) + 
                       #list(input_path.glob('*.jpeg')) + list(input_path.glob('*.JPEG')))
    
    if not jpg_files:
        print(f"No JPG images found in '{input_dir}'.")
        return
    
    print(f"Found {len(jpg_files)} JPG image(s) to convert.")
    
    # Load all images into a list
    images_list = []
    filenames = []
    
    for img_file in jpg_files:
        try:
            # Open image and convert to numpy array
            # This preserves the original data type (typically uint8 for JPG)
            img = Image.open(img_file)
            img_array = np.array(img)
            
            images_list.append(img_array)
            filenames.append(img_file.name)
            
            print(f"Loaded: '{img_file.name}' - Shape: {img_array.shape}, Dtype: {img_array.dtype}")
            
        except Exception as e:
            print(f"Error loading '{img_file.name}': {e}")
    
    if not images_list:
        print("No images were successfully loaded.")
        return
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to HDF5 file
    try:
        with h5py.File(output_path, 'w') as hf:
            # Check if all images have the same shape
            shapes = [img.shape for img in images_list]
            if len(set(shapes)) == 1:
                # All images have the same shape - store as a single array
                images_array = np.stack(images_list, axis=0)
                hf.create_dataset('images', data=images_array, compression='gzip')
                print(f"\nAll images have the same shape {shapes[0]}.")
                print(f"Stored as array with shape: {images_array.shape}, "
                f"dtype: {images_array.dtype}, "
                f"min: {images_array.min()}, "
                f"max: {images_array.max()}")
            else:
                # Images have different shapes
                print(f"\nImages have different shapes. not processing.")                
            
            # Store filenames as metadata
            hf.create_dataset('filenames', data=np.array(filenames, dtype='S'))
            
        print(f"\nSuccessfully saved {len(images_list)} images to '{output_file}'")
        print(f"HDF5 key: 'images'")
        
    except Exception as e:
        print(f"Error saving to HDF5: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert JPG images to HDF5 format while preserving original data type and scale.'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to directory containing input JPG images'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output HDF5 file (e.g., images.h5)'
    )
    
    args = parser.parse_args()
    
    jpg_to_h5(args.input_dir, args.output_file)


if __name__ == '__main__':
    main()
