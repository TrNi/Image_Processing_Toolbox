import argparse
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


def extract_and_save_image(h5_path, dataset_key, image_index, save_dir):
    """
    Extract an image from an H5 file and save it as JPG.
    
    Parameters:
    -----------
    h5_path : str
        Path to the H5 file
    dataset_key : str
        Key for the dataset in the H5 file
    image_index : int
        Index of the image in the dataset
    save_dir : str
        Directory where the JPG will be saved
    """
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Read the image from H5 file
    with h5py.File(h5_path, 'r') as f:
        if dataset_key not in f:
            raise KeyError(f"Dataset key '{dataset_key}' not found in H5 file. Available keys: {list(f.keys())}")
        
        dataset = f[dataset_key]
        
        if image_index >= len(dataset):
            raise IndexError(f"Image index {image_index} out of range. Dataset has {len(dataset)} images.")
        print("\n\n\n")
        print(dataset.shape)        
        print("\n\n\n")
        print(dataset)
        # Read the image
        image_data = dataset[image_index]
    
    # Store original data for info
    original_dtype = image_data.dtype
    original_min = image_data.min()
    original_max = image_data.max()
    
    print(f"Original data info:")
    print(f"  Shape: {image_data.shape}")
    print(f"  Dtype: {original_dtype}")
    print(f"  Min value: {original_min}")
    print(f"  Max value: {original_max}")
    
    # Convert to appropriate format for PIL
    # Handle different data types and ranges
    if image_data.dtype == np.float32 or image_data.dtype == np.float64:
        # Assume float images are in range [0, 1]
        if original_max <= 1.0:
            image_data = (image_data * 255).astype(np.uint8)
        else:
            # Normalize to 0-255 range
            image_data = ((image_data - original_min) / (original_max - original_min) * 255).astype(np.uint8)
    elif image_data.dtype != np.uint8:
        # For integer types (like depth maps), normalize to 0-255 range
        if original_max > 255 or original_max < 50:  # Likely needs normalization
            print(f"  Normalizing to 0-255 range...")
            image_data = ((image_data - original_min) / (original_max - original_min) * 255).astype(np.uint8)
        else:
            image_data = image_data.astype(np.uint8)
    
    # Create PIL Image
    if len(image_data.shape) == 2:
        # Grayscale image
        img = Image.fromarray(image_data, mode='L')
    elif len(image_data.shape) == 3:
        if image_data.shape[2] == 3:
            # RGB image
            img = Image.fromarray(image_data, mode='RGB')
        elif image_data.shape[2] == 4:
            # RGBA image
            img = Image.fromarray(image_data, mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels: {image_data.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {image_data.shape}")
    
    # Visualize the image before saving
    plt.figure(figsize=(10, 8))
    if len(image_data.shape) == 2:
        plt.imshow(image_data, cmap='gray')
        plt.colorbar(label='Normalized pixel value (0-255)')
    else:
        plt.imshow(image_data)
    plt.title(f"Image from {Path(h5_path).name}\nDataset: {dataset_key}, Index: {image_index}\nOriginal range: [{original_min}, {original_max}]")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Generate output filename
    h5_filename = Path(h5_path).stem
    output_filename = f"{h5_filename}_{dataset_key}_idx{image_index}.jpg"
    output_path = save_path / output_filename
    
    # Save as JPG
    img.save(output_path, 'JPEG', quality=95)
    
    print(f"Image saved to: {output_path}")
    print(f"Image shape: {image_data.shape}")
    print(f"Image dtype: {image_data.dtype}")
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Extract an image from an H5 file and save as JPG')
    parser.add_argument('h5_path', type=str, help='Path to the H5 file')
    parser.add_argument('dataset_key', type=str, help='Key for the dataset in the H5 file')
    parser.add_argument('image_index', type=int, help='Index of the image in the dataset')
    parser.add_argument('save_dir', type=str, help='Directory where the JPG will be saved')
    
    args = parser.parse_args()
    
    extract_and_save_image(args.h5_path, args.dataset_key, args.image_index, args.save_dir)


if __name__ == '__main__':
    main()
