import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np


def illustrate_h5(h5_file_path, dataset_key, indices):
    """
    Display images from an HDF5 file at specified indices as interactive plots.
    
    Args:
        h5_file_path (str): Path to the HDF5 file
        dataset_key (str): Key/name of the dataset in the HDF5 file
        indices (list): List of indices to display
    """
    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            # Check if dataset exists
            if dataset_key not in h5_file:
                print(f"Error: Dataset '{dataset_key}' not found in HDF5 file.")
                print(f"Available datasets: {list(h5_file.keys())}")
                return
            
            dataset = h5_file[dataset_key]
            dataset_length = len(dataset)
            
            # Validate indices
            valid_indices = []
            for idx in indices:
                if 0 <= idx < dataset_length:
                    valid_indices.append(idx)
                else:
                    print(f"Warning: Index {idx} is out of range (dataset length: {dataset_length}). Skipping.")
            
            if not valid_indices:
                print("Error: No valid indices to display.")
                return
            
            # Determine subplot layout
            num_images = len(valid_indices)
            cols = min(3, num_images)  # Max 3 columns
            rows = (num_images + cols - 1) // cols  # Ceiling division
            
            # Create figure with subplots
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            
            # Handle single subplot case
            if num_images == 1:
                axes = np.array([axes])
            axes = axes.flatten() if num_images > 1 else axes
            
            # Display images
            for i, idx in enumerate(valid_indices):
                image = dataset[idx]
                
                # Display image without normalization to preserve original values (e.g., depth maps)
                if len(image.shape) == 2:  # Grayscale
                    axes[i].imshow(image, cmap='gray')
                else:  # Color (RGB or RGBA)
                    axes[i].imshow(image)
                
                axes[i].set_title(f'Index: {idx}\nShape: {image.shape}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_images, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
            
    except FileNotFoundError:
        print(f"Error: File '{h5_file_path}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Display images from an HDF5 file at specified indices as interactive plots.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python illustrate_h5.py data.h5 images 0 5 10
  python illustrate_h5.py data.h5 train_images 0 1 2 3 4
        """
    )
    
    parser.add_argument('h5_file', type=str, help='Path to the HDF5 file')
    parser.add_argument('dataset_key', type=str, help='Dataset key/name in the HDF5 file')
    parser.add_argument('indices', type=int, nargs='+', help='Indices of images to display (space-separated)')
    
    args = parser.parse_args()
    
    illustrate_h5(args.h5_file, args.dataset_key, args.indices)


if __name__ == '__main__':
    main()
