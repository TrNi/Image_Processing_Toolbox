import os
import numpy as np
import h5py

def combine_and_save_npy_to_hdf5(dir_path, output_h5="images_"):
    """
    Reads left_rectified_{0..5}.npy files in dir_path, stacks them into (6,C,H,W),
    and writes to an HDF5 file with dataset 'left'.
    """
    images = []
    for i in range(6):
        npy_file = os.path.join(dir_path, f"left_rectified_{i}.npy")
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"Expected file not found: {npy_file}")
        
        img = np.load(npy_file)  # shape: (C, H, W)
        if img.ndim != 3:
            raise ValueError(f"File {npy_file} does not have shape (C,H,W); got {img.shape}")
        images.append(img)
    
    stacked = np.stack(images, axis=0)  # shape: (6, C, H, W)
    
    # Save to HDF5
    output_path = os.path.join(dir_path, output_h5)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("left", data=stacked)
    
    print(f"Saved combined array with shape {stacked.shape} to {output_path}")

# Example usage:
combine_and_save_npy_to_hdf5("I:\\My Drive\\scene5-f28.0mm-a1.4mm\\stereocal_results_f28.0mm_a1.4mm\\rectified")

pass