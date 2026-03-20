

import h5py
import numpy as np
import os
from glob import glob



def concatenate_h5_files(file_pattern, output_file):
    # Get all matching files
    files = sorted(glob(file_pattern))
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    print(f"Found {len(files)} files to concatenate")
    
    # First pass: collect shapes and determine output shape
    shapes = []
    for f in files:
        with h5py.File(f, 'r') as hf:
            # Get first dataset (assuming one dataset per file)
            dset = next(iter(hf.values()))
            data = dset[()]  # Load data
            
            # Handle both (H,W,C) and (1,H,W,C) cases
            if data.ndim == 3:
                data = data[None, ...]  
            shapes.append(data.shape)
    
    # Check all shapes match except for batch dimension
    if len(set([s[1:] for s in shapes])) != 1:
        raise ValueError("All input arrays must have the same shape except for the first dimension")
    
    # Pre-allocate output array
    total_samples = sum(s[0] for s in shapes)
    output_shape = (len(shapes),)  + shapes[0][1:]
    output_data = np.zeros(output_shape, dtype=np.float32)
    print(output_shape, output_data.shape)
    # Second pass: concatenate data
    idx = 0
    for i, (f, shape) in enumerate(zip(files, shapes)):
        if i>=5:
            break
        
        with h5py.File(f, 'r') as hf:
            dset = next(iter(hf.values()))
            data = dset[()]
            if data.ndim == 3:
                data = data[None, ...]
            
            output_data[idx:idx+shape[0]] = data
            idx += shape[0]
      
    # Save concatenated data
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('data', data=output_data,compression='gzip')
    print(f"Saved concatenated data to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Concatenate HDF5 files')
    rdir = "J:\\My Drive\\scene5-f28.0mm-a2.0mm\\stereocal_results_f28.0mm_a2.0mm\\rectified_h5"
    # rdir = "J:\\My Drive\\scene5-f28.0mm-a5.0mm\\stereocal_results_f28.0mm_a5.0mm\\rectified_h5"
    parser.add_argument('--input-dir', type=str, default=rdir,
                        help='Directory containing input HDF5 files')
    parser.add_argument('--output-dir', type=str, default=rdir,
                        help='Directory to save output HDF5 files')
    
    args = parser.parse_args()
    
    # Process left and right files
    for prefix in ['left', 'right']:
        file_pattern = os.path.join(args.input_dir, f'{prefix}_*.h5')
        output_file = os.path.join(args.output_dir, f'concatenated_5_{prefix}.h5')
        concatenate_h5_files(file_pattern, output_file)



