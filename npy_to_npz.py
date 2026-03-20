import numpy as np

# Input and output file paths
# input_file = r"J:\\My Drive\\scene5-f28.0mm-a2.0mm\\stereocal_results_f28.0mm_a2.0mm\\stereocal_params.npy"
input_file = r"J:\\My Drive\\scene5-f28.0mm-a5.0mm\\stereocal_results_f28.0mm_a5.0mm\\stereocal_params.npy"
output_file = input_file.replace('.npy', '.npz')

try:
    # Load the .npy file
    data = np.load(input_file, allow_pickle=True).item()
    
    # Save as .npz
    np.savez_compressed(output_file, **data)
    print(f"Successfully converted {input_file} to {output_file}")
    
    # Print the keys in the dictionary for verification
    print("\n Dictionary keys in the saved file:")
    with np.load(output_file, allow_pickle=True) as npz_file:
        for key in npz_file.files:
            print(f"- {key}")
            
except Exception as e:
    print(f"An error occurred: {str(e)}")