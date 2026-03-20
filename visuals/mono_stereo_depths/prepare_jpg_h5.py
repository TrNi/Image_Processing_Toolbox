import h5py
import os
import numpy as np

stereonames = ["monster", "foundation", "defom", "selective"]
mononames = ["depthpro", "metric3d", "unidepth", "anything"]


def process_h5_pair(left_path, left_id, right_path, right_id, output_folder, stereo_folder=None, mono_folder=None):
    """
    Process a pair of H5 files and save specific indices as new H5 files.
    
    Args:
        left_path (str): Path to the left H5 file
        left_id (int): Index to extract from left H5 file
        right_path (str): Path to the right H5 file
        right_id (int): Index to extract from right H5 file
        output_folder (str): Path to the output folder
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder+"\\monodepth", exist_ok=True)    
    
    # Read and process left image
    with h5py.File(left_path, 'r') as f:
        # Assuming the dataset name is 'rectified_lefts' and shape is NxCxHxW
        left_data = f['rectified_lefts'][left_id:left_id+1]  # Get 1xCxHxW slice
    
    # Read and process right image
    with h5py.File(right_path, 'r') as f:
        # Assuming the dataset name is 'rectified_lefts' and shape is NxCxHxW
        right_data = f['rectified_rights'][right_id:right_id+1]  # Get 1xCxHxW slice
    
    # Save left image
    left_h5_path = os.path.join(output_folder, "_left.h5")
    with h5py.File(left_h5_path, 'w') as f:
        f.create_dataset("rectified_lefts", data=left_data)
    
    # Save right image
    right_h5_path = os.path.join(output_folder, "_right.h5")
    with h5py.File(right_h5_path, 'w') as f:
        f.create_dataset("rectified_rights", data=right_data)

    if stereo_folder is not None:
        filelist = [f for f in os.listdir(stereo_folder) if f.endswith('.h5')]
        for file in filelist:
            if any(name in file for name in stereonames):
                with h5py.File(os.path.join(stereo_folder, file), 'r') as f:
                    # Assuming the dataset name is 'rectified_lefts' and shape is NxCxHxW
                    depth_data = f['depth'][left_id:left_id+1]  # Get 1xCxHxW slice
                
                depth_h5_path = os.path.join(output_folder, file)
                with h5py.File(depth_h5_path, 'w') as f:
                    f.create_dataset("depth", data=depth_data)

    if mono_folder is not None:
        filelist = [f for f in os.listdir(mono_folder) if f.endswith('.h5')]
        for file in filelist:
            if any(name in file for name in mononames):
                with h5py.File(os.path.join(mono_folder, file), 'r') as f:
                    # Assuming the dataset name is 'rectified_lefts' and shape is NxCxHxW
                    depth_data = f['depth'][left_id:left_id+1]  # Get 1xCxHxW slice
                
                depth_h5_path = os.path.join(output_folder+"\\monodepth", file)
                if "anything" in file:
                    depth_h5_path = depth_h5_path.replace(".h5", "_dav2.h5")
                with h5py.File(depth_h5_path, 'w') as f:
                    f.create_dataset("depth", data=depth_data)                
            

def process_multiple_pairs(pairs_dict):
    """
    Process multiple pairs of H5 files from a dictionary.
    
    Args:
        pairs_dict (dict): Dictionary containing entries with:
            - left_path: path to left H5 file
            - left_id: index to extract from left file
            - right_path: path to right H5 file
            - right_id: index to extract from right file
            - output_folder: output directory path
    """
    for pair_id, pair_info in pairs_dict.items():
        left_path = pair_info['left_path']
        left_id = pair_info['left_id']
        right_path = pair_info['right_path']
        right_id = pair_info['right_id']
        output_folder = pair_info['output_folder']
        stereo_folder = pair_info['stereo_folder']
        mono_folder = pair_info['mono_folder']
        
        try:
            process_h5_pair(left_path, left_id, right_path, right_id, output_folder, stereo_folder = stereo_folder, mono_folder = mono_folder)
            print(f"Successfully processed pair {pair_id}")
        except Exception as e:
            print(f"Error processing pair {pair_id}: {str(e)}")

# Example usage:
if __name__ == "__main__":
    # Example dictionary structure
    pairs = {
        "pair1": {
            "left_path": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\rectified\\rectified_lefts.h5",
            "left_id": 2,
            "right_path": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_B_Right\\fl_70mm\\inference\\F16.0\\rectified\\rectified_rights.h5",
            "right_id": 2,
            "output_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\illusions_used_S6_3",
            "stereo_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\rectified",
            "mono_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\monodepth"
        },
        "pair2": {
            "left_path": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\rectified\\rectified_lefts.h5",
            "left_id":17,
            "right_path": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_B_Right\\fl_70mm\\inference\\F16.0\\rectified\\rectified_rights.h5",
            "right_id": 17,
            "output_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\illusions_used_S6_16",
            "stereo_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\rectified",
            "mono_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\monodepth"
        },
        "pair3": {
            "left_path": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\rectified\\rectified_lefts.h5",
            "left_id": 16,
            "right_path": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_B_Right\\fl_70mm\\inference\\F16.0\\rectified\\rectified_rights.h5",
            "right_id": 16,
            "output_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\illusions_used_S6_15",
            "stereo_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\rectified",
            "mono_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\monodepth"
        },
        "pair4": {
            "left_path": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\rectified\\rectified_lefts.h5",
            "left_id": 9,
            "right_path": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_B_Right\\fl_70mm\\inference\\F16.0\\rectified\\rectified_rights.h5",
            "right_id": 9,
            "output_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\illusions_used_S6_8",
            "stereo_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\rectified",
            "mono_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\monodepth"
        },
        "pair5": {
            "left_path": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\rectified\\rectified_lefts.h5",
            "left_id": 11,
            "right_path": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_B_Right\\fl_70mm\\inference\\F16.0\\rectified\\rectified_rights.h5",
            "right_id": 11,
            "output_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\illusions_used_S6_10",
            "stereo_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\rectified",
            "mono_folder": "I:\\My Drive\\Pubdata\\Scene6_illusions\\EOS6D_A_Left\\fl_70mm\\inference\\F16.0\\monodepth"
        }
        # "pair1": {
        #     "left_path": "I:\\My Drive\\Pubdata\\Scene9\\EOS6D_B_Left\\fl_40mm\\inference\\F2.8\\rectified\\rectified_lefts.h5",
        #     "left_id": 7,
        #     "right_path": "I:\\My Drive\\Pubdata\\Scene9\\EOS6D_A_Right\\fl_40mm\\inference\\F2.8\\rectified\\rectified_rights.h5",
        #     "right_id": 7,
        #     "output_folder": "I:\\My Drive\\Pubdata\\illusion_crops_new\\S9_f40_img_3901"
        # },
        # "pair2": {
        #     "left_path": "I:\\My Drive\\Pubdata\\Scene9\\EOS6D_B_Left\\fl_45mm\\inference\\F2.8\\rectified\\rectified_lefts.h5",
        #     "left_id": 2,
        #     "right_path": "I:\\My Drive\\Pubdata\\Scene9\\EOS6D_A_Right\\fl_45mm\\inference\\F2.8\\rectified\\rectified_rights.h5",
        #     "right_id": 2,
        #     "output_folder": "I:\\My Drive\\Pubdata\\illusion_crops_new\\S9_f45_img_3962"
        # },
        # "pair3": {
        #     "left_path": "I:\\My Drive\\Pubdata\\Scene9\\EOS6D_B_Left\\fl_45mm\\inference\\F2.8\\rectified\\rectified_lefts.h5",
        #     "left_id": 15,
        #     "right_path": "I:\\My Drive\\Pubdata\\Scene9\\EOS6D_A_Right\\fl_45mm\\inference\\F2.8\\rectified\\rectified_rights.h5",
        #     "right_id": 15,
        #     "output_folder": "I:\\My Drive\\Pubdata\\illusion_crops_new\\S9_f45_img_4023"
        # },
        # "pair4": {
        #     "left_path": "I:\\My Drive\\Pubdata\\Scene9\\EOS6D_B_Left\\fl_60mm\\inference\\F2.8\\rectified\\rectified_lefts.h5",
        #     "left_id": 8,
        #     "right_path": "I:\\My Drive\\Pubdata\\Scene9\\EOS6D_A_Right\\fl_60mm\\inference\\F2.8\\rectified\\rectified_rights.h5",
        #     "right_id": 8,
        #     "output_folder": "I:\\My Drive\\Pubdata\\illusion_crops_new\\S9_f60_img_4378"
        # },
        # "pair5": {
        #     "left_path": "I:\\My Drive\\Pubdata\\Scene9\\EOS6D_B_Left\\fl_60mm\\inference\\F2.8\\rectified\\rectified_lefts.h5",
        #     "left_id": 10,
        #     "right_path": "I:\\My Drive\\Pubdata\\Scene9\\EOS6D_A_Right\\fl_60mm\\inference\\F2.8\\rectified\\rectified_rights.h5",
        #     "right_id": 10,
        #     "output_folder": "I:\\My Drive\\Pubdata\\illusion_crops_new\\S9_f60_img_4380"
        # }
        # Add more pairs as needed
    }
    
    process_multiple_pairs(pairs)