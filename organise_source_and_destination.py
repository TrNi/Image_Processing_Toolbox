import os
import shutil
from pathlib import Path


def move_non_rectified_files(source_dir, destination_dir, cameras, focal_lengths):
    """
    Move files from source calibration folders to destination, excluding rectified_*.h5 files.
    
    Args:
        source_dir (str): Source directory path
        destination_dir (str): Destination directory path
        cameras (list): List of camera names (e.g., ['left_camera', 'right_camera'])
        focal_lengths (list): List of focal length folders (e.g., ['fl_28mm', 'fl_35mm'])
    """
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    
    # Check if source directory exists
    if not source_path.exists():
        print(f"Skipping source directory {source_dir}: does not exist.")
        return
    
    files_moved = 0
    Fnums = ["F2.8", "F5.0", "F9.0", "F16.0", "F22.0"]
    # Iterate through each camera
    for camera in cameras:
        # Iterate through each focal length
        for focal_length in focal_lengths:
            # Construct the source folder path
            #source_folder = source_path / camera / focal_length / "calibration" / "rectified"
            for Fnum in Fnums:
              source_folder = source_path / camera / focal_length / "Output" # "inference" / Fnum
            
              # Check if source folder exists
              if not source_folder.exists():
                  print(f"Skipping: does not exist {source_folder}")
                  continue
              
              print(f"\nProcessing: {source_folder}")
              
              # Get all files or subdirs in the source folder
              try:
                  files = [f for f in source_folder.iterdir()] # if f.is_file()]
              except PermissionError:
                  print(f"Skipping: Permission denied accessing {source_folder}")
                  continue
              
              # Filter out rectified_*.h5 files
              files_to_move = [f for f in files if not f.name.startswith("rectified_") or not f.name.endswith(".h5")]
              
              if not files_to_move:
                  print(f"Skipping: No files to move: only rectified_*.h5 files found")
                  continue
              
              # Create destination folder with identical structure
              dest_folder = dest_path / camera / focal_length / "inference" / "mono"
              dest_folder.mkdir(parents=True, exist_ok=True)
              
              # Move each file / folder
              for file in files_to_move:
                  dest_file = dest_folder / file.name
                  try:
                      shutil.move(str(file), str(dest_file))
                      print(f"Moved: {file.name} -> {dest_folder}")
                      files_moved += 1
                  except Exception as e:
                      print(f"Skipping: error moving {file.name}: {e}")
      
    print(f"\n{'='*60}")
    print(f"Total files moved: {files_moved}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Configuration
    scene = 9
    SOURCE_DIR = r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene"+str(scene)
    DESTINATION_DIR = r"I:\My Drive\Pubdata\Results\Scene"+str(scene)  
    
    # Define cameras and focal lengths
    CAMERAS = [
        "EOS6D_B_Left",
        "EOS6D_A_Right"
        # "EOS6D_A_Left",
        # "EOS6D_B_Right"
    ]
    
    FOCAL_LENGTHS = [
        "fl_28mm",
        "fl_32mm",
        "fl_36mm",
        "fl_40mm",
        "fl_45mm",
        "fl_50mm",
        "fl_55mm",
        "fl_60mm",
        "fl_65mm",
        "fl_70mm"
    ]
    
    # Run the function
    print("Starting file organization...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DESTINATION_DIR}")
    print(f"Cameras: {CAMERAS}")
    print(f"Focal Lengths: {FOCAL_LENGTHS}")
    print("="*60)
    
    move_non_rectified_files(SOURCE_DIR, DESTINATION_DIR, CAMERAS, FOCAL_LENGTHS)