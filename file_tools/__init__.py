"""
file_tools — file format conversion and data management utilities
=================================================================

Modules
-------
extract_from_h5   Extract individual images from HDF5 datasets and save as JPEG/PNG.
illustrate_h5     Display multiple images from an HDF5 dataset interactively.
jpg_to_h5         Pack a directory of JPEG images into a single HDF5 file.
merge_h5          Concatenate multiple HDF5 files along the batch dimension.
npy_to_h5         Combine a sequence of .npy arrays into a single HDF5 file.
npy_to_npz        Convert a .npy dict-pickle file to compressed .npz format.
pkl_to_h5         Convert gzip-pickle error files to HDF5.
create_csv        Build a CSV index from one or more image directories with URL prefixes.
move_images       Move images at specified indices from one directory to another.
organise_files    Reorganise files from a calibration source tree into a processing tree.
get_gdrive_ids    Resolve Google Drive file IDs from local mirrored paths.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
