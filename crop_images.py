import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


def crop_images_multiple_regions(image_paths, crop_regions, save_dir, color_mode='rgb'):
    """
    Crop multiple images using multiple crop regions and save them.
    
    Parameters:
    -----------
    image_paths : list of str
        List of paths to JPG images to crop
    crop_regions : list of tuples
        List of crop regions, each as (h_start, h_end, w_start, w_end)
    save_dir : str
        Directory where cropped images and info file will be saved
    color_mode : str
        'rgb' for RGB images (3 channels) or 'gray' for grayscale (1 channel)
    """
    # Validate color mode
    if color_mode not in ['rgb', 'gray']:
        raise ValueError(f"color_mode must be 'rgb' or 'gray', got '{color_mode}'")
    
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Validate that all images exist and have the same dimensions
    print("Validating input images...")
    valid_image_paths = []
    image_dimensions = None
    
    for img_path in image_paths:
        img_path = Path(img_path)
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Load image to check dimensions
        img = Image.open(img_path)
        img_array = np.array(img)
        img_h, img_w = img_array.shape[:2]
        
        if image_dimensions is None:
            image_dimensions = (img_h, img_w)
            print(f"Reference dimensions: {img_h} x {img_w}")
        else:
            if (img_h, img_w) != image_dimensions:
                raise ValueError(
                    f"Image dimension mismatch!\n"
                    f"  Expected: {image_dimensions[0]} x {image_dimensions[1]}\n"
                    f"  Got: {img_h} x {img_w} for {img_path.name}"
                )
        
        valid_image_paths.append(img_path)
        print(f"  {img_path.name}: {img_h} x {img_w}")
    
    if len(valid_image_paths) == 0:
        raise ValueError("No valid images found!")
    
    print(f"\nAll {len(valid_image_paths)} images have matching dimensions: {image_dimensions[0]} x {image_dimensions[1]}\n")
    
    # Prepare text file for crop information
    info_file_path = save_path / "crop_info.txt"
    
    crop_info_list = []
    total_crops = 0
    
    print(f"Processing {len(valid_image_paths)} images with {len(crop_regions)} crop region(s)...")
    print(f"Total crops to generate: {len(valid_image_paths) * len(crop_regions)}")
    print(f"Color mode: {color_mode.upper()}\n")
    
    # Process each image and collect all its crops
    for img_idx, img_path in enumerate(valid_image_paths):
        # Load image
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Convert to grayscale if needed
        if color_mode == 'gray':
            if len(img_array.shape) == 3:
                # Convert RGB to grayscale using standard weights
                img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        
        # Collect all crops for this image
        crops_list = []
        
        for region_idx, (h_start, h_end, w_start, w_end) in enumerate(crop_regions):
            # Crop the image
            cropped_array = img_array[h_start:h_end, w_start:w_end]
            
            # Add channel dimension if grayscale
            if color_mode == 'gray':
                if len(cropped_array.shape) == 2:
                    cropped_array = cropped_array[:, :, np.newaxis]  # Add channel dim
            
            # Convert to channel-first format: (H, W, C) -> (C, H, W)
            if len(cropped_array.shape) == 3:
                cropped_array = np.transpose(cropped_array, (2, 0, 1))
            else:
                # Grayscale without channel dim, add it
                cropped_array = cropped_array[np.newaxis, :, :]
            
            crops_list.append(cropped_array)
        
        # Stack all crops for this image: shape (p, C, H, W)
        crops_batch = np.stack(crops_list, axis=0)
        
        # Save crop batch as numpy array
        crop_batch_filename = f"{img_path.stem}_crops.npy"
        crop_batch_path = save_path / crop_batch_filename
        np.save(crop_batch_path, crops_batch)
        
        print(f"[{img_idx+1}/{len(valid_image_paths)}] Saved crop batch: {crop_batch_filename}")
        print(f"  Shape: {crops_batch.shape} (p={len(crop_regions)}, C={crops_batch.shape[1]}, H={crops_batch.shape[2]}, W={crops_batch.shape[3]})")
        print(f"  Dtype: {crops_batch.dtype}\n")
    
    # Save coordinate arrays (once per crop region)
    print(f"Saving coordinate arrays for {len(crop_regions)} crop region(s)...\n")
    
    coord_arrays_list = []
    
    for region_idx, (h_start, h_end, w_start, w_end) in enumerate(crop_regions):
        print(f"\n--- Crop Region {region_idx + 1}/{len(crop_regions)} ---")
        print(f"Coordinates: H[{h_start}:{h_end}], W[{w_start}:{w_end}]")
        print(f"Crop size: {h_end - h_start} x {w_end - w_start}\n")
        
        # Validate crop coordinates
        if h_end > image_dimensions[0] or w_end > image_dimensions[1] or h_start < 0 or w_start < 0:
            raise ValueError(
                f"Crop coordinates out of bounds for region {region_idx}!\n"
                f"  Image size: {image_dimensions[0]} x {image_dimensions[1]}\n"
                f"  Crop region: H[{h_start}:{h_end}], W[{w_start}:{w_end}]"
            )      
        
        # Create meshgrid of coordinates (pixel coordinates)
        y_coords = np.arange(h_start, h_end)
        x_coords = np.arange(w_start, w_end)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Normalize coordinates to [-1, 1] range with center of image as origin
        # Center of full image
        img_h, img_w = image_dimensions
        center_y = (img_h - 1) / 2.0
        center_x = (img_w - 1) / 2.0
        
        # Normalize: shift to center, then scale to [-1, 1]
        # For y: range is [0, img_h-1], center is (img_h-1)/2
        # Normalized: (y - center_y) / center_y gives range approximately [-1, 1]
        yy_normalized = (yy - center_y) / center_y
        xx_normalized = (xx - center_x) / center_x
        
        # Stack to create coordinate array of shape (1, 1, h_crop, w_crop, 2)
        # where last dimension is [y, x] coordinates (normalized)
        coord_array = np.stack([yy_normalized, xx_normalized], axis=-1)
        coord_array = coord_array[np.newaxis, np.newaxis, :, :, :]  # Add batch and channel dims
        
        # Save coordinate array
        coord_filename = f"coords_region{region_idx}.npy"
        coord_path = save_path / coord_filename
        np.save(coord_path, coord_array)
        print(f"Saved coordinate array: {coord_filename}")
        print(f"  Shape: {coord_array.shape}")
        print(f"  Dtype: {coord_array.dtype}")
        print(f"  Value range: Y[{yy_normalized.min():.4f}, {yy_normalized.max():.4f}], X[{xx_normalized.min():.4f}, {xx_normalized.max():.4f}]\n")
        
        # Store for overall array (remove the first two singleton dimensions)
        coord_arrays_list.append(coord_array[0, 0, :, :, :])  # Shape: (H, W, 2)
        
        for idx, img_path in enumerate(valid_image_paths):
            # Load image
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Crop the image
            cropped_array = img_array[h_start:h_end, w_start:w_end]
            cropped_img = Image.fromarray(cropped_array)
            
            # Generate output filename with region index
            output_filename = f"{img_path.stem}_crop_region{region_idx}.jpg"
            output_path = save_path / output_filename
            
            # Save cropped image
            cropped_img.save(output_path, 'JPEG', quality=95)
            
            # Store crop information
            crop_info = f"{img_path.name},{h_start},{h_end},{w_start},{w_end},{region_idx}"
            crop_info_list.append(crop_info)
            total_crops += 1
            
            print(f"  [{idx+1}/{len(valid_image_paths)}] Saved: {output_filename}")
    
    # Save overall coordinate array with all regions stacked
    overall_coords = np.stack(coord_arrays_list, axis=0)  # Shape: (p, H, W, 2)
    overall_coords_filename = "coords_all_regions.npy"
    overall_coords_path = save_path / overall_coords_filename
    np.save(overall_coords_path, overall_coords)
    
    print(f"\n{'='*60}")
    print(f"Saved overall coordinate array: {overall_coords_filename}")
    print(f"  Shape: {overall_coords.shape} (p={overall_coords.shape[0]}, H={overall_coords.shape[1]}, W={overall_coords.shape[2]}, coords=2)")
    print(f"  Dtype: {overall_coords.dtype}")
    print(f"{'='*60}\n")
    
    # Write crop information to text file
    with open(info_file_path, 'w') as f:
        f.write("image_name,h_start,h_end,w_start,w_end,region_index\n")
        for info in crop_info_list:
            f.write(info + "\n")
    
    print(f"\n{'='*60}")
    print(f"Crop information saved to: {info_file_path}")
    print(f"Successfully created {total_crops} cropped images")
    print(f"Successfully created {len(crop_regions)} coordinate arrays")
    print(f"Successfully created {len(valid_image_paths)} crop batch arrays")
    print(f"Successfully created 1 overall coordinate array")
    print(f"{'='*60}")
    
    return crop_info_list


def main():
    # ========== USER CONFIGURATION ==========
    # Specify your image paths here
    image_paths = [
        # r"I:\My Drive\DoF_Datasets\toy_set\ip_s1left_fl70_F22_IMG_4652.JPG",
        # r"I:\My Drive\DoF_Datasets\toy_set\tgt_s1left_fl70_F2.8_IMG_4660.JPG",
        r"I:\My Drive\DoF_Datasets\toy_set\depth_anything_v2_depths_depth_idx8.JPG",
    ]
    
    # Specify crop regions as list of tuples: (h_start, h_end, w_start, w_end)
    crop_regions = [
        (2175, 2815, 4596, 5236),  # Region 0
        (2175, 2815, 3956, 4596),   # Region 1
        (1535, 2175, 3743, 4383),  # Region 2
    ]
    
    # Specify output directory
    save_dir = r"I:\My Drive\DoF_Datasets\toy_set\cropped_output"
    
    # Specify color mode: 'rgb' or 'gray'
    color_mode = 'gray'  # Change to 'gray' for grayscale
    
    # ========================================
    
    # Process all images with all crop regions
    crop_images_multiple_regions(image_paths, crop_regions, save_dir, color_mode)


if __name__ == '__main__':
    main()