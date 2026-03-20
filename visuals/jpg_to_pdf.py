import os
from PIL import Image
import numpy as np
import h5py
import rawpy


def trim_and_save_as_pdf(image_path, row_trim_per_side, col_trim_per_side, pdf_dpi, save_path):
    """
    Read an image, trim rows from top and bottom, columns from left and right,
    and save as PDF with specified DPI.
    
    Args:
        image_path (str): Path to the input image (supports JPG, PNG, CR2, etc.)
        row_trim_per_side (int): Number of rows to trim from top and bottom
        col_trim_per_side (int): Number of columns to trim from left and right
        pdf_dpi (int): DPI for the output PDF
        save_path (str): Path where the PDF will be saved
    """
    # Check if it's a CR2 file
    if image_path.lower().endswith('.cr2'):
        # Read CR2 file using rawpy
        with rawpy.imread(image_path) as raw:
            # Process the raw image to RGB with camera white balance
            rgb = raw.postprocess(
                use_camera_wb=True,        # Use camera's white balance
                use_auto_wb=False,          # Don't use auto white balance
                output_color=rawpy.ColorSpace.sRGB,  # Use sRGB color space
                output_bps=8,               # 8-bit output
                no_auto_bright=False,       # Allow auto brightness
                gamma=(2.222, 4.5),         # Standard sRGB gamma
            )
        # Convert numpy array to PIL Image
        img = Image.fromarray(rgb)
    else:
        # Read regular image file
        img = Image.open(image_path)
    
    # Convert to RGB if necessary (PDFs work best with RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Get image dimensions
    width, height = img.size
    
    # Calculate crop box (left, upper, right, lower)
    left = col_trim_per_side
    upper = row_trim_per_side
    right = width - col_trim_per_side
    lower = height - row_trim_per_side
    
    # Ensure valid crop dimensions
    if left >= right or upper >= lower:
        raise ValueError(f"Invalid trim parameters for image {image_path}. "
                        f"Trimming would result in zero or negative dimensions.")
    
    # Crop the image
    img_cropped = img.crop((left, upper, right, lower))
    
    # Save as PDF with specified DPI
    img_cropped.save(save_path, 'PDF', resolution=pdf_dpi, quality=95)
    
    # Also save as PNG with the same DPI
    png_path = save_path.replace('.pdf', '.png')
    img_cropped.save(png_path, 'PNG', dpi=(pdf_dpi, pdf_dpi))
    
    print(f"Processed: {os.path.basename(image_path)} -> {os.path.basename(save_path)}")


def trim_array_and_save_as_pdf(image_array, row_trim_per_side, col_trim_per_side, pdf_dpi, save_path):
    """
    Trim a numpy array and save as PDF with specified DPI.
    
    Args:
        image_array (np.ndarray): Input image as numpy array
        row_trim_per_side (int): Number of rows to trim from top and bottom
        col_trim_per_side (int): Number of columns to trim from left and right
        pdf_dpi (int): DPI for the output PDF
        save_path (str): Path where the PDF will be saved
    """
    # Handle different array shapes
    if image_array.ndim == 2:
        # Grayscale image (H x W)
        img_data = image_array
    elif image_array.ndim == 3:
        # Color image (H x W x C) or (C x H x W)
        if image_array.shape[0] in [1, 3, 4]:  # Likely (C x H x W)
            img_data = np.transpose(image_array, (1, 2, 0))
        else:  # Likely (H x W x C)
            img_data = image_array
    else:
        raise ValueError(f"Unsupported array shape: {image_array.shape}")
    
    # Normalize to 0-255 if needed
    if img_data.dtype == np.float32 or img_data.dtype == np.float64:
        if img_data.max() <= 1.0:
            img_data = (img_data * 255).astype(np.uint8)
        else:
            img_data = img_data.astype(np.uint8)
    elif img_data.dtype != np.uint8:
        img_data = img_data.astype(np.uint8)
    
    # Convert BGR to RGB if it's a 3-channel image (common for OpenCV/HDF5 data)
    if img_data.ndim == 3 and img_data.shape[2] == 3:
        img_data = img_data[..., ::-1]  # Reverse the channel order (BGR -> RGB)
    
    # Convert to PIL Image
    if img_data.ndim == 2:
        img = Image.fromarray(img_data, mode='L')
        img = img.convert('RGB')  # Convert grayscale to RGB for PDF
    elif img_data.shape[2] == 1:
        img = Image.fromarray(img_data.squeeze(), mode='L')
        img = img.convert('RGB')
    elif img_data.shape[2] == 3:
        img = Image.fromarray(img_data, mode='RGB')
    elif img_data.shape[2] == 4:
        img = Image.fromarray(img_data, mode='RGBA')
        img = img.convert('RGB')
    else:
        raise ValueError(f"Unsupported number of channels: {img_data.shape[2]}")
    
    # Get image dimensions
    width, height = img.size
    
    # Calculate crop box (left, upper, right, lower)
    left = col_trim_per_side
    upper = row_trim_per_side
    right = width - col_trim_per_side
    lower = height - row_trim_per_side
    
    # Ensure valid crop dimensions
    if left >= right or upper >= lower:
        raise ValueError(f"Invalid trim parameters. "
                        f"Trimming would result in zero or negative dimensions.")
    
    # Crop the image
    img_cropped = img.crop((left, upper, right, lower))
    
    # Save as PDF with specified DPI
    img_cropped.save(save_path, 'PDF', resolution=pdf_dpi, quality=95)
    
    # Also save as PNG with the same DPI
    png_path = save_path.replace('.pdf', '.png')
    img_cropped.save(png_path, 'PNG', dpi=(pdf_dpi, pdf_dpi))
    
    print(f"Processed array -> {os.path.basename(save_path)} and {os.path.basename(png_path)}")


def process_image_list(image_list, save_dir):
    """
    Process a list of images with their respective parameters.
    
    Args:
        image_list (list): List of tuples/lists, each containing either:
                          - 4 elements: (image_path, row_trim_per_side, col_trim_per_side, pdf_dpi)
                          - 6 elements: (h5_name, dataset_name, index, row_trim_per_side, col_trim_per_side, pdf_dpi)
        save_dir (str): Directory where PDFs will be saved
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Process each image in the list
    for idx, item in enumerate(image_list):
        try:
            if len(item) == 4:
                # Regular image file
                image_path, row_trim_per_side, col_trim_per_side, pdf_dpi = item
                
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = os.path.join(save_dir, f"{base_name}.pdf")
                
                # Process the image
                trim_and_save_as_pdf(image_path, row_trim_per_side, col_trim_per_side, 
                                    pdf_dpi, save_path)
                
            elif len(item) == 6:
                # HDF5 file with dataset and index
                h5_name, dataset_name, index, row_trim_per_side, col_trim_per_side, pdf_dpi = item
                
                # Read from HDF5 file
                with h5py.File(h5_name, 'r') as hdf5_file:
                    if dataset_name not in hdf5_file:
                        raise ValueError(f"Dataset '{dataset_name}' not found in {h5_name}")
                    
                    dataset = hdf5_file[dataset_name]
                    
                    # Extract the image at the specified index
                    if dataset.ndim >= 3:
                        image_array = dataset[index]
                    else:
                        raise ValueError(f"Dataset has insufficient dimensions: {dataset.shape}")
                
                # Generate output filename
                h5_base = os.path.splitext(os.path.basename(h5_name))[0]
                save_path = os.path.join(save_dir, f"{h5_base}_{dataset_name}_idx{index}.pdf")
                
                # Process the array
                trim_array_and_save_as_pdf(image_array, row_trim_per_side, col_trim_per_side, 
                                          pdf_dpi, save_path)
            else:
                print(f"Error processing item {idx}: Expected 4 or 6 elements, got {len(item)}")
                continue
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            continue
    
    print(f"\nProcessing complete! PDFs saved to: {save_dir}")


# Example usage
if __name__ == "__main__":
    # Example list of images with their parameters
    image_list = [
        # Regular image files: (image_path, row_trim_per_side, col_trim_per_side, pdf_dpi)
        # (r"H:\My Drive\Research_collabs\MODEST Research Collab\CVPR_visuals\IMG_6425.JPG", 474, 711, 900),
        (r"H:\My Drive\Research_collabs\MODEST Research Collab\CVPR_visuals\IMG_6425.CR2", 118, 178, 900),
        
        
        # HDF5 files: (h5_name, dataset_name, index, row_trim_per_side, col_trim_per_side, pdf_dpi)
        # (r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset" +\
        #  r"\Scene1\EOS6D_B_Left\fl_40mm\inference\F2.8\rectified\rectified_lefts.h5", 
        #   "rectified_lefts", 7, 85, 128, 600),
        # (r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset" +\
        #  r"\Scene1\EOS6D_A_Right\fl_40mm\inference\F2.8\rectified\rectified_rights.h5", 
        #   "rectified_rights", 7, 85, 128, 600),
        # ("path/to/data.h5", "images", 0, 10, 20, 300),
        # ("path/to/data.h5", "images", 1, 10, 20, 300),
    ]
    
    # Directory where PDFs will be saved
    save_dir = r"H:\My Drive\Research_collabs\MODEST Research Collab\CVPR_visuals\rebuttal_pdfs"
    
    # Process all images
    process_image_list(image_list, save_dir)