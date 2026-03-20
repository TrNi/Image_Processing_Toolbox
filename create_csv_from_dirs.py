import os
import csv
from pathlib import Path

def create_csv_from_directories(dir_prefix_list, output_csv="output.csv"):
    """
    Creates a CSV file from multiple directories with prefixes.
    
    Args:
        dir_prefix_list: List of [directory, prefix] pairs
        output_csv: Output CSV filename
    
    The CSV will have k rows (k = number of JPG files in each directory)
    Each row has n values in the format: prefix/filename, matched by index
    """
    if not dir_prefix_list:
        print("Error: Empty directory list provided")
        return
    
    # Get JPG files from each directory separately and sort them
    all_jpg_lists = []
    for directory, prefix in dir_prefix_list:
        if not os.path.exists(directory):
            print(f"Error: Directory '{directory}' does not exist")
            return
        
        # Get all JPG files from this directory (sorted alphabetically)
        jpg_files = sorted([f for f in os.listdir(directory) 
                           if f.lower().endswith('.jpg')])
        
        if not jpg_files:
            print(f"Error: No JPG files found in '{directory}'")
            return
        
        all_jpg_lists.append(jpg_files)
        print(f"Found {len(jpg_files)} JPG files in '{directory}'")
    
    # Determine the number of rows (minimum count across all directories)
    num_rows = min(len(jpg_list) for jpg_list in all_jpg_lists)
    
    if num_rows == 0:
        print("Error: No JPG files found in directories")
        return
    
    print(f"Creating CSV with {num_rows} rows")
    
    # Create CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # For each index, create a row with prefix/filename from each directory
        for i in range(num_rows):
            row = []
            for j, (directory, prefix) in enumerate(dir_prefix_list):
                jpg_file = all_jpg_lists[j][i]
                # Construct the path as prefix/filename
                path_entry = f"{prefix}/{jpg_file}"
                row.append(path_entry)
            
            writer.writerow(row)
    
    print(f"CSV file '{output_csv}' created successfully with {num_rows} rows")

def cat_csvs_as_text(csv1, csv2, output_text):
    """
    Concatenates two CSV files as text files.
    
    Args:
        csv1: First CSV file
        csv2: Second CSV file
        output_text: Output text file
    """
    with open(csv1, 'r') as f1:
        with open(csv2, 'r') as f2:
            with open(output_text, 'w') as fo:
                fo.write(f1.read())
                fo.write(f2.read())


if __name__ == "__main__":
    # Hardcode your inputs here
    # Example format:
    # dir_prefix_list = [
    #     [r"C:\path\to\dir1", "prefix1"],
    #     [r"C:\path\to\dir2", "prefix2"],
    #     [r"C:\path\to\dir3", "prefix3"],
    # ]
    
    dir_prefix_list = [                
        [r"I:\My Drive\DOF_benchmarking\inference\GT_fl70_F2.8", "https://f004.backblazeb2.com/file/dof-mos/GT_fl70_F2.8"],
        [r"I:\My Drive\DOF_benchmarking\inference\bokehdiff\cropped", "https://f004.backblazeb2.com/file/dof-mos/bokehdiff"],
        [r"I:\My Drive\DOF_benchmarking\inference\Bokehme_fl70_scale20_K40", "https://f004.backblazeb2.com/file/dof-mos/bokehme"],
        [r"I:\My Drive\DOF_benchmarking\inference\Drbokeh_fl70_K25_fp0.25", "https://f004.backblazeb2.com/file/dof-mos/drbokeh"],
        [r"I:\My Drive\DOF_benchmarking\inference\Bokehliciouslg_intp_fl70", "https://f004.backblazeb2.com/file/dof-mos/bokehlicious"],
    ]
    
    output_csv1 = "I:\My Drive\DOF_benchmarking\inference\images_used.csv"
    
    create_csv_from_directories(dir_prefix_list, output_csv1)

    dir_prefix_list = [                
        [r"I:\My Drive\DOF_benchmarking\inference\GT_fl70_F2.8\subcropped", "https://f004.backblazeb2.com/file/dof-mos/GT_fl70_F2.8"],
        [r"I:\My Drive\DOF_benchmarking\inference\bokehdiff\subcropped", "https://f004.backblazeb2.com/file/dof-mos/bokehdiff"],
        [r"I:\My Drive\DOF_benchmarking\inference\Bokehme_fl70_scale20_K40\subcropped", "https://f004.backblazeb2.com/file/dof-mos/bokehme"],
        [r"I:\My Drive\DOF_benchmarking\inference\Drbokeh_fl70_K25_fp0.25\subcropped", "https://f004.backblazeb2.com/file/dof-mos/drbokeh"],
        [r"I:\My Drive\DOF_benchmarking\inference\Bokehliciouslg_intp_fl70\subcropped", "https://f004.backblazeb2.com/file/dof-mos/bokehlicious"],
    ]
    
    output_csv2 = "I:\My Drive\DOF_benchmarking\inference\images_used_subcrops.csv"
    
    create_csv_from_directories(dir_prefix_list, output_csv2)

    cat_csvs_as_text(output_csv1, output_csv2, "I:\My Drive\DOF_benchmarking\inference\images_used_all.csv")
