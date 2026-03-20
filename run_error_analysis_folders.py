from get_errors import Get_errors_and_GT
import visualize_error_analysis as viz
from pathlib import Path

def main():
    # Define the dataset folders to process
    datalist = [
        # "E:\\pub_results\\scene9_fl28mm_F2.8\\err_GT",
        # "E:\\pub_results\\scene9_fl32mm_F2.8\\err_GT",
        # "E:\\pub_results\\scene9_fl36mm_F2.8\\err_GT",
        # "E:\\pub_results\\scene9_fl40mm_F2.8\\err_GT",
        # "E:\\pub_results\\scene9_fl45mm_F2.8\\err_GT",
        # "E:\\pub_results\\scene9_fl60mm_F2.8\\err_GT",
        # "E:\\pub_results\\scene9_fl65mm_F2.8\\err_GT",
        # "E:\\pub_results\\scene9_fl70mm_F2.8\\err_GT",
        # r"E:\\pub_results\\scene9_fl70mm_F2.8\\err_GT",
        "E:\\pub_results\\scene9_fl70mm_F5.0\\err_GT",
        # "E:\\pub_results\\scene9_fl70mm_F9.0\\err_GT",
        # "E:\\pub_results\\scene9_fl70mm_F16.0\\err_GT",
        # "E:\\pub_results\\scene9_fl70mm_F22.0\\err_GT",
        # Add more folder paths as needed
    ]

    # Define models to analyze
    MONO_MODELS = ['depthpro', 'metric3d', 'unidepth', 'depth_anything']
    STEREO_MODELS = ['monster', 'foundation', 'defom', 'selective']
    ALL_MODELS = MONO_MODELS + STEREO_MODELS

    print("Step 1: Computing errors...")
    # Initialize and run error computation
    # error_computer = Get_errors_and_GT(datalist, MONO_MODELS, STEREO_MODELS)
    # error_computer.save_errors()
    print("Error computation complete.")

    print("\nStep 2: Generating visualizations...")
    # Run visualization analysis
    for base_folder in datalist:
        base = Path(base_folder)
        error_data_path = base / "error_data.pkl"
        viz.main(datalist=[{"base": base_folder}], specific_path=error_data_path)

    print("\nAnalysis complete! Check the output directories for:")
    print("1. error_data.pkl - Raw error data")
    print("2. error_percentiles.csv - Statistical analysis")
    print("3. error_cdf_*.png - Error distribution plots")
    print("4. depth_maps_*.png - Depth map visualizations")
    print("5. error_maps_*.png - Error map visualizations")

if __name__ == "__main__":
    main()
