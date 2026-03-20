from get_errors import Get_errors_and_GT
import visualize_error_analysis as viz
from pathlib import Path

def main():
    # Define the dataset to process
    datalist = [
        # {
        #     "base": r"I:\\My Drive\\Pubdata\\Scene6_illusions",
        #     "cameras": ['EOS6D_A_Left', 'EOS6D_B_Right'],
        #     "configs": [
        #         {"fl": 70, "F": 16},
        #     ]
        # },
        {
            "base": r"I:\\My Drive\\Pubdata\\Scene9",
            "cameras": ['EOS6D_B_Left', 'EOS6D_A_Right'],
            "configs": [
                {"fl":28, "F":2.8},
                #{"fl":40, "F":2.8},
                #{"fl":45, "F":2.8},
                #{"fl":60, "F":2.8},
                #{"fl":65, "F":2.8},
                #{"fl":70, "F":2.8},
                # {"fl":70, "F":2.8},
                # {"fl":70, "F":5.0},
                # {"fl":70, "F":9.0},
                # {"fl":70, "F":16.0},
                # {"fl":70, "F":22.0},
            ]
        },
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
    for entry in datalist:
        base = Path(entry['base'])
        left_cam = entry['cameras'][0]
        for cfg in entry['configs']:
            fl_folder = f"fl_{int(cfg['fl'])}mm"
            F_folder = f"F{cfg['F']:.1f}"
            save_dir = base / left_cam / fl_folder / "inference" / F_folder / "err_GT"
            
            print(f"\nProcessing {fl_folder} {F_folder}...")
            # Load and visualize error data
            error_data_path = save_dir / "error_data.pkl"
            viz.main(datalist=datalist, specific_path=error_data_path)

    print("\nAnalysis complete! Check the output directories for:")
    print("1. error_data.pkl - Raw error data")
    print("2. error_percentiles.csv - Statistical analysis")
    print("3. error_cdf_*.png - Error distribution plots")
    print("4. depth_maps_*.png - Depth map visualizations")
    print("5. error_maps_*.png - Error map visualizations")

if __name__ == "__main__":
    main()
