"""
run_depth_analysis.py
=====================
Full end-to-end pipeline: compute per-image error maps → save → visualise.

Wraps :class:`depth_analysis.get_errors.Get_errors_and_GT` and
:func:`visualization.visualize_error_analysis.main` into a single CLI entry
point.

Usage
-----
::

    python pipelines/run_depth_analysis.py \\
        --base      /path/to/scene \\
        --left_cam  <left_camera_dir> \\
        --right_cam <right_camera_dir> \\
        --fl  70 --F 2.8 \\
        --mono_models  model_a model_b \\
        --stereo_models model_c model_d \\
        --out_root /path/to/output \\
        --visualise
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from depth_analysis.get_errors import Get_errors_and_GT
from visualization.visualize_error_analysis import main as visualise_main


def run_pipeline(
    base: str,
    left_cam: str,
    right_cam: str,
    fl: int,
    F: float,
    mono_models: list,
    stereo_models: list,
    out_root: str = None,
    visualise: bool = False,
):
    """Run the full depth analysis pipeline for one scene configuration.

    Parameters
    ----------
    base : str
        Root directory of the scene.
    left_cam, right_cam : str
        Camera directory names.
    fl : int
        Focal length in mm.
    F : float
        Aperture f-number.
    mono_models : list of str
        Keywords for monocular model depth files.
    stereo_models : list of str
        Keywords for stereo model depth files.
    out_root : str, optional
        Root directory for output files.  Defaults to alongside input data.
    visualise : bool
        If ``True``, run :func:`visualize_error_analysis.main` after saving.
    """
    datalist = [{
        "base":    base,
        "cameras": [left_cam, right_cam],
        "configs": [{"fl": fl, "F": F}],
    }]

    print(f"\n{'='*70}")
    print(f"Scene:  {base}")
    print(f"Config: fl={fl}mm  F={F:.1f}  "
          f"left={left_cam}  right={right_cam}")
    print(f"Mono:   {mono_models}")
    print(f"Stereo: {stereo_models}")
    print(f"{'='*70}\n")

    computer = Get_errors_and_GT(datalist, mono_models, stereo_models)
    computer.save_errors(out_root=out_root)

    if visualise:
        # Find the generated error_data.pkl and visualise it
        if out_root:
            tag = (f"{Path(base).name.lower()}_"
                   f"fl{fl}mm_F{F:.1f}".replace('_', '').replace('.', ''))
            pkl_path = Path(out_root) / tag / "err_GT" / "error_data.pkl"
        else:
            pkl_path = (Path(base) /
                        f"{left_cam}" /
                        f"fl_{fl}mm" /
                        "inference" /
                        f"F{F:.1f}" /
                        "rectified" /
                        "err_GT" /
                        "error_data.pkl")

        if pkl_path.exists():
            print(f"\nVisualising: {pkl_path}")
            visualise_main(specific_path=pkl_path)
        else:
            print(f"WARNING: error_data.pkl not found at expected path: {pkl_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full depth analysis pipeline: compute errors → save → (optionally) visualise.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
    python pipelines/run_depth_analysis.py \\
        --base /path/to/your_dataset/scene \\
        --left_cam <left_camera_dir> \\
        --right_cam <right_camera_dir> \\
        --fl 70 --F 2.8 \\
        --mono_models model_a model_b \\
        --stereo_models model_c model_d \\
        --out_root /path/to/output \\
        --visualise
""",
    )
    parser.add_argument('--base',      required=True,
                        help="Root directory of the scene.")
    parser.add_argument('--left_cam',  required=True,
                        help="Left camera folder name.")
    parser.add_argument('--right_cam', required=True,
                        help="Right camera folder name.")
    parser.add_argument('--fl',        type=int,   required=True,
                        help="Focal length in mm.")
    parser.add_argument('--F',         type=float, required=True,
                        help="Aperture f-number.")
    parser.add_argument('--mono_models',   nargs='+', required=True,
                        help="Keywords identifying monocular depth model files "
                             "(substrings matched against HDF5 filenames).")
    parser.add_argument('--stereo_models', nargs='+', required=True,
                        help="Keywords identifying stereo depth model files "
                             "(substrings matched against HDF5 filenames).")
    parser.add_argument('--out_root',  default=None,
                        help="Root directory for output files.")
    parser.add_argument('--visualise', action='store_true',
                        help="Run visualisation after computing errors.")
    args = parser.parse_args()

    run_pipeline(
        base=args.base,
        left_cam=args.left_cam,
        right_cam=args.right_cam,
        fl=args.fl,
        F=args.F,
        mono_models=args.mono_models,
        stereo_models=args.stereo_models,
        out_root=args.out_root,
        visualise=args.visualise,
    )


if __name__ == '__main__':
    main()
