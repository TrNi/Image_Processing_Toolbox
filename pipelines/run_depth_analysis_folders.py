"""
run_depth_analysis_folders.py
==============================
Folder-based pipeline variant: point at directories of pre-computed
``error_data.pkl`` files and run visualisation across all of them.

Usage
-----
::

    python pipelines/run_depth_analysis_folders.py \\
        --root /path/to/output_root \\
        --pattern "**/err_GT/error_data.pkl"

or for a multi-scene / multi-config sweep::

    python pipelines/run_depth_analysis_folders.py \\
        --folders /data/scene1/err_GT /data/scene2/err_GT \\
        --plot_types cdf percentiles
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visualization.visualize_error_analysis import main as visualise_main


def run_on_folders(
    root: str = None,
    folders: list = None,
    pattern: str = "**/err_GT/error_data.pkl",
    plot_types: list = None,
):
    """Run visualisation on all discovered ``error_data.pkl`` files.

    Parameters
    ----------
    root : str, optional
        Root directory to search recursively using *pattern*.
    folders : list of str, optional
        Explicit list of directories containing ``error_data.pkl`` files.
    pattern : str
        Glob pattern for auto-discovery under *root*.
    plot_types : list of str, optional
        Visualisation types to run (reserved for future extension).
    """
    pkl_paths = []

    if root is not None:
        root_path = Path(root)
        found = sorted(root_path.glob(pattern))
        pkl_paths.extend(found)
        print(f"Found {len(found)} pkl file(s) under {root}")

    if folders is not None:
        for d in folders:
            p = Path(d) / "error_data.pkl"
            if p.exists():
                pkl_paths.append(p)
            else:
                print(f"WARNING: not found: {p}")

    if not pkl_paths:
        print("No error_data.pkl files found.")
        return

    print(f"\nWill process {len(pkl_paths)} pkl file(s).\n")
    for i, pkl_path in enumerate(pkl_paths):
        print(f"[{i+1}/{len(pkl_paths)}] {pkl_path}")
        try:
            visualise_main(specific_path=pkl_path)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()
        print()

    print("All done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run error-analysis visualisation on a set of pkl result folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
    # Auto-discover all error_data.pkl files under an output root
    python pipelines/run_depth_analysis_folders.py \\
        --root /data/output

    # Explicit folder list
    python pipelines/run_depth_analysis_folders.py \\
        --folders /data/scene1/err_GT /data/scene2/err_GT

    # Custom discovery glob
    python pipelines/run_depth_analysis_folders.py \\
        --root /data/output --pattern "*/err_GT/error_data.pkl"
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--root',    help="Root directory to search for pkl files.")
    group.add_argument('--folders', nargs='+',
                       help="Explicit directories containing error_data.pkl.")

    parser.add_argument('--pattern', default="**/err_GT/error_data.pkl",
                        help="Glob pattern for auto-discovery (default: **/err_GT/error_data.pkl).")
    parser.add_argument('--plot_types', nargs='+', default=['cdf', 'percentiles'],
                        help="Visualisation types to run.")
    args = parser.parse_args()

    run_on_folders(
        root=args.root,
        folders=args.folders,
        pattern=args.pattern,
        plot_types=args.plot_types,
    )


if __name__ == '__main__':
    main()
