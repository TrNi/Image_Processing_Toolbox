"""
charuco_board.py
================
Generate a ChArUco calibration board PDF and save its parameters as pickle.

A ChArUco board combines a chessboard pattern with ArUco fiducial markers,
enabling robust single-image camera calibration and pose estimation.

The output is:
- A single-page PDF at print resolution (DPI is configurable).
- A ``charuco_board.pkl`` pickle containing the ``cv2.aruco.CharucoBoard``
  object for use in calibration pipelines.

Usage
-----
::

    python calibration/charuco_board.py \\
        --cols 9 --rows 6 \\
        --square_mm 40 --marker_mm 30 \\
        --out_dir calibration_board/

Default prints a standard A3-compatible 9×6 board.
"""

import sys
import pickle
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_charuco_board(
    cols: int = 9,
    rows: int = 6,
    square_length_mm: float = 40.0,
    marker_length_mm: float = 30.0,
    aruco_dict_name: str = "DICT_5X5_250",
    dpi: int = 300,
    out_dir: str = "calibration_board",
) -> cv2.aruco.CharucoBoard:
    """Generate a ChArUco board image, PDF, and parameter pickle.

    Parameters
    ----------
    cols : int
        Number of squares along the horizontal axis.
    rows : int
        Number of squares along the vertical axis.
    square_length_mm : float
        Physical square side length in millimetres.
    marker_length_mm : float
        Physical ArUco marker side length in millimetres.
    aruco_dict_name : str
        ArUco dictionary name (from ``cv2.aruco.*``).
        E.g. ``'DICT_5X5_250'``, ``'DICT_6X6_250'``.
    dpi : int
        Print resolution (dots per inch).
    out_dir : str
        Output directory.

    Returns
    -------
    cv2.aruco.CharucoBoard
        The generated board object (also saved as a pickle).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Resolve ArUco dictionary
    dict_attr = getattr(cv2.aruco, aruco_dict_name, None)
    if dict_attr is None:
        raise ValueError(
            f"Unknown ArUco dictionary: {aruco_dict_name!r}.  "
            "Check cv2.aruco.DICT_* constants."
        )
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)

    # Create board
    square_m = square_length_mm / 1000.0
    marker_m = marker_length_mm / 1000.0

    board = cv2.aruco.CharucoBoard(
        (cols, rows), square_m, marker_m, aruco_dict
    )

    # Compute image size in pixels
    px_per_mm = dpi / 25.4
    board_w_px = int(np.ceil(cols * square_length_mm * px_per_mm))
    board_h_px = int(np.ceil(rows * square_length_mm * px_per_mm))

    print(f"Board:      {cols} × {rows} squares")
    print(f"Square:     {square_length_mm} mm  ({square_m * 1000:.1f} mm)")
    print(f"Marker:     {marker_length_mm} mm  ({marker_m * 1000:.1f} mm)")
    print(f"Dictionary: {aruco_dict_name}")
    print(f"DPI:        {dpi}")
    print(f"Image size: {board_w_px} × {board_h_px} px")

    img = board.generateImage((board_w_px, board_h_px), marginSize=0)
    img_pil = Image.fromarray(img, mode='L').convert('RGB')

    # Save PNG
    png_path = out_path / "charuco_board.png"
    img_pil.save(str(png_path), 'PNG')
    print(f"Saved PNG → {png_path}")

    # Save PDF
    pdf_path = out_path / "charuco_board.pdf"
    img_pil.save(str(pdf_path), 'PDF', resolution=dpi)
    print(f"Saved PDF → {pdf_path}")

    # Save parameters as pickle
    pkl_path = out_path / "charuco_board.pkl"
    with open(str(pkl_path), 'wb') as fh:
        pickle.dump(board, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved PKL → {pkl_path}")

    # Print calibration usage hint
    print(f"\nCalibration info:")
    print(f"  board.chessboardCorners: {board.getChessboardCorners().shape}")
    print(f"  board.getIds():          {board.getIds().flatten()[:10]} …")

    return board


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a ChArUco calibration board PDF and pickle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
    # Standard 9×6 board with 40 mm squares at 300 DPI
    python calibration/charuco_board.py \\
        --cols 9 --rows 6 \\
        --square_mm 40 --marker_mm 30 \\
        --dpi 300 --out_dir board_output/

    # Larger board with different dictionary
    python calibration/charuco_board.py \\
        --cols 11 --rows 8 --square_mm 35 --marker_mm 26 \\
        --dict DICT_6X6_250 --dpi 600 --out_dir board_large/
""",
    )
    parser.add_argument("--cols",      type=int,   default=9,
                        help="Number of columns (default: 9).")
    parser.add_argument("--rows",      type=int,   default=6,
                        help="Number of rows (default: 6).")
    parser.add_argument("--square_mm", type=float, default=40.0,
                        help="Square side length in mm (default: 40.0).")
    parser.add_argument("--marker_mm", type=float, default=30.0,
                        help="Marker side length in mm (default: 30.0).")
    parser.add_argument("--dict",      default="DICT_5X5_250",
                        dest="aruco_dict",
                        help="ArUco dictionary name (default: DICT_5X5_250).")
    parser.add_argument("--dpi",       type=int,   default=300,
                        help="Print resolution in DPI (default: 300).")
    parser.add_argument("--out_dir",   default="calibration_board",
                        help="Output directory (default: calibration_board).")
    args = parser.parse_args()

    generate_charuco_board(
        cols=args.cols,
        rows=args.rows,
        square_length_mm=args.square_mm,
        marker_length_mm=args.marker_mm,
        aruco_dict_name=args.aruco_dict,
        dpi=args.dpi,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
