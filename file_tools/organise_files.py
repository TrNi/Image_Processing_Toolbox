"""
organise_files.py
=================
Reorganise files from a calibration capture tree into a structured
processing/inference directory tree.

Source tree assumption
----------------------
::

    <source_root>/
        <camera_A>/
            fl_<F>mm/
                F<aperture>/
                    *.JPG  (or *.jpg)

Destination tree
----------------
::

    <dest_root>/
        <scene_name>/
            <camera_A>/
                fl_<F>mm/
                    inference/
                        F<aperture>/
                            *.jpg

Usage
-----
::

    python file_tools/organise_files.py \\
        --src    /path/to/source_root \\
        --dst    /path/to/dest_root \\
        --scene  SceneName \\
        --cameras EOS6D_A EOS6D_B \\
        --fls    28 40 70 100 \\
        --apers  2.8 5.6 8.0 11.0 16.0 22.0
"""

import sys
import shutil
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.cr2', '.raw'}


def organise_files(src_root: str, dst_root: str, scene: str,
                   cameras: list, focal_lengths: list, apertures: list,
                   dry_run: bool = False, copy: bool = False):
    """Reorganise image files from source tree into processing tree.

    Parameters
    ----------
    src_root : str
        Root of the source capture tree.
    dst_root : str
        Root of the destination processing tree.
    scene : str
        Scene sub-directory name inside *dst_root*.
    cameras : list of str
        Camera directory names to process.
    focal_lengths : list of int
        Focal length values to look for (mm).
    apertures : list of float
        Aperture f-numbers to look for.
    dry_run : bool
        If ``True``, print what would happen without moving/copying files.
    copy : bool
        If ``True``, copy files; otherwise move them.
    """
    src_path = Path(src_root)
    dst_path = Path(dst_root) / scene
    verb     = "copy" if copy else "move"

    moved = skipped = 0

    for cam in cameras:
        for fl in focal_lengths:
            fl_dir = src_path / cam / f"fl_{fl}mm"
            if not fl_dir.exists():
                print(f"  [SKIP] not found: {fl_dir}")
                continue
            for ap in apertures:
                ap_str    = f"F{float(ap):.1f}"
                src_ap    = fl_dir / ap_str
                if not src_ap.exists():
                    print(f"  [SKIP] not found: {src_ap}")
                    continue

                dst_ap = dst_path / cam / f"fl_{fl}mm" / "inference" / ap_str
                if not dry_run:
                    dst_ap.mkdir(parents=True, exist_ok=True)

                img_files = sorted(
                    f for f in src_ap.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                )
                if not img_files:
                    print(f"  [SKIP] no images in: {src_ap}")
                    continue

                print(f"  {cam}/fl_{fl}mm/{ap_str}  →  {dst_ap}  "
                      f"({len(img_files)} files)")
                for f in img_files:
                    dst_f = dst_ap / f.name.lower()   # Normalise to lowercase
                    if dry_run:
                        print(f"    [DRY] {verb}: {f.name} → {dst_f}")
                    else:
                        if copy:
                            shutil.copy2(str(f), str(dst_f))
                        else:
                            shutil.move(str(f), str(dst_f))
                    moved += 1

    if dry_run:
        print(f"\nDry run: {moved} file(s) would be {verb}d.")
    else:
        print(f"\nDone.  {moved} file(s) {verb}d.  Skipped: {skipped}")
        print(f"Destination: {dst_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganise captures from source tree into processing tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
    python file_tools/organise_files.py \\
        --src /data/raw_captures --dst /data/processed \\
        --scene Scene1 \\
        --cameras EOS6D_A EOS6D_B \\
        --fls 28 40 70 100 \\
        --apers 2.8 5.6 8.0 22.0 \\
        --dry_run
""",
    )
    parser.add_argument("--src",     required=True, help="Source root directory.")
    parser.add_argument("--dst",     required=True, help="Destination root directory.")
    parser.add_argument("--scene",   required=True, help="Scene sub-directory name.")
    parser.add_argument("--cameras", nargs="+", required=True,
                        help="Camera directory names.")
    parser.add_argument("--fls",     nargs="+", type=int, required=True,
                        help="Focal lengths in mm.")
    parser.add_argument("--apers",   nargs="+", type=float, required=True,
                        help="Aperture f-numbers.")
    parser.add_argument("--copy",    action="store_true",
                        help="Copy files instead of moving.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview changes without modifying the filesystem.")
    args = parser.parse_args()

    organise_files(
        src_root=args.src,
        dst_root=args.dst,
        scene=args.scene,
        cameras=args.cameras,
        focal_lengths=args.fls,
        apertures=args.apers,
        dry_run=args.dry_run,
        copy=args.copy,
    )


if __name__ == "__main__":
    main()
