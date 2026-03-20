"""
get_errors.py
=============
High-level class that orchestrates per-image error computation and saving.

Overview
--------
:class:`Get_errors_and_GT` loads rectified stereo images and depth maps for a
configured dataset, computes five no-reference error types per depth model
(gradient, planarity variants, IQR uncertainty) and serialises the results as
gzip-compressed pickle files for later visualisation.

Typical workflow
----------------
1. Define a ``datalist`` (see :func:`main` for the required schema).
2. Instantiate :class:`Get_errors_and_GT`.
3. Call :meth:`Get_errors_and_GT.save_errors`.
4. Use :mod:`visualization.visualize_error_analysis` to plot the results.

Usage
-----
::

    python depth_analysis/get_errors.py --help

Or via the pipeline wrapper::

    python pipelines/run_depth_analysis.py

"""

import sys
import os
import gzip
import pickle
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np
import cv2
import traceback

from depth_analysis.depth_reproj_eval import (
    load_camera_params, get_Kinv_uv1, get_errors,
)
from depth_analysis.uncertainty_and_weights import get_iqr_uncertainty
from depth_analysis.geometric_structure_errors import compute_grad


# ---------------------------------------------------------------------------
# H5 / dataset keys
# ---------------------------------------------------------------------------
RECT_LEFT_KEY  = 'rectified_lefts'
RECT_RIGHT_KEY = 'rectified_rights'
DEPTH_KEY      = 'depth'


# ---------------------------------------------------------------------------
# Name-mapping helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# User-extendable name map: populate with your model keywords → display names.
# e.g. _PRETTY_NAME_MAP = {'my_model': 'My Model v2', 'baseline': 'Baseline'}
# ---------------------------------------------------------------------------
_PRETTY_NAME_MAP: dict = {}


def get_pretty_name(name: str) -> str:
    """Return a display-friendly model name from a raw filename keyword.

    Looks up *name* (case-insensitive substring) in :data:`_PRETTY_NAME_MAP`.
    Populate that dict with your model keywords before calling this function.
    Falls back to *name* unchanged if no match is found.
    """
    n = name.lower()
    for keyword, display in _PRETTY_NAME_MAP.items():
        if keyword.lower() in n:
            return display
    return name


def find_h5_by_keywords(folder: Path, keywords: list) -> dict:
    """Return ``{keyword: Path}`` for the best-matching H5 file in *folder*.

    Parameters
    ----------
    folder : Path
        Directory to search.
    keywords : list of str
        Model keyword strings (case-insensitive substring match).

    Returns
    -------
    dict
        Maps each matched keyword to the first matching ``Path``.
    """
    found = {}
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in ('.h5', '.hdf5')]
    lower_names = {f: f.name.lower() for f in files}
    for kw in keywords:
        for f, lname in lower_names.items():
            if kw.lower() in lname:
                found[kw] = f
                break
    return found


def load_h5_dataset(h5path: Path, key: str) -> np.ndarray:
    """Load a single dataset from an H5 file.

    Parameters
    ----------
    h5path : Path
        Path to the HDF5 file.
    key : str
        Dataset key.

    Returns
    -------
    np.ndarray
    """
    with h5py.File(h5path, 'r') as fh:
        if key not in fh:
            raise KeyError(f"Key '{key}' not found in {h5path}. Available: {list(fh.keys())}")
        data = fh[key][()]
    return data


# ---------------------------------------------------------------------------
# Resize helpers
# ---------------------------------------------------------------------------

def resize_image_hwc(img_hwc: np.ndarray, target_h: int, target_w: int,
                     interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(img_hwc, (target_w, target_h), interpolation=interpolation)


def resize_batch_nhwc(batch: np.ndarray, target_h: int, target_w: int,
                      interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    return np.stack([resize_image_hwc(img, target_h, target_w, interpolation) for img in batch])


def resize_image_chw(img_chw: np.ndarray, target_h: int, target_w: int,
                     interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    resized_hwc = cv2.resize(img_hwc, (target_w, target_h), interpolation=interpolation)
    return np.transpose(resized_hwc, (2, 0, 1))


def resize_batch_nchw(batch: np.ndarray, target_h: int, target_w: int,
                      interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    return np.stack([resize_image_chw(img, target_h, target_w, interpolation) for img in batch])


def sorted_k(arr: np.ndarray, k: int = 10_000) -> np.ndarray:
    """Return *k* evenly-spaced quantiles from *arr*, excluding outer 5 %."""
    arr = arr.flatten()
    p5, p95 = np.percentile(arr, [5, 95])
    arr = arr[(arr >= p5) & (arr <= p95)]
    x_sorted = np.sort(arr)
    idx = np.linspace(0, len(x_sorted) - 1, k).astype(int)
    return x_sorted[idx]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Get_errors_and_GT:
    """Compute and save per-model depth error maps for a configured dataset.

    Parameters
    ----------
    datalist : list of dict
        Each entry must have:

        - ``'base'``: ``str`` — root directory of the scene.
        - ``'cameras'``: ``[left_cam_name, right_cam_name]``.
        - ``'configs'``: list of ``{'fl': int, 'F': float}`` dicts.

    MONO_MODELS : list of str
        Keywords identifying monocular depth model files (substring match).
    STEREO_MODELS : list of str
        Keywords identifying stereo depth model files.
    """

    def __init__(self, datalist: list, MONO_MODELS: list, STEREO_MODELS: list):
        self.datalist     = datalist
        self.MONO_MODELS  = MONO_MODELS
        self.STEREO_MODELS = STEREO_MODELS

    # ------------------------------------------------------------------
    def load_rects(self, base: Path, left_cam: str, right_cam: str, cfg: dict):
        """Locate and load rectified stereo image arrays for one configuration."""
        fl     = cfg['fl']
        F      = float(cfg['F'])
        fl_folder = f"fl_{int(fl)}mm"
        F_folder  = f"F{F:.1f}"
        self.fl_folder, self.F_folder = fl_folder, F_folder

        self.left_rectified_dir  = base / left_cam  / fl_folder / "inference" / F_folder / "rectified"
        self.right_rectified_dir = base / right_cam / fl_folder / "inference" / F_folder / "rectified"
        self.stereocal_params    = base / f'stereocal_params_{fl}mm.npz'

        print(f"Config: {base.name}  fl={fl}mm  F={F:.1f}  "
              f"left={left_cam}  right={right_cam}")

        self.left_rect_h5  = self.left_rectified_dir  / "rectified_lefts.h5"
        self.right_rect_h5 = self.right_rectified_dir / "rectified_rights.h5"

        if not self.left_rect_h5.exists():
            raise FileNotFoundError(f"Not found: {self.left_rect_h5}")
        if not self.right_rect_h5.exists():
            raise FileNotFoundError(f"Not found: {self.right_rect_h5}")

        left_rects  = load_h5_dataset(self.left_rect_h5,  RECT_LEFT_KEY)
        right_rects = load_h5_dataset(self.right_rect_h5, RECT_RIGHT_KEY)

        self.left_rects  = np.asarray(left_rects)
        self.right_rects = np.asarray(right_rects)
        self.N = min(self.left_rects.shape[0], self.right_rects.shape[0])

    # ------------------------------------------------------------------
    def load_depths(self):
        """Discover and load depth arrays for all configured models."""
        mono_found   = find_h5_by_keywords(
            self.left_rectified_dir.parent / "monodepth", self.MONO_MODELS)
        stereo_found = find_h5_by_keywords(
            self.left_rectified_dir, self.STEREO_MODELS)

        depth_paths  = []
        depth_titles = []

        for kw in self.MONO_MODELS:
            p = mono_found.get(kw)
            depth_paths.append(p)
            depth_titles.append(get_pretty_name(p.name) if p else kw)

        for kw in self.STEREO_MODELS:
            p = stereo_found.get(kw)
            depth_paths.append(p)
            depth_titles.append(get_pretty_name(p.name) if p else kw)

        depth_arrays = []
        for p in depth_paths:
            if p is None:
                depth_arrays.append(None)
                continue
            try:
                d = load_h5_dataset(p, DEPTH_KEY).astype(np.float32)
                if d.shape[0] > self.N:
                    d = d[:self.N]
                elif d.shape[0] < self.N:
                    self.N = d.shape[0]
                print(f"Loaded depth: {p.name}  {d.shape}")
                depth_arrays.append(d)
            except Exception as exc:
                print(f"Failed to load {p}: {exc}")
                depth_arrays.append(None)

        self.depth_arrays = depth_arrays
        self.depth_titles = depth_titles
        self.depth_paths  = depth_paths

    # ------------------------------------------------------------------
    def save_errors(self, out_root: str = None):
        """Compute all error maps and save to disk.

        Parameters
        ----------
        out_root : str, optional
            Root directory for output files.  If ``None``, results are saved
            next to the rectified images under ``<scene>/err_GT/``.
        """
        error_types = ['grad', 'plan', 'rms_orth', 'Prel', 'Pnorm', 'icp', 'iqr']
        aggr_points = 20_000

        for entry in self.datalist:
            base      = Path(entry['base'])
            left_cam  = entry['cameras'][0]
            right_cam = entry['cameras'][1]

            for cfg in entry['configs']:
                self.load_rects(base, left_cam, right_cam, cfg)
                self.load_depths()

                params = load_camera_params(str(self.stereocal_params))
                P2     = params['P2']
                K_inv  = params['K_inv']

                # Determine common spatial resolution
                N, C, H, W = self.left_rects.shape
                aspect_ratio = W / H
                min_N, min_h, min_w = N, H, W

                for name, arr in zip(self.depth_titles, self.depth_arrays):
                    if arr is None:
                        continue
                    N_d, h_d, w_d = arr.shape
                    if abs(w_d / h_d - aspect_ratio) > 0.01:
                        print(f"Skipping {name}: aspect ratio mismatch "
                              f"({w_d/h_d:.3f} vs {aspect_ratio:.3f})")
                        continue
                    min_N = min(min_N, N_d)
                    min_h = min(min_h, h_d)
                    min_w = min(min_w, w_d)

                K_inv_uv1 = get_Kinv_uv1(K_inv, min_h, min_w)
                print(f"Common size: {min_h}×{min_w} px  "
                      f"(scale {min_h/H:.2f}×{min_w/W:.2f})")

                self.left_rects  = resize_batch_nchw(self.left_rects,  min_h, min_w)
                self.right_rects = resize_batch_nchw(self.right_rects, min_h, min_w)

                for i, (name, arr) in enumerate(zip(self.depth_titles, self.depth_arrays)):
                    if arr is None:
                        continue
                    self.depth_arrays[i] = resize_batch_nhwc(arr, min_h, min_w)

                # Set up output directory
                if out_root is not None:
                    tag = (f"{base.name.lower()}_"
                           f"{self.fl_folder.replace('_', '')}_"
                           f"{self.F_folder.replace('_', '')}")
                    out_dir = Path(out_root) / tag / "err_GT"
                else:
                    out_dir = self.left_rectified_dir.parent / "err_GT"
                out_dir.mkdir(parents=True, exist_ok=True)

                num_models = len(self.depth_titles)
                error_maps = {}
                error_aggr = {}

                for name, arr in zip(self.depth_titles, self.depth_arrays):
                    if arr is None:
                        continue
                    error_maps[name] = {k: np.zeros((min_N, min_h, min_w))
                                        for k in error_types}
                    error_aggr[name] = {k: np.zeros((min_N, aggr_points))
                                        for k in error_types}

                for current_idx in range(min_N):
                    rect_left  = self.left_rects[current_idx].transpose(1, 2, 0)
                    rect_right = self.right_rects[current_idx].transpose(1, 2, 0)

                    depth_stack = np.stack(
                        [self.depth_arrays[i][current_idx]
                         for i in range(num_models)
                         if self.depth_arrays[i] is not None],
                        axis=0,
                    )

                    alpha  = 0.1
                    kernel = 5
                    g_i    = compute_grad(rect_left, k=kernel)
                    g_i   /= g_i.max() + 1e-8

                    try:
                        iqr_errors = get_iqr_uncertainty(depth_stack)
                    except Exception as exc:
                        print(f"  get_iqr_uncertainty failed at idx {current_idx}: {exc}")
                        continue

                    valid_idx = 0
                    for i, (name, arr) in enumerate(zip(self.depth_titles, self.depth_arrays)):
                        if arr is None:
                            continue
                        err_data = get_errors(
                            arr[current_idx], rect_left, rect_right,
                            K_inv, K_inv_uv1, g_i, P2, alpha, kernel,
                        )
                        for k in ["grad", "plan", "rms_orth", "Prel", "Pnorm"]:
                            error_maps[name][k][current_idx] = err_data[k]
                            error_aggr[name][k][current_idx] = sorted_k(err_data[k], k=aggr_points)

                        error_maps[name]["iqr"][current_idx] = iqr_errors[valid_idx]
                        error_aggr[name]["iqr"][current_idx] = sorted_k(
                            iqr_errors[valid_idx], k=aggr_points)
                        valid_idx += 1

                # Persist results
                save_path = out_dir / "error_data.pkl"
                with gzip.open(save_path, 'wb') as fh:
                    pickle.dump(
                        {'error_maps': error_maps, 'error_aggr': error_aggr},
                        fh, protocol=pickle.HIGHEST_PROTOCOL,
                    )
                print(f"Saved error data → {save_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute depth-map error metrics for a stereo dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
Edit the datalist inside this script (or import Get_errors_and_GT into your
own pipeline script) then run:

    python depth_analysis/get_errors.py \\
        --base /path/to/scene \\
        --left_cam <left_camera_dir> --right_cam <right_camera_dir> \\
        --fl 70 --F 2.8 \\
        --mono_models model_a model_b \\
        --stereo_models model_c model_d \\
        --out_root /path/to/output
""",
    )
    parser.add_argument('--base',      required=True,
                        help="Root directory of the scene.")
    parser.add_argument('--left_cam',  required=True,
                        help="Left camera folder name.")
    parser.add_argument('--right_cam', required=True,
                        help="Right camera folder name.")
    parser.add_argument('--fl',  type=int,   required=True,
                        help="Focal length in mm (e.g. 70).")
    parser.add_argument('--F',   type=float, required=True,
                        help="Aperture f-number (e.g. 2.8).")
    parser.add_argument('--out_root', default=None,
                        help="Root directory for output files (default: next to input).")
    parser.add_argument('--mono_models',   nargs='+', required=True,
                        help="Keywords identifying monocular depth model files "
                             "(substrings matched against HDF5 filenames).")
    parser.add_argument('--stereo_models', nargs='+', required=True,
                        help="Keywords identifying stereo depth model files "
                             "(substrings matched against HDF5 filenames).")
    args = parser.parse_args()

    datalist = [{
        "base":    args.base,
        "cameras": [args.left_cam, args.right_cam],
        "configs": [{"fl": args.fl, "F": args.F}],
    }]

    computer = Get_errors_and_GT(datalist, args.mono_models, args.stereo_models)
    computer.save_errors(out_root=args.out_root)


if __name__ == '__main__':
    main()
