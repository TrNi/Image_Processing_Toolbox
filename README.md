# Image Processing Toolbox

A modular Python toolbox for computational photography and depth-estimation research.
It provides end-to-end utilities for **depth quality assessment**, **no-reference error metrics**, **stereo image processing**, **HDF5 dataset management**, and **publication-quality figure generation** — designed for computer vision researchers and university students working with multi-optics or stereo camera datasets.

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Module Reference](#module-reference)
  - [depth\_analysis](#depth_analysis)
  - [visualization](#visualization)
  - [image\_processing](#image_processing)
  - [file\_tools](#file_tools)
  - [calibration](#calibration)
  - [pipelines](#pipelines)
  - [visuals](#visuals)
- [Quick-Start Examples](#quick-start-examples)
- [Dataset Conventions](#dataset-conventions)
- [Contributing](#contributing)

---

## Features

| Category | Highlights |
|---|---|
| **Depth Analysis** | Gradient-consistency, local planarity (PCA), IQR ensemble uncertainty, ICP point-cloud error, photometric reprojection error |
| **Visualization** | Interactive depth viewer, CDF plots, publication 3-column comparison figures, bokeh blur-ROI comparisons |
| **Image Processing** | Phase-correlation & SIFT alignment, white-balance matching, interactive batch crop, quadrant splitting |
| **File Tools** | HDF5 pack/extract/merge, `.npy → .h5`, `.pkl → .h5`, CSV manifests, Google Drive ID resolution |
| **Calibration** | ChArUco board PDF generator (configurable square / marker size, dictionary) |
| **Pipelines** | Single-command full error-computation → save → visualise workflow |

---

## Repository Structure

```
Image_Processing_Toolbox/
│
├── depth_analysis/                  ← Core depth quality metrics
│   ├── __init__.py
│   ├── depth_reproj_eval.py         Camera geometry, reprojection, photometric errors
│   ├── geometric_structure_errors.py Gradient-consistency & planarity error maps
│   ├── uncertainty_and_weights.py   IQR / MAD uncertainty, weighted fusion
│   ├── point_cloud_opt.py           Point-cloud ICP / GICP consistency (open3d)
│   └── get_errors.py                Orchestrator class: compute & save all error types
│
├── visualization/                   ← Plotting & visual analysis
│   ├── __init__.py
│   ├── visualize_depth.py           Interactive per-image depth viewer + fusion
│   ├── visualize_error_analysis.py  CDF plots, error-map figures, percentile tables
│   ├── plots_from_csvs.py           Error trend plots vs. focal length / aperture
│   ├── plot_one_row.py              Single-row ECCV/CVPR publication figure
│   ├── generate_comparison_figure.py 3-column × N-row comparison figure
│   └── vis_blur_rois.py             Interactive bokeh/depth-of-field ROI comparison
│
├── image_processing/                ← Alignment, cropping, transforms
│   ├── __init__.py
│   ├── align_images.py              Phase-correlation (+ ORB fallback) alignment
│   ├── align_whitebal.py            Focus-masked SIFT + white-balance matching
│   ├── apply_crop_from_json.py      Apply saved crop regions to image directories
│   ├── interactive_crop.py          Interactive batch rectangle-crop GUI
│   ├── visualize_and_crop.py        View image at folder index, then crop
│   ├── crop_images.py               Multi-region crop with coordinate array export
│   ├── crop_jpg.py                  Simple width-based JPEG crop
│   ├── resize_images.py             Batch resize with bilinear interpolation
│   └── split_quadrants.py           Split images into 4 equal quadrants
│
├── file_tools/                      ← File format conversion & data management
│   ├── __init__.py
│   ├── extract_from_h5.py           Extract individual frames from HDF5 as JPEG/PNG
│   ├── illustrate_h5.py             Display HDF5 dataset images in a grid
│   ├── jpg_to_h5.py                 Pack image directory → single HDF5 file
│   ├── merge_h5.py                  Concatenate multiple HDF5 files
│   ├── npy_to_h5.py                 Combine .npy sequence → HDF5
│   ├── npy_to_npz.py                Convert .npy dict-pickle → compressed .npz
│   ├── pkl_to_h5.py                 Convert gzip-pickle error files → HDF5
│   ├── create_csv.py                Build CSV index from image directories
│   ├── move_images.py               Move images at specified indices
│   ├── organise_files.py            Reorganise capture tree → processing tree
│   └── get_gdrive_ids.py            Resolve Google Drive file IDs from sidecar files
│
├── calibration/                     ← Camera calibration board utilities
│   ├── __init__.py
│   └── charuco_board.py             ChArUco board PDF + parameter pickle generator
│
├── pipelines/                       ← End-to-end analysis pipelines
│   ├── __init__.py
│   ├── run_depth_analysis.py        Full pipeline: compute → save → visualise
│   └── run_depth_analysis_folders.py Folder-based visualisation sweep
│
├── visuals/                         ← Publication figure utilities
│   ├── __init__.py
│   ├── jpg_to_pdf.py                Trim + export images/HDF5 frames as PDF
│   ├── merge_imgs.py                Combine 4 images into a 1×4 CVPR figure
│   └── mono_stereo_depths/          Scene-specific depth visualisation helpers
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-org/Image_Processing_Toolbox.git
cd Image_Processing_Toolbox
pip install -r requirements.txt
```

**Optional extras:**

```bash
# For ICP / GICP point-cloud error maps (large dependency)
pip install open3d>=0.17.0

# For camera RAW (CR2) support
pip install rawpy>=0.18.0
```

Python ≥ 3.9 is recommended.

---

## Module Reference

### depth\_analysis

> Core no-reference depth quality metrics for evaluating mono and stereo
> depth estimation models without ground-truth depth.

| Script | Key functions / classes |
|---|---|
| `depth_reproj_eval.py` | `load_h5_images`, `load_camera_params`, `get_Kinv_uv1`, `px_to_camera`, `project_to_view`, `photometric_errors`, `get_errors` |
| `geometric_structure_errors.py` | `compute_grad`, `compute_grad_error`, `get_planarity_error` |
| `uncertainty_and_weights.py` | `get_iqr_uncertainty`, `calculate_individual_mad_uncertainty`, `simple_weighted_fusion` |
| `point_cloud_opt.py` | `PointCloudConsistencyAnalyzer`, `get_point_cloud_errors` |
| `get_errors.py` | `Get_errors_and_GT` (CLI + class) |

**Error types computed per depth map:**

| Name | Description |
|---|---|
| `grad` | Gradient-consistency: large depth edges in smooth image regions |
| `plan` | Local planarity (smallest PCA eigenvalue of 3-D patch covariance) |
| `rms_orth` | RMS orthogonal distance to local best-fit plane |
| `Prel` | Relative planarity λ₃/(λ₁+λ₂+λ₃) |
| `Pnorm` | Depth-normalised planarity λ₃/Z̄² |
| `iqr` | IQR-normalised ensemble uncertainty (leave-one-out) |
| `icp` | ICP point-cloud alignment error w.r.t. median depth reference |
| `photo_l1` / `photo_ssim` | Photometric reprojection L1 / SSIM error from stereo pair |

---

### visualization

> Plotting utilities for depth maps and error-metric analysis.

| Script | Description |
|---|---|
| `visualize_depth.py` | Interactive per-image depth viewer with IQR+ICP error overlay and weighted fusion output |
| `visualize_error_analysis.py` | CDF figures, per-model error maps, percentile CSV tables |
| `plots_from_csvs.py` | Two-row trend plot (focal-length sweep × aperture sweep) from CSV |
| `plot_one_row.py` | One-row figure: 2 full panels + 2 × (2×2 quadrant) panels |
| `generate_comparison_figure.py` | Text-file-driven 3-column × N-row comparison figure with zoomed crops |
| `vis_blur_rois.py` | Interactive ROI picker → two-row bokeh comparison figure (PNG/SVG/PDF) |

---

### image\_processing

> Image alignment, cropping, and geometric transformation utilities.

| Script | CLI synopsis |
|---|---|
| `align_images.py` | `align_images.py img1 img2 --out_dir out/ --threshold 0.02` |
| `align_whitebal.py` | `align_whitebal.py ref_dir/ src_dir/ --model similarity` |
| `apply_crop_from_json.py` | `apply_crop_from_json.py json_dir/ img_dir1/ img_dir2/` |
| `interactive_crop.py` | `interactive_crop.py image_list.txt save_dir/` |
| `visualize_and_crop.py` | `visualize_and_crop.py folder/ --index 7 --view` |
| `crop_images.py` | `crop_images.py --images *.jpg --regions 100,500,200,600 --out_dir out/` |
| `crop_jpg.py` | `crop_jpg.py input_dir/ save_dir/ --width 5472` |
| `resize_images.py` | `resize_images.py input_dir/ output_dir/ --width 1920 --height 1080` |
| `split_quadrants.py` | `split_quadrants.py input_dir/ output_dir/` |

---

### file\_tools

> File format conversion, dataset packaging, and data management.

| Script | CLI synopsis |
|---|---|
| `extract_from_h5.py` | `extract_from_h5.py data.h5 --key images --out_dir extracted/` |
| `illustrate_h5.py` | `illustrate_h5.py data.h5 --rows 3 --cols 4 --start 0` |
| `jpg_to_h5.py` | `jpg_to_h5.py images/ output.h5 --max 200` |
| `merge_h5.py` | `merge_h5.py --input_dir h5s/ --output merged.h5` |
| `npy_to_h5.py` | `npy_to_h5.py npy_dir/ output.h5 --glob depth_*.npy` |
| `npy_to_npz.py` | `npy_to_npz.py data.npy --out data.npz` |
| `pkl_to_h5.py` | `pkl_to_h5.py error_data.pkl --out error_data.h5` |
| `create_csv.py` | `create_csv.py --dirs img_dir1/ img_dir2/ --out dataset.csv` |
| `move_images.py` | `move_images.py --src src/ --dst dst/ --indices 3 5 7` |
| `organise_files.py` | `organise_files.py --src raw/ --dst processed/ --scene S1 ...` |
| `get_gdrive_ids.py` | `get_gdrive_ids.py --search_dir gdrive/ --glob *.gdoc` |

---

### calibration

> Camera calibration board generation.

```bash
# Generate a 9×6 ChArUco board at 300 DPI
python calibration/charuco_board.py \
    --cols 9 --rows 6 \
    --square_mm 40 --marker_mm 30 \
    --dpi 300 --out_dir board_output/
```

Outputs: `charuco_board.png`, `charuco_board.pdf`, `charuco_board.pkl`.

---

### pipelines

> End-to-end depth analysis pipelines.

**Single configuration:**
```bash
python pipelines/run_depth_analysis.py \
    --base /data/MODEST/Scene6 \
    --left_cam EOS6D_B_Left --right_cam EOS6D_A_Right \
    --fl 70 --F 2.8 \
    --mono_models depthpro metric3d unidepth depth_anything \
    --stereo_models monster foundation defom selective \
    --out_root /data/output \
    --visualise
```

**Batch visualisation over existing results:**
```bash
python pipelines/run_depth_analysis_folders.py \
    --root /data/output \
    --pattern "**/err_GT/error_data.pkl"
```

---

### visuals

> Publication-quality figure utilities for CVPR/ECCV papers.

```bash
# Trim and export a camera RAW as PDF
python visuals/jpg_to_pdf.py \
    --image /path/to/IMG_6425.CR2 \
    --row_trim 118 --col_trim 178 --dpi 900 \
    --out_dir /path/to/output

# Combine 4 images into a 1×4 paper figure
python visuals/merge_imgs.py \
    --images charuco.png rect_left.png rect_right.png foundation.png \
    --titles "ChArUco" "Rectified Left" "Rectified Right" "Foundation Stereo" \
    --out_dir output/ --dpi 900
```

---

## Quick-Start Examples

### 1 — Compute and visualise depth errors for one scene

```bash
python pipelines/run_depth_analysis.py \
    --base /data/MODEST/Scene9 \
    --left_cam EOS6D_B_Left --right_cam EOS6D_A_Right \
    --fl 70 --F 2.8 \
    --out_root /data/output \
    --visualise
```

### 2 — Pack a folder of JPEGs into HDF5

```bash
python file_tools/jpg_to_h5.py /data/images rectified.h5 --max 200
```

### 3 — Align two images and white-balance

```bash
python image_processing/align_whitebal.py ref_dir/ src_dir/ --model similarity
```

### 4 — Generate a ChArUco calibration board

```bash
python calibration/charuco_board.py \
    --cols 9 --rows 6 --square_mm 40 --marker_mm 30 --dpi 300
```

### 5 — Plot error trends from pre-computed CSVs

```bash
python visualization/plots_from_csvs.py \
    --focal_csvs fl28:/data/out/fl28/error_percentiles.csv \
                 fl70:/data/out/fl70/error_percentiles.csv \
    --aperture_csvs F2.8:/data/out/F2.8/error_percentiles.csv \
                    F22:/data/out/F22/error_percentiles.csv \
    --out_dir /data/plots
```

### 6 — Programmatic API

```python
import numpy as np
from depth_analysis import get_iqr_uncertainty, simple_weighted_fusion

depth_stack = np.random.rand(4, 480, 640).astype(np.float32) * 10 + 1.0
uncertainty  = get_iqr_uncertainty(depth_stack)   # (4, 480, 640)
fused        = simple_weighted_fusion(depth_stack, uncertainty)  # (480, 640)
```

---

## Dataset Conventions

The toolbox assumes a directory structure compatible with the
**MODEST (Multi-Optics Depth Estimation Stereo) dataset** layout:

```
<scene_root>/
    <left_camera>/
        fl_<F>mm/
            inference/
                F<aperture>/
                    rectified/
                        rectified_lefts.h5
                        monodepth/
                            <model>_depth.h5
                    stereodepth/
                        <model>_depth.h5
    <right_camera>/
        fl_<F>mm/
            inference/
                F<aperture>/
                    rectified/
                        rectified_rights.h5
    stereocal_params_<F>mm.npz
```

The `stereocal_params_*.npz` files must contain keys `P1`, `P2`, `baseline`, `fB`.

---

## Contributing

1. Follow the existing module structure — new tools belong in the appropriate sub-package.
2. All scripts must be runnable both as CLI (`python module/script.py --help`) and importable as library functions.
3. Replace hard-coded paths with `argparse` arguments.
4. All dependencies must be listed in `requirements.txt`.
