"""
align_whitebal.py
=================
Focus-masked SIFT alignment with optional white-balance correction.

For each image pair, this script:

1. Computes a focus mask on the source image (restricts SIFT detection to
   in-focus regions whose descriptor space is reliable).
2. Estimates a translation (or similarity) transform via SIFT + Lowe ratio
   test + RANSAC.
3. Warps the source image and crops both images to the common overlap region.
4. Applies per-channel mean/std white-balance matching.
5. Saves alignment JSON, cropped images, white-balanced images, and a
   3×3 summary plot.

Usage
-----
::

    python image_processing/align_whitebal.py ref_dir/ src_dir/ \\
        --model translation --alpha 0.5

"""

import sys
import json
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def luminance(img_bgr: np.ndarray) -> np.ndarray:
    """BT.601 luminance from BGR uint8, returned as float32."""
    return (0.299 * img_bgr[..., 2] +
            0.587 * img_bgr[..., 1] +
            0.114 * img_bgr[..., 0]).astype(np.float32)


def focus_measure_map(img_bgr: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Per-pixel windowed-variance of Laplacian focus measure, normalised to [0,1].

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format.
    ksize : int
        Box-filter window size.

    Returns
    -------
    np.ndarray
        Float32 focus map with values in [0, 1].
    """
    g   = luminance(img_bgr)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=5)
    k   = (ksize, ksize)
    mu  = cv2.boxFilter(lap,    cv2.CV_32F, k)
    mu2 = cv2.boxFilter(lap**2, cv2.CV_32F, k)
    var = np.clip(mu2 - mu**2, 0, None)
    vmax = var.max()
    return var / vmax if vmax > 0 else var


def focus_mask(img_bgr: np.ndarray, ksize: int = 15,
               percentile_thresh: float = 60.0,
               morph_open_r: int = 15,
               morph_close_r: int = 25) -> np.ndarray:
    """Binary focus mask (uint8, 0/255) for in-focus regions.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format.
    ksize : int
        Window size for :func:`focus_measure_map`.
    percentile_thresh : float
        Pixels above this percentile of focus measure are considered in focus.
    morph_open_r, morph_close_r : int
        Radii for morphological open (noise removal) and close (hole fill).

    Returns
    -------
    np.ndarray
        Binary mask, uint8, values 0 or 255.
    """
    F     = focus_measure_map(img_bgr, ksize=ksize)
    thresh = np.percentile(F, percentile_thresh)
    mask  = (F >= thresh).astype(np.uint8) * 255

    def disk(r):
        d = 2 * r + 1
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        return (x**2 + y**2 <= r**2).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  disk(morph_open_r))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(morph_close_r))
    return mask


def sift_translation(img1_bgr: np.ndarray, img2_bgr: np.ndarray,
                     mask2=None,
                     ransac_thresh: float = 2.0,
                     ratio_thresh: float = 0.75,
                     min_inliers: int = 20,
                     model: str = "translation"):
    """SIFT keypoint matching with Lowe ratio test + RANSAC.

    Parameters
    ----------
    img1_bgr, img2_bgr : np.ndarray
        Input images in BGR format.
    mask2 : np.ndarray, optional
        uint8 mask (0/255) restricting keypoint detection in I2.
    ransac_thresh : float
        RANSAC inlier threshold in pixels.
    ratio_thresh : float
        Lowe ratio test threshold.
    min_inliers : int
        Minimum required inliers.
    model : str
        ``'translation'`` (2 DOF) or ``'similarity'`` (4 DOF — handles
        aperture breathing).

    Returns
    -------
    tuple or None
        ``(M_2x3 float64, info_dict)`` or ``None`` on failure.
        Info keys: ``tx``, ``ty``, ``scale``, ``n_inliers``, ``n_matches``.
    """
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02, edgeThreshold=15)
    kp1, des1 = sift.detectAndCompute(img1_bgr, None)
    kp2, des2 = sift.detectAndCompute(img2_bgr, mask2)

    if des1 is None or des2 is None:
        return None
    if len(kp1) < min_inliers or len(kp2) < min_inliers:
        print(f"  Too few keypoints: I1={len(kp1)}, I2={len(kp2)}")
        return None

    index_params  = dict(algorithm=1, trees=8)
    search_params = dict(checks=128)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw   = flann.knnMatch(des1, des2, k=2)
    good  = [m for m, n in raw if m.distance < ratio_thresh * n.distance]
    print(f"  SIFT: {len(kp1)}/{len(kp2)} kpts, {len(good)} matches after ratio test")

    if len(good) < min_inliers:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    if model == "translation":
        deltas   = pts1 - pts2
        t_med    = np.median(deltas, axis=0)
        residuals = np.linalg.norm(deltas - t_med, axis=1)
        inliers  = residuals < ransac_thresh
        if inliers.sum() < min_inliers:
            print(f"  Too few translation inliers: {inliers.sum()}")
            return None
        tx, ty = deltas[inliers].mean(axis=0)
        M    = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)
        info = dict(tx=float(tx), ty=float(ty), scale=1.0,
                    n_inliers=int(inliers.sum()), n_matches=len(good))

    elif model == "similarity":
        M, inlier_mask = cv2.estimateAffinePartial2D(
            pts2, pts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            confidence=0.999,
            maxIters=10000,
        )
        if M is None:
            return None
        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if n_inliers < min_inliers:
            print(f"  Too few similarity inliers: {n_inliers}")
            return None
        scale = float(np.sqrt(M[0, 0]**2 + M[1, 0]**2))
        tx, ty = float(M[0, 2]), float(M[1, 2])
        info = dict(tx=tx, ty=ty, scale=scale,
                    n_inliers=n_inliers, n_matches=len(good))
        M = M.astype(np.float64)
    else:
        raise ValueError(f"Unknown model: {model!r}")

    return M, info


def compute_common_crop(M: np.ndarray, h: int, w: int) -> dict:
    """Compute the overlapping region in I1's frame after warping I2 by M.

    Parameters
    ----------
    M : np.ndarray
        2×3 affine transform matrix.
    h, w : int
        Image height and width.

    Returns
    -------
    dict
        ``{x0, y0, x1, y1}`` — axis-aligned bounding box (exclusive end).
    """
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    ones    = np.ones((4, 1), dtype=np.float32)
    warped  = (M @ np.hstack([corners, ones]).T).T

    x_min = max(0, int(np.ceil(warped[:, 0].min())))
    y_min = max(0, int(np.ceil(warped[:, 1].min())))
    x_max = min(w, int(np.floor(warped[:, 0].max())))
    y_max = min(h, int(np.floor(warped[:, 1].max())))
    return {"x0": x_min, "y0": y_min, "x1": x_max, "y1": y_max}


def match_white_balance(img_src: np.ndarray,
                        img_ref: np.ndarray) -> tuple:
    """Match white balance of *img_src* to *img_ref* via mean/std scaling.

    Parameters
    ----------
    img_src, img_ref : np.ndarray
        BGR uint8 images.

    Returns
    -------
    tuple
        ``(white_balanced_image_bgr, wb_params_dict)``
    """
    src_f  = img_src.astype(np.float32)
    ref_f  = img_ref.astype(np.float32)
    result = np.zeros_like(src_f)
    params = {"method": "mean_std_scaling", "channels": {}}

    for c, name in enumerate(['B', 'G', 'R']):
        s_mu  = float(src_f[:, :, c].mean())
        s_std = float(src_f[:, :, c].std())
        r_mu  = float(ref_f[:, :, c].mean())
        r_std = float(ref_f[:, :, c].std())
        params["channels"][name] = dict(
            source_mean=round(s_mu, 2), source_std=round(s_std, 2),
            reference_mean=round(r_mu, 2), reference_std=round(r_std, 2),
            scale_factor=round(r_std / s_std, 4) if s_std > 0 else 1.0,
        )
        if s_std > 0:
            result[:, :, c] = (src_f[:, :, c] - s_mu) * (r_std / s_std) + r_mu
        else:
            result[:, :, c] = src_f[:, :, c]

    return np.clip(result, 0, 255).astype(np.uint8), params


# ---------------------------------------------------------------------------
# Directory-level processing
# ---------------------------------------------------------------------------

def process_image_directories(
    ref_dir: str,
    src_dir: str,
    model: str = "translation",
    focus_percentile: float = 60.0,
    focus_ksize: int = 15,
    ransac_thresh: float = 2.0,
    ratio_thresh: float = 0.75,
    min_inliers: int = 20,
    alpha: float = 0.4,
):
    """Align all images in *src_dir* to their counterparts in *ref_dir*.

    Images are matched by sorted order (lexicographic).  For each pair the
    script produces:

    - ``<src_dir>_align/``         — aligned-and-cropped source images.
    - ``<ref_dir>_align_whitebal/`` — reference images cropped to common region.
    - ``<src_dir>_align_whitebal/`` — source images aligned + white-balanced.
    - Per-pair ``alignment.json`` and a 3×3 diagnostic plot PNG.

    Parameters
    ----------
    ref_dir, src_dir : str
        Reference and source image directories.
    model : str
        ``'translation'`` or ``'similarity'``.
    focus_percentile : float
        Focus-mask threshold percentile.
    focus_ksize : int
        Focus-mask window size.
    ransac_thresh : float
        RANSAC inlier threshold in pixels.
    ratio_thresh : float
        Lowe ratio test threshold.
    min_inliers : int
        Minimum required SIFT inliers.
    alpha : float
        Alpha blending weight (0–1) for overlay images.
    """
    ref_path = Path(ref_dir)
    src_path = Path(src_dir)
    exts     = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

    ref_images = sorted(f for f in ref_path.glob("*") if f.suffix.lower() in exts)
    src_images = sorted(f for f in src_path.glob("*") if f.suffix.lower() in exts)

    if len(ref_images) != len(src_images):
        raise ValueError(f"Image count mismatch: ref={len(ref_images)}, src={len(src_images)}")
    if not ref_images:
        raise ValueError("No images found.")

    print(f"Processing {len(ref_images)} image pair(s) …")

    src_align_dir      = src_path.parent / f"{src_path.name}_align"
    ref_align_wb_dir   = ref_path.parent / f"{ref_path.name}_align_whitebal"
    src_align_wb_dir   = src_path.parent / f"{src_path.name}_align_whitebal"

    for d in [src_align_dir, ref_align_wb_dir, src_align_wb_dir]:
        d.mkdir(exist_ok=True)

    for idx, (ref_img_path, src_img_path) in enumerate(zip(ref_images, src_images)):
        print(f"\n{'='*60}")
        print(f"Pair {idx+1}/{len(ref_images)}")
        print(f"  Ref: {ref_img_path.name}")
        print(f"  Src: {src_img_path.name}")
        print(f"{'='*60}")

        img1 = cv2.imread(str(ref_img_path))
        img2 = cv2.imread(str(src_img_path))
        if img1 is None or img2 is None:
            print("  ERROR: Failed to load images, skipping.")
            continue

        h, w = img1.shape[:2]

        print(f"[focus mask] ksize={focus_ksize}, percentile={focus_percentile}")
        mask2 = focus_mask(img2, ksize=focus_ksize, percentile_thresh=focus_percentile)

        print(f"[SIFT+RANSAC] model={model}")
        result = sift_translation(img1, img2, mask2=mask2,
                                   ransac_thresh=ransac_thresh,
                                   ratio_thresh=ratio_thresh,
                                   min_inliers=min_inliers, model=model)

        if result is None:
            print("  Masked SIFT failed; retrying without mask …")
            result = sift_translation(img1, img2, mask2=None,
                                       ransac_thresh=ransac_thresh,
                                       ratio_thresh=ratio_thresh,
                                       min_inliers=min_inliers, model=model)
        if result is None:
            print("  ERROR: SIFT alignment failed, skipping.")
            continue

        M, info = result
        print(f"  tx={info['tx']:.3f}px, ty={info['ty']:.3f}px, "
              f"scale={info['scale']:.6f}, inliers={info['n_inliers']}")

        img2_aligned = cv2.warpAffine(img2, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0, 0, 0))

        crop = compute_common_crop(M, h, w)
        x0, y0, x1, y1 = crop["x0"], crop["y0"], crop["x1"], crop["y1"]
        print(f"[crop] x:[{x0},{x1}), y:[{y0},{y1}), size={x1-x0}×{y1-y0}")

        img1_crop = img1[y0:y1, x0:x1]
        img2_crop = img2_aligned[y0:y1, x0:x1]

        cv2.imwrite(str(src_align_dir / f"{src_img_path.stem}.png"), img2_crop)

        alignment_data = {
            "reference_image": ref_img_path.name,
            "source_image":    src_img_path.name,
            "transform_matrix": M.tolist(),
            "tx": info["tx"], "ty": info["ty"], "scale": info["scale"],
            "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
            "n_inliers": info["n_inliers"], "n_matches": info["n_matches"],
            "model": model,
        }

        print("[white balance] matching source to reference …")
        img2_wb, wb_params = match_white_balance(img2_crop, img1_crop)
        alignment_data["white_balance"] = wb_params

        cv2.imwrite(str(ref_align_wb_dir / f"{ref_img_path.stem}.png"),  img1_crop)
        cv2.imwrite(str(src_align_wb_dir / f"{src_img_path.stem}.png"),  img2_wb)

        for d_out, stem_key in [(ref_align_wb_dir, ref_img_path.stem),
                                 (src_align_wb_dir, src_img_path.stem)]:
            with open(d_out / f"{stem_key}_alignment.json", "w") as fh:
                json.dump(alignment_data, fh, indent=2)

        # Diagnostic plot
        overlay_align = cv2.addWeighted(img1_crop, alpha, img2_crop, alpha, 0.0)
        overlay_wb    = cv2.addWeighted(img1_crop, alpha, img2_wb,   alpha, 0.0)

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        to_rgb = lambda b: cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

        data = [
            (img1,       f"Reference (original)\n{ref_img_path.name}"),
            (img2,       f"Source (original)\n{src_img_path.name}"),
            (None,       None),
            (img1_crop,  f"Reference (cropped)\n{img1_crop.shape[1]}×{img1_crop.shape[0]}"),
            (img2_crop,  f"Source (aligned & cropped)\ntx={info['tx']:.1f}, ty={info['ty']:.1f}"),
            (overlay_align, f"Alpha blend (α={alpha})\nAligned only"),
            (img1_crop,  "Reference (same)"),
            (img2_wb,    "Source (white balanced)"),
            (overlay_wb, f"Alpha blend (α={alpha})\nAligned + White balanced"),
        ]
        for ax, (img, title) in zip(axes.flatten(), data):
            if img is None:
                ax.axis('off')
                continue
            ax.imshow(to_rgb(img))
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        plt.suptitle(f"Pair {idx+1}/{len(ref_images)}: Alignment & White Balance",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_path = ref_align_wb_dir.parent / f"alignment_plot_{idx+1:03d}.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[plot] saved → {plot_path}")

    print(f"\n{'='*60}")
    print(f"Done.  Aligned:        {src_align_dir}")
    print(f"       White-balanced: {ref_align_wb_dir}, {src_align_wb_dir}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Align and white-balance image directories with focus-masked SIFT.",
    )
    parser.add_argument("ref_dir", help="Reference image directory.")
    parser.add_argument("src_dir", help="Source image directory.")
    parser.add_argument("--model",      default="translation",
                        choices=["translation", "similarity"],
                        help="translation (2 DOF) or similarity (4 DOF).")
    parser.add_argument("--focus_pct",  type=float, default=60.0,
                        help="Focus-mask percentile threshold (default: 60).")
    parser.add_argument("--focus_k",    type=int,   default=15,
                        help="Focus-mask kernel size (default: 15).")
    parser.add_argument("--ransac",     type=float, default=2.0,
                        help="RANSAC inlier threshold in pixels (default: 2.0).")
    parser.add_argument("--ratio",      type=float, default=0.75,
                        help="Lowe ratio test threshold (default: 0.75).")
    parser.add_argument("--min_inliers", type=int,  default=20)
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="Alpha blend factor for overlay (default: 0.5).")
    args = parser.parse_args()

    process_image_directories(
        ref_dir=args.ref_dir,
        src_dir=args.src_dir,
        model=args.model,
        focus_percentile=args.focus_pct,
        focus_ksize=args.focus_k,
        ransac_thresh=args.ransac,
        ratio_thresh=args.ratio,
        min_inliers=args.min_inliers,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
