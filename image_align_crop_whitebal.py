import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def luminance(img_bgr: np.ndarray) -> np.ndarray:
    """BT.601 luminance from BGR uint8, returned as float32."""
    return (0.299 * img_bgr[..., 2] +
            0.587 * img_bgr[..., 1] +
            0.114 * img_bgr[..., 0]).astype(np.float32)


def focus_measure_map(img_bgr: np.ndarray, ksize: int = 15) -> np.ndarray:
    """
    Per-pixel focus measure via windowed variance of the Laplacian (float32, [0,1]).

    F(x) = Var_{N(x)}( nabla^2 g )  where N(x) is a ksize x ksize window.

    Laplacian computed with a 5x5 kernel (ksize=5 in cv2.Laplacian) to capture
    mid-frequency sharpness without excessive noise sensitivity.
    """
    g = luminance(img_bgr)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=5)           # nabla^2 g
    # Local variance: Var = E[x^2] - E[x]^2
    k = (ksize, ksize)
    mu  = cv2.boxFilter(lap,    cv2.CV_32F, k)
    mu2 = cv2.boxFilter(lap**2, cv2.CV_32F, k)
    var = mu2 - mu**2
    var = np.clip(var, 0, None)
    # Normalise to [0, 1]
    vmax = var.max()
    if vmax > 0:
        var /= vmax
    return var


def focus_mask(
    img_bgr: np.ndarray,
    ksize: int = 15,
    percentile_thresh: float = 60.0,
    morph_open_r: int = 15,
    morph_close_r: int = 25,
) -> np.ndarray:
    """
    Binary focus mask for img_bgr (uint8, 0/255).

    Threshold at `percentile_thresh`-th percentile of F to keep the sharpest
    regions. Morphological open removes isolated noise; close fills holes.
    """
    F = focus_measure_map(img_bgr, ksize=ksize)
    thresh = np.percentile(F, percentile_thresh)
    mask = (F >= thresh).astype(np.uint8) * 255

    def disk(r):
        d = 2 * r + 1
        y, x = np.ogrid[-r:r+1, -r:r+1]
        return (x**2 + y**2 <= r**2).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  disk(morph_open_r))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(morph_close_r))
    return mask


def sift_translation(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    mask2: np.ndarray | None = None,
    ransac_thresh: float = 2.0,
    ratio_thresh: float = 0.75,
    min_inliers: int = 20,
    model: str = "translation",          # "translation" | "similarity"
) -> tuple[np.ndarray, dict] | None:
    """
    SIFT keypoint matching with Lowe ratio test + RANSAC.

    mask2   : uint8 mask (0/255) restricting keypoint detection in I2.
              I1 is searched unrestricted.
    model   : "translation" -- 2 DOF, M = [[1,0,tx],[0,1,ty]]
              "similarity"  -- 4 DOF, handles small aperture breathing
                               (scale + rotation + translation)

    Returns (M_2x3 float64, info_dict) or None on failure.
    info_dict keys: tx, ty, scale, n_inliers, n_matches
    """
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02, edgeThreshold=15)

    kp1, des1 = sift.detectAndCompute(img1_bgr, None)
    kp2, des2 = sift.detectAndCompute(img2_bgr, mask2)

    if des1 is None or des2 is None:
        return None
    if len(kp1) < min_inliers or len(kp2) < min_inliers:
        print(f"  Too few keypoints: I1={len(kp1)}, I2={len(kp2)}")
        return None

    # Lowe ratio test with FLANN (L2, exact for SIFT)
    index_params = dict(algorithm=1, trees=8)   # FLANN_INDEX_KDTREE=1
    search_params = dict(checks=128)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < ratio_thresh * n.distance]

    print(f"  SIFT: {len(kp1)} / {len(kp2)} kpts (I1/I2), "
          f"{len(good)} matches after ratio test")

    if len(good) < min_inliers:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    if model == "translation":
        # Pure translation RANSAC: consensus on per-match displacement
        deltas = pts1 - pts2
        t_med = np.median(deltas, axis=0)
        residuals = np.linalg.norm(deltas - t_med, axis=1)
        inliers = residuals < ransac_thresh
        if inliers.sum() < min_inliers:
            print(f"  Too few translation inliers: {inliers.sum()}")
            return None
        tx, ty = deltas[inliers].mean(axis=0)
        M = np.array([[1.0, 0.0, tx],
                      [0.0, 1.0, ty]], dtype=np.float64)
        info = dict(tx=float(tx), ty=float(ty), scale=1.0,
                    n_inliers=int(inliers.sum()), n_matches=len(good))

    elif model == "similarity":
        # estimateAffinePartial2D: similarity (s*R + t), 4 DOF
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
        # Decompose: M = s*[[cos θ, -sin θ],[sin θ, cos θ]] + t
        scale = float(np.sqrt(M[0, 0]**2 + M[1, 0]**2))
        tx, ty = float(M[0, 2]), float(M[1, 2])
        info = dict(tx=tx, ty=ty, scale=scale,
                    n_inliers=n_inliers, n_matches=len(good))
    else:
        raise ValueError(f"Unknown model: {model}")

    return M.astype(np.float64), info


def compute_common_crop(M: np.ndarray, h: int, w: int) -> dict:
    """
    Compute the axis-aligned bounding box of the valid overlap region in I1's
    frame, accounting for a general 2x3 affine warp of I2.

    Maps the four corners of I2 through M, then intersects the convex hull
    of those projected corners with [0,W) x [0,H).

    Returns {x0, y0, x1, y1} with exclusive end, consistent with NumPy slicing.
    """
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    corners_h = np.hstack([corners, ones])  # homogeneous
    warped = (M @ corners_h.T).T  # (4,2)

    x_min = max(0, int(np.ceil(warped[:, 0].min())))
    y_min = max(0, int(np.ceil(warped[:, 1].min())))
    x_max = min(w, int(np.floor(warped[:, 0].max())))
    y_max = min(h, int(np.floor(warped[:, 1].max())))

    return {"x0": x_min, "y0": y_min, "x1": x_max, "y1": y_max}


def match_white_balance(img_src: np.ndarray, img_ref: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Match the white balance of img_src to img_ref using mean/std scaling per channel.
    
    Both images should be BGR uint8 numpy arrays of the same shape.
    Returns tuple of (white-balanced image as BGR uint8, parameters dict).
    """
    img_src_float = img_src.astype(np.float32)
    img_ref_float = img_ref.astype(np.float32)
    
    result = np.zeros_like(img_src_float)
    wb_params = {
        "method": "mean_std_scaling",
        "channels": {}
    }
    
    channel_names = ['B', 'G', 'R']
    
    for c in range(3):  # B, G, R channels
        src_mean = float(img_src_float[:, :, c].mean())
        src_std = float(img_src_float[:, :, c].std())
        ref_mean = float(img_ref_float[:, :, c].mean())
        ref_std = float(img_ref_float[:, :, c].std())
        
        # Store parameters
        wb_params["channels"][channel_names[c]] = {
            "source_mean": round(src_mean, 2),
            "source_std": round(src_std, 2),
            "reference_mean": round(ref_mean, 2),
            "reference_std": round(ref_std, 2),
            "scale_factor": round(ref_std / src_std, 4) if src_std > 0 else 1.0
        }
        
        # Avoid division by zero
        if src_std > 0:
            result[:, :, c] = ((img_src_float[:, :, c] - src_mean) * (ref_std / src_std)) + ref_mean
        else:
            result[:, :, c] = img_src_float[:, :, c]
    
    # Clip to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result, wb_params

'''
def align_images(
    path1: str,
    path2: str,
    out_dir: str = ".",
    model: str = "translation",
    focus_percentile: float = 60.0,
    focus_ksize: int = 15,
    ransac_thresh: float = 2.0,
    ratio_thresh: float = 0.75,
    min_inliers: int = 20,
    alpha: float = 0.4,
    save_focus_mask: bool = True,
) -> dict:
    """
    Align I2 to I1 using focus-masked SIFT matching.

    The focus mask is computed on I2 to restrict keypoint detection to its
    in-focus (spectrally valid) regions, avoiding correspondences from blurred
    regions whose descriptor space is distorted.

    Parameters
    ----------
    model           : "translation" (2 DOF) or "similarity" (4 DOF, handles
                      aperture breathing / slight magnification change)
    focus_percentile: percentile threshold on per-pixel focus measure F(x);
                      higher value = smaller, more conservative mask
    ransac_thresh   : RANSAC inlier threshold in pixels
    ratio_thresh    : Lowe ratio test threshold
    alpha           : I1 weight in alpha overlay (I2 weight = 1 - alpha)

    Outputs (in out_dir)
    --------------------
    aligned.png       warped I2 on I1's canvas
    overlay.png       alpha blend over common crop
    I1_crop.png       I1 cropped to common region
    I2_crop.png       aligned I2 cropped to common region
    focus_mask.png    binary focus mask on I2 (if save_focus_mask=True)
    alignment.json    M (2x3), crop, tx, ty, scale, n_inliers, method
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    assert img1 is not None, f"Cannot load {path1}"
    assert img2 is not None, f"Cannot load {path2}"
    assert img1.shape == img2.shape, (
        f"Shape mismatch: {img1.shape} vs {img2.shape}"
    )
    h, w = img1.shape[:2]

    # --- Focus mask on I2 ---
    print("[focus mask] computing on I2 ...")
    mask2 = focus_mask(
        img2,
        ksize=focus_ksize,
        percentile_thresh=focus_percentile,
    )
    masked_frac = mask2.astype(bool).mean()
    print(f"  Masked fraction of I2: {masked_frac:.2%}")
    if save_focus_mask:
        cv2.imwrite(str(out_path / "focus_mask.png"), mask2)

    # --- Focus-masked SIFT ---
    print(f"[SIFT+RANSAC] model={model}")
    result = sift_translation(
        img1, img2,
        mask2=mask2,
        ransac_thresh=ransac_thresh,
        ratio_thresh=ratio_thresh,
        min_inliers=min_inliers,
        model=model,
    )

    if result is None:
        # Retry without mask as last resort
        print("  Masked SIFT failed; retrying without focus mask ...")
        result = sift_translation(
            img1, img2,
            mask2=None,
            ransac_thresh=ransac_thresh,
            ratio_thresh=ratio_thresh,
            min_inliers=min_inliers,
            model=model,
        )
    if result is None:
        raise RuntimeError("SIFT alignment failed with and without focus mask.")

    M, info = result
    print(f"  tx={info['tx']:.3f}px, ty={info['ty']:.3f}px, "
          f"scale={info['scale']:.6f}, inliers={info['n_inliers']}")

    # --- Warp I2 ---
    img2_aligned = cv2.warpAffine(
        img2, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    cv2.imwrite(str(out_path / "aligned.png"), img2_aligned)

    # --- Common crop ---
    crop = compute_common_crop(M, h, w)
    x0, y0, x1, y1 = crop["x0"], crop["y0"], crop["x1"], crop["y1"]
    print(f"[crop] x:[{x0},{x1}), y:[{y0},{y1}), size={x1-x0}x{y1-y0}")

    img1_crop        = img1[y0:y1, x0:x1]
    img2_crop        = img2_aligned[y0:y1, x0:x1]
    cv2.imwrite(str(out_path / "I1_crop.png"),  img1_crop)
    cv2.imwrite(str(out_path / "I2_crop.png"),  img2_crop)

    # --- Alpha overlay ---
    overlay = cv2.addWeighted(img1_crop, alpha, img2_crop, alpha, 0.0)
    cv2.imwrite(str(out_path / "overlay.png"), overlay)

    # --- Persist ---
    out = {
        "tx":       info["tx"],
        "ty":       info["ty"],
        "scale":    info["scale"],
        "M":        M.tolist(),
        "crop":     crop,
        "n_inliers": info["n_inliers"],
        "n_matches": info["n_matches"],
        "model":    model,
    }
    # with open(out_path / "alignment.json", "w") as f:
    #     json.dump(out, f, indent=2)

    print(f"[done] outputs in {out_path.resolve()}")
    return out
    '''


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
    """
    Process directories of images: align source to reference, apply white balance, and save results.
    
    Args:
        ref_dir: Directory containing reference images (img1)
        src_dir: Directory containing source images (img2)
        model: "translation" or "similarity"
        focus_percentile: Percentile threshold for focus mask
        focus_ksize: Kernel size for focus measure
        ransac_thresh: RANSAC inlier threshold in pixels
        ratio_thresh: Lowe ratio test threshold
        min_inliers: Minimum number of inliers required
        alpha: Alpha blending factor for overlay
    """
    ref_path = Path(ref_dir)
    src_path = Path(src_dir)
    
    # Get image files
    ref_images = sorted([f for f in ref_path.glob("*") if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']])
    src_images = sorted([f for f in src_path.glob("*") if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']])
    
    # Check equal number of images
    if len(ref_images) != len(src_images):
        raise ValueError(f"Number of images mismatch: reference={len(ref_images)}, source={len(src_images)}")
    
    if len(ref_images) == 0:
        raise ValueError("No images found in the specified directories")
    
    print(f"Found {len(ref_images)} image pairs to process")
    
    # Create output directories
    #ref_align_dir = ref_path.parent / f"{ref_path.name}_align"
    src_align_dir = src_path.parent / f"{src_path.name}_align"
    ref_align_wb_dir = ref_path.parent / f"{ref_path.name}_align_whitebal"
    src_align_wb_dir = src_path.parent / f"{src_path.name}_align_whitebal"
    
    for d in [src_align_dir, ref_align_wb_dir, src_align_wb_dir]:
        d.mkdir(exist_ok=True)
    
    # Process each image pair
    for idx, (ref_img_path, src_img_path) in enumerate(zip(ref_images, src_images)):
        print(f"\n{'='*60}")
        print(f"Processing pair {idx+1}/{len(ref_images)}")
        print(f"  Reference: {ref_img_path.name}")
        print(f"  Source:    {src_img_path.name}")
        print(f"{'='*60}")
        
        # Load images
        img1 = cv2.imread(str(ref_img_path))
        img2 = cv2.imread(str(src_img_path))
        
        if img1 is None or img2 is None:
            print(f"  ERROR: Failed to load images, skipping pair")
            continue
        
        h, w = img1.shape[:2]
        
        # Generate focus mask for img2
        print(f"[focus mask] ksize={focus_ksize}, percentile={focus_percentile}")
        mask2 = focus_mask(
            img2,
            ksize=focus_ksize,
            percentile_thresh=focus_percentile,
        )
        
        # SIFT alignment
        print(f"[SIFT+RANSAC] model={model}")
        result = sift_translation(
            img1, img2,
            mask2=mask2,
            ransac_thresh=ransac_thresh,
            ratio_thresh=ratio_thresh,
            min_inliers=min_inliers,
            model=model,
        )
        
        if result is None:
            print("  Masked SIFT failed; retrying without focus mask ...")
            result = sift_translation(
                img1, img2,
                mask2=None,
                ransac_thresh=ransac_thresh,
                ratio_thresh=ratio_thresh,
                min_inliers=min_inliers,
                model=model,
            )
        
        if result is None:
            print(f"  ERROR: SIFT alignment failed, skipping pair")
            continue
        
        M, info = result
        print(f"  tx={info['tx']:.3f}px, ty={info['ty']:.3f}px, "
              f"scale={info['scale']:.6f}, inliers={info['n_inliers']}")
        
        # Warp img2
        img2_aligned = cv2.warpAffine(
            img2, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        
        # Compute common crop
        crop = compute_common_crop(M, h, w)
        x0, y0, x1, y1 = crop["x0"], crop["y0"], crop["x1"], crop["y1"]
        print(f"[crop] x:[{x0},{x1}), y:[{y0},{y1}), size={x1-x0}x{y1-y0}")
        
        # Crop both images
        img1_crop = img1[y0:y1, x0:x1]
        img2_crop = img2_aligned[y0:y1, x0:x1]
        
        # Save aligned images
        ref_stem = ref_img_path.stem
        src_stem = src_img_path.stem
        
        #cv2.imwrite(str(ref_align_dir / f"{ref_stem}.png"), img1_crop)
        cv2.imwrite(str(src_align_dir / f"{src_stem}.png"), img2_crop)
        
        # Save alignment JSON
        alignment_data = {
            "reference_image": ref_img_path.name,
            "source_image": src_img_path.name,
            "transform_matrix": M.tolist(),
            "tx": info["tx"],
            "ty": info["ty"],
            "scale": info["scale"],
            "crop_x0": x0,
            "crop_y0": y0,
            "crop_x1": x1,
            "crop_y1": y1,
            "n_inliers": info["n_inliers"],
            "n_matches": info["n_matches"],
            "model": model,
        }
        
        with open(ref_align_wb_dir / f"{ref_stem}_alignment.json", "w") as f:
            json.dump(alignment_data, f, indent=2)
        with open(src_align_wb_dir / f"{src_stem}_alignment.json", "w") as f:
            json.dump(alignment_data, f, indent=2)
        
        # White balance matching
        print("[white balance] matching source to reference")
        img2_wb, wb_params = match_white_balance(img2_crop, img1_crop)
        
        # Save white-balanced images
        cv2.imwrite(str(ref_align_wb_dir / f"{ref_stem}.png"), img1_crop)
        cv2.imwrite(str(src_align_wb_dir / f"{src_stem}.png"), img2_wb)
        
        # Add white balance parameters to alignment data
        alignment_data_wb = alignment_data.copy()
        alignment_data_wb["white_balance"] = wb_params
        
        # Save updated JSON with white balance info
        with open(ref_align_wb_dir / f"{ref_stem}_alignment.json", "w") as f:
            json.dump(alignment_data_wb, f, indent=2)
        with open(src_align_wb_dir / f"{src_stem}_alignment.json", "w") as f:
            json.dump(alignment_data_wb, f, indent=2)

        # Create alpha blends
        overlay_align = cv2.addWeighted(img1_crop, alpha, img2_crop, alpha, 0.0) #1.0 - alpha
        overlay_wb = cv2.addWeighted(img1_crop, alpha, img2_wb, alpha, 0.0) #1.0 - alpha
        
        # Create visualization plot
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Convert BGR to RGB for matplotlib
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1_crop_rgb = cv2.cvtColor(img1_crop, cv2.COLOR_BGR2RGB)
        img2_crop_rgb = cv2.cvtColor(img2_crop, cv2.COLOR_BGR2RGB)
        img2_wb_rgb = cv2.cvtColor(img2_wb, cv2.COLOR_BGR2RGB)
        overlay_align_rgb = cv2.cvtColor(overlay_align, cv2.COLOR_BGR2RGB)
        overlay_wb_rgb = cv2.cvtColor(overlay_wb, cv2.COLOR_BGR2RGB)
        
        # Row 1: Original images
        axes[0, 0].imshow(img1_rgb)
        axes[0, 0].set_title(f"Reference (original)\n{ref_img_path.name}", fontsize=10)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img2_rgb)
        axes[0, 1].set_title(f"Source (original)\n{src_img_path.name}", fontsize=10)
        axes[0, 1].axis('off')
        
        axes[0, 2].axis('off')  # Empty
        
        # Row 2: Aligned and cropped
        axes[1, 0].imshow(img1_crop_rgb)
        axes[1, 0].set_title(f"Reference (cropped)\n{img1_crop.shape[1]}x{img1_crop.shape[0]}", fontsize=10)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(img2_crop_rgb)
        axes[1, 1].set_title(f"Source (aligned & cropped)\ntx={info['tx']:.1f}, ty={info['ty']:.1f}", fontsize=10)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(overlay_align_rgb)
        axes[1, 2].set_title(f"Alpha blend (α={alpha})\nAligned only", fontsize=10)
        axes[1, 2].axis('off')
        
        # Row 3: White balanced
        axes[2, 0].imshow(img1_crop_rgb)
        axes[2, 0].set_title(f"Reference (same as row 2)", fontsize=10)
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(img2_wb_rgb)
        axes[2, 1].set_title(f"Source (white balanced)", fontsize=10)
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(overlay_wb_rgb)
        axes[2, 2].set_title(f"Alpha blend (α={alpha})\nAligned + White balanced", fontsize=10)
        axes[2, 2].axis('off')
        
        plt.suptitle(f"Image Pair {idx+1}/{len(ref_images)}: Alignment & White Balance", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = ref_align_wb_dir.parent / f"alignment_plot_{idx+1:03d}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[plot] saved to {plot_path}")
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"  Aligned images:        {src_align_dir}")
    print(f"  White-balanced images: {ref_align_wb_dir}, {src_align_wb_dir}")
    print(f"  Plots saved in:        {ref_align_wb_dir.parent}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Align and white-balance image directories using focus-masked SIFT + RANSAC."
    )
    parser.add_argument("ref_dir", help="Reference image directory")
    parser.add_argument("src_dir", help="Source image directory")
    parser.add_argument("--model", default="translation",
                        choices=["translation", "similarity"],
                        help="translation (2DOF) or similarity (4DOF, for aperture breathing)")
    parser.add_argument("--focus_pct", type=float, default=60.0,
                        help="Percentile threshold for focus mask on source images (default 60)")
    parser.add_argument("--ransac", type=float, default=2.0,
                        help="RANSAC inlier threshold in pixels (default 2.0)")
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="Lowe ratio test threshold (default 0.75)")
    parser.add_argument("--min_inliers", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha blending factor for overlay (default 0.5)")
    args = parser.parse_args()

    process_image_directories(
        args.ref_dir,
        args.src_dir,
        model=args.model,
        focus_percentile=args.focus_pct,
        ransac_thresh=args.ransac,
        ratio_thresh=args.ratio,
        min_inliers=args.min_inliers,
        alpha=args.alpha,
    )