import cv2
import numpy as np
import json
from pathlib import Path


def luminance(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 image to float32 luminance via BT.601 coefficients."""
    return (0.299 * img_bgr[..., 2] +
            0.587 * img_bgr[..., 1] +
            0.114 * img_bgr[..., 0]).astype(np.float32)


def phase_correlate(g1: np.ndarray, g2: np.ndarray) -> tuple[float, float, float]:
    """
    Estimate sub-pixel translation (tx, ty) such that I2 is shifted by (tx, ty)
    relative to I1, using windowed phase correlation.

    Returns (tx, ty, peak_response). Peak response in [0,1]; low value indicates
    unreliable estimate.
    """
    h, w = g1.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (tx, ty), response = cv2.phaseCorrelate(g1 * win, g2 * win)
    return tx, ty, response


def orb_translation_fallback(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    max_features: int = 5000,
    ransac_thresh: float = 3.0,
) -> tuple[float, float] | None:
    """
    ORB + brute-force Hamming matching + RANSAC with translation-only model (2 DOF).
    Returns (tx, ty) or None if insufficient inliers.
    """
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(img1_bgr, None)
    kp2, des2 = orb.detectAndCompute(img2_bgr, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 4:
        return None

    # Sort by descriptor distance, keep top 500
    matches = sorted(matches, key=lambda m: m.distance)[:500]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # RANSAC: translation-only -- inliers satisfy ||( pts2 - pts1 ) - t|| < thresh
    deltas = pts1 - pts2  # I2->I1 displacement per correspondence
    # Median as initial estimate; then RANSAC-style consensus
    t_init = np.median(deltas, axis=0)
    residuals = np.linalg.norm(deltas - t_init, axis=1)
    inliers = residuals < ransac_thresh

    if inliers.sum() < 4:
        return None

    tx, ty = deltas[inliers].mean(axis=0)
    return float(tx), float(ty)


def compute_common_crop(
    tx: float, ty: float, h: int, w: int
) -> dict[str, int]:
    """
    Compute the axis-aligned bounding box of the overlapping region in I1's
    coordinate frame after aligning I2 by (tx, ty).

    The valid x-range is [max(0, tx), min(W, W+tx)],
    the valid y-range is [max(0, ty), min(H, H+ty)].

    Returns a dict with keys x0, y0, x1, y1 (integer, inclusive-exclusive).
    """
    x0 = int(np.ceil(max(0.0, tx)))
    y0 = int(np.ceil(max(0.0, ty)))
    x1 = int(np.floor(min(w, w + tx)))
    y1 = int(np.floor(min(h, h + ty)))
    assert x1 > x0 and y1 > y0, "No overlapping region; translation too large."
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def align_images(
    path1: str,
    path2: str,
    out_dir: str = ".",
    phase_corr_threshold: float = 0.02,
    alpha: float = 0.5,
) -> dict:
    """
    Align img2 to img1 via phase correlation (with ORB fallback).

    Outputs:
        aligned.png         -- I2 warped to I1's frame (full canvas)
        overlay.png         -- alpha-blended I1 and aligned I2 over common crop
        I1_crop.png         -- I1 cropped to common region
        I2_crop.png         -- aligned I2 cropped to common region
        alignment.json      -- translation matrix M (2x3), crop box, method used

    Parameters
    ----------
    path1, path2            : input image paths
    out_dir                 : output directory
    phase_corr_threshold    : minimum phase correlation peak response to trust result
    alpha                   : blending weight for I1 in overlay (I2 weight = 1-alpha)

    Returns
    -------
    dict with keys: tx, ty, M (list[list[float]]), crop (dict), method (str)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    assert img1 is not None, f"Failed to load {path1}"
    assert img2 is not None, f"Failed to load {path2}"
    assert img1.shape == img2.shape, (
        f"Image shapes must match: {img1.shape} vs {img2.shape}"
    )

    h, w = img1.shape[:2]
    g1 = luminance(img1)
    g2 = luminance(img2)

    # --- Phase correlation ---
    tx, ty, response = phase_correlate(g1, g2)
    method = "phase_correlation"
    print(f"[phase_correlation] tx={tx:.3f}, ty={ty:.3f}, response={response:.4f}")

    if response < phase_corr_threshold:
        print(f"  Peak response {response:.4f} < threshold {phase_corr_threshold}; "
              f"falling back to ORB+RANSAC.")
        result = orb_translation_fallback(img1, img2)
        if result is None:
            raise RuntimeError("Both phase correlation and ORB fallback failed.")
        tx, ty = result
        method = "orb_ransac"
        print(f"[orb_ransac] tx={tx:.3f}, ty={ty:.3f}")

    # --- Build 2x3 affine (translation-only) matrix ---
    # Warps I2 so that a point at (x,y) in I2 maps to (x+tx, y+ty) in I1's frame.
    M = np.array([[1.0, 0.0, tx],
                  [0.0, 1.0, ty]], dtype=np.float64)

    # --- Warp I2 ---
    img2_aligned = cv2.warpAffine(
        img2, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    cv2.imwrite(str(out_path / "aligned.png"), img2_aligned)

    # --- Common crop ---
    crop = compute_common_crop(tx, ty, h, w)
    x0, y0, x1, y1 = crop["x0"], crop["y0"], crop["x1"], crop["y1"]
    print(f"[crop] x:[{x0},{x1}), y:[{y0},{y1}), "
          f"size={x1-x0}x{y1-y0}")

    img1_crop = img1[y0:y1, x0:x1]
    img2_crop = img2_aligned[y0:y1, x0:x1]
    cv2.imwrite(str(out_path / "I1_crop.png"), img1_crop)
    cv2.imwrite(str(out_path / "I2_crop.png"), img2_crop)

    # --- Alpha overlay over common crop ---
    overlay = cv2.addWeighted(img1_crop, alpha, img2_crop, 1.0 - alpha, 0.0)
    cv2.imwrite(str(out_path / "overlay.png"), overlay)

    # --- Persist results ---
    result = {
        "tx": float(tx),
        "ty": float(ty),
        "M": M.tolist(),          # 2x3 list-of-lists, row-major
        "crop": crop,             # {x0, y0, x1, y1} in I1 frame, exclusive end
        "method": method,
        "phase_response": float(response),
    }
    with open(out_path / "alignment.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"[done] outputs written to {out_path.resolve()}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Align img2 to img1 via phase correlation or ORB fallback."
    )
    parser.add_argument("img1", help="Reference image path")
    parser.add_argument("img2", help="Image to align")
    parser.add_argument("--out_dir", default="alignment_out",
                        help="Output directory (default: alignment_out)")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Phase correlation response threshold (default: 0.02)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Blend weight for I1 in overlay (default: 0.5)")
    args = parser.parse_args()

    result = align_images(
        args.img1, args.img2,
        out_dir=args.out_dir,
        phase_corr_threshold=args.threshold,
        alpha=args.alpha,
    )

    print("\nAlignment result:")
    print(f"  Method : {result['method']}")
    print(f"  (tx,ty): ({result['tx']:.3f}, {result['ty']:.3f}) px")
    print(f"  M      : {result['M']}")
    print(f"  Crop   : {result['crop']}")