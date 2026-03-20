"""
align_images.py
===============
Align two images via phase correlation (with ORB fallback).

Produces four output files in the output directory:

- ``aligned.png``      — I2 warped to I1's coordinate frame.
- ``overlay.png``      — alpha-blended I1 and aligned I2 over the common crop.
- ``I1_crop.png``      — I1 cropped to the overlapping region.
- ``I2_crop.png``      — aligned I2 cropped to the overlapping region.
- ``alignment.json``   — transform matrix M (2×3), crop box, method used.

Usage
-----
::

    python image_processing/align_images.py img1.png img2.png \\
        --out_dir alignment_out --threshold 0.02 --alpha 0.5
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def luminance(img_bgr: np.ndarray) -> np.ndarray:
    """BT.601 luminance from a BGR uint8 image, returned as float32."""
    return (0.299 * img_bgr[..., 2] +
            0.587 * img_bgr[..., 1] +
            0.114 * img_bgr[..., 0]).astype(np.float32)


def phase_correlate(g1: np.ndarray, g2: np.ndarray) -> tuple:
    """Sub-pixel translation estimate via windowed phase correlation.

    Parameters
    ----------
    g1, g2 : np.ndarray
        Float32 grayscale images of the same shape.

    Returns
    -------
    tx, ty : float
        Translation such that I2 shifted by (tx, ty) aligns with I1.
    response : float
        Peak correlation response in [0, 1].  Low values indicate unreliable
        estimates.
    """
    h, w = g1.shape
    win  = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (tx, ty), response = cv2.phaseCorrelate(g1 * win, g2 * win)
    return float(tx), float(ty), float(response)


def orb_translation_fallback(img1_bgr: np.ndarray, img2_bgr: np.ndarray,
                              max_features: int = 5000,
                              ransac_thresh: float = 3.0):
    """ORB + brute-force Hamming + RANSAC translation-only fallback.

    Parameters
    ----------
    img1_bgr, img2_bgr : np.ndarray
        Input images in BGR format.
    max_features : int
        Maximum ORB keypoints to detect.
    ransac_thresh : float
        Inlier threshold in pixels.

    Returns
    -------
    tuple or None
        ``(tx, ty)`` or ``None`` if insufficient inliers.
    """
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(img1_bgr, None)
    kp2, des2 = orb.detectAndCompute(img2_bgr, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)[:500]
    if len(matches) < 4:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    deltas   = pts1 - pts2
    t_init   = np.median(deltas, axis=0)
    residuals = np.linalg.norm(deltas - t_init, axis=1)
    inliers  = residuals < ransac_thresh

    if inliers.sum() < 4:
        return None

    tx, ty = deltas[inliers].mean(axis=0)
    return float(tx), float(ty)


def compute_common_crop(tx: float, ty: float, h: int, w: int) -> dict:
    """Compute the axis-aligned common crop after a (tx, ty) translation.

    Parameters
    ----------
    tx, ty : float
        Translation applied to I2 to align with I1.
    h, w : int
        Image height and width.

    Returns
    -------
    dict
        Keys ``x0``, ``y0``, ``x1``, ``y1`` (exclusive end).
    """
    x0 = int(np.ceil(max(0.0, tx)))
    y0 = int(np.ceil(max(0.0, ty)))
    x1 = int(np.floor(min(w, w + tx)))
    y1 = int(np.floor(min(h, h + ty)))
    assert x1 > x0 and y1 > y0, (
        "No overlapping region — translation too large."
    )
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


# ---------------------------------------------------------------------------
# Main alignment function
# ---------------------------------------------------------------------------

def align_images(path1: str, path2: str, out_dir: str = ".",
                 phase_corr_threshold: float = 0.02,
                 alpha: float = 0.5) -> dict:
    """Align *path2* to *path1* via phase correlation (ORB fallback).

    Parameters
    ----------
    path1 : str
        Reference image path.
    path2 : str
        Image to align.
    out_dir : str
        Output directory.
    phase_corr_threshold : float
        Minimum phase-correlation peak response.  Values below this trigger
        the ORB fallback.
    alpha : float
        I1 weight in the alpha-blended overlay (0–1).

    Returns
    -------
    dict
        Keys: ``tx``, ``ty``, ``M`` (2×3 list), ``crop`` (dict),
        ``method`` (str), ``phase_response`` (float).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img1 = cv2.imread(str(path1))
    img2 = cv2.imread(str(path2))
    assert img1 is not None, f"Failed to load: {path1}"
    assert img2 is not None, f"Failed to load: {path2}"
    assert img1.shape == img2.shape, (
        f"Image shapes must match: {img1.shape} vs {img2.shape}"
    )

    h, w = img1.shape[:2]
    g1   = luminance(img1)
    g2   = luminance(img2)

    tx, ty, response = phase_correlate(g1, g2)
    method = "phase_correlation"
    print(f"[phase_correlation] tx={tx:.3f}  ty={ty:.3f}  response={response:.4f}")

    if response < phase_corr_threshold:
        print(f"  Response below threshold ({response:.4f} < {phase_corr_threshold}); "
              f"falling back to ORB+RANSAC.")
        result = orb_translation_fallback(img1, img2)
        if result is None:
            raise RuntimeError("Both phase correlation and ORB fallback failed.")
        tx, ty = result
        method = "orb_ransac"
        print(f"[orb_ransac] tx={tx:.3f}  ty={ty:.3f}")

    M = np.array([[1.0, 0.0, tx],
                  [0.0, 1.0, ty]], dtype=np.float64)

    img2_aligned = cv2.warpAffine(
        img2, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    cv2.imwrite(str(out_path / "aligned.png"), img2_aligned)

    crop = compute_common_crop(tx, ty, h, w)
    x0, y0, x1, y1 = crop["x0"], crop["y0"], crop["x1"], crop["y1"]
    print(f"[crop] x:[{x0},{x1})  y:[{y0},{y1})  size={x1-x0}×{y1-y0}")

    img1_crop = img1[y0:y1, x0:x1]
    img2_crop = img2_aligned[y0:y1, x0:x1]
    cv2.imwrite(str(out_path / "I1_crop.png"), img1_crop)
    cv2.imwrite(str(out_path / "I2_crop.png"), img2_crop)

    overlay = cv2.addWeighted(img1_crop, alpha, img2_crop, 1.0 - alpha, 0.0)
    cv2.imwrite(str(out_path / "overlay.png"), overlay)

    result_dict = {
        "tx":             float(tx),
        "ty":             float(ty),
        "M":              M.tolist(),
        "crop":           crop,
        "method":         method,
        "phase_response": float(response),
    }
    with open(out_path / "alignment.json", "w") as fh:
        json.dump(result_dict, fh, indent=2)

    print(f"[done] outputs written to: {out_path.resolve()}")
    return result_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Align img2 to img1 via phase correlation or ORB fallback.",
    )
    parser.add_argument("img1", help="Reference image path.")
    parser.add_argument("img2", help="Image to align.")
    parser.add_argument("--out_dir",   default="alignment_out",
                        help="Output directory (default: alignment_out).")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Phase correlation response threshold (default: 0.02).")
    parser.add_argument("--alpha",     type=float, default=0.5,
                        help="I1 weight in alpha overlay (default: 0.5).")
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
    print(f"  Crop   : {result['crop']}")


if __name__ == "__main__":
    main()
