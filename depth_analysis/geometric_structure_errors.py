"""
geometric_structure_errors.py
==============================
Gradient-consistency and local planarity error metrics for depth maps.

Key functions
-------------
compute_grad         Compute per-pixel image gradient magnitude (Sobel).
compute_grad_error   Penalise large depth gradients in smooth image regions.
get_planarity_error  Local PCA-based planarity error (smallest eigenvalue of 3D patch covariance).

Usage
-----
::

    from depth_analysis.geometric_structure_errors import (
        compute_grad, compute_grad_error, get_planarity_error
    )
"""

import numpy as np
import cv2
from scipy.ndimage import uniform_filter


def compute_grad(image: np.ndarray, k: int = 7) -> np.ndarray:
    """Compute per-pixel gradient magnitude of a colour image using Sobel filters.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape ``(H, W, 3)``, dtype uint8.
    k : int
        Sobel kernel size.  Larger values smooth out noise for high-resolution
        images.  Recommended: 5–7 for full-frame sensors.

    Returns
    -------
    np.ndarray
        Gradient magnitude map of shape ``(H, W)``, dtype float32, normalised
        to ``[0, 1]``.
    """
    I_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    I_gray = (I_gray - I_gray.min()) / (I_gray.max() - I_gray.min() + 1e-8)
    g = (
        np.abs(cv2.Sobel(I_gray, cv2.CV_32F, 1, 0, ksize=k))
        + np.abs(cv2.Sobel(I_gray, cv2.CV_32F, 0, 1, ksize=k))
    )
    return g


def compute_grad_error(depth_map: np.ndarray, g_i: np.ndarray,
                       alpha: float = 1.0, k: int = 7) -> np.ndarray:
    """Gradient-consistency error map.

    Penalises large depth gradients in image regions that are locally smooth,
    using the formula::

        error = |∇D| / c  ·  exp(-α · |∇I|)

    where ``c`` is a fixed normalisation constant, ``∇D`` is the depth
    gradient, and ``∇I`` is the image gradient (``g_i``).

    Parameters
    ----------
    depth_map : np.ndarray
        2-D depth map of shape ``(H, W)``.
    g_i : np.ndarray
        Normalised image gradient magnitude, shape ``(H, W)``.
    alpha : float
        Penalty strength.  Higher values suppress errors more aggressively in
        textured regions.
    k : int
        Sobel kernel size for depth gradient computation.

    Returns
    -------
    np.ndarray
        Error map of shape ``(H, W)``, dtype float32.
    """
    d = depth_map.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)

    g_d_x = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=k)
    g_d_y = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=k)
    g_d = np.abs(g_d_x) + np.abs(g_d_y)

    fixed_min_grad = 0.1
    g_d = g_d / fixed_min_grad

    error_map = g_d * np.exp(-alpha * g_i)
    return error_map


def get_planarity_error(X_c: np.ndarray,
                        patch_size: int = 7) -> tuple:
    """Local PCA planarity error derived from the 3-D point cloud.

    For every pixel a ``patch_size × patch_size`` neighbourhood of 3-D points
    (in camera coordinates) is considered.  The covariance matrix of those
    points is estimated via box-filtered outer products.  The smallest
    eigenvalue ``λ₃`` quantifies how far the patch deviates from a perfect
    plane.

    Parameters
    ----------
    X_c : np.ndarray
        3-D points in camera frame, shape ``(H, W, 3)``, float32.
    patch_size : int
        Local neighbourhood size for the box-filter PCA estimate.

    Returns
    -------
    planarity_error : np.ndarray
        ``λ₃`` clipped to ``[0, ∞)``, shape ``(H, W)``.
    rms_orth : np.ndarray
        RMS orthogonal distance to the local best-fit plane, shape ``(H, W)``.
    Prel : np.ndarray
        Relative planarity ``λ₃ / (λ₁ + λ₂ + λ₃)``.  Smaller → more planar.
    Pnorm : np.ndarray
        Depth-normalised planarity ``λ₃ / Z̄²``.
    """
    H, W, _ = X_c.shape

    mu = np.stack(
        [uniform_filter(X_c[..., c], size=patch_size, mode='nearest') for c in range(3)],
        axis=-1,
    ).astype(np.float32)

    XX = X_c[..., :, None] * X_c[..., None, :]
    E = np.stack(
        [uniform_filter(XX[..., i, j], size=patch_size, mode='nearest')
         for i in range(3) for j in range(3)],
        axis=-1,
    ).reshape(H, W, 3, 3)

    mu_outer = mu[..., :, None] * mu[..., None, :]
    Cov = E - mu_outer

    eigs = np.linalg.eigvalsh(Cov)     # (H, W, 3) — ascending order
    lam1 = eigs[..., 2]
    lam2 = eigs[..., 1]
    lam3 = eigs[..., 0]                # smallest eigenvalue

    rms_orth = np.sqrt(np.clip(lam3, a_min=0.0, a_max=None))

    denom = lam1 + lam2 + lam3 + 1e-12
    Prel = lam3 / denom

    Z = X_c[..., 2]
    Z_mean = uniform_filter(Z, size=patch_size, mode='nearest')
    Pnorm = lam3 / (Z_mean ** 2 + 1e-12)

    planarity_error = np.clip(lam3, a_min=0, a_max=None)
    return planarity_error, rms_orth, Prel, Pnorm


if __name__ == "__main__":
    print("geometric_structure_errors module loaded successfully.")
