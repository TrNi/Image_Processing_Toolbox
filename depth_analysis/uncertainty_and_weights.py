"""
uncertainty_and_weights.py
===========================
Ensemble uncertainty estimation and weighted depth fusion.

Key functions
-------------
calculate_individual_mad_uncertainty  Per-map MAD uncertainty relative to ensemble median.
get_iqr_uncertainty                   IQR-normalised uncertainty for each depth map.
simple_weighted_fusion                Fuse depth maps using inverse-uncertainty weights.

Usage
-----
::

    from depth_analysis.uncertainty_and_weights import get_iqr_uncertainty, simple_weighted_fusion
    iqr_maps = get_iqr_uncertainty(depth_stack)   # shape (K, H, W)
    fused    = simple_weighted_fusion(depth_stack, iqr_maps)
"""

import numpy as np


def calculate_individual_mad_uncertainty(depth_maps: np.ndarray) -> np.ndarray:
    """Compute per-map MAD uncertainty relative to the leave-one-out ensemble median.

    For each depth map *k*, the uncertainty is the absolute deviation from the
    median of the remaining *K-1* maps::

        U_k = |D_k - median({D_j : j ≠ k})|

    Parameters
    ----------
    depth_maps : np.ndarray
        Stack of depth maps, shape ``(K, H, W)``.

    Returns
    -------
    np.ndarray
        Uncertainty maps, shape ``(K, H, W)``.
    """
    num_maps = depth_maps.shape[0]
    uncertainty_maps = np.zeros_like(depth_maps)

    for k in range(num_maps):
        other_maps = np.concatenate(
            (depth_maps[:k], depth_maps[k + 1:]), axis=0
        )
        ensemble_median = np.median(other_maps, axis=0)
        uncertainty_maps[k] = np.abs(depth_maps[k] - ensemble_median)

    return uncertainty_maps


def get_iqr_uncertainty(depth_maps: np.ndarray) -> np.ndarray:
    """Compute IQR-normalised uncertainty for each depth map.

    For each map *k*, the uncertainty is the deviation from the leave-one-out
    median divided by the leave-one-out interquartile range (IQR)::

        U_k = |D_k - median| / max(IQR, p1)

    This makes the uncertainty scale-invariant across different depth ranges.
    Low IQR indicates high ensemble consensus; high IQR signals disagreement.

    Parameters
    ----------
    depth_maps : np.ndarray
        Stack of depth maps, shape ``(K, H, W)``.

    Returns
    -------
    np.ndarray
        Uncertainty maps, shape ``(K, H, W)``, clipped at the 99th percentile
        to suppress outliers.
    """
    num_maps = depth_maps.shape[0]
    uncertainty_maps = np.zeros((num_maps, depth_maps.shape[1], depth_maps.shape[2]),
                                dtype=np.float32)
    epsilon = 1e-4

    for k in range(num_maps):
        other_maps = np.concatenate(
            (depth_maps[:k], depth_maps[k + 1:]), axis=0
        )
        ensemble_median = np.median(other_maps, axis=0)
        q0 = np.percentile(other_maps, 1, axis=0)
        q1 = np.percentile(other_maps, 25, axis=0)
        q3 = np.percentile(other_maps, 75, axis=0)

        iqr = np.maximum(q3 - q1, q0)
        umap = np.abs(depth_maps[k] - ensemble_median) / (iqr + epsilon)
        uncertainty_maps[k] = np.clip(umap, 0, np.percentile(umap, 99))

    return uncertainty_maps


def simple_weighted_fusion(depth_maps: np.ndarray,
                           uncertainty_maps: np.ndarray) -> np.ndarray:
    """Fuse depth maps using inverse-uncertainty weights.

    Pixels with lower uncertainty receive higher weight::

        w_k = 1 / (U_k + ε)
        D_fused = Σ_k (w_k · D_k) / Σ_k w_k

    Parameters
    ----------
    depth_maps : np.ndarray
        Stack of depth maps, shape ``(K, H, W)``.
    uncertainty_maps : np.ndarray
        Corresponding uncertainty maps, same shape as ``depth_maps``.

    Returns
    -------
    np.ndarray
        Fused depth map, shape ``(H, W)``.
    """
    epsilon = 1e-4
    weights = 1.0 / (uncertainty_maps + epsilon)
    numerator = np.sum(depth_maps * weights, axis=0)
    denominator = np.sum(weights, axis=0)
    return numerator / denominator


if __name__ == "__main__":
    print("uncertainty_and_weights module loaded successfully.")
    rng = np.random.default_rng(42)
    dummy = rng.random((4, 16, 16)).astype(np.float32) * 10 + 1.0
    iqr = get_iqr_uncertainty(dummy)
    fused = simple_weighted_fusion(dummy, iqr)
    print(f"  iqr maps shape : {iqr.shape}")
    print(f"  fused map shape: {fused.shape}")
    print("  OK")
