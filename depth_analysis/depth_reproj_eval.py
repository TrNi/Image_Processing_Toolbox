"""
depth_reproj_eval.py
====================
Camera geometry utilities, depth-map reprojection, and photometric error maps.

Key functions
-------------
load_h5_images          Load an image array from an HDF5 dataset.
load_camera_params      Load stereo calibration parameters from a .npz file.
get_Kinv_uv1            Pre-compute K⁻¹·[u,v,1]ᵀ ray directions for a full image grid.
px_to_camera            Back-project a depth map to 3-D camera coordinates.
project_to_view         Project 3-D points from one camera frame to another image plane.
photometric_errors      Compute L1 / L2 / SSIM error maps via right-image warping.
get_errors              Bundle all error types (gradient, planarity, photometric) for one image.

Usage
-----
Run as a script to verify imports::

    python depth_analysis/depth_reproj_eval.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from math import e  # noqa: F401 (kept for backward compatibility)
import os
import h5py
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim
from depth_analysis.geometric_structure_errors import compute_grad_error, get_planarity_error, compute_grad


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_h5_images(h5_path: str) -> np.ndarray:
    """Load images from an HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.  The dataset is expected to be stored under
        the key ``'data'``.

    Returns
    -------
    np.ndarray
        Image array of shape ``(N, H, W, C)`` (transposed from ``(N, C, H, W)``
        if the on-disk layout is channel-first).
    """
    with h5py.File(h5_path, 'r') as f:
        images = f['data'][()]

    if images.ndim == 4:
        return images.transpose(0, 2, 3, 1)
    return images


def load_camera_params(npz_path: str) -> dict:
    """Load stereo camera calibration parameters from a ``.npz`` file.

    Parameters
    ----------
    npz_path : str
        Path to a NumPy archive produced by a stereo calibration pipeline.
        Expected keys: ``P1``, ``P2``, ``baseline``, ``fB``.

    Returns
    -------
    dict
        Dictionary with keys ``P1``, ``P2``, ``baseline``, ``fB``,
        ``K_new``, ``K_inv``, ``T``.
    """
    data = np.load(npz_path)
    params = {
        'P1': data['P1'],
        'P2': data['P2'],
        'baseline': data['baseline'],
        'fB': data['fB'],
    }
    params['K_new'] = params['P1'][:, :3]
    params['K_inv'] = np.linalg.inv(params['K_new'])
    params['T'] = np.array([params['baseline'], 0, 0])
    return params


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------

def get_Kinv_uv1(K_inv: np.ndarray, H: int, W: int, uv=None) -> np.ndarray:
    """Pre-compute ``K⁻¹ · [u, v, 1]ᵀ`` ray directions for a full image grid.

    Parameters
    ----------
    K_inv : np.ndarray
        3×3 inverse intrinsic matrix.
    H, W : int
        Image height and width in pixels.
    uv : tuple of np.ndarray, optional
        Custom pixel coordinate grids ``(uu, vv)``.  When ``None`` a full
        meshgrid is created automatically.

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 3)`` — one ray direction per pixel.
    """
    if uv is None:
        uu, vv = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    else:
        uu, vv = uv[0], uv[1]

    rays = np.stack(
        [
            K_inv[0, 0] * uu + K_inv[0, 1] * vv + K_inv[0, 2],
            K_inv[1, 0] * uu + K_inv[1, 1] * vv + K_inv[1, 2],
            K_inv[2, 0] * uu + K_inv[2, 1] * vv + K_inv[2, 2],
        ],
        axis=-1,
    )
    return rays


def px_to_camera(depth_map: np.ndarray, K_inv: np.ndarray,
                 K_inv_uv1=None, uv1=None) -> np.ndarray:
    """Back-project a depth map to 3-D camera coordinates.

    Parameters
    ----------
    depth_map : np.ndarray
        2-D depth map of shape ``(H, W)``.
    K_inv : np.ndarray
        3×3 inverse intrinsic matrix.
    K_inv_uv1 : np.ndarray, optional
        Pre-computed ray directions ``(H, W, 3)`` from :func:`get_Kinv_uv1`.
        Providing this avoids redundant computation when the function is
        called in a loop.
    uv1 : np.ndarray, optional
        Custom pixel coordinate array of shape ``(H*W, 3)``.

    Returns
    -------
    np.ndarray
        3-D point cloud of shape ``(H, W, 3)`` (or ``(H*W, 3)`` when
        ``K_inv_uv1`` is ``None``).
    """
    if K_inv_uv1 is not None:
        return depth_map[..., None] * K_inv_uv1

    H, W = depth_map.shape
    if uv1 is None:
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)

    Z_c = depth_map.ravel()
    X_c = Z_c[:, None] * (K_inv @ uv1.T).T
    return X_c


def project_to_view(X_one: np.ndarray, P_two: np.ndarray) -> np.ndarray:
    """Project 3-D points from camera frame 1 to image plane 2.

    Parameters
    ----------
    X_one : np.ndarray
        3-D points in camera 1 frame, shape ``(H, W, 3)`` or ``(N, 3)``.
    P_two : np.ndarray
        3×4 projection matrix of camera 2.

    Returns
    -------
    np.ndarray
        2-D pixel coordinates in camera 2, same leading dimensions as
        ``X_one`` with last dimension 2 (u, v).
    """
    x_two = X_one @ P_two[:, :3].T + P_two[:, 3]
    return x_two[..., :2] / x_two[..., 2, None]


# ---------------------------------------------------------------------------
# Photometric errors
# ---------------------------------------------------------------------------

def photometric_error_ssim(I_L: np.ndarray, I_R_warped: np.ndarray) -> np.ndarray:
    """Per-pixel SSIM dissimilarity map (1 − SSIM).

    Parameters
    ----------
    I_L : np.ndarray
        Left image, shape ``(H, W, 3)``, dtype uint8.
    I_R_warped : np.ndarray
        Right image warped to left frame, same shape and dtype.

    Returns
    -------
    np.ndarray
        Dissimilarity map of shape ``(H, W)`` in ``[0, 2]``.
    """
    I_L_gray = cv2.cvtColor(I_L, cv2.COLOR_RGB2GRAY)
    I_R_gray = cv2.cvtColor(I_R_warped, cv2.COLOR_RGB2GRAY)
    ssim_map = ssim(I_L_gray, I_R_gray, data_range=255, full=True)[1]
    return 1.0 - ssim_map


def photometric_errors(I_L: np.ndarray, I_R: np.ndarray,
                       x_right: np.ndarray,
                       error_types=('l1', 'ssim')) -> dict:
    """Compute photometric error maps between left image and warped right image.

    Parameters
    ----------
    I_L : np.ndarray
        Left image, shape ``(H, W, 3)``, dtype uint8.
    I_R : np.ndarray
        Right image, shape ``(H, W, 3)``, dtype uint8.
    x_right : np.ndarray
        Per-pixel reprojected coordinates in the right image, shape ``(H, W, 2)``.
    error_types : sequence of str
        Which error types to compute.  Choices: ``'l1'``, ``'l2'``, ``'ssim'``.

    Returns
    -------
    dict
        Keys from *error_types*, each mapping to an ``(H, W)`` float32 array.
    """
    H, W, _ = I_L.shape
    u_R = x_right[..., 0].astype(np.float32).reshape(H, W)
    v_R = x_right[..., 1].astype(np.float32).reshape(H, W)

    I_R_warped = cv2.remap(
        I_R, u_R, v_R,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    errors = {}
    if 'l1' in error_types:
        errors['photo_l1'] = np.mean(np.abs(I_L.astype(np.float32) - I_R_warped.astype(np.float32)), axis=-1)
    if 'l2' in error_types:
        errors['photo_l2'] = np.mean((I_L.astype(np.float32) - I_R_warped.astype(np.float32)) ** 2, axis=-1)
    if 'ssim' in error_types:
        errors['photo_ssim'] = photometric_error_ssim(I_L, I_R_warped)
    return errors


def get_errors(depth_left: np.ndarray, rectified_left: np.ndarray,
               rectified_right: np.ndarray, K_inv: np.ndarray,
               K_inv_uv1: np.ndarray, g_i: np.ndarray,
               P2: np.ndarray, alpha: float, kernel: int) -> dict:
    """Bundle all no-reference error types for a single depth map.

    Combines gradient consistency, local planarity, and photometric
    (L1, SSIM) errors computed from the stereo pair.

    Parameters
    ----------
    depth_left : np.ndarray
        Depth map for the left view, shape ``(H, W)``.
    rectified_left, rectified_right : np.ndarray
        Rectified stereo images, shape ``(H, W, 3)``.
    K_inv : np.ndarray
        3×3 inverse intrinsic matrix.
    K_inv_uv1 : np.ndarray
        Pre-computed ray directions, shape ``(H, W, 3)``.
    g_i : np.ndarray
        Normalised image gradient magnitude, shape ``(H, W)``.
    P2 : np.ndarray
        Projection matrix of the right camera, 3×4.
    alpha : float
        Gradient-error tuning parameter (see :func:`compute_grad_error`).
    kernel : int
        Sobel kernel size used for gradient computation.

    Returns
    -------
    dict
        Keys: ``'grad'``, ``'plan'``, ``'rms_orth'``, ``'Prel'``, ``'Pnorm'``,
        ``'photo_l1'``, ``'photo_ssim'``.
    """
    X_c_left = px_to_camera(depth_left, K_inv, K_inv_uv1)
    x_right_2d = project_to_view(X_c_left, P2)
    grad_error = compute_grad_error(depth_left, g_i, alpha, kernel)
    planarity_error, rms_orth, Prel, Pnorm = get_planarity_error(X_c_left)
    photo_errors = photometric_errors(rectified_left, rectified_right, x_right_2d,
                                      error_types=['l1', 'ssim'])
    errors = {
        "grad": grad_error,
        "plan": planarity_error,
        "rms_orth": rms_orth,
        "Prel": Prel,
        "Pnorm": Pnorm,
        **photo_errors,
    }
    return errors


if __name__ == "__main__":
    print("depth_reproj_eval module loaded successfully.")
    print("Available functions:", [
        "load_h5_images", "load_camera_params", "get_Kinv_uv1",
        "px_to_camera", "project_to_view", "photometric_errors",
        "photometric_error_ssim", "get_errors",
    ])
