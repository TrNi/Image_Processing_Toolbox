import numpy as np
from skimage.util import view_as_windows
import cv2
from scipy.ndimage import uniform_filter

def get_planarity_error(X_c, patch_size=7):
    H, W, _ = X_c.shape
    # 1) compute per‐channel local means μ_c = box_filter(X_c[...,c])    
    mu = np.stack([
        uniform_filter(X_c[..., c], size=patch_size, mode='nearest')
        for c in range(3)
    ], axis=-1).astype(np.float32)
    # 2) compute box‐filtered second moments E[x_c * x_d] for c≤d
    XX = X_c[..., :, None] * X_c[..., None, :]
    E = np.stack([
        uniform_filter(XX[..., i, j], size=patch_size, mode='nearest')
        for i in range(3) for j in range(3)
    ], axis=-1).reshape(H, W, 3, 3)
    
    # 3) form covariance matrix per‐pixel: Cov = E - μ⊗μ
    #    Cov_{c,d} = E[x_c x_d] - μ_c μ_d
    mu_outer = mu[..., :, None] * mu[..., None, :]
    Cov = E - mu_outer

    # 4) batched eigensolve: smallest eigenvalue at index 0
    #    np.linalg.eigvalsh works on stacked last‐two dims
    eigs = np.linalg.eigvalsh(Cov)        # (H, W, 3)
    lam1 = eigs[..., 2]                   # largest
    lam2 = eigs[..., 1]
    lam3 = eigs[..., 0]                   # smallest

    # 1. RMS orthogonal error (meters)
    rms_orth = np.sqrt(np.clip(lam3, a_min=0.0, a_max=None))

    # 2. Prel = lam3 / (lam1 + lam2 + lam3) # relative smallest/sum ratio. smaller means more planar
    denom = lam1 + lam2 + lam3 + 1e-12
    Prel = lam3 / denom

    # 3. Pnorm = lam3 / (mean depth in patch)^2
    # compute patch-mean depth
    Z = X_c[..., 2]
    Z_mean = uniform_filter(Z, size=patch_size, mode='nearest')
    Pnorm = lam3 / (Z_mean**2 + 1e-12)

    # your original return of λ3 as “planarity error”
    planarity_error = np.clip(lam3, a_min=0, a_max=None)

    return planarity_error, rms_orth, Prel, Pnorm
'''
def get_planarity_error(X_c, patch_size=7):
    """
    Computes a planarity error map for a given depth map using PCA.
    For each pixel in the image, take a small kxk patch and unproject it to 3D camera frame.
    PCA on these 3D points: 
    - principal components are the eigenvectors of the covariance matrix and give orientation of the best plane fitting the 3D points.
    - Lambda1 >= Lambda2 >= Lambda3.
    - The largest eigenvalue Lambda1 is the variance along the plane
    - The smallest eigenvalue Lambda3 is the variance orthogonal to the plane
    - The ratio of these two eigenvalues Lambda1/Lambda3 tells how co-planar the 3D points are.
    - Lambda3 is the planarity error.

    Args:
        X_c_left (np.ndarray): The 3D points in left camera frame (H, W, 3).
        patch_size (int): The size of the local patch (e.g., 5, 7).

    Returns:
        np.ndarray: The planarity error map (H, W).
    """
    H, W, _ = X_c.shape
    # 1) compute per‐channel local means μ_c = box_filter(X_c[...,c])    
    mu = np.stack([
        uniform_filter(X_c[..., c], size=patch_size, mode='nearest')
        for c in range(3)
    ], axis=-1).astype(np.float32)                    # shape (H, W, 3)
    # 2) compute box‐filtered second moments E[x_c * x_d] for c≤d
    XX = X_c[..., :, None] * X_c[..., None, :]  # (H,W,3,3)
    E = np.stack([
        uniform_filter(XX[..., i, j], size=patch_size, mode='nearest')
        for i in range(3) for j in range(3)
    ], axis=-1).reshape(H, W, 3, 3)

    
    # E = np.zeros((X_c.shape[0], X_c.shape[1], 3, 3), dtype=np.float32)
    # for c in range(3):
    #     for d in range(c, 3):
    #         prod = X_c[..., c] * X_c[..., d]
    #         Ecd = uniform_filter(prod, size=patch_size, mode='nearest')
    #         E[..., c, d] = Ecd
    #         E[..., d, c] = Ecd   # symmetry

    # 3) form covariance matrix per‐pixel: Cov = E - μ⊗μ
    #    Cov_{c,d} = E[x_c x_d] - μ_c μ_d
    mu_outer = mu[..., :, None] * mu[..., None, :]  # (H,W,3,3)
    Cov = E - mu_outer

    # 4) batched eigensolve: smallest eigenvalue at index 0
    #    np.linalg.eigvalsh works on stacked last‐two dims
    eigs = np.linalg.eigvalsh(Cov) # shape (H, W, 3)    
    return np.clip(eigs[..., 0], min=0)#.astype(np.float16)       # smallest eigenvalue    
'''

def compute_grad(image, k=7):
    # 1. Convert image to grayscale if it's a color image
    I_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    I_gray = (I_gray - I_gray.min()) / (I_gray.max() - I_gray.min())
    return np.abs(cv2.Sobel(I_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=k)) + \
           np.abs(cv2.Sobel(I_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=k))
    

def compute_grad_error(depth_map, g_i, alpha=1, k=7):
    """
    Computes the gradient consistency error map based on the provided formula,
    using cv2.Sobel for gradient calculation.

    This function calculates a penalty for a depth map by penalizing large
    depth gradients in regions where the corresponding image is smooth.

    Args:
        depth_map (np.ndarray): A 2D NumPy array representing the depth map.
        g_i (np.ndarray): A 2D NumPy array representing the gradient of the image.
        alpha (float): A tuning parameter that controls the penalty strength.
        k (int): The size of the Sobel kernel, recommended larger for ful-frame, 
                 high-res images, to avoid pixel-level noisy edges.

    Returns:
        np.ndarray: A 2D NumPy array of the same shape as the depth map,
                    representing the gradient consistency error map.
    """
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())    

    # 2. Compute the gradients using cv2.Sobel for both depth and grayscale image
    # cv2.Sobel calculates either x or y gradient at a time    
    g_d_x = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=k)
    g_d_y = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=k)    
    g_d = np.abs(g_d_x) + np.abs(g_d_y)
    fixed_min_grad = 0.1
    g_d = g_d/fixed_min_grad
    # 4. Compute the final consistency error map element-wise
    error_map = g_d * np.exp(-alpha * g_i)
    print('max ∇I:', g_i.max())
    print('min exp(-alpha ∇I):', np.exp(-alpha * g_i).min())
    return error_map