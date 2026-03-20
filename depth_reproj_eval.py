from math import e
import os
import h5py
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim
from geometric_structure_errors import compute_grad_error, get_planarity_error, compute_grad

def load_h5_images(h5_path):
    """Load images from .h5 file."""
    with h5py.File(h5_path, 'r') as f:
        images = f['data'][()] #[:]  # Shape: (24, 3, 2280, 3420)

    if images.ndim == 4:
        return images.transpose(0, 2, 3, 1)
    else:
        return images


def load_camera_params(npz_path):
    """Load stereo camera parameters from .npz file."""
    data = np.load(npz_path)
    params = {        
        'P1': data['P1'], 'P2': data['P2'],
        'baseline': data['baseline'], 'fB': data['fB']
    }

    params['K_new'] = params['P1'][:, :3]
    params['K_inv'] = np.linalg.inv(params['K_new'])
    params['T'] = np.array([params['baseline'], 0, 0])

    return params

def get_Kinv_uv1(K_inv, H, W, uv=None):
    """
    Returns K_inv @ uv1.T
    """
    if uv is None:
        uu, vv = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    else:
        uu, vv = uv[0], uv[1]
        
    rays = np.stack(
        [
            K_inv[0,0]*uu + K_inv[0,1]*vv + K_inv[0,2],
            K_inv[1,0]*uu + K_inv[1,1]*vv + K_inv[1,2],
            K_inv[2,0]*uu + K_inv[2,1]*vv + K_inv[2,2],
        ], axis=-1)
    return rays


def px_to_camera(depth_map, K_inv, K_inv_uv1=None, uv1=None):
    """
    Convert pixel coordinates to 3D camera coordinates for the same view,
    scaled to match depth.
    """
    if K_inv_uv1 is not None:
        X_c = depth_map[...,None] * K_inv_uv1
        return X_c
    else:
        if uv1 is None:
            H, W = depth_map.shape
            u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
            uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)  # (H*W, 3)

        # Camera coordinates: X_c = Z_c * K^-1 * [u,v,1]
        Z_c = depth_map.ravel()    
        X_c = Z_c[:,None] * (K_inv @ uv1.T).T  # (H*W, 3)
        return X_c

def project_to_view(X_one, P_two):
    """Reproject first camera frame 3D points to second camera 2D image view."""    
    #x_two = X_one_hom @ P_two.T
    x_two = X_one @ P_two[:, :3].T + P_two[:, 3] # H, W, 3
    return x_two[..., :2] / x_two[..., 2, None]    


def depth_cycle_errors(D_left, I_left, x_right, K_inv, P_one, T, fx_B):
    """
    The cycle errors only produce numerical differences - in our framework, since right depth map is unavailable.
    These O(1e-13) numerical cycle errors are unuseful. 
    """    
    return {}
    # H, W = D_left.shape
    # u_l, v_l = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    # x_left = np.stack([u_l, v_l], axis=-1).reshape(-1, 2)

    # # Split u_R and v_R
    # u_r = x_right[..., 0].astype(np.float32).reshape(H, W)
    # v_r = x_right[..., 1].astype(np.float32).reshape(H, W)    
    # ur_vr_1 = np.stack([u_r, v_r, np.ones_like(u_r)], axis=-1).reshape(-1, 3)

    # D_right_prime = fx_B / (u_l-u_r)
    # X_c_right_prime1 = px_to_camera(D_right_prime, K_inv, ur_vr_1)
    # X_c_right_prime_left = X_c_right_prime1 - T
    # D_left_prime = X_c_right_prime_left[:,2].reshape(H, W)

    # x_left_2d_prime = project_to_view(X_c_right_prime_left, P_one)
    # u_l_prime = x_left_2d_prime[..., 0].astype(np.float32).reshape(H, W)
    # v_l_prime = x_left_2d_prime[..., 1].astype(np.float32).reshape(H, W)

    # Warp left-depth map in the same left view but at right image pixel location
    # D_left_prime_uv = cv2.remap(D_left.astype(np.float32), u_r, v_r, interpolation=cv2.INTER_LINEAR,
    #               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # X_c_right_prime = px_to_camera(D_left_prime_uv, K_inv, ur_vr_1)
    # x_left_2d_prime = project_to_view(X_c_right_prime, P_one)
    # if 'depth' in error_types:
    #     errors.append(np.abs(D_left - D_left_prime_uv))

    # Warp each channel using remap
    # I_left_warped = cv2.remap(I_left, u_l_prime, v_l_prime, interpolation=cv2.INTER_LINEAR,
    #               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # errors = {}
    
    #errors['cycle_px'] = np.nan_to_num(np.abs(x_left_2d_prime - x_left).sum(-1).astype(np.float32).reshape(H, W),nan=500, posinf=500,neginf=500)
    # errors['cycle_image_l1'] = np.nan_to_num(np.mean(np.abs(I_left_warped - I_left), axis=-1), nan=500, posinf=500,neginf=500)
    # errors['cycle_image_ssim'] = np.nan_to_num(photometric_error_ssim(I_left_warped,I_left), nan=500, posinf=500,neginf=500)
    # errors['cycle_depth_ratio'] = np.nan_to_num(np.abs(D_left - D_left_prime)/D_left, nan=500, posinf=500,neginf=500)  
    
    # return {}

def photometric_error_ssim(I_L, I_R_warped):
    """
    Returns 1 - SSIM index per pixel (approximate perceptual error)
    """
    # Convert to grayscale
    I_L_gray = cv2.cvtColor(I_L, cv2.COLOR_RGB2GRAY)
    I_R_gray = cv2.cvtColor(I_R_warped, cv2.COLOR_RGB2GRAY)

    ssim_map = ssim(I_L_gray, I_R_gray, data_range=255, full=True)[1]  # Returns (mean SSIM score, full map)
    error_map = 1.0 - ssim_map  # dissimilarity

    return error_map

def photometric_errors(I_L, I_R, x_right, error_types=['l1', 'l2', 'ssim']):
    """
    Computes L1 and L2 error maps between I_L and I_R(x_right), per pixel.

    I_L: HxWx3 left image
    I_R: HxWx3 right image
    x_right: HxWx2 of (u_R, v_R) reprojected coordinates in right image
    """
    H, W, _ = I_L.shape

    # Split u_R and v_R
    u_R = x_right[..., 0].astype(np.float32).reshape(H, W)
    v_R = x_right[..., 1].astype(np.float32).reshape(H, W)

    valid = (
        (u_R >= 0) & (u_R < W) &
        (v_R >= 0) & (v_R < H)
    )

    # Warp each channel using remap
    I_R_warped = cv2.remap(I_R, u_R, v_R, interpolation=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    errors = {}
    if 'l1' in error_types:
        errors['photo_l1'] = np.mean(np.abs(I_L - I_R_warped), axis=-1)  # H x W scalar map
    if 'l2' in error_types:
        errors['photo_l2'] = np.mean((I_L - I_R_warped)**2, axis=-1)  # H x W scalar map
    if 'ssim' in error_types:        
        #ssim_err[~valid] = 5 # does not make a difference.
        errors['photo_ssim'] = photometric_error_ssim(I_L, I_R_warped)  # H x W scalar map
    
    return errors

def get_errors(depth_left, rectified_left,rectified_right, K_inv, K_inv_uv1, g_i,P2, alpha, kernel): #, P1,P2, T, fB):
    # H,W = depth_left.shape      
    X_c_left = px_to_camera(depth_left, K_inv, K_inv_uv1)                
    x_right_2d = project_to_view(X_c_left, P2)
    grad_error = compute_grad_error(depth_left, g_i, alpha, kernel)
    planarity_error, rms_orth, Prel, Pnorm = get_planarity_error(X_c_left)
    photo_errors = photometric_errors(rectified_left, rectified_right, x_right_2d, error_types=['l1','ssim'])
    #depth_errors = depth_cycle_errors(depth_left, rectified_left, x_right_2d, K_inv, P1, T, fB)
    errors = {"grad":grad_error, "plan":planarity_error, "rms_orth":rms_orth, "Prel":Prel, "Pnorm":Pnorm, **photo_errors} #, **depth_errors} #, **photo_errors}
    return errors







if __name__ == "__main__":
    pass