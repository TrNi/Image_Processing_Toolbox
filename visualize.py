import matplotlib
# Set the backend to 'TkAgg' for better VSCode compatibility
matplotlib.use('TkAgg')
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os
import matplotlib.colors as colors
import cv2
from depth_reproj_eval import *
import traceback
from scipy.ndimage import median_filter
from uncertainty_and_weights import get_iqr_uncertainty
from collections import defaultdict
from point_cloud_opt import get_point_cloud_errors
from geometric_structure_errors import compute_grad_error, get_planarity_error, compute_grad

# get_cmap=colormaps.get_cmap

'''
visualize_depth_maps(
    base_path='/content/your_data_folder/',
    original_path='sample_data/left_rectified_{}.h5',
    depth_paths={
        'monster': 'depth_outputs/monster/leftview_disp_depth.h5',
        'foundation': 'depth_outputs/foundation_stereo/leftview_disp_depth.h5',
        'selective': 'depth_outputs/selective_stereo_igev/leftview_disp_depth.h5',
        'deform': 'depth_outputs/defom/leftview_disp_depth.h5'
    }
)

Key features of this implementation:

Efficient Memory Usage: Loads and processes one image at a time.
Interactive Visualization: Shows colorbars for depth maps and allows interactive zooming/panning.
Flexible Input: Handles different H5 file structures and image formats.
User Control:
Shows one image at a time by default
After 5 images, asks if you want to continue automatically
Press Enter to proceed to the next image
Error Handling: Properly closes H5 files even if an error occurs.
The script will display the original image alongside the four depth maps in a single row, 
with colorbars for each depth map. The visualization is interactive, 
allowing you to zoom and pan each subplot independently.
'''
def get_stats(arr, maxval=1000):
    return {
        'min': np.clip(np.nanmin(arr), 1e-6, maxval),
        'max': np.clip(np.nanmax(arr), 1e-6, maxval),
        '5': np.percentile(arr, 5),
        '95': np.percentile(arr, 95),
        'num_nan': np.isnan(arr).sum(),
        'pct_nan': np.mean(np.isnan(arr)) * 100
    }


def resize_image_hwc(img_hwc, target_h, target_w, interpolation=cv2.INTER_LINEAR):
    # img_hwc: H x W x C numpy array    
    resized_hwc = cv2.resize(img_hwc, (target_w, target_h), interpolation=interpolation)    
    return resized_hwc

def visualize_depth_maps(base_path='/content/', 
                        left_rectified_path='sample_data/left_rectified_{}.h5',                        
                        depth_paths={
                            'monster': 'depth_outputs/monster/leftview_disp_depth.h5',
                            'foundation': 'depth_outputs/foundation_stereo/leftview_disp_depth.h5',
                            'selective': 'depth_outputs/selective_stereo_igev/leftview_disp_depth.h5',
                            'defom': 'depth_outputs/defom/leftview_disp_depth.h5'
                        },
                        params_path='stereocal_params.npz',
                        chunk_size=1,
                        max_plots=5):
    """
    Visualize depth maps from multiple sources interactively.
    
    Args:
        base_path (str): Base path where all files are located
        left_rectified_path (str): Format string for original image paths
        depth_paths (dict): Dictionary mapping model names to their depth map file paths
        chunk_size (int): Number of images to process in each chunk
        max_plots (int): Number of plots to show before asking to continue
    """
    # Initialize variables
    current_idx = 0    
    model_names = list(depth_paths.keys())
    num_models = len(model_names)

    params = load_camera_params(os.path.join(base_path, params_path))
    P1, P2 = params['P1'], params['P2']
    K, K_inv = params['K_new'], params['K_inv']
    #T, baseline, fB = params['T'], params['baseline'], params['fB']
    right_rectified_path = os.path.join(base_path, left_rectified_path.replace("left", "right"))
    bottom_plot = "error_types" #  "total_error"       
    
    # Open all H5 files
    h5_files = {}
    try:       
        h5_files['left_rectified'] = h5py.File(os.path.join(base_path, left_rectified_path), 'r')        
        h5_files['right_rectified'] = h5py.File(os.path.join(base_path, right_rectified_path), 'r')        
        
        for name, path in depth_paths.items():
            h5_files[name] = h5py.File(os.path.join(base_path, path), 'r')
        
        
        plot_cols = 6 #len(h5_files)
        if bottom_plot == "error_types":
            bottom_cols = 6 #5        
        elif bottom_plot == "total_error":
            bottom_cols = 6 #plot_cols
        figsize = (2.5*plot_cols, 9) 

        # Get the minimum number of images across all files
        min_images = min([h5_files[f]['data'].shape[0] if 'data' in h5_files[f] else h5_files[f]['depth'].shape[0] 
                         for f in h5_files])    

        # Create the first figure with adjusted layout
        fig = plt.figure(figsize=figsize)
        width_ratios = [1]*plot_cols
        width_ratios[0] = 1.5
        gs = fig.add_gridspec(3, plot_cols, height_ratios=[1, 1, 1], hspace=0.1, wspace=0.2, width_ratios=width_ratios) #num_models + 1
        axes = np.full((3, plot_cols), None, dtype=object)
        axes[0,0] = fig.add_subplot(gs[:, 0]) # Top-left for left image
        #axes[1, 0] = fig.add_subplot(gs[1, 0])  # Bottom-left for right image

        for i in range(1, plot_cols): #num_models + 1   
            if i<5:         
                axes[0, i] = fig.add_subplot(gs[0, i])
            if 4+i<=len(depth_paths)+1:
                axes[1, i] = fig.add_subplot(gs[1, i])            
            if i<=bottom_cols:                
                axes[2, i] = fig.add_subplot(gs[2, i])        
        axes_flat = axes.flatten()[1:]

        plt.subplots_adjust(left=0.015, bottom=0.01, top=0.95, right=0.98, hspace=0.1, wspace=0.2)

        base_title = base_path 
        os.makedirs(os.path.join(base_path, 'ml_data'), exist_ok=True)
        
        col_clip = 400
        reuse_calculation = False
        min_w, min_h = np.inf, np.inf
        while current_idx < min_images:
            if current_idx<8:
                current_idx+=1
                continue
            fig.suptitle(f'{base_title} : Image {current_idx }', fontsize=10, y=0.98, color='black')
            rectified_left = h5_files['left_rectified']['data'][()][current_idx].transpose(1,2,0)
            rectified_right =h5_files['right_rectified']['data'][()][current_idx].transpose(1,2,0)
            h_l,w_l,c_l = rectified_left.shape
            h_r,w_r,c_r = rectified_right.shape
            assert h_l==h_r and w_l==w_r and c_l==c_r, f"rectified_left shape: {rectified_left.shape} but rectified_right shape: {rectified_right.shape}"            
            aspect_ratio = w_l/h_l
            if not reuse_calculation:
                min_w, min_h = w_l, h_l
            
            depth_names = []

            for name in depth_paths.keys():
                h_d, w_d = h5_files[name]['depth'].shape[1:] \
                            if 'depth' in h5_files[name] else \
                            h5_files[name]['data'].shape[1:] 

                if abs(w_d/h_d - aspect_ratio) > 0.01:
                    print(f"Warning: not using {name} depth map: aspect ratio mismatch: {w_d/h_d} vs {aspect_ratio}")
                    continue

                depth_names.append(name)
                if not reuse_calculation:
                    min_w = min(min_w, w_d)
                    min_h = min(min_h, h_d)

            assert depth_names, "No valid depth maps found due to aspect ratio mismatches, consider regenerating depth maps."
            if not reuse_calculation:
                print(f"Common resizing factor: {min_h/h_l:0.2f}x{min_w/w_l:0.2f}, reduced size: {min_h}x{min_w} px")
            rectified_left = resize_image_hwc(rectified_left, min_h, min_w)
            rectified_right = resize_image_hwc(rectified_right, min_h, min_w)   

            # Combine left and right images vertically in the first subplot
            combined = np.vstack((rectified_left[:,col_clip:,:],\
                       255*np.ones((200, rectified_left[:,col_clip:,:].shape[1], 3), dtype=np.uint8),\
                       rectified_right[:,:-col_clip,:]))
            axes[0, 0].imshow(combined)
            #axes[0, 0].imshow(cv2.cvtColor(combined, cv2.COLOR_RGB2GRAY), cmap='gray')
            axes[0, 0].set_title('Left Image', fontsize=9)
            axes[0,0].text(0.5, 0.5, 'Right Image', fontsize=9, ha='center', va='center', color='black',
                           transform=axes[0,0].transAxes, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            axes[0, 0].axis('off')
            # if axes[1, 0] is not None:
            #     fig.delaxes(axes[1, 0])
            #axes = axes[:, 1:]
            if not reuse_calculation:
                K_inv_uv1 = get_Kinv_uv1(K_inv, min_h, min_w)
                reuse_calculation = True 
                # image resizing and K_inv_uv1 calculation will be reused for subsequent images and depth maps.
            alpha=0.1
            kernel=5
            g_i = compute_grad(rectified_left, k=kernel)
            g_i /= g_i.max()
            depth_data = {}
            depth_stats = {}
            err_data = {}
            err_stats = {}            

            for name in depth_paths.keys():
                if "depth" in h5_files[name]:
                    keyname = "depth"
                elif "depths" in h5_files[name]:
                    keyname = "depths"
                elif "data" in h5_files[name]:
                    keyname = "data"
                else:
                    raise ValueError(f"Unknown key in h5 file: {name}")

                depth_data[name] = h5_files[name][keyname][()][current_idx]                
                depth_data[name] = depth_data[name].squeeze()                                
                depth_data[name] = resize_image_hwc(depth_data[name].astype(np.float32), min_h, min_w)
                err_data[name] = get_errors(depth_data[name], rectified_left, rectified_right, K_inv, K_inv_uv1, g_i,P2, alpha, kernel) #, P1, P2, T, fB)
                                
            depth_data_arr = np.stack(list(depth_data.values()), axis=0)
            try:
                iqr_errors = get_iqr_uncertainty(depth_data_arr, depth_names)
                icp_errors, global_error_maps = get_point_cloud_errors(depth_data_arr, depth_names,  K_inv)
            except Exception as e:
                print(f"Error in get_iqr_uncertainty or get_point_cloud_errors: {e}")
                continue

            for i,name in enumerate(depth_names):
                err_data[name]['icp_error'] = icp_errors['error_maps'][i+1].reshape(min_h, min_w)
                #err_data[name]['global_error'] = global_error_maps[i]
                err_data[name]['iqr'] = iqr_errors[name]
                
            error_min, error_max = defaultdict(lambda: np.inf), defaultdict(lambda: -np.inf)    
            for j,name in enumerate(err_data):
                if j==0:
                    all_err = err_data[name]  
                for err_type in err_data[name]:
                    error_min[err_type] = min(error_min[err_type],np.percentile(err_data[name][err_type][:,col_clip:], 1))
                    error_max[err_type] = max(error_max[err_type],np.percentile(err_data[name][err_type][:,col_clip:], 99))              
                #err_data[name] = np.stack(list(err_data[name].values()), axis=0)[:,:,col_clip:].sum(axis=0)

            for j,name in enumerate(err_data):
                err_data[name] = 0.3* (err_data[name]['grad_error'] + 1e-8 - error_min['grad_error'])/ (error_max['grad_error'] - error_min['grad_error'])+ \
                                0.1* (err_data[name]['planarity_error'] + 1e-8 - error_min['planarity_error'])/ (error_max['planarity_error'] - error_min['planarity_error'])+ \
                                5.5* (err_data[name]['iqr'] + 1e-8 - error_min['iqr'])/ (error_max['iqr'] - error_min['iqr'])+ \
                                4.1* (err_data[name]['icp_error'] + 1e-8 - error_min['icp_error'])/ (error_max['icp_error'] - error_min['icp_error'])
                                #1* (err_data[name]['photo_l1'] + 1e-4 - error_min['photo_l1'])/ (error_max['photo_l1'] - error_min['photo_l1'])+ \
                                #1* (err_data[name]['photo_ssim'] + 1e-4 - error_min['photo_ssim'])/ (error_max['photo_ssim'] - error_min['photo_ssim'])+ \
                                
                err_data[name] = np.maximum(err_data[name],0) / 10

                err_data[name] = err_data[name][:,col_clip:]
                err_stats[name] = get_stats(err_data[name], maxval=500)
                #depth_data[name] = depth_data[name][:,col_clip:]
                depth_stats[name] = get_stats(depth_data[name][:,col_clip:], maxval=100)
            # Find global min and max for consistent color scaling (excluding NaNs)
            depth_mins = [stats['min'] for stats in depth_stats.values() if not np.isnan(stats['min'])]
            depth_maxs = [stats['max'] for stats in depth_stats.values() if not np.isnan(stats['max'])]
            #err_min = np.array([err for err in err_data.values()]).min(0)

            # for name in err_data:                               
                # err_data[name] = cv2.bilateralFilter(median_filter(err, size=3).astype(np.float32), \
                #                                     d=7, sigmaColor=75, sigmaSpace=50)
                # err_stats[name] = get_stats(err_data[name], maxval=6000)                

            err_mins = [stats['min'] for stats in err_stats.values() if not np.isnan(stats['min'])]
            err_maxs = [stats['max'] for stats in err_stats.values() if not np.isnan(stats['max'])]
            

            if depth_mins and depth_maxs:
                dmin = max(min(depth_mins), 1e-3)
                dmax = max(depth_maxs)
            else:
                dmin, dmax = 1e-3, 100  
            
            if err_mins and err_maxs:  
                emin = max(min(err_mins), 1e-3)
                emax = max(err_maxs)
            else:
                emin, emax = 1e-3, 500
            
            
            # Create a new colormap that's the same as turbo_r but without the last colors
            turbo_r = plt.get_cmap('turbo_r')            
            turbo_r_colors = turbo_r(np.linspace(0, 0.8, 256))  # Stop at 0.95 instead of 1.0 to avoid black
            custom_cmap = colors.ListedColormap(turbo_r_colors)
            n_levels = 50            
            cmap1 = plt.get_cmap(custom_cmap, n_levels)
            turbo = plt.get_cmap('turbo')            
            turbo_colors = turbo(np.linspace(0.12, 1, 256))  # Start at 0.12 instead of 0.0 to avoid black
            custom_cmap = colors.ListedColormap(turbo_colors)
            cmap2 = plt.get_cmap(custom_cmap, n_levels)

            fused_depth = np.zeros((min_h, min_w - col_clip))
            weights = np.zeros((min_h, min_w - col_clip))
            weight_map = {}
            for name in depth_names:
                weights += 1 / (err_data[name] + 1e-8)
                fused_depth += depth_data[name][:,col_clip:] * 1 / (err_data[name] + 1e-8)
                weight_map[name] = 1 / (err_data[name] + 1e-8)
            fused_depth /= weights

            ax_iter = iter(axes_flat[axes_flat!=None])
            
            for i, (name, path) in enumerate(depth_paths.items()):                                                                
                # Top row: depth maps
                ax_depth = next(ax_iter) #axes[0, 1+i]
                im_depth = ax_depth.imshow(depth_data[name][:,col_clip:].round(3), cmap=cmap1, 
                                        norm=colors.LogNorm(vmin=dmin, vmax=dmax), 
                                        interpolation='nearest')                
                subtitle = f'{name}: \n {depth_stats[name]["5"]:.2f} - {depth_stats[name]["95"]:.2f}, '+\
                            f'#NaNs: {depth_stats[name]["num_nan"]}({depth_stats[name]["pct_nan"]:.0f}%)'
                ax_depth.set_title(subtitle, fontsize=9)
                cbar_depth = plt.colorbar(im_depth, ax=ax_depth, fraction=0.035, pad=0.04)                
                ax_depth.axis('off')
                # Set ticks for depth colorbar
                extra_ticks = np.array([1.0, 1.5, 3.0])
                extra_ticks = extra_ticks[(extra_ticks >= dmin) & (extra_ticks <= dmax)]
                log_ticks = np.logspace(np.log10(dmin), np.log10(dmax), num=5)
                ticks = np.unique(np.concatenate([extra_ticks, log_ticks]))
                cbar_depth.set_ticks(ticks)
                cbar_depth.set_ticklabels([f"{tick:.1f}" for tick in ticks])
                cbar_depth.ax.tick_params(labelsize=7)

                # Bottom row: error maps
                if bottom_plot=="error_types":
                    if i==0:
                        err_types = ["grad_error", "planarity_error", "iqr", "icp_error"] #list(all_err.keys())
                        for j in range(0,len(err_types)):
                            ax_err = axes[2, j+1]
                            this_err = (all_err[err_types[j]][:,col_clip:] + 1e-8 - error_min[err_types[j]])/ (error_max[err_types[j]] - error_min[err_types[j]])
                            lo = np.percentile(this_err, 1)
                            hi = np.percentile(this_err, 99)
                            print(err_types[j], lo,hi) #, np.percentile(this_err, 0.1), np.percentile(this_err, 99))
                            this_err = np.clip(this_err, lo, hi)
                            im_err = ax_err.imshow(this_err, cmap=cmap2, 
                                # norm=colors.Normalize(vmin=emin, vmax=emax), 
                                interpolation='nearest')
                            subtitle = f"Error component {j}"#err_types[j]
                            #f"{name}: \n {err_stats[name]["min"]:.2f} - {err_stats[name]["max"]:.2f}, #NaNs: {err_stats[name]["num_nan"]}({err_stats[name]["pct_nan"]:.2f}%)"
                            ax_err.set_title(subtitle, fontsize=9)
                            cbar_err = plt.colorbar(im_err, ax=ax_err, fraction=0.035, pad=0.04)                
                            ax_err.axis('off')
                            log_ticks = np.logspace(np.log10(this_err.min()), np.log10(this_err.max()), num=7)
                            print(err_types[j], [f"{tick:.3f}" for tick in log_ticks])
                            cbar_err.set_ticks(log_ticks)
                            cbar_err.set_ticklabels([f"{tick:.1f}" for tick in log_ticks])         
                            cbar_err.ax.tick_params(labelsize=7)
                    
                        ax_err = axes[2, plot_cols-1]
                        this_err = weight_map[name]/weights#err_data[name]
                        # lo = np.percentile(this_err, 1)
                        # hi = np.percentile(this_err, 99)
                        # print("total weight", lo,hi) #, np.percentile(this_err, 0.1), np.percentile(this_err, 99))
                        # this_err = np.clip(this_err, lo, hi)
                        im_err = ax_err.imshow(this_err, cmap=cmap2, 
                            # norm=colors.Normalize(vmin=emin, vmax=emax), 
                            interpolation='nearest')
                        subtitle = f"Total Weight for {name} depth"#"Total Error"
                        ax_err.set_title(subtitle, fontsize=9)
                        cbar_err = plt.colorbar(im_err, ax=ax_err, fraction=0.035, pad=0.04)                
                        ax_err.axis('off')
                        log_ticks = np.logspace(np.log10(this_err.min()+1e-3), np.log10(this_err.max()), num=7)
                        #print("Total weight", [f"{tick:.3f}" for tick in log_ticks])
                        cbar_err.set_ticks(log_ticks)
                        cbar_err.set_ticklabels([f"{tick:.1f}" for tick in log_ticks])  
                        cbar_err.ax.tick_params(labelsize=7)
                        # Set ticks for error colorbar 
                        # log_ticks = np.logspace(np.log10(0.01), np.log10(emax), num=7)                
                        # cbar_err.set_ticks(log_ticks)

                        # lin_ticks = np.linspace(emin, emax, num=7)
                        # cbar_err.set_ticks(lin_ticks)
                        # cbar_err.set_ticklabels([f"{tick:.1f}" for tick in lin_ticks])   
                elif bottom_plot=="total_error":
                    if i<plot_cols:
                        ax_err = axes[2, 1+i]
                        this_err = weight_map[name]/weights#
                        # lo = np.percentile(this_err, 1)
                        # hi = np.percentile(this_err, 99)
                        # print("total error", lo,hi) #, np.percentile(this_err, 0.1), np.percentile(this_err, 99))
                        # this_err = np.clip(this_err, lo, hi)
                        im_err = ax_err.imshow(this_err, cmap=cmap2, 
                            # norm=colors.Normalize(vmin=emin, vmax=emax), 
                            interpolation='nearest')
                        subtitle = "Total Weight" #"Total Error" # 
                        ax_err.set_title(subtitle, fontsize=9)
                        cbar_err = plt.colorbar(im_err, ax=ax_err, fraction=0.035, pad=0.04)                
                        ax_err.axis('off')
                        log_ticks = np.logspace(np.log10(this_err.min()+1e-3), np.log10(this_err.max()), num=7)
                        #print("Total weight", [f"{tick:.3f}" for tick in log_ticks])
                        cbar_err.set_ticks(log_ticks)
                        cbar_err.set_ticklabels([f"{tick:.1f}" for tick in log_ticks])                 
                        cbar_err.ax.tick_params(labelsize=7)
            
            # Top row: fused depth map
            ax_depth = next(ax_iter)

            im_depth = ax_depth.imshow(fused_depth.round(3), cmap=cmap1, 
                                    norm=colors.LogNorm(vmin=dmin, vmax=dmax), 
                                    interpolation='nearest')                
            depth_stats['fused'] = {
                'min': fused_depth.min(),
                'max': fused_depth.max(),
                '5': np.percentile(fused_depth, 5),
                '95': np.percentile(fused_depth, 95),
                'num_nan': np.sum(np.isnan(fused_depth)),
                'pct_nan': np.sum(np.isnan(fused_depth)) / fused_depth.size * 100
            }
            subtitle = f'Fused: \n {depth_stats["fused"]["5"]:.2f} - {depth_stats["fused"]["95"]:.2f}, '+\
                         f'#NaNs: {depth_stats["fused"]["num_nan"]}({depth_stats["fused"]["pct_nan"]:.0f}%)'
            ax_depth.set_title(subtitle, fontsize=9)
            cbar_depth = plt.colorbar(im_depth, ax=ax_depth, fraction=0.035, pad=0.04)                
            ax_depth.axis('off')
            # Set ticks for depth colorbar
            extra_ticks = np.array([1.0, 1.5, 3.0])
            extra_ticks = extra_ticks[(extra_ticks >= dmin) & (extra_ticks <= dmax)]
            log_ticks = np.logspace(np.log10(dmin), np.log10(dmax), num=5)
            ticks = np.unique(np.concatenate([extra_ticks, log_ticks]))
            cbar_depth.set_ticks(ticks)
            cbar_depth.set_ticklabels([f"{tick:.1f}" for tick in ticks])
            cbar_depth.ax.tick_params(labelsize=7)

            # first_ax = axes[0,0]
            second_ax = axes[0,1]
            for ax in axes_flat:#[*axes[0,1:plot_cols].flatten(), *axes[1,1:bottom_cols].flatten()]:
                if ax:
                    ax.sharex(second_ax)
                    ax.sharey(second_ax)

            # axes[0,1].get_shared_x_axes().join(*[*axes[0,1:], *axes[1,1:]])
            # axes[0,1].get_shared_y_axes().join(*[*axes[0,1:], *axes[1,1:]])                
            # plt.tight_layout()            
            with h5py.File(os.path.join(base_path, 'ml_data', f"img_{current_idx}.h5"), 'w') as f:
                
                for t,name in enumerate(depth_names):
                    if t==0:
                        subarr = []
                        subarr.append(rectified_left)
                        subarr.append(rectified_right)
                        subarr.append(depth_data[name][...,None])
                        for k,v in all_err.items():
                            subarr.append(v[...,None])
                            print(k)
                        subarr = np.concatenate(subarr, axis=2)                                
                        f.create_dataset('data', data=subarr, compression='gzip', compression_opts=9)
                        f.create_dataset('total_weight', data=weight_map[name]/weights, compression='gzip', compression_opts=9)
                        f.create_dataset('fused_depth', data=fused_depth, compression='gzip', compression_opts=9)



            plt.show(block=True)
            plt.close('all')
            
            # Create a new figure for the next iteration if there are more images
            if current_idx < min_images - 1:
                fig = plt.figure(figsize=figsize)
                gs = fig.add_gridspec(3, plot_cols, height_ratios=[1, 1, 1], #num_models + 1
                hspace=0.1, wspace=0.2, width_ratios = width_ratios)
                axes = np.empty((3, plot_cols), dtype=object) #num_models + 1
                axes[0,0] = fig.add_subplot(gs[:, 0]) # Top-left for left image
                #axes[1,0] = fig.add_subplot(gs[1, 0])  # Bottom-left for right image
                for i in range(1, plot_cols):                   
                    if i<5:         
                        axes[0, i] = fig.add_subplot(gs[0, i])
                    if 4+i<=len(depth_paths)+1:
                        axes[1, i] = fig.add_subplot(gs[1, i])            
                    if i<=bottom_cols:                
                        axes[2, i] = fig.add_subplot(gs[2, i])
                #fig.delaxes(axes[1,0])
                #axes = axes[:, 1:]                
                plt.subplots_adjust(left=0.015, bottom=0.01, top=0.95, right=0.98, hspace=0.1, wspace=0.2)
                axes_flat = axes.flatten()[1:]
                # Update title for new figure
                fig.suptitle(f'{base_title} : Image {current_idx + 1}', fontsize=10, y=0.98, color='black')
            
            current_idx += 1
            
    except Exception as e:
        print("Error occurred:")
        print(traceback.format_exc())
        print(f"Error details: {str(e)}")
    finally:
        # Close all H5 files
        for f in h5_files.values():
            try:
                f.close()
            except:
                pass

#left_ple usage:

if __name__ == '__main__':
    # base_path = 'I:\\My Drive\\Scene-5\\f-28.0mm\\a-1.27mm\\stereocal_results_f28.0mm_a1.27mm'
    # left_rectified_path = 'I:\\My Drive\\Scene-5\\f-28.0mm\\a-1.27mm\\stereocal_results_f28.0mm_a1.27mm\\rectified\\rectified_lefts.h5'
    base_path = 'I:\\My Drive\\Scene-6\\stereocal_results_f28mm_a22mm'
    left_rectified_path = 'I:\\My Drive\\Scene-6\\stereocal_results_f28mm_a22mm\\rectified_h5\\rectified_lefts.h5'
    depth_paths = {
        #'traditional': 'raw_depth_h5\\raw_depth_lefts.h5',
        'monster': 'stereodepth\\leftview_disp_depth_monster.h5',                
        'selective': 'stereodepth\\leftview_disp_depth_selective_igev.h5',
        'defom': 'stereodepth\\leftview_disp_depth_defom.h5',
        'foundation': 'stereodepth\\leftview_disp_depth_foundation.h5',        
        "depth_anythingv2": "monodepths_rectified_left_1.5\\depth_anything_v2_depths.h5",
        "apple_depthpro": "monodepths_rectified_left_1.5\\depthpro_depths.h5",
    }
    anonymous = True
    stereo_id = 0
    mono_id = 0
    if anonymous:
        new_depth_paths = {}
        for k,v in depth_paths.items():
            if k in ["monster", "selective", "defom", "foundation"]:
                new_depth_paths[f"stereo {stereo_id}"] = v
                stereo_id += 1
            else:
                new_depth_paths[f"mono {mono_id}"] = v
                mono_id += 1
        depth_paths = new_depth_paths
    
    params_path = 'stereocal_params.npz'
    visualize_depth_maps(base_path, left_rectified_path, depth_paths, params_path)
    