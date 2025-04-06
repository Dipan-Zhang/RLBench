from typing import Union
import logging

import numpy as np
import hashlib
import open3d as o3d
import torch
import cv2
from addict import Dict
import math
import torchvision.transforms as T
import json
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator, UnivariateSpline
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import RANSACRegressor, LinearRegression
import matplotlib.pyplot as plt
import torch.nn.functional as F
import ipdb
from benchmark.generate_masked_object import compute_cropping_params, crop_images, compute_cropped_intrinsics
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def save_pickle(pickle_file, data):
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(data, pfile)


def apply_motion_plan(pose_init, motion_plan):
    """apply motion plan to the initial pose, from relative R, t to absolute pose
    Args:
    - pose_init: [4, 4], 
    - motion_plan: List[(R, t, success), ...], R: [3, 3], t: [3, ]
    Returns:
    - poses: [4, 4], list of *absolute* poses
    """
    poses = [pose_init.copy()]
    current_pose = poses[0]
    
    for R, t, success in motion_plan:
        if not success:
            print("Skip invalid motion plan")
            continue
        new_pose = np.eye(4)
        new_pose[:3, :3] = current_pose[:3, :3] @ R
        pos = current_pose[:3, 3].copy()
        new_pose[:3, 3] = np.matmul(R, pos[..., None]).squeeze() + t
        poses.append(new_pose)
        current_pose = new_pose
    return poses

def visualize_motion_plan(contact_pt, motion_plan):
    "visualize motion plan with contact point, contact_pt: [3,], motion_plan: [(R, t, success), ..."
    assert motion_plan[0][0].shape == (3, 3) and motion_plan[0][1].shape == (3,), "motion plan should be (R, t)"
    assert contact_pt.shape == (3,)

    contact_pt_transform = np.eye(4)
    contact_pt_transform[:3, 3] = contact_pt
    poses = apply_motion_plan(contact_pt_transform, motion_plan)
    
    poses_vis = []
    for pose in poses:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        mesh.transform(pose)
        poses_vis.append(mesh)

    world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    return [world] + poses_vis

def underscore_string_to_camel_case(string):
    """
    Convert a string from underscore format to camel case format.
    'my_variable_name' -> 'MyVariableName'.
    """
    components = string.split('_')
    return ''.join(x.title() for x in components) 

def visualize_points(points, colors=None):
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    except Exception as e:
        logger.error(f"Error in visualize_points: {e}")
        raise

def spline_interpolation(fill_indices, traj, NUM_POINTS, smoothing_factor=0.1):
    try:
        fill_times = np.array(fill_indices, dtype=np.float32)
        fill_traj = np.array(
            [traj[ii] for ii, idx in enumerate(fill_indices)], dtype=np.float32
        )
        dt = fill_times[2:] - fill_times[:-2]
        dt = np.hstack([fill_times[1] - fill_times[0], dt, fill_times[-1] - fill_times[-2]])
        dx = fill_traj[2:] - fill_traj[:-2]
        dx = np.hstack([fill_traj[1] - fill_traj[0], dx, fill_traj[-1] - fill_traj[-2]])
        dxdt = dx / dt
        # curve = CubicHermiteSpline(fill_times, fill_traj, dxdt)
        curve = PchipInterpolator(fill_times, fill_traj)
        # curve = UnivariateSpline(fill_times, fill_traj, s=smoothing_factor)

        full_traj = curve(
            np.linspace(fill_indices[0], fill_indices[-1], NUM_POINTS+1, dtype=np.float32)
        )
        return full_traj, curve
    except Exception as e:
        # logger.error(f"Error in spline_interpolation: {e}")
        raise

def ransac_filter(fill_indices, traj, residual_threshold=0.08, min_samples=0.5):
    """
    Uses RANSAC to filter out outliers in the z axis.
    fill_indices: list or array of indices/times corresponding to the z measurements.
    traj_z: corresponding z-axis values.
    residual_threshold: maximum residual for a data point to be considered an inlier.
    min_samples: minimum number (or fraction) of samples for RANSAC.
    Returns a boolean mask indicating inliers.
    """
    X = np.array(fill_indices).reshape(-1, 1)
    y = np.array(traj)
    # Use a simple linear model as base estimator.
    base_estimator = LinearRegression()
    ransac = RANSACRegressor(estimator=base_estimator,
                             min_samples=min_samples,
                             residual_threshold=residual_threshold,
                             random_state=0)
    ransac.fit(X, y)
    return ransac.inlier_mask_


def interpolate_trajectory(fill_indices, traj, NUM_POINTS=120):
    ""
    try:
        full_traj_x, curve_x = spline_interpolation(fill_indices, traj[:, 0], NUM_POINTS)
        full_traj_y, curve_y = spline_interpolation(fill_indices, traj[:, 1], NUM_POINTS)

        traj_z = traj[:, 2]
        inlier_mask = ransac_filter(fill_indices, traj_z)
        # If RANSAC filtering leaves too few points, fallback to the original data.
        if np.sum(inlier_mask) < 3:
            filtered_fill_indices = np.array(fill_indices)
            filtered_z = traj_z
        else:
            filtered_fill_indices = np.array(fill_indices)[inlier_mask]
            filtered_z = traj_z[inlier_mask]
        full_traj_z, curve_z = spline_interpolation(filtered_fill_indices, filtered_z, NUM_POINTS)
        full_traj = np.stack([full_traj_x, full_traj_y, full_traj_z], axis=1)

        curve = (curve_x, curve_y, curve_z)
        return full_traj, curve
    except Exception as e:
        # logger.error(f"Error in interpolate_trajectory: {e}")
        raise

def get_heatmap(values, cmap_name="turbo", invert=False):
    try:
        if invert:
            values = -values
        values = (values - values.min()) / (values.max() - values.min())
        colormaps = plt.get_cmap(cmap_name)
        rgb = colormaps(values)[..., :3]  # don't need alpha channel
        return rgb
    except Exception as e:
        # logger.error(f"Error in get_heatmap: {e}")
        raise

def visualize_sphere_o3d(center, color=[1, 0, 0], size=0.03):
    try:
        center_o3d = o3d.geometry.TriangleMesh.create_sphere()
        center_o3d.compute_vertex_normals()
        center_o3d.scale(size, [0, 0, 0])
        center_o3d.translate(center)
        center_o3d.paint_uniform_color(color)
        return center_o3d
    except Exception as e:
        # logger.error(f"Error in visualize_sphere_o3d: {e}")
        raise

def visualize_3d_trajectory(trajectory, size=0.03, cmap_name="plasma", invert=False):
    """Visualize a 3D trajectory as a series of spheres that can be rendered in Open3D."""
    try:
        vis_o3d = []
        traj_color = get_heatmap(
            np.arange(len(trajectory)), cmap_name=cmap_name, invert=invert
        ) # change from purples to yellow
        for i, traj_point in enumerate(trajectory):
            vis_o3d.append(visualize_sphere_o3d(traj_point, color=traj_color[i], size=size))
        return vis_o3d
    except Exception as e:
        # logger.error(f"Error in visualize_3d_trajectory: {e}")
        raise


def preprocess_target_data(tgt_rgb, tgt_depth, cam_K, dataset, obj_name:str):
    """preprocess target data for affordance transfer
    Args:
        tgt_rgb (np.array): target rgb image
        tgt_depth (np.array): target depth image
        cam_K (np.array): camera intrinsic matrix
        dataset_type (str): dataset type, 'kinect' or 'rendered'
        prompt (str): text prompt for SAM
    Returns:
        dict: target data
    """
    if dataset == 'kinect':
        assert obj_name is not None, "Please provide a text object name for SAM"
        from thirdparty.GroundedSAM.grounded_sam_utils import prepare_gsam_model, inference_one_image
        grounded_dino_model, sam_predictor = prepare_gsam_model(device='cuda')
        tgt_masks = inference_one_image(tgt_rgb, grounded_dino_model, sam_predictor,\
                                    box_threshold=0.3, text_threshold=0.25, text_prompt=obj_name,\
                                    device="cuda").cpu().numpy() # you can set point_prompt to traj[0]
    
    
        tgt_mask = (tgt_masks[0][0] > 0).astype(np.uint8)
        center, crop_scale, resize_ratio = compute_cropping_params(tgt_mask, pad_ratio=1.25, resolution=512)
        cam_K_resized = compute_cropped_intrinsics(cam_K, resize_ratio, center, 512)
        tgt_rgb_cropped, tgt_mask_cropped, tgt_depth_cropped = crop_images(tgt_rgb, tgt_mask, tgt_depth, center, crop_scale, 512)
        color_cropped_rgba = np.concatenate([tgt_rgb_cropped, tgt_mask_cropped], axis=-1)
        # TODO: put this in proper location
        cropped_rgb_fname = './outputs/rlbench/cropped_color.png'
        cv2.imwrite(cropped_rgb_fname, color_cropped_rgba)

        target_data = {
            'dataset': dataset,
            'obj_name': obj_name,
            'rgb': tgt_rgb,
            'depth': tgt_depth,
            'mask': tgt_mask,
            'camera_intrinsic': cam_K,
            'cropped_rgb': tgt_rgb_cropped,
            'cropped_rgb_fname': cropped_rgb_fname, # for DINO 
            'cropped_depth': tgt_depth_cropped, 
            'cropped_mask': tgt_mask_cropped,
            'camera_intrinsic_resized': cam_K_resized
        }
        del sam_predictor, grounded_dino_model # free up memory
    elif dataset == 'rendered' or dataset == 'generative': #TODO: is this necc??? 
        tgt_mask = tgt_depth > 0
        tgt_mask = tgt_mask.astype(np.uint8)
        tgt_rgba = np.concatenate([tgt_rgb, np.expand_dims(tgt_mask, axis=-1)], axis=-1)
        cropped_rgb_fname = './outputs/rlbench/cropped_color.png'
        cv2.imwrite(cropped_rgb_fname, tgt_rgba)
        target_data = {
            'dataset': dataset,
            'obj_name': obj_name,
            'rgb': tgt_rgb,
            'depth': tgt_depth,
            'mask': tgt_mask,
            'cropped_rgb_fname': cropped_rgb_fname, # for DINOe
            'camera_intrinsic': cam_K
        }

    return target_data


def backproject(depth, intrinsics, instance_mask, NOCS_convention=True):
    """backproject depth image to 3d points
    Args:
        depth: [h, w]
        intrinsics: [3, 3]
        instance_mask: [h, w]
    return: pts: [num_pixel, 3], idxs: [2, num_pixel]
    """
    try:
        intrinsics_inv = np.linalg.inv(intrinsics)
        non_zero_mask = depth > 0
        final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

        idxs = np.where(final_instance_mask)
        grid = np.array([idxs[1], idxs[0]])

        length = grid.shape[1]
        ones = np.ones([1, length])
        uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

        xyz = intrinsics_inv @ uv_grid  # [3, num_pixsel]
        xyz = np.transpose(xyz)  # [num_pixel, 3]

        z = depth[idxs[0], idxs[1]]

        pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
        if NOCS_convention:
            pts[:, 1] = -pts[:, 1]
            pts[:, 2] = -pts[:, 2]
        return pts, idxs
    except Exception as e:
        logger.error(f"Error in backproject: {e}")
        raise

def backproject_with_color(depth, color, intrinsic, mask, NOCS_convention= False):
    "backproject depth to 3d points and get color"
    pts, pts_idx = backproject(depth, intrinsic, mask, NOCS_convention=False)
    color = (color / 255.0).astype(np.float32)
    colors = color[pts_idx[0], pts_idx[1]]
    return pts, colors

def visualize_points(points, colors=None):
    "take points and return open3d pcd"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def get_time():
    """Get the current date as a string."""
    import time
    curr_time = time.strftime("%Y-%m-%d-%H-%M")
    return curr_time