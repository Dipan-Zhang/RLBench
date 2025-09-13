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
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from scipy.signal import savgol_filter
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
    - poses: list of *absolute* poses in shape [4, 4]
    """
    poses = [pose_init.copy()]
    current_pose = poses[0]
    
    for R, t, success in motion_plan:
        if not success:
            print("Skip invalid motion plan")
            continue
        new_pose = np.eye(4)
        new_pose[:3, :3] = R @ current_pose[:3, :3] 
        pos = current_pose[:3, 3].copy()
        new_pose[:3, 3] = np.matmul(R, pos[..., None]).squeeze() + t
        poses.append(new_pose)
        current_pose = new_pose
    return poses

def transform_motion_plan(motion_plan, T_cam_obj):
    """
    Transforms a motion plan of relative transforms (R, t, success) from object frame to camera frame.

    Each (R, t) in motion_plan is a relative transform in the object frame.
    """
    T_obj_cam = np.linalg.inv(T_cam_obj)  # Needed for change of basis
    motion_plan_cam = []

    for R_obj, t_obj, success in motion_plan:
        T_rel_obj = np.eye(4)
        T_rel_obj[:3, :3] = R_obj
        T_rel_obj[:3, 3] = t_obj

        T_rel_cam = T_cam_obj @ T_rel_obj @ T_obj_cam

        R_cam = T_rel_cam[:3, :3]
        t_cam = T_rel_cam[:3, 3]
        motion_plan_cam.append((R_cam, t_cam, success))

    return motion_plan_cam


def scale_abs_trajectory(traj, scale, reciprocal=False):
    "scale trajectory poses traj: [H, 4, 4]"
    if reciprocal:
        scale = 1 / scale

    if isinstance(traj, list):
        traj = np.array(traj)
    traj_pos = traj[:, :3, 3]  # [H, 3]
    traj_pos_init = traj_pos[0:1]  # [1, 3]
    traj_dist = traj_pos - traj_pos_init  # [H, 3]
    traj_pos = traj_pos_init + traj_dist * scale  # [H, 3]
    traj[:, :3, 3] = traj_pos
    return traj

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


def interpolate_trajectory(waypoints, distance_threshold=0.01, min_points=2):
    """
    Interpolates a trajectory based on distance threshold between consecutive waypoints.
    
    Args:
        waypoints (np.ndarray): Array of waypoint positions.
        distance_threshold (float): Maximum distance between consecutive points after interpolation.
        min_points (int): Minimum number of points to interpolate between waypoints.
        
    Returns:
        np.ndarray: Array of interpolated waypoints.
    """
    if len(waypoints) < 2:
        return np.array(waypoints)
    
    interpolated = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        
        # Calculate the distance between consecutive waypoints
        distance = np.linalg.norm(end - start)
        
        # Calculate how many points we need based on the distance threshold
        num_points = max(min_points, int(np.ceil(distance / distance_threshold)))
        
        # Generate the interpolated points
        for t in np.linspace(0, 1, num_points):
            interpolated.append((1 - t) * start + t * end)
    
    return np.array(interpolated)


def generate_postgrasp_trajectory(grasp_T, post_grasp_dir):
    # Generate a post-grasp trajectory
    post_grasp_trajectory = []
    for i in range(10):
        t = grasp_T[:3, 3] + (post_grasp_dir * (i + 1) * 0.03)

        post_grasp_trajectory.append(t)
    return np.array(post_grasp_trajectory)


def vis_pose(pos, ori, size=0.05):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0,0,0])
    T_w_obj = np.eye(4)
    T_w_obj[:3, :3] = ori
    T_w_obj[:3, 3] = pos
    frame.transform(T_w_obj)
    return frame


def plan_gripper_trajectory(obs, affordance_traj_world, save_fn, smooth=False, vis=False):
    "plan smooth gripper trajectory based on affordance trajectory"
    current_gripper_pose = obs.gripper_pose[:7]
    offset = affordance_traj_world[0] - current_gripper_pose[:3]
    affordance_traj_world -= offset
    affordance_traj_world = np.concatenate([affordance_traj_world[0].reshape(-1,3), affordance_traj_world], axis=0)
    
    # smoothen the trajectory
    if smooth:
        affordance_traj_world = interpolate_trajectory(affordance_traj_world, distance_threshold=0.01)
        
    post_gripper_ori = current_gripper_pose[3:7]
    post_gripper_poses = np.concatenate((
        affordance_traj_world, 
        np.repeat(post_gripper_ori.reshape(-1, 4), affordance_traj_world.shape[0], axis=0)), axis=1)
    post_gripper_poses = np.concatenate([post_gripper_poses, np.zeros((affordance_traj_world.shape[0], 1))], axis=1)
    
    # add noise to avoid devide by zero
    noise = np.random.normal(0, 1e-4, post_gripper_poses.shape)
    post_gripper_poses[:, :3] += noise[:, :3]
    actions = post_gripper_poses

    # world frame pcd
    current_pts = obs.left_shoulder_point_cloud
    current_pcd = visualize_points(current_pts.reshape(-1, 3))

    action_vis = []
    for i in range(len(actions)):
        action_vis.append(vis_pose(actions[i][:3], Rot.from_quat(actions[i][3:7]).as_matrix()))
    world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    plan_trajectory = visualize_3d_trajectory(affordance_traj_world, size=0.02, cmap_name="plasma", invert=False)
    
    if vis:
        o3d.visualization.draw_geometries(action_vis + plan_trajectory + [current_pcd ,world])
    else: 
        mesh = o3d.geometry.TriangleMesh()
        for wp_vis in plan_trajectory+action_vis:
            mesh += wp_vis
        o3d.io.write_triangle_mesh(save_fn, mesh)
        point_cloud_save_fn = save_fn.replace('.ply', '_pcd.ply')
        o3d.io.write_point_cloud(point_cloud_save_fn, current_pcd)
    return actions


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
        text_prompt = obj_name
        from thirdparty.GroundedSAM.grounded_sam_utils import prepare_gsam_model, inference_one_image
        grounded_dino_model, sam_predictor = prepare_gsam_model(device='cuda',use_sam_hq=True)
        tgt_masks = inference_one_image(tgt_rgb, grounded_dino_model, sam_predictor,\
                                    box_threshold=0.3, text_threshold=0.25, text_prompt=text_prompt,\
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


def smooth_trajectory(poses, max_translation_step=0.01, max_rotation_deg=2.0):
    """
    Interpolates a trajectory of SE(3) poses to ensure smooth translation and rotation.
    Args:
        poses: (N, 7) array of [x, y, z, qx, qy, qz, qw]
        max_translation_step: maximum translation (meters) between waypoints
        max_rotation_deg: maximum rotation (degrees) between waypoints
    Returns:
        (M, 7) array of smoothed poses
    """
    smoothed = [poses[0]]
    for i in range(len(poses) - 1):
        p0, p1 = poses[i], poses[i+1]
        t0, t1 = p0[:3], p1[:3]
        r0, r1 = Rot.from_quat(p0[3:]), Rot.from_quat(p1[3:])
        
        # Compute translation and rotation difference
        trans_dist = np.linalg.norm(t1 - t0)
        rot_dist = r0.inv() * r1
        rot_angle = np.degrees(np.linalg.norm(rot_dist.as_rotvec()))
        
        # Number of steps needed for translation and rotation
        n_trans = int(np.ceil(trans_dist / max_translation_step))
        n_rot = int(np.ceil(rot_angle / max_rotation_deg))
        n_steps = max(n_trans, n_rot, 1)
        
        # Interpolate
        if n_steps > 1:
            times = np.linspace(0, 1, n_steps+1)[1:]  # skip 0, include 1
            slerp = Slerp([0, 1], Rot.from_quat([p0[3:], p1[3:]]))
            for t in times:
                interp_pos = (1-t) * t0 + t * t1
                interp_rot = slerp([t])[0].as_quat()
                smoothed.append(np.concatenate([interp_pos, interp_rot]))
        else:
            smoothed.append(p1)
    return np.array(smoothed)


def smooth_pose_sequence(poses, window_length=11, polyorder=3):
    """
    Smooths a sequence of SE(3) poses using Savitzky-Golay filter.
    Args:
        poses: (N, 7) array of [x, y, z, qx, qy, qz, qw]
        window_length: length of the filter window (must be odd and <= N)
        polyorder: order of the polynomial used to fit the samples
    Returns:
        (N, 7) array of smoothed poses
    """
    poses = np.array(poses)
    N = poses.shape[0]
    if N < window_length:
        window_length = N if N % 2 == 1 else N-1
    # Smooth translation
    smoothed_xyz = savgol_filter(poses[:, :3], window_length, polyorder, axis=0)
    # Smooth rotation (convert to rotvec, smooth, convert back)
    rots = Rot.from_quat(poses[:, 3:])
    rotvecs = rots.as_rotvec()
    smoothed_rotvecs = savgol_filter(rotvecs, window_length, polyorder, axis=0)
    smoothed_rots = Rot.from_rotvec(smoothed_rotvecs)
    smoothed_quat = smoothed_rots.as_quat()
    return np.hstack([smoothed_xyz, smoothed_quat])
