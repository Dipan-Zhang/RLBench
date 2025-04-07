
import numpy as np
import ipdb
import cv2
import matplotlib.pyplot as plt # for debugging
import open3d as o3d
import copy
from omegaconf import OmegaConf
import torch

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from pyrep.errors import ConfigurationPathError
from benchmark.helpers import (
                        visualize_points,
                        visualize_3d_trajectory,
                        preprocess_target_data,
                        load_pickle,
                        save_pickle,
                        visualize_motion_plan,
                        apply_motion_plan
                        )
# from thirdparty.graspNet.gsnet_wrapper import GSNetWrapper
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from benchmark.sim_utils import create_obs_config, vis_pose, compute_gripper_poses,\
      convert_camera_name, draw_trajectory, interpolate_trajectory,\
          get_robot_pose, pose_to_matrix, hide_robot_temporarily, restore_robot_position, \
          adjust_camera_pose, set_camera_pose, CAMERA_POSES
import importlib
import os
import pandas as pd
from typing import List, Tuple
import gc
import pickle

def transform_motion_plan(motion_plan, T_cam_obj, scale=1.0):
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
        t_cam = T_rel_cam[:3, 3] / scale
        motion_plan_cam.append((R_cam, t_cam, success))

    return motion_plan_cam

def generate_postgrasp_trajectory(grasp_T, post_grasp_dir):
    # Generate a post-grasp trajectory
    post_grasp_trajectory = []
    for i in range(10):
        t = grasp_T[:3, 3] + (post_grasp_dir * (i + 1) * 0.04)
        print(t)
        post_grasp_trajectory.append(t)
    return np.array(post_grasp_trajectory)

def plan_gripper_trajectory(obs, affordance_traj_world, vis=False):
    "plan smooth gripper trajectory based on affordance trajectory"
    current_gripper_pose =obs.gripper_pose[:7]
    offset = affordance_traj_world[0] - current_gripper_pose[:3]
    affordance_traj_world -= offset
    affordance_traj_world = np.concatenate([affordance_traj_world[0].reshape(-1,3), affordance_traj_world], axis=0)
    post_gripper_ori = current_gripper_pose[3:7]

    post_gripper_poses = np.concatenate((
        affordance_traj_world, 
        np.repeat(post_gripper_ori.reshape(-1, 4), affordance_traj_world.shape[0], axis=0)), axis=1)
    post_gripper_poses = np.concatenate([post_gripper_poses, np.zeros((affordance_traj_world.shape[0], 1))], axis=1)
    
    # add noise to avoid devide by zero
    noise = np.random.normal(0, 0.005, post_gripper_poses.shape)
    post_gripper_poses[:, :3] += noise[:, :3]
    actions =  post_gripper_poses

    if vis:
        # ipdb.set_trace()
        current_pts = obs.left_shoulder_point_cloud
        current_pcd = visualize_points(current_pts.reshape(-1, 3))
        action_vis = []
        for i in range(len(actions)):
            action_vis.append(vis_pose(actions[i][:3], Rot.from_quat(actions[i][3:7]).as_matrix()))
        world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
        plan_trajectory = visualize_3d_trajectory(affordance_traj_world, size=0.02, cmap_name="plasma", invert=False)
        # best_grasp_world_vis = best_grasp_world.to_open3d_geometry()
        o3d.visualization.draw_geometries(action_vis + plan_trajectory + [current_pcd ,world])
    return actions

def save_frame(frame_idx, camera, obs, image_save_dir):
    # Save the RGB frame
    frame = getattr(obs, f'{camera}_rgb')
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_path = os.path.join(image_save_dir, f"{frame_idx:06d}.png")
    cv2.imwrite(frame_path, frame_bgr)

def act_sparse(obs, actions, trajectory_idx, distance_threshold=0.05):
    """Execute sparse actions with fallback for motion planning failures"""
    # Handle end of trajectory or empty trajectory
    if actions is None or len(actions) == 0:
        return obs.gripper_pose  # Stay in place
        
    if trajectory_idx >= len(actions):
        return actions[-1]  # Stay at last position

    current_action = actions[trajectory_idx]
    current_pos = obs.gripper_pose[:3]
    target_pos = current_action[:3]
    
    # Calculate distance to target
    distance = np.linalg.norm(current_pos - target_pos)
    
    # Check if we've reached the current waypoint
    if distance < distance_threshold:
        old_idx = trajectory_idx
        print(f'Waypoint {old_idx} reached. Distance: {distance:.4f}')
        # If we reached the end of trajectory
        if trajectory_idx >= len(actions):
            print("✓ Complete trajectory executed successfully!")
            return actions[-1]
    
    try:
        return current_action
    except Exception as e:
        print(f"Error executing action {trajectory_idx}: {e}")
        # Emergency fallback - move a tiny bit toward target
        fallback_action = obs.gripper_pose.copy()
        direction = target_pos - current_pos
        if np.linalg.norm(direction) > 0:
            fallback_action[:3] += direction * 0.01 / np.linalg.norm(direction)
        return fallback_action

def plan_motion_plan(obs, motion_plan_world, vis=False):
    """Plan smooth gripper trajectory based on motion plan"""
    current_gripper_pose = copy.deepcopy(obs.gripper_pose[:7])
    
    gripper_pose_matrix = np.eye(4)
    gripper_pose_matrix[:3, :3] = Rot.from_quat(current_gripper_pose[3:7]).as_matrix()
    gripper_pose_matrix[:3, 3] = current_gripper_pose[:3]

    post_gripper_matrices = apply_motion_plan(
        gripper_pose_matrix, motion_plan_world)
    
    post_gripper_poses = []
    for i in range(len(post_gripper_matrices)):
        post_gripper_pose = post_gripper_matrices[i]
        post_gripper_rotation = Rot.from_matrix(post_gripper_pose[:3, :3])
        post_gripper_translation = post_gripper_pose[:3, 3]
        post_gripper_poses.append(np.concatenate([post_gripper_translation, post_gripper_rotation.as_quat()]))
        
    # vis unsmooothed trajectory
    if vis:
        current_pts = obs.left_shoulder_point_cloud
        current_pcd = visualize_points(current_pts.reshape(-1, 3))
        action_vis = []
        for i in range(len(post_gripper_poses)):
            action_vis.append(vis_pose(post_gripper_poses[i][:3], 
                            Rot.from_quat(post_gripper_poses[i][3:7]).as_matrix()))
        world = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0,0,0])
        plan_trajectory = visualize_3d_trajectory(
            np.array(post_gripper_poses)[:, :3], size=0.02, 
            cmap_name="plasma", invert=False)
        o3d.visualization.draw_geometries(
            action_vis + plan_trajectory + [current_pcd, world])
    
    # Process the motion plan with smooth interpolation
    interpolated_poses = []
            
    # If we already have poses, make sure they're properly smoothed
    if len(post_gripper_poses) > 1:
        # Add the first pose
        interpolated_poses.append(post_gripper_poses[0])
        
        # For each consecutive pair of poses
        for i in range(len(post_gripper_poses) - 1):
            current_pose = post_gripper_poses[i]
            next_pose = post_gripper_poses[i+1]
            
            # Calculate distance between poses
            translation_diff = next_pose[:3] - current_pose[:3]
            translation_norm = np.linalg.norm(translation_diff)
            
            max_step_distance = 0.03
            
            if translation_norm > max_step_distance:
                steps = int(np.ceil(translation_norm / max_step_distance))
                
                # Create rotation keypoints for Slerp
                current_rot = Rot.from_quat(current_pose[3:7])
                next_rot = Rot.from_quat(next_pose[3:7])
                
                key_rots = Rot.from_quat([current_rot.as_quat(), next_rot.as_quat()])
                key_times = [0, 1]
                
                # Create the Slerp object for rotation interpolation
                slerp = Slerp(key_times, key_rots)
                
                for step in range(1, steps):
                    step_fraction = step / steps
                    step_pos = current_pose[:3] + translation_diff * step_fraction
                    step_rot = slerp([step_fraction])[0]
                    step_quat = step_rot.as_quat()
                    interpolated_poses.append(np.concatenate([step_pos, step_quat]))
                
                # Add the final pose in this segment
                interpolated_poses.append(next_pose)
            else:
                # Small enough translation, add directly
                interpolated_poses.append(next_pose)
        
        # Replace with interpolated poses
        post_gripper_poses = interpolated_poses
        print(f"Smoothed trajectory from {len(motion_plan_world)} steps to {len(post_gripper_poses)} steps")
    
    if len(post_gripper_poses) == 0:
        print("WARNING: No valid poses generated from motion plan!")
        return
    # Stack poses and add gripper state (open)
    post_gripper_poses = np.stack(post_gripper_poses)
    # Add gripper state (ones for open)
    post_gripper_poses = np.concatenate(
        [post_gripper_poses, np.zeros((post_gripper_poses.shape[0], 1))], 
        axis=1)
    
    # add noise to avoid devide by zero
    noise = np.random.normal(0, 0.005, post_gripper_poses.shape)
    post_gripper_poses[:, :3] += noise[:, :3]

    actions = post_gripper_poses
    print(f'Generated {len(actions)} action steps from motion plan')
    
    if vis:
        # Visualization code remains the same
        current_pts = obs.left_shoulder_point_cloud
        current_pcd = visualize_points(current_pts.reshape(-1, 3))
        action_vis = []
        for i in range(len(actions)):
            action_vis.append(vis_pose(actions[i][:3], 
                            Rot.from_quat(actions[i][3:7]).as_matrix()))
        world = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0,0,0])
        plan_trajectory = visualize_3d_trajectory(
            post_gripper_poses[:, :3], size=0.02, 
            cmap_name="plasma", invert=False)
        o3d.visualization.draw_geometries(
            action_vis + plan_trajectory + [current_pcd, world])
    return actions

def main(args, sim_cfg):
    DEBUG_VIS = args.DEBUG_VIS
    task_name = args.task_name # TODO make this using underscore_string_to_camel_case
    method = args.method
    SAVE_ROOT = args.trial_dir
    trial_name = SAVE_ROOT.split('/')[-1]
    # set up env
    cameras =  ["front", "left_shoulder", "right_shoulder", "wrist", "overhead"]
    camera_resolution = [sim_cfg['cam_w'], sim_cfg['cam_h']]
    obs_config = create_obs_config(cameras, camera_resolution, method_name="")
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=False), 
            gripper_action_mode=Discrete()
            ),
        obs_config=obs_config,
        headless=args.headless
    )
    env.launch()

    mod = importlib.import_module("rlbench.tasks")
    mod = importlib.reload(mod)
    task_class = getattr(mod, args.task_name)
    task = env.get_task(task_class)
    obs = None

    # load affordance
    if method == 'RAM':
        traj_fn = os.path.join(SAVE_ROOT, 'retrieved_motion_all.pkl')
        traj_data = pickle.load(open(traj_fn, 'rb'))
        num_trial = len(traj_data.keys())
    elif method == 'ours':
        motion_data_fp = os.path.join(SAVE_ROOT, 'transferred_motion_all.pkl')
        motion_data = load_pickle(motion_data_fp)
        camera_names = list(motion_data.keys())
        num_trial = len(motion_data[camera_names[0]].keys())
    else:
        raise ValueError('Invalid affordance method name')
    
    # num_trial = 1 #! TEMP FIX for testing ours 

    exp_results_all = {}
    for camera in camera_names:
        result_list = []
        for i in range(num_trial):
            print(f'Camera {camera}, Episode {i}')
            if args.save_video:
                image_save_dir = "./outputs/{}/exp_results/{}/{}/video_{}/obs_{}/trial_{}".format(
                    task_name, method, trial_name, args.video_camera, camera, i
                )
                os.makedirs(image_save_dir, exist_ok=True)
                frame_idx = 0  # to number frames

            PREDEFINED_CAM = CAMERA_POSES[task_name]
            set_camera_pose(camera, PREDEFINED_CAM[camera]['pos'], PREDEFINED_CAM[camera]['ori'] )
            
            # smoothen the trajectory
            print('Reset Episode')
            descriptions, obs = task.reset()
            obs = task.get_observation()
            if args.save_video:
                save_frame(frame_idx, args.video_camera, obs, image_save_dir)
                frame_idx += 1

            if method == 'ours':
                motion_plan_c2 = motion_data[camera][i]['motion_plan']
                # convert the motion plan from c2 to world frame
                cam_key = convert_camera_name(camera)
                T_world_cam = obs.misc[cam_key+'_extrinsics'].copy()

                # converting coppliaSIM camera convention to OpenGL
                R_z_180 = np.array(
                    [[ -1,  0,  0],
                    [  0, -1,  0],
                    [  0,  0,  1]])
                T_world_cam[:3, :3] = T_world_cam[:3, :3] @ R_z_180
                motion_plan_world = transform_motion_plan(motion_plan_c2, T_world_cam)
            elif method == 'RAM':
                grasp_array = traj_data[str(i)]['grasp_array'] # TODO, make this into int
                grasp_array = np.array(grasp_array)
                grasp_R = grasp_array[4:13].reshape(3, 3)
                grasp_t = grasp_array[13:16]
                grasp_T = np.eye(4)
                grasp_T[:3, :3] = grasp_R
                grasp_T[:3, 3] = grasp_t

                post_grasp_dir = traj_data[i]['post_grasp_dir']
                post_grasp_dir = np.array(post_grasp_dir)

                post_grasp_trajectory = generate_postgrasp_trajectory(grasp_T, post_grasp_dir)

            task.move_to_grasp()
            obs = task.get_observation()
            if args.save_video:
                save_frame(frame_idx, args.video_camera, obs, image_save_dir)
                frame_idx += 1
            # get new obs and plan actions
            if method == 'ours':
                actions = plan_motion_plan(obs, motion_plan_world, vis=DEBUG_VIS)
            elif method == 'RAM':
                actions = plan_gripper_trajectory(obs, post_grasp_trajectory, vis=DEBUG_VIS)
                
            episode_length = 20
            trajectory_idx = 0
            for ii in range(episode_length):
                action = act_sparse(obs, actions, trajectory_idx, distance_threshold=0.01)
                obs, reward, terminate = task.step(action)
                trajectory_idx+=1
                if args.save_video:
                    save_frame(frame_idx, args.video_camera, obs, image_save_dir)
                    frame_idx += 1
        
                if terminate:
                    if not reward:
                        print('All fails condition are met, task terminated')
                    else:
                        print('Task Success!')
                    break
            
            result = 0  # Default to failure
            if terminate:
                if reward:
                    result = 1
            else:
                print('Task Timeout!')
                result = 0
            
            result_list.append(result)

            # compose video
            if args.save_video:
                cmd = "ffmpeg -framerate 30 -start_number 0 -i {}/%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p {}/output.mp4 -y".format(
                    image_save_dir, image_save_dir
                )
                os.system(cmd)
        if not args.no_save_result:
            # Save the observation
            save_fn = os.path.join(SAVE_ROOT, camera, 'exp_result.csv')
            to_write = {
                "camera_name": camera,
                "ID": np.arange(len(result_list)),
                "scores": result_list,
            }
            df = pd.DataFrame(to_write)
            df = df.to_csv(save_fn, mode="w", index=None)
        
        exp_results_all[camera] = result_list
    
    exp_save_dir = os.path.join(SAVE_ROOT, 'exp_results')
    os.makedirs(exp_save_dir, exist_ok=True)
    exp_results_all_save_fp = os.path.join(exp_save_dir, 'exp_results_all.pkl')
    save_pickle(exp_results_all_save_fp, exp_results_all)
    # save the results
    # if not args.no_save_result:
    #     save_results_dir = "./outputs/{}/{}/exp_results/{}/".format(
    #         task_name, args.method, get_time()
    #     )
    #     os.makedirs(save_results_dir, exist_ok=True)
    #     save_results_path = os.path.join(save_results_dir, 'result.csv')
    #     to_write = {
    #         "ID": np.arange(len(exp_results)),
    #         "scores": exp_results,
    #     }
    #     df = pd.DataFrame(to_write)
    #     df = df.to_csv(save_results_path, mode="w", index=None)

    print('Done')
    env.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--camera', type=str, default='cam_front', help='camera for affordance transfer')
    parser.add_argument('--task_name', type=str, default='PickUpCup', help='task name')
    parser.add_argument('--method', type=str, default='ours', help='affordance method name')
    parser.add_argument('--sim_config_fp', type=str, default='./cfgs/config.yaml', help='config file path')
    parser.add_argument('--no_save_result', type=bool, default=False, help='whether to save images')
    parser.add_argument('--trial_dir', type=str, help='directory to save results')
    parser.add_argument('--save_video', type=bool, default=False, help='whether to save video')
    parser.add_argument('--video_camera', type=str, default='front', help='camera name for video')
    parser.add_argument('--trial_save_dir', type=str, default='./outputs/', help='save directory')
    parser.add_argument('--DEBUG_VIS', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    sim_cfg_fp = args.sim_config_fp
    sim_cfg = OmegaConf.load(sim_cfg_fp)

    main(args, sim_cfg)

# Portable 
# python benchmark/test_affordance.py --task_name PickUpCup --method ours --DEBUG_VIS 
# python benchmark/test_affordance.py --task_name PickUpBottle --method ours  --DEBUG_VIS 
# python benchmark/test_affordance.py --task_name PickUpMug --method ours  --DEBUG_VIS 
# python benchmark/test_affordance.py --task_name PickUpBowl --method ours  --DEBUG_VIS 

# Articulate
# python benchmark/test_affordance.py --task_name OpenMicrowave --method ours --save_video True --video_camera left_shoulder
# python benchmark/test_affordance.py --task_name OpenMicrowave --method ours  --DEBUG_VIS

# python benchmark/test_affordance.py --task_name CloseMicrowave --method ours --save_video True --video_camera left_shoulder




# RAM 
# python benchmark/test_affordance.py --task_name OpenDrawer --method RAM --save_video True --video_camera left_shoulder
# python benchmark/test_affordance.py --task_name OpenMicrowave --method RAM --save_video True --video_camera left_shoulder