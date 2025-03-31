
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

from benchmark.helpers import (
                        visualize_points,
                        visualize_3d_trajectory,
                        preprocess_target_data,
                        )
# from dataset_utils.generate_masked_object import compute_cropping_params, crop_images, compute_cropped_intrinsics

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

def get_date():
    """Get the current date as a string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")
def transform_trajectory(affordance_cam, T_world_cam):
    affordance_trajectory = affordance_cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    # print(f'transformed affordance trajectory: {affordance_trajectory}')
    return affordance_trajectory

def transform_motion_plan(motion_plan, T_world_cam):
    R_world_cam = T_world_cam[:3, :3]
    motion_plan_world = []
    for (R, t, success) in motion_plan:
        R = R_world_cam @ R @ R_world_cam.T
        t = R_world_cam @ t
        motion_plan_world.append((R, t, success))
    return motion_plan_world

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

class Agent(object):
    def __init__(self, method_name, config_fp=None):
        self.method_name = method_name # affordance method
        self.rgbs = None
        self.depths = None
        self.cam_K = None
        self.T_co_optimized = None
        self.trajectory_idx = 0
        self.T_world_cam = np.eye(4)  # Assuming identity matrix as default
        self.affordance_trajectory = None
        self.gripper_closed = False
        self.grasp_waypoint_idx = 1
        self.distance_threshold = 0.05
        self.pregrasp_offset = 0.1
        self.gripper_states = ['home', 'home', 'pregrasp', 'grasp', 'attach', 'postgrasp']
        self.grounded_dino_model = None
        self.sam_predictor = None

        if config_fp is not None:
            print(f'affordance transfer using config from: {config_fp}')
            self.config = OmegaConf.load(config_fp)
            if self.method_name == 'ours':
                pass
            elif self.method_name == 'RAM':
                self.method = None
            else:
                raise ValueError('Invalid affordance method name')
        else:
            print('No config file provided, affordance transfer not initialized')
        
        # self.gsNet = GSNetWrapper(self.config)


    
    def run_method(self, obs, cam_name='cam_front', tgt_obj_prompt='', DEBUG=False):
        """
        Run the affordance transfer method on the given observation and camera name.
        Returns the 3D correspondences, the affordance trajectory in camera frame, and the affordance trajectory in world frame.
        """
        # generate the required data for affordance transfer
        self.init_ori = obs.gripper_pose[3:7]
        self._get_images(obs, SAVE=False)
        self._get_camera_intrinsics_and_pose(obs, camera_name=cam_name)
        key_name = convert_camera_name(cam_name)
        tgt_rgb = self.rgbs[key_name[:-7]]
        tgt_depth = self.depths[key_name[:-7]]
        
        ################## temp for testing RAM offline ###########
        save_base_dir = f'./NeuS/exp/sim_results/{tgt_obj_prompt}'
        os.makedirs(save_base_dir, exist_ok=True)
        task_rgb_save_fn = os.path.join(save_base_dir, 'rgb.png')
        cv2.imwrite(task_rgb_save_fn, tgt_rgb[:,:,::-1])
        tgt_pointcloud = getattr(obs, f'{key_name[:-7]}_point_cloud')
        tgt_pointcloud_reshaped = tgt_pointcloud.reshape(-1, 3)
        tgt_pointcloud_save_fn = os.path.join(save_base_dir, 'pcd.ply')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tgt_pointcloud_reshaped)
        o3d.io.write_point_cloud(tgt_pointcloud_save_fn, pcd)
        print(f'saved tgt pointcloud to {tgt_pointcloud_save_fn}')
        ###########################################################
        
        tgt_data = preprocess_target_data(tgt_rgb, tgt_depth, self.cam_K, 'kinect', obj_name=tgt_obj_prompt)

        ##################### run method ####################
        # self.T_co_optimized, corres_3d_c2, self.affordance_cam = self.method.run(tgt_data)

        self.T_co_optimized, corres_3d_c2, self.motion_plan_c2 = self.method.run(tgt_data)


        # if DEBUG:
        #     tgt_pts, colors = backproject_with_color(tgt_data['depth'], tgt_data['rgb'],
        #                                      tgt_data['camera_intrinsic'], tgt_data['mask'],
        #                                      NOCS_convention=False)
        #     tgt_pcd = visualize_points(tgt_pts, colors)
        #     # visualize backprojected object and affordance trajectory
        #     visualize_affordance_with_scene(tgt_pcd, self.affordance_cam, corres_3d_c2)

        #! convert left hand camera convention (CoppeliaSIM) to opengl convention !
        R_z_180 = np.array([[ -1,  0,  0],
                            [  0, -1,  0],
                            [  0,  0,  1]])
        # transfer the affordance trajectory to world frame for robot to execute
        self.trajectory_idx = 0
        self.T_world_cam[:3, :3] = self.T_world_cam[:3,:3] @ R_z_180

        # self.affordance_trajectory = transform_trajectory(self.affordance_cam, self.T_world_cam) # camera to world for exection
        self.motion_plan_world = transform_motion_plan(self.motion_plan_c2, self.T_world_cam) # camera to world for exection
        #####################################################
        

        return corres_3d_c2, self.motion_plan_c2, self.motion_plan_world

    def plan_gripper_trajectory(self, obs, affordance_traj_world, vis=False):
        "plan smooth gripper trajectory based on affordance trajectory"
        current_gripper_pose =obs.gripper_pose[:7]
        offset = affordance_traj_world[0] - current_gripper_pose[:3]
        affordance_traj_world -= offset
        affordance_traj_world_downsampled = affordance_traj_world[::10]
        affordance_traj_world_downsampled = np.concatenate([affordance_traj_world[0].reshape(-1,3), affordance_traj_world_downsampled], axis=0)
        post_gripper_ori = current_gripper_pose[3:7]

        post_gripper_poses = np.concatenate((affordance_traj_world_downsampled, np.repeat(post_gripper_ori.reshape(-1, 4), affordance_traj_world_downsampled.shape[0], axis=0)), axis=1)
        post_gripper_poses = np.concatenate([post_gripper_poses, np.zeros((affordance_traj_world_downsampled.shape[0], 1))], axis=1)
        
        # add noise to avoid devide by zero
        noise = np.random.normal(0, 0.005, post_gripper_poses.shape)
        post_gripper_poses[:, :3] += noise[:, :3]
        self.actions =  post_gripper_poses

        if vis:
            # ipdb.set_trace()
            current_pts = obs.left_shoulder_point_cloud
            current_pcd = visualize_points(current_pts.reshape(-1, 3))
            action_vis = []
            for i in range(len(self.actions)):
                action_vis.append(vis_pose(self.actions[i][:3], Rot.from_quat(self.actions[i][3:7]).as_matrix()))
            world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
            plan_trajectory = visualize_3d_trajectory(affordance_traj_world, size=0.02, cmap_name="plasma", invert=False)
            # best_grasp_world_vis = best_grasp_world.to_open3d_geometry()
            o3d.visualization.draw_geometries(action_vis + plan_trajectory + [current_pcd ,world])

    def plan_motion_plan(self, obs, motion_plan_world, vis=False):
        """Plan smooth gripper trajectory based on motion plan"""
        current_gripper_pose = copy.deepcopy(obs.gripper_pose[:7])
        
        post_gripper_poses = []
        current_gripper_rotation = Rot.from_quat(current_gripper_pose[3:7])
        current_gripper_translation = current_gripper_pose[:3]

        for (R, t, success) in motion_plan_world:
            if success:
                motion_rotation = Rot.from_matrix(R)
                new_gripper_rotation = current_gripper_rotation * motion_rotation
                new_pos = (np.matmul(R, current_gripper_translation[:, np.newaxis]) + t[:, np.newaxis]).squeeze()

                post_gripper_poses.append(np.concatenate([new_pos, new_gripper_rotation.as_quat()]))
                current_gripper_translation = new_pos
                current_gripper_rotation = new_gripper_rotation
            else:
                print('Failed to find a valid grasp pose, skipping this pose')
                continue
        ipdb.set_trace()
            
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
        
        self.actions = post_gripper_poses
        print(f'Generated {len(self.actions)} action steps from motion plan')
        
        if vis:
            # Visualization code remains the same
            current_pts = obs.left_shoulder_point_cloud
            current_pcd = visualize_points(current_pts.reshape(-1, 3))
            action_vis = []
            for i in range(len(self.actions)):
                action_vis.append(vis_pose(self.actions[i][:3], 
                                Rot.from_quat(self.actions[i][3:7]).as_matrix()))
            world = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0,0,0])
            plan_trajectory = visualize_3d_trajectory(
                post_gripper_poses[:, :3], size=0.02, 
                cmap_name="plasma", invert=False)
            o3d.visualization.draw_geometries(
                action_vis + plan_trajectory + [current_pcd, world])

    
    def act_sparse(self, obs):
        """Execute sparse actions with fallback for motion planning failures"""
        # Handle end of trajectory or empty trajectory
        if self.actions is None or len(self.actions) == 0:
            return obs.gripper_pose  # Stay in place
            
        if self.trajectory_idx >= len(self.actions):
            return self.actions[-1]  # Stay at last position

        current_action = self.actions[self.trajectory_idx]
        current_pos = obs.gripper_pose[:3]
        target_pos = current_action[:3]
        
        # Calculate distance to target
        distance = np.linalg.norm(current_pos - target_pos)
        
        # Check if we've reached the current waypoint
        if distance < self.distance_threshold:
            old_idx = self.trajectory_idx
            self.trajectory_idx += 1
            print(f'Waypoint {old_idx} reached. Distance: {distance:.4f}')
            # If we reached the end of trajectory
            if self.trajectory_idx >= len(self.actions):
                print("✓ Complete trajectory executed successfully!")
                return self.actions[-1]
        
        try:
            return current_action
        except Exception as e:
            print(f"Error executing action {self.trajectory_idx}: {e}")
            # Emergency fallback - move a tiny bit toward target
            fallback_action = obs.gripper_pose.copy()
            direction = target_pos - current_pos
            if np.linalg.norm(direction) > 0:
                fallback_action[:3] += direction * 0.01 / np.linalg.norm(direction)
            return fallback_action
        
    def reset(self):
        """Call this function at the start of each episode to reset the agent's state."""
        self.phase = 'pregrasp'
        self.trajectory_idx = 0
        self.gripper_closed = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _get_images(self, obs, SAVE=False):
        "camera_name: cam_front cam_left_shoulder etc"
        camera_keys = ['front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist']
        self.rgbs = {key: getattr(obs, f'{key}_rgb') for key in camera_keys}
        self.depths = {key: getattr(obs, f'{key}_depth') for key in camera_keys}

        if SAVE:
            for key in camera_keys:
                cv2.imwrite(f'./outputs/rlbench/{key}_rgb.png', self.rgbs[key])
                cv2.imwrite(f'./outputs/rlbench/{key}_depth.png', (self.depths[key] * 1000).astype(np.float32))
            print(f'saved images to ./outputs/rlbench/')
    
    def _get_camera_intrinsics_and_pose(self, obs, camera_name='cam_front'):
        cam_key = convert_camera_name(camera_name)
        self.cam_K = obs.misc[cam_key+'_intrinsics']

        # fix the negative focal length
        self.cam_K[0, 0] = np.abs(self.cam_K[0,0])
        self.cam_K[1, 1] = np.abs(self.cam_K[1,1])
        self.T_world_cam = obs.misc[cam_key+'_extrinsics'].copy()

def main(args, sim_cfg):
    # set up env
    cameras =  ["front", "left_shoulder", "right_shoulder", "wrist"]
    camera_resolution = [sim_cfg['cam_w'], sim_cfg['cam_h']]
    obs_config = create_obs_config(cameras, camera_resolution, method_name="")
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=False), 
            gripper_action_mode=Discrete()
            ),
    obs_config=obs_config,
        headless=False)
    env.launch()

    mod = importlib.import_module("rlbench.tasks")
    mod = importlib.reload(mod)
    task_class = getattr(mod, args.task_name)
    task = env.get_task(task_class)
    obs = None

    # set up the method
    task_name = args.task_name
    method = args.method
    # load affordance
    task_data_fp = os.path.join(args.save_dir, task_name, 'task_data.npz')
    task_data = np.load(task_data_fp, allow_pickle=True)
    if method == 'RAM':
        traj_fn = os.path.join(args.save_dir, task_name, method, 'result_all.pkl')
        traj_data = pickle.load(open(traj_fn, 'rb'))
    elif method == 'ours':
        pass
    else:
        raise ValueError('Invalid affordance method name')
    
    num_trial = len(traj_data.keys())

    exp_results = []
    for i in range(num_trial):
        print(f'Episode {i}')
        if args.save_video:
            image_save_dir = "./outputs/{}/{}/exp_results/{}/video_{}/trial_{}".format(
                task_name, args.method, get_date(), args.video_camera, i
            )
            os.makedirs(image_save_dir, exist_ok=True)
            frame_idx = 0  # to number frames

        if method == 'ours':
            pass
        elif method == 'RAM':
            grasp_array = traj_data[str(i)]['grasp_array']
            grasp_array = np.array(grasp_array)
            grasp_R = grasp_array[4:13].reshape(3, 3)
            grasp_t = grasp_array[13:16]
            grasp_T = np.eye(4)
            grasp_T[:3, :3] = grasp_R
            grasp_T[:3, 3] = grasp_t

            post_grasp_dir = traj_data[str(i)]['post_grasp_dir']
            post_grasp_dir = np.array(post_grasp_dir)

            post_grasp_trajectory = generate_postgrasp_trajectory(grasp_T, post_grasp_dir)

        # smoothen the trajectory
        print('Reset Episode')
        descriptions, obs = task.reset()
        obs = task.get_observation()
        task.move_to_grasp()
        obs = task.get_observation()
        actions = plan_gripper_trajectory(obs, post_grasp_trajectory, vis=True)
            
        # execute the trajectory
        episode_length = 40
        trajectory_idx = 0
        for ii in range(episode_length):
            action = act_sparse(obs, actions, trajectory_idx, distance_threshold=0.05)
            obs, reward, terminate = task.step(action)
            trajectory_idx+=1

            if args.save_video:
                frame = getattr(obs, f'{args.video_camera}_rgb')
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_path = os.path.join(image_save_dir, f"{frame_idx:06d}.png")
                cv2.imwrite(frame_path, frame_bgr)
                frame_idx += 1
                     
            if terminate:
                if not reward:
                    print('All fails condition are met, task terminated')
                else:
                    print('Task Success!')
                break
        
        result = -1  # Default to failure
        if terminate:
            if reward:
                result = 1
        else:
            print('Task Timeout!')
            result = 0
        exp_results.append(result)

        # compose video
        if args.save_video:
            cmd = "ffmpeg -framerate 30 -start_number 0 -i {}/%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p {}/output.mp4".format(
                image_save_dir, image_save_dir
            )
            os.system(cmd)

    # save the results
    if args.save:
        save_results_dir = "./outputs/{}/{}/exp_results/{}/".format(
            task_name, args.method, get_date()
        )
        os.makedirs(save_results_dir, exist_ok=True)
        save_results_path = os.path.join(save_results_dir, 'result.csv')
        to_write = {
            "ID": np.arange(len(exp_results)),
            "scores": exp_results,
        }
        df = pd.DataFrame(to_write)
        df = df.to_csv(save_results_path, mode="w", index=None)

    print('Done')
    env.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--camera', type=str, default='cam_front', help='camera for affordance transfer')
    parser.add_argument('--task_name', type=str, default='PickUpCup', help='task name')
    parser.add_argument('--method', type=str, default='ours', help='affordance method name')
    parser.add_argument('--sim_config_fp', type=str, default='./cfgs/config.yaml', help='config file path')
    parser.add_argument('--save', type=bool, default=True, help='whether to save images')
    parser.add_argument('--save_video', type=bool, default=False, help='whether to save video')
    parser.add_argument('--video_camera', type=str, default='front', help='camera name for video')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default='./outputs/', help='save directory')
    args = parser.parse_args()

    sim_cfg_fp = args.sim_config_fp
    sim_cfg = OmegaConf.load(sim_cfg_fp)

    main(args, sim_cfg)

# Portable 
# python benchmark/test_affordance.py --task_name PickUpCup --method ours --debug True
# python benchmark/test_affordance.py --task_name PickUpBottle --method ours  --debug True
# python benchmark/test_affordance.py --task_name PickUpMug --method ours  --debug True
# python benchmark/test_affordance.py --task_name PickUpBowl --method ours  --debug True

# Articulate
# python benchmark/test_affordance.py --task_name OpenDrawer --method ours  --debug True
# python benchmark/test_affordance.py --task_name OpenMicrowave --method ours  --debug True



# RAM 
# python benchmark/test_affordance.py --task_name OpenDrawer --method RAM --save_video True --video_camera left_shoulder
# python benchmark/test_affordance.py --task_name OpenMicrowave --method RAM --save_video True --video_camera left_shoulder