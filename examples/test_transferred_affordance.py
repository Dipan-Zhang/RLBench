
import numpy as np
import ipdb
import cv2
import matplotlib.pyplot as plt # for debugging
import open3d as o3d
import copy
from omegaconf import OmegaConf

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment

from affordance.helpers import (
                        visualize_points,
                        visualize_3d_trajectory,
                        pick_points_in_viewer,
                        draw_line, 
                        backproject_with_color,
                        preprocess_target_data
                        )
from visualization import visualize_affordance_with_scene
from NeuS.models.utils import backproject
from thirdparty.graspNet.gsnet_wrapper import GSNetWrapper
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from simulation.sim_utils import create_obs_config, vis_pose, compute_gripper_poses,\
      convert_camera_name, draw_trajectory, interpolate_trajectory,\
          get_robot_pose, pose_to_matrix, hide_robot_temporarily, restore_robot_position, \
          adjust_camera_pose, set_camera_pose, CAMERA_POSES
import importlib
import os
import pandas as pd
from typing import List, Tuple

#TODO temp solution before task retrieval ready => should be ok bc each demo is very different
DEMO_CFG= {
  'PickUpCup': './cfgs/task/cup0.yaml',
  'SlideCabinetOpen': './cfgs/task/cupboard1.yaml',
  'OpenDrawerFixed': './cfgs/task/drawer1.yaml',
  # 'take_bowl': './cfgs/task/bowl0.yaml',
}

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
        

        if config_fp is not None:
            print(f'affordance transfer using config from: {config_fp}')
            self.config = OmegaConf.load(config_fp)
            if self.method_name == 'ours':
                from affordance.affordance_transfer import AffordanceTransfer
                self.method = AffordanceTransfer(self.config) # TODO Split the method and sim config
            elif self.method_name == 'RAM':
                pass
            else:
                raise ValueError('Invalid affordance method name')
        else:
            print('No config file provided, affordance transfer not initialized')
        
        self.gsNet = GSNetWrapper(self.config)


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
            ipdb.set_trace()
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
        "plan smooth gripper trajectory based on affordance trajectory"
        current_gripper_pose = copy.deepcopy(obs.gripper_pose[:7])

        ipdb.set_trace()
        post_gripper_poses = []
        current_gripper_rotation = Rot.from_quat(current_gripper_pose[3:7])
        current_gripper_translation = current_gripper_pose[:3]
        for (R, t, success) in motion_plan_world:
            if success:
                motion_rotation = Rot.from_matrix(R)
                current_gripper_rotation = current_gripper_rotation * motion_rotation 
                post_gripper_ori = current_gripper_rotation.as_quat()
                
                current_gripper_translation = current_gripper_translation[:3] + t

                post_gripper_poses.append(np.concatenate([current_gripper_translation, post_gripper_ori]))
            else:
                print('Failed to find a valid grasp pose, skipping this pose')
                continue
        
        post_gripper_poses = np.stack(post_gripper_poses)
        post_gripper_poses = np.concatenate([post_gripper_poses, np.zeros((post_gripper_poses.shape[0], 1))], axis=1)
        
        # add noise to avoid devide by zero
        noise = np.random.normal(0, 0.005, post_gripper_poses.shape)
        post_gripper_poses[:, :3] += noise[:, :3]
        self.actions =  post_gripper_poses

        if vis:
            current_pts = obs.left_shoulder_point_cloud
            current_pcd = visualize_points(current_pts.reshape(-1, 3))
            action_vis = []
            for i in range(len(self.actions)):
                action_vis.append(vis_pose(self.actions[i][:3], Rot.from_quat(self.actions[i][3:7]).as_matrix()))
            world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
            plan_trajectory = visualize_3d_trajectory(post_gripper_poses[:, :3], size=0.02, cmap_name="plasma", invert=False)
            ipdb.set_trace()
            o3d.visualization.draw_geometries(action_vis + plan_trajectory + [current_pcd ,world])


    def act_sparse(self, obs):
        if self.trajectory_idx >= len(self.actions):
            random_noise = np.random.normal(0, 0.005, self.actions.shape[0])
            return self.actions[-1] + random_noise[:3]

        current_action = self.actions[self.trajectory_idx]
        current_pos = obs.gripper_pose[:3]
        target_pos = current_action[:3]

        distance = np.linalg.norm(current_pos - target_pos)

        if distance < self.distance_threshold:
            self.trajectory_idx += 1
            print(f'Gripper from current pose {current_pos} to target pose {target_pos}')

        return current_action
    
    def reset(self):
        """Call this function at the start of each episode to reset the agent's state."""
        self.phase = 'pregrasp'
        self.trajectory_idx = 0
        self.gripper_closed = False

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
            arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, ), 
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
    # TEMP fix
    if args.task_name =='OpenDrawer':
        task_name = 'OpenDrawerFixed'
    else:
        task_name = args.task_name
    affordance_agent = Agent(args.method, config_fp=DEMO_CFG[task_name])

    episode_length = 100
    num_try = 3
    exp_results = []
    for i in range(num_try):
        print(f'Episode {i}')
        for ii in range(episode_length):
            if ii % episode_length == 0:
                print('Reset Episode')
                descriptions, obs = task.reset()
                affordance_agent.reset()
                obs = task.get_observation()
                print('<===== Task description: ====>\n', descriptions)

                obj_name = descriptions[0][11:]
                if args.task_name == 'PickUpCup':
                    corres_3d_c2, motion_plan_c2, motion_plan_world = \
                        affordance_agent.run_method(obs, cam_name=args.camera, tgt_obj_prompt=obj_name, DEBUG=False)
                else:
                    hide_robot_temporarily('Panda')
                    PREDEFINED_CAM = CAMERA_POSES[task_name]
                    set_camera_pose(PREDEFINED_CAM['camera_name'], PREDEFINED_CAM['pos'], PREDEFINED_CAM['ori'] ) # get overview of the workspace
                    obs = task.get_observation()
                    affordance_agent.reset()
                    corres_3d_c2, motion_plan_c2, motion_plan_world = \
                        affordance_agent.run_method(obs, cam_name=args.camera, tgt_obj_prompt=obj_name, DEBUG=True)
                    restore_robot_position('Panda')
                
                task.move_to_grasp()
                obs = task.get_observation()
                # affordance_agent.plan_gripper_trajectory(obs, affordance_traj_world, vis=True)
                affordance_agent.plan_motion_plan(obs, motion_plan_world, vis=True)

            action = affordance_agent.act_sparse(obs)
            obs, reward, terminate = task.step(action)
        
            if terminate:
                if not reward:
                    print('All fails condition are met, task terminated')
                else:
                    print('Task Success!')
                break


    # record the feedback/ video
    task_env = 'rlbench'
    task_name = args.task_name

    # read yarr for this part
    if args.save:
        trial = 0
        save_results_dir = "results/{}/{}/{}/{}".format(
            trial, task_env, task_name, args.camera
        )
        os.makedirs(save_results_dir, exist_ok=True)
        save_results_path = "{}/{}.csv".format(save_results_dir, args.method)
        to_write = {
            "ID": np.arange(len(exp_results)),
            "scores": exp_results,
        }
        df = pd.DataFrame(to_write)
        df = df.to_csv(save_results_path, mode="w", index=None)

    # if args.save_video:
    #     image_save_dir = "result_videos/{}/{}/{}/{}/".format(
    #         task_env, task_name, args.video_camera, args.method
    #     )
    #     cmd = "ffmpeg -framerate 30 -start_number 10 -i {}/%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p {}/output.mp4".format(
    #         image_save_dir, image_save_dir
    #     )
    #     os.system(cmd)

    print('Done')
    env.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, default='cam_front', help='camera for affordance transfer')
    parser.add_argument('--task_name', type=str, default='PickUpCup', help='task name')
    parser.add_argument('--method', type=str, default='ours', help='affordance method name')
    parser.add_argument('--sim_config_fp', type=str, default=None, help='config file path')
    parser.add_argument('--save', type=bool, default=True, help='whether to save images')
    parser.add_argument('--save_video', type=bool, default=False, help='whether to save video')
    parser.add_argument('--video_camera', type=str, default='front', help='camera name for video')
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()

    sim_cfg_fp = args.sim_config_fp
    sim_cfg = OmegaConf.load(sim_cfg_fp)

    main(args, sim_cfg)


# python simulation/test_transferred_affordance.py --task_name PickUpCup --method ours --sim_config_fp /home/stud/zanr/code/MastertThesisAR/cfgs/simulation/config.yaml --debug True
# python simulation/test_transferred_affordance.py --task_name OpenDrawer --camera cam_over_shoulder_left --method ours --sim_config_fp /home/stud/zanr/code/MastertThesisAR/cfgs/simulation/config.yaml --debug True
# python simulation/test_transferred_affordance.py --task_name OpenDrawer --camera cam_over_shoulder_left --method ours --sim_config_fp /home/stud/zanr/code/MastertThesisAR/cfgs/simulation/config.yaml --debug True