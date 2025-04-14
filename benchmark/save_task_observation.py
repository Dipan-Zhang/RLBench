
import os
import importlib
import numpy as np
import ipdb
import cv2
import matplotlib.pyplot as plt # for debugging
import open3d as o3d
from omegaconf import OmegaConf

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment

from benchmark.helpers import (
                        visualize_points,
                        visualize_3d_trajectory,
                        preprocess_target_data,
                        underscore_string_to_camel_case
                        )
from benchmark.sim_utils import (
    create_obs_config,
    convert_camera_name,
    hide_robot_temporarily, 
    restore_robot_position,
    set_camera_pose,
    CAMERA_POSES,
    )


def save_observation(obs, cam_name='cam_front', task_name='', object_name='', save_dir='./outputs'):
    "save observation to disk for affordance prediction"
    assert object_name != '', 'object name cannot be empty'

    key_name = convert_camera_name(cam_name)
    rgb = getattr(obs, f'{key_name[:-7]}_rgb')
    depth = getattr(obs, f'{key_name[:-7]}_depth')
    pointcloud = getattr(obs, f'{key_name[:-7]}_point_cloud')
    pointcloud_reshaped = pointcloud.reshape(-1, 3)
    cam_K = obs.misc[key_name+'_intrinsics']
    # fix the negative focal length
    cam_K[0, 0] = np.abs(cam_K[0,0])
    cam_K[1, 1] = np.abs(cam_K[1,1])
    T_world_cam = obs.misc[key_name+'_extrinsics'].copy()


    task_rgb_save_fn = os.path.join(save_dir, 'rgb.png')
    cv2.imwrite(task_rgb_save_fn, rgb[:,:,::-1])

    pointcloud_save_fn = os.path.join(save_dir, 'pcd.ply')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_reshaped)
    o3d.io.write_point_cloud(pointcloud_save_fn, pcd)

    task_data = preprocess_target_data(rgb, depth, cam_K, 'kinect', obj_name=object_name)
    task_data['pointcloud'] = pointcloud_reshaped
    task_data['T_world_cam'] = T_world_cam
    
    task_data_save_fn = os.path.join(save_dir, 'task_data.npz')
    np.savez(task_data_save_fn, **task_data)
    print(f'saved observation to {save_dir}')
    
    # for debugging
    cropped_rgb = task_data['cropped_rgb']
    cv2.imwrite(os.path.join(save_dir, 'cropped_rgb.png'), cropped_rgb[:,:,::-1])


def main(args, sim_cfg, task_list):
    # set up env
    save_dir = args.save_dir
    cameras =  ["front", "left_shoulder", "right_shoulder", "wrist", "overhead"]
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
    # num_obs_per_task = args.obs_per_task

    for task_name in task_list:
        print(f'Processing task: {task_name}')
        try:
            mod = importlib.import_module("rlbench.tasks")
            mod = importlib.reload(mod)
            task_class = getattr(mod, task_name)
            task = env.get_task(task_class)
            obs = None
        except Exception as e:
            print(f"Error processing task {task_name}: {str(e)}")
            continue
        
        print('Reset Episode')
        descriptions, obs = task.reset()
        obs = task.get_observation()
        print('<===== Task description: ====>\n', descriptions)
        obj_name = descriptions[0].split(' ')[-1]
        PREDEFINED_CAMS = CAMERA_POSES[task_name]
        for camera_name in PREDEFINED_CAMS.keys():
            save_base_dir = os.path.join(save_dir, task_name, 'obs', f'{camera_name}')
            os.makedirs(save_base_dir, exist_ok=True)
            hide_robot_temporarily('Panda')
            set_camera_pose(camera_name, PREDEFINED_CAMS[camera_name]['pos'], PREDEFINED_CAMS[camera_name]['ori'] ) # get overview of the workspace
            obs = task.get_observation()
            save_observation(obs, cam_name=camera_name, task_name=task_name, object_name=obj_name, save_dir=save_base_dir)
            restore_robot_position('Panda')

    print('Done')
    env.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='pick_up_cup', help='task name')
    # parser.add_argument('--obs_per_task', type=int, default=1, help='number of observations per task')
    parser.add_argument('--sim_config_fp', type=str, default='./cfgs/config.yaml', help='config file path')
    parser.add_argument('--save', type=bool, default=True, help='whether to save images')
    parser.add_argument('--DEBUG', action='store_true', default=False, help='debug mode')
    parser.add_argument('--save_dir', type=str, default='./outputs/', help='save directory')
    args = parser.parse_args()


    sim_cfg_fp = args.sim_config_fp
    sim_cfg = OmegaConf.load(sim_cfg_fp)
    if args.task_name == 'all':
        # task_list = ['PickUpCup', 'PickUpBottle', 'PickUpMug', 'PickUpBowl', 'PickUpKnife']
        # task_list = ['OpenDrawerFixed', 'OpenMicrowave']
        task_list = list(CAMERA_POSES.keys())
        # TODO: finish the camera poses for all tasks
        # task_list = sim_cfg['PORTABLE_TASK_LIST'] + sim_cfg['ARTICULATE_TASK_LIST']
    elif args.task_name == 'portable':
        task_list = [underscore_string_to_camel_case(x) for x in sim_cfg['PORTABLE_TASK_LIST']]
    elif args.task_name == 'articulate':
        task_list = [underscore_string_to_camel_case(x) for x in sim_cfg['ARTICULATE_TASK_LIST']]
    else:
        task_list = [underscore_string_to_camel_case(args.task_name)]
    
    main(args, sim_cfg, task_list)

# python benchmark/save_task_observation.py --task_name all --DEBUG
# python benchmark/save_task_observation.py --task_name open_microwave --DEBUG