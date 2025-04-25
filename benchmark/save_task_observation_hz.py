
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
from tqdm import tqdm
import json
import shutil

from benchmark.helpers import (
                        visualize_points,
                        visualize_3d_trajectory,
                        preprocess_target_data,
                        underscore_string_to_camel_case,
                        backproject
                        )
from benchmark.sim_utils import (
    create_obs_config,
    convert_camera_name,
    hide_robot_temporarily, 
    restore_robot_position,
    set_camera_pose,
    CAMERA_POSES,
    get_T_world_cam_gl
    )


def save_observation(task, obs, cam_name='cam_front', task_name='', object_name='', save_dir='./outputs'):
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
    T_world_cam = get_T_world_cam_gl(obs, cam_name)

    task_rgb_save_fn = os.path.join(save_dir, 'color_000000.png')
    cv2.imwrite(task_rgb_save_fn, rgb[:,:,::-1])

    task_depth_save_fn = os.path.join(save_dir, 'depth_000000.png')
    cv2.imwrite(task_depth_save_fn, (depth*1000).astype(np.uint16))

    pointcloud_save_fn = os.path.join(save_dir, 'pcd.ply')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_reshaped)
    o3d.io.write_point_cloud(pointcloud_save_fn, pcd)

    task_data = preprocess_target_data(rgb, depth, cam_K, 'kinect', obj_name=object_name)
    task_data['T_world_cam'] = T_world_cam
    
    # save mask
    if task_name == 'open_cabinet' or task_name == 'open_slide_cabinet' or task_name == 'down_toilet_seat':
        "use the cropped mask for open cabinet tasks"
        cropped_mask_fn = os.path.join(save_dir, 'mask_000001.png')
        shutil.copyfile(cropped_mask_fn, os.path.join(save_dir, 'mask_000000.png'))
    else:
        object_mask = task_data['mask']
        object_mask = (object_mask * 255).astype(np.uint8)
        mask_save_fn = os.path.join(save_dir, 'mask_000000.png')
        cv2.imwrite(mask_save_fn, object_mask)
    
    # save the task data
    task_data_save_fn = os.path.join(save_dir, 'task_data.npz')
    np.savez(task_data_save_fn, **task_data)
    print(f'saved observation to {save_dir}')
    
    # for debugging
    cropped_rgb = task_data['cropped_rgb']
    cv2.imwrite(os.path.join(save_dir, 'cropped_rgb.png'), cropped_rgb[:,:,::-1])

    # get the gripper pose
    scene_waypoints = task._scene.task.get_waypoints()
    T_world_gripper = np.eye(4)
    found_close_gripper = False
    for wp in scene_waypoints:
        ext_description = wp.get_ext()
        if 'close_gripper' in ext_description:
            T_world_gripper = wp.get_waypoint_object().get_matrix()
            found_close_gripper = True
    if not found_close_gripper:
        T_world_gripper = scene_waypoints[1].get_waypoint_object().get_matrix()

    # save results into a meta.json file
    T_world_cam = get_T_world_cam_gl(obs, cam_name)
    T_cam_gripper = np.linalg.inv(T_world_cam) @ T_world_gripper
    meta_data = {}
    meta_data['contact_point'] = T_cam_gripper[:3,3].tolist()
    meta_data['T_cam_gripper'] = T_cam_gripper.reshape(-1).tolist()
    meta_data['T_world_cam'] = T_world_cam.reshape(-1).tolist()
    meta_data['T_world_gripper'] = T_world_gripper.reshape(-1).tolist()
    meta_data['intrinsics'] = cam_K.reshape(-1).tolist()

    meta_data_save_fn = os.path.join(save_dir, 'meta_000000.json')
    with open(meta_data_save_fn, 'w') as f:
        json.dump(meta_data, f)

def main(args, sim_cfg, task_list):
    # set up env
    save_dir = args.save_dir
    cameras =  ["front", "left_shoulder", "right_shoulder", "wrist", "overhead"]
    # camera_resolution = [sim_cfg['cam_w'], sim_cfg['cam_h']]
    camera_resolution = [456, 256]
    obs_config = create_obs_config(cameras, camera_resolution, method_name="")
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=False), 
            gripper_action_mode=Discrete()
            ),
        obs_config=obs_config,
        headless=args.headless)
    env.launch()
    # num_obs_per_task = args.obs_per_task

    for task_name in tqdm(task_list):
        print(f'Processing task: {task_name}')
        taskName = underscore_string_to_camel_case(task_name)
        try:
            mod = importlib.import_module("rlbench.tasks")
            mod = importlib.reload(mod)
            task_class = getattr(mod, taskName)
            task = env.get_task(task_class)
            obs = None
        except Exception as e:
            print(f"Error processing task {taskName}: {str(e)}")
            continue
        
        print('Reset Episode')
        descriptions, obs = task.reset()
        obs = task.get_observation()
        print('<===== Task description: ====>\n', descriptions)
        obj_name = descriptions[0].split(' ')[-1]
        PREDEFINED_CAMS = CAMERA_POSES[taskName]
        hz_cam_convert = {
            'cam_overhead': 'default',
            'cam_front': 'default',
            'cam_over_shoulder_left': 'left',
            'cam_over_shoulder_right': 'right',
        }
        for camera_name in PREDEFINED_CAMS.keys():
            # save_base_dir = os.path.join(save_dir, taskName, f'{camera_name}')
            hz_save_name = hz_cam_convert.get(camera_name, camera_name)
            save_base_dir = os.path.join(save_dir, taskName, f'{hz_save_name}')

            os.makedirs(save_base_dir, exist_ok=True)
            hide_robot_temporarily('Panda')
            set_camera_pose(camera_name, PREDEFINED_CAMS[camera_name]['pos'], PREDEFINED_CAMS[camera_name]['ori'] ) # get overview of the workspace
            obs = task.get_observation()
            save_observation(task, obs, cam_name=camera_name, task_name=task_name, object_name=obj_name, save_dir=save_base_dir)
            restore_robot_position('Panda')

    print('Done')
    env.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='pick_up_cup', help='task name')
    # parser.add_argument('--obs_per_task', type=int, default=1, help='number of observations per task')
    parser.add_argument('--sim_config_fp', type=str, default='./cfgs/config.yaml', help='config file path')
    parser.add_argument('--headless', action='store_true', help='run in headless mode')
    parser.add_argument('--save', type=bool, default=True, help='whether to save images')
    parser.add_argument('--DEBUG', action='store_true', default=False, help='debug mode')
    parser.add_argument('--save_dir', type=str, default='./benchmark_dataset/', help='save directory')
    args = parser.parse_args()


    sim_cfg_fp = args.sim_config_fp
    sim_cfg = OmegaConf.load(sim_cfg_fp)
    PORTABLE_TASK_LIST = sim_cfg['PORTABLE_TASK_LIST']
    ARTICULATE_TASK_LIST = sim_cfg['ARTICULATE_TASK_LIST']

    if args.task_name == 'all':
        # TODO: finish the camera poses for all tasks
        task_list = PORTABLE_TASK_LIST + ARTICULATE_TASK_LIST
    elif args.task_name == 'portable':
        task_list = PORTABLE_TASK_LIST
    elif args.task_name == 'articulate':
        task_list = ARTICULATE_TASK_LIST
    else:
        task_list = [args.task_name]
    
    main(args, sim_cfg, task_list)

# python benchmark/save_task_observation.py --task_name all --DEBUG
# python benchmark/save_task_observation.py --task_name open_microwave --DEBUG