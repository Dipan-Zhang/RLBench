# from transformations import rotation_matrix
import numpy as np
from thirdparty.graspNet.gsnet_wrapper import GSNetWrapper
from omegaconf import OmegaConf
import torch
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from transforms3d.quaternions import mat2quat, quat2mat
from affordance.helpers import (hash_filename, read_image, visualize_points,\
resize_img, load_optimization_result, get_configs, interpolate_trajectory,\
visualize_3d_trajectory, pick_points_in_viewer, draw_line, backproject_with_color)
import open3d as o3d
import numpy as np
import copy




# def compute_gripper_pose(grasp, approach_dist, quat=False):
#     "borrowed from HZ codebase, but upside down"
#     grasp_ = copy.deepcopy(grasp)
#     grasp_pos = grasp_.translation
#     grasp_ori = grasp_.rotation_matrix # already in world frame
#     grasp_depth = grasp_.depth
#     grasp_width = grasp_.width
#     grasp_height = grasp_.height


#     grasp_pos += grasp_ori[:3, 2] * (
#         grasp_depth - 0.11 - approach_dist
#     )
#     print(grasp_depth - 0.11 - approach_dist)
#     # grasp_pos -= grasp_ori[:3, 0] * (grasp_height)
#     # grasp_pos += grasp_ori[:3, 0] * (approach_dist) # approaching axis
#     print(grasp_pos)

#     if quat:
#         grasp_ori = Rot.from_matrix(grasp_ori).as_quat()

#     return grasp_pos, grasp_ori

def compute_gripper_poses(grasp, T_wc):
    "double check this"
    T_ee_grasp = np.eye(4)
    T_ee_grasp[:3, :3] = Rot.from_euler('y', -90, degrees=True).as_matrix()
    # Rx_180 = Rot.from_euler('x', 180, degrees=True).as_matrix()
    grasp_ee = copy.deepcopy(grasp)
    # grasp_ee.translation = grasp.translation @ T_ee_grasp[:3, :3].T + T_ee_grasp[:3, 3]
    grasp_ee.rotation_matrix = grasp.rotation_matrix @ T_ee_grasp[:3, :3].T

    pregrasp_pos, pregrasp_ori = compute_gripper_pose(grasp_ee, 0.1, quat=False)
    # pregrasp_ori[:3,:3] = Rx_180 @ pregrasp_ori[:3,:3]    
    grasp_pos, grasp_ori = compute_gripper_pose(grasp_ee, 0, quat=False)
    # grasp_ori[:3,:3] = Rx_180 @ grasp_ori[:3,:3]

    pregrasp_ori_quat = Rot.from_matrix(pregrasp_ori).as_quat()
    grasp_ori_quat = Rot.from_matrix(grasp_ori).as_quat()
    
    pregrasp_gripper_pose = np.concatenate([pregrasp_pos, pregrasp_ori_quat])
    grasp_gripper_pose = np.concatenate([grasp_pos, grasp_ori_quat])

    return pregrasp_gripper_pose, grasp_gripper_pose


def compute_gripper_pose(grasp, approach_dist, quat=True):
    grasp_ = copy.deepcopy(grasp)
    grasp_pos = grasp_.translation
    grasp_ori = grasp_.rotation_matrix # already in world frame
    grasp_depth = grasp_.depth
    grasp_width = grasp_.width
    grasp_height = grasp_.height


    # grasp_pos -= grasp_ori[:3, 2] * (
    #     grasp_depth - 0.11
    # )
    # grasp_pos -= grasp_ori[:3, 0] * (grasp_height)
    # grasp_pos += grasp_ori[:3, 2] * (approach_dist)
    # grasp_pos -= grasp_ori[:3, 2] * (
    #     grasp_depth - 0.11
    # )
    #
    grasp_pos -= grasp_ori[:3, 2] * (grasp_depth - 0.01)
    grasp_pos -= grasp_ori[:3, 2] * (approach_dist)
    print(grasp_pos)

    if quat:
        grasp_ori = Rot.from_matrix(grasp_ori).as_quat()

    return grasp_pos, grasp_ori


def vis_pose(pos, ori, size=0.05, color=None):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0,0,0])
    if color is not None:
        frame.paint_uniform_color(color)    
    T_w_obj = np.eye(4)
    T_w_obj[:3, :3] = ori
    T_w_obj[:3, 3] = pos
    frame.transform(T_w_obj)
    return frame


import ipdb
if __name__ == '__main__':
    "script for fast testing grasp using the data saved from simulation" 
    config_fp = './affordance/task_conf/cup0.yaml'
    config = OmegaConf.load(config_fp)
    gsNet = GSNetWrapper(config)

    # Load the point cloud
    # scene_dir = './grasp_test.npz'
    # scene_dir = './scene_front.npz'
    scene_dir = './gripper_test.npz'
    scene_data = np.load(scene_dir)
    points = scene_data['pcd']
    tgt_pt = scene_data['tgt_pt']
    T_cw = scene_data['T_cw']
    home_gripper_pose = scene_data['home_pose']
    #TODO add geometryies

    points_wrist_world = points
    points_wrist_cam = points_wrist_world @ T_cw[:3, :3].T + T_cw[:3, 3]
    
    pcd = visualize_points(points_wrist_world.reshape(-1, 3))
    pcd_wrist_cam = visualize_points(points_wrist_cam.reshape(-1, 3))
    tgt_pt_cam = tgt_pt @ T_cw[:3, :3].T + T_cw[:3, 3] + np.array([0,0,-0.05])
    # tgt_pt_cam_noised = tgt_pt_cam + np.array([0,0,-0.05]) + np.random.normal(0, 0.01, tgt_pt_cam.shape)

    ipdb.set_trace()
    # best_grasp_world = gsNet.infer_best_grasp(pcd, tgt_pt, max_dis=0.06)
    best_grasp_cam = gsNet.infer_best_grasp(pcd_wrist_cam, tgt_pt_cam, max_dis=0.06)
    # best_grasp_cam_noised = gsNet.infer_best_grasp(pcd_wrist_cam, tgt_pt_cam_noised, max_dis=0.06)

    ############### vis
    # tgt_pt_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=tgt_pt)
    tgt_pt_cam_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=tgt_pt_cam)
    # tgt_pt_cam_noised_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=tgt_pt_cam_noised)
    
    world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    o3d.visualization.draw_geometries([pcd_wrist_cam, world, tgt_pt_cam_vis]) # visualize the scene
    
    # ipdb.set_trace()
    gripper_vis = best_grasp_cam.to_open3d_geometry()
    # gripper_vis_world = best_grasp_world.to_open3d_geometry()
    # gripper_vis_noised = best_grasp_cam_noised.to_open3d_geometry()

    # o3d.visualization.draw_geometries([pcd, world, tgt_pt_vis, gripper_vis_world])
    o3d.visualization.draw_geometries([pcd_wrist_cam, world, tgt_pt_cam_vis, gripper_vis])
    # o3d.visualization.draw_geometries([pcd_wrist_cam, world, tgt_pt_cam_noised_vis, gripper_vis_noised])

    ##############
    T_wc = np.linalg.inv(T_cw)
    best_grasp_cam_pos = best_grasp_cam.translation
    best_grasp_cam_ori = best_grasp_cam.rotation_matrix
    best_grasp_world_pos = best_grasp_cam_pos @ T_wc[:3, :3].T + T_wc[:3, 3]
    best_grasp_world_ori = best_grasp_cam_ori @ T_wc[:3, :3].T
    best_grasp_world = copy.deepcopy(best_grasp_cam)
    best_grasp_world.translation = best_grasp_world_pos
    best_grasp_world.rotation_matrix = best_grasp_world_ori


    # compute the gripper pose
    pregrasp_gripper_pose, grasp_gripper_pose = compute_gripper_poses(best_grasp_world, T_wc)

    gripper_states = ['home', 'home', 'pregrasp', 'grasp', 'attach', 'postgrasp']
    gripper_poses = np.stack([
            home_gripper_pose,
            home_gripper_pose,
            pregrasp_gripper_pose, 
            grasp_gripper_pose, 
            grasp_gripper_pose, 
            grasp_gripper_pose,
            grasp_gripper_pose,
        ])
    
        # generate smooth trajectory
    gripper_smooth_steps = [10, 20, 40, 20, 10, 2]
    gripper_poses_smooth = []
    gripper_efforts_smooth = []

    affordance_traj_world = np.repeat(grasp_gripper_pose.reshape(-1,1), 3, axis = 1)

    for i in range(len(gripper_poses)-1):
        gripper_state = gripper_states[i]
        smooth_steps = gripper_smooth_steps[i]
        # smooth the efforts
        if gripper_state == "attach":
            gripper_effort = 0.0
        elif gripper_state == "postgrasp":
            gripper_effort = 0.0 #Task2Effort[name] read from file
        else:
            gripper_effort = 1.0 # effort -> the force applied

        lower_pos = gripper_poses[i, :3]
        upper_pos = gripper_poses[i + 1, :3]
        lower_qua = gripper_poses[i, 3:7]
        upper_qua = gripper_poses[i + 1, 3:7]
        interp_rot = Slerp([0, 1], Rot.from_quat([lower_qua, upper_qua]))

        for i_smooth in range(smooth_steps):
            i_smooth_quat = interp_rot(i_smooth / smooth_steps).as_quat()
            if gripper_state != "postgrasp":
                i_smooth_pos = (
                    lower_pos + (upper_pos - lower_pos) * i_smooth / smooth_steps
                )
            else:
                # directly us pose traj
                i_smooth_pos = affordance_traj_world[i_smooth]
            gripper_poses_smooth.append(
                np.concatenate([i_smooth_pos, i_smooth_quat])
            )
            gripper_efforts_smooth.append(gripper_effort)
        
    gripper_poses_smooth = np.array(gripper_poses_smooth)
    gripper_efforts_smooth = np.array(gripper_efforts_smooth)
    actions = np.hstack([gripper_poses_smooth, gripper_efforts_smooth.reshape(-1, 1)])

    ##############################
    action_vis2 = []

    for i in range(len(actions)):
        action_vis2.append(vis_pose(actions[i][:3], Rot.from_quat(actions[i][3:7]).as_matrix()))

    world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    # plan_trajectory = visualize_3d_trajectory(affordance_traj_world, size=0.02, cmap_name="plasma", invert=False)
    best_grasp_world_vis = best_grasp_world.to_open3d_geometry()
    ipdb.set_trace()
    # o3d.visualization.draw_geometries(action_vis2 + plan_trajectory + [pcd ,world, best_grasp_world_vis])
    o3d.visualization.draw_geometries(action_vis2 + [pcd ,world, best_grasp_world_vis])
    ##########################################

