# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import yaml
import csv
import torch
import cv2
import shutil
import numpy as np

import open3d as o3d
from benchmark.helpers import visualize_3d_trajectory
import ipdb


from omegaconf import OmegaConf
from multiprocessing import Value
# from tensorflow.python.summary.summary_iterator import summary_iterator
from copy import deepcopy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode

from pyrep.backend import sim
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import copy
import transforms3d as t3d
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects import Object


# figure out how to parse target name??
CAMERA_POSES = {
    "PickUpBottle":{
        'cam_front':{
        'pos': [1.35, 0, 1.13],
        'ori': [-180, -75, 90],
        },
        'cam_over_shoulder_left':{
            'pos': [1.35, -0.3, 1.13],
            'ori': [-110, -70, 160]
        },
        'cam_over_shoulder_right':{
            'pos': [1.35, 0.3, 1.13],
            'ori': [135, -75, 45]
        },
    },
    "PickUpCup":{
        'cam_front':{
        'pos': [1.35, 0, 1.13],
        'ori': [-180, -75, 90],
        },
        'cam_over_shoulder_left':{
            'pos': [1.35, -0.3, 1.13],
            'ori': [-110, -70, 160]
        },
        'cam_over_shoulder_right':{
            'pos': [1.35, 0.3, 1.13],
            'ori': [135, -75, 45]
        },
        },
    "PickUpMug":{
        'cam_front':{
        'pos': [1.35, 0, 1.13],
        'ori': [-180, -75, 90],
        },
        'cam_over_shoulder_left':{
            'pos': [1.35, -0.3, 1.13],
            'ori': [-110, -70, 160]
        },
        'cam_over_shoulder_right':{
            'pos': [1.35, 0.3, 1.13],
            'ori': [135, -75, 45]
        },
        },
    "PickUpBowl":{
        'cam_front':{
        'pos': [1.35, 0, 1.13],
        'ori': [-180, -75, 90],
        },
        'cam_over_shoulder_left':{
            'pos': [1.35, -0.3, 1.13],
            'ori': [-110, -70, 160]
        },
        'cam_over_shoulder_right':{
            'pos': [1.35, 0.3, 1.13],
            'ori': [135, -75, 45]
        },
        },


    "OpenDrawer":{
        'cam_over_shoulder_left':{
            'pos': [0.5, 1.6, 1.4],
            'ori': [110, -5, 0]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.1, 1.6, 1.4],
            'ori': [110, 5, 0]
        },
        'cam_overhead':{
            'pos': [0.2, 1.6, 1.7],
            'ori': [110, 0, 0]
        }
    },
    "CloseDrawer":{
        'cam_over_shoulder_left':{
            'pos': [0.5, 1.6, 1.4],
            'ori': [110, -5, 0]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.1, 1.6, 1.4],
            'ori': [110, 5, 0]
        },
        'cam_overhead':{
            'pos': [0.2, 1.6, 1.7],
            'ori': [110, 0, 0]
        }
    },

    "OpenMicrowave":{
        'cam_over_shoulder_left':{
            'pos': [0.8, -1.5, 1.2],
            'ori': [-100, -14, 180]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.2, -1.5, 1.2],
            'ori': [-100, 14, 180]
        },
        'cam_overhead':{
            'pos': [0.2, -1.6, 1.7],
            'ori': [-115, 0, -180]
        }
    },
    "CloseMicrowave":{
        'cam_over_shoulder_left':{
            'pos': [0.8, -1.5, 1.2],
            'ori': [-100, -14, 180]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.2, -1.5, 1.2],
            'ori': [-100, 14, 180]
        },
        'cam_overhead':{
            'pos': [0.2, -1.6, 1.7],
            'ori': [-115, 0, -180]
        }
    },

    "OpenDishwasher":{
        'cam_over_shoulder_left':{
            'pos': [0.8, -1.5, 1.2],
            'ori': [-100, -14, 180]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.2, -1.5, 1.2],
            'ori': [-100, 14, 180]
        },
        'cam_overhead':{
            'pos': [0.2, -1.6, 1.7],
            'ori': [-115, 0, -180]
        }
    },
    "CloseLaptop":{
        'cam_over_shoulder_left':{
            'pos': [0.8, -1.5, 1.2],
            'ori': [-100, -14, 180]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.2, -1.5, 1.2],
            'ori': [-100, 14, 180]
        },
        'cam_overhead':{
            'pos': [0.2, -1.6, 1.7],
            'ori': [-115, 0, -180]
        }
    },
    "DownToiletSeat":{
        'cam_over_shoulder_left':{
            'pos': [0.5, 1.6, 1.4],
            'ori': [110, -5, 0]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.1, 1.6, 1.4],
            'ori': [110, 5, 0]
        },
        'cam_overhead':{
            'pos': [0.2, 1.6, 1.7],
            'ori': [110, 0, 0]
        }
    },

    "OpenSlideCabinet":{
        'cam_over_shoulder_left':{
        'pos': [-0.8, 0.2, 1.7],
        'ori': [150, 67, -60]
        },
        'cam_over_shoulder_right':{
        'pos': [-0.8, -0.2, 1.7],
        'ori': [-170, 70, -100]
        },
        'cam_overhead':{
        'pos': [-0.8, 0.0, 1.9],
        'ori': [-180, 55, -90]
        }
    },
    "CloseSlideCabinet":{
        'cam_over_shoulder_left':{
        'pos': [-1.0, 0.2, 1.7],
        'ori': [150, 67, -60]
        },
        'cam_over_shoulder_right':{
        'pos': [-1.0, -0.2, 1.7],
        'ori': [-170, 70, -100]
        },
        'cam_overhead':{
        'pos': [-1.2, 0.0, 1.9],
        'ori': [166, 70, -80]
        }
    },

    "OpenFridge":{
        'cam_over_shoulder_left':{
        'pos': [-1.0, 0.2, 1.7],
        'ori': [150, 67, -60]
        },
        'cam_over_shoulder_right':{
        'pos': [-1.0, -0.2, 1.7],
        'ori': [-170, 70, -100]
        },
        'cam_overhead':{
        'pos': [-1.2, 0.0, 1.9],
        'ori': [166, 70, -80]
        }
    },
    "OpenCabinet":{
        'cam_over_shoulder_left':{
        'pos': [-1.0, 0.2, 1.7],
        'ori': [150, 67, -60]
        },
        'cam_over_shoulder_right':{
        'pos': [-1.0, -0.2, 1.7],
        'ori': [-170, 70, -100]
        },
        'cam_overhead':{
        'pos': [-1.2, 0.0, 1.9],
        'ori': [166, 70, -80]
        }
    },
    "CloseCabinet":{
        'cam_over_shoulder_left':{
        'pos': [-1.0, 0.2, 1.7],
        'ori': [150, 67, -60]
        },
        'cam_over_shoulder_right':{
        'pos': [-1.0, -0.2, 1.7],
        'ori': [-170, 70, -100]
        },
        'cam_overhead':{
        'pos': [-1.2, 0.0, 1.9],
        'ori': [166, 70, -80]
        }
    },

    # "ToiletSeatUp":{
    #     'cam_over_shoulder_left':{
    #         'pos': [0.5, 1.6, 1.4],
    #         'ori': [110, -5, 0]
    #     },
    #     'cam_over_shoulder_right':{
    #         'pos': [-0.1, 1.6, 1.4],
    #         'ori': [110, 5, 0]
    #     },
    #     'cam_overhead':{
    #         'pos': [0.2, 1.6, 1.7],
    #         'ori': [110, 0, 0]
    #     }
    # },

    
    # still missing: open/ close cabinet
    # needs to tune pos


    # "OpenWashingMachin":{},
    # "OpenOven":{},
    # "OpenCabinet": {}, 
    # "OpenDoor":{
    #     'camera_name': 'cam_over_shoulder_left',
    #     'pos': [-1.0, 0.0, 1.2],
    #     'ori': [180, 75, -90]
    # },
    # "CloseDoor":{
    #     'camera_name': 'cam_over_shoulder_left',
    #     'pos': [-1.0, 0.0, 1.2],
    #     'ori': [180, 75, -90]
    # },

}
def task_file_to_task_class(task_file):
  import importlib
  name = task_file.replace('.py', '')
  class_name = ''.join([w[0].upper() + w[1:] for w in name.split('_')])
  mod = importlib.import_module("rlbench.tasks.%s" % name)
  mod = importlib.reload(mod)
  task_class = getattr(mod, class_name)
  return task_class

def hide_robot_temporarily(robot_base_name='Panda', ZOFFFSET=2):
    """
    Moves the robot's base to a designated hiding position (outside the camera view)
    and stores the original position so that it can be restored later.
    
    Args:
        robot_base_name (str): The name of the robot's base object in the scene.
        hiding_position (list or tuple): The [x, y, z] position where the robot should be moved
                                         to (this should be out of the camera's view).
    """
    # Get the handle of the robot's base.
    robot_handle = sim.simGetObjectHandle(robot_base_name)
    if robot_handle is None:
        raise RuntimeError(f"Robot '{robot_base_name}' not found.")
    
    # Store the original position for later restoration.
    original_pos = sim.simGetObjectPosition(robot_handle, -1)  # Relative to world frame (-1)
    hiding_position = original_pos.copy()
    hiding_position[-1] += ZOFFFSET
    
    # Set the robot's position to the hiding position.
    sim.simSetObjectPosition(robot_handle, -1, hiding_position)
    print('Robot moves to %s', hiding_position)


def restore_robot_position(robot_base_name, ZOFFFSET=2):
    """
    Moves the robot's base to a designated hiding position (outside the camera view)
    and stores the original position so that it can be restored later.
    
    Args:
        robot_base_name (str): The name of the robot's base object in the scene.
        hiding_position (list or tuple): The [x, y, z] position where the robot should be moved
                                         to (this should be out of the camera's view).
    """
    # Get the handle of the robot's base.
    robot_handle = sim.simGetObjectHandle(robot_base_name)
    if robot_handle is None:
        raise RuntimeError(f"Robot '{robot_base_name}' not found.")
    
    # Store the original position for later restoration.
    original_pos = sim.simGetObjectPosition(robot_handle, -1)  # Relative to world frame (-1)
    hiding_position = original_pos.copy()
    hiding_position[-1] -= ZOFFFSET
    
    # Set the robot's position to the hiding position.
    sim.simSetObjectPosition(robot_handle, -1, hiding_position)
    print('Robot moves to %s', hiding_position)

def adjust_camera_pose(camera_name, position_offset, orientation_offset):
    """
    Adjusts the pose of the specified camera using its full transformation matrix.
    
    The function applies:
      - A position offset (in world coordinates).
      - A full rotation offset defined as [d_rx, d_ry, d_rz] (in radians) applied
        in the camera's local coordinate system.
    
    The rotation offsets are applied sequentially: first around the camera's local x axis,
    then around its local y axis, then around its local z axis.
    
    Args:
        camera_name (str): The name of the camera in the scene (e.g. 'cam_front').
        position_offset (list or tuple): [dx, dy, dz] added to the camera's world position.
        orientation_offset (list or tuple): [d_rx, d_ry, d_rz] in radians to rotate around
            the camera's local x, y, and z axes, respectively.
    """
    # Get the camera handle.
    cam_handle = sim.simGetObjectHandle(camera_name)
    if cam_handle is None:
        raise RuntimeError(f"Camera {camera_name} not found in the scene.")

    # Get the current transformation matrix (flat list representing a 3x4 matrix) relative to world.
    mat_flat = sim.simGetObjectMatrix(cam_handle, -1)
    # Reshape to 3x4.
    mat_3x4 = np.array(mat_flat).reshape(3, 4)
    # Convert to a homogeneous 4x4 matrix.
    current_mat = np.vstack([mat_3x4, [0, 0, 0, 1]])
    
    # Extract the rotation block and translation.
    current_R = current_mat[:3, :3]
    current_t = current_mat[:3, 3]
    
    # Build rotation matrices for each axis from the provided offsets.
    d_rx, d_ry, d_rz = orientation_offset
    R_x = t3d.axangles.axangle2mat([1, 0, 0], d_rx)  # rotation about local x
    R_y = t3d.axangles.axangle2mat([0, 1, 0], d_ry)  # rotation about local y
    R_z = t3d.axangles.axangle2mat([0, 0, 1], d_rz)  # rotation about local z
    
    # Combine the local rotations.
    # The order here is important: the rotations are applied in sequence.
    # In this example, the local rotations are applied in the order:
    #   1. Rotate around x axis,
    #   2. then around y axis,
    #   3. then around z axis.
    # Adjust the order (e.g., R_x @ R_y @ R_z) if needed.
    R_delta = np.dot(np.dot(R_x, R_y), R_z)
    
    # Apply the local rotation: new_R = current_R * R_delta.
    new_R = np.dot(current_R, R_delta)
    
    # Apply the position offset (world coordinates).
    new_t = current_t + np.array(position_offset)
    
    # Construct the new 4x4 transformation matrix.
    new_mat = np.eye(4)
    new_mat[:3, :3] = new_R
    new_mat[:3, 3] = new_t
    
    new_mat_float = new_mat[:3, :].flatten().tolist()
    
    # Update the camera's transformation.
    sim.simSetObjectMatrix(cam_handle, -1, new_mat_float)
    
    print(f"Camera '{camera_name}' adjusted:")
    print("  New position:", new_t)
    print("  New rotation matrix:\n", new_R)

def set_camera_pose(camera_name: str, position: list, orientation_deg: list):
    """
    Sets the camera pose in RLBench simulation.
    
    The provided Euler angles (alpha, beta, gamma) are assumed to represent
    extrinsic rotations in ZYX order (i.e. a rotation about the world Z axis,
    then Y, then X). This function converts those angles to an equivalent set
    (in radians) that is used by the simulator (assumed to use intrinsic XYZ order).

    :param camera_name: Name of the camera in RLBench.
    :param position: List of [x, y, z] coordinates.
    :param orientation_deg: List of [alpha, beta, gamma] in degrees (ZYX extrinsic).
    """
    camera = VisionSensor(camera_name)
    
    # Convert input degrees to radians.
    orientation_rad = np.radians(orientation_deg)
    
    # Set the camera position and orientation (in radians).
    camera.set_position(position)
    camera.set_orientation(orientation_rad)

def compute_gripper_pose(grasp, approach_dist, quat=True):
    grasp_ = copy.deepcopy(grasp)
    grasp_pos = grasp_.translation
    grasp_ori = grasp_.rotation_matrix # already in world frame
    grasp_depth = grasp_.depth
    grasp_width = grasp_.width
    grasp_height = grasp_.height
    # TODO understand this
    grasp_pos -= grasp_ori[:3, 2] * (grasp_depth - 0.05)
    grasp_pos -= grasp_ori[:3, 2] * (approach_dist)
    print(grasp_pos)

    if quat:
        grasp_ori = Rot.from_matrix(grasp_ori).as_quat()

    return grasp_pos, grasp_ori


def compute_gripper_poses(grasp, T_wc):
    "double check this"
    T_ee_grasp = np.eye(4)
    T_ee_grasp[:3, :3] = Rot.from_euler('y', -90, degrees=True).as_matrix()
    grasp_ee = copy.deepcopy(grasp)
    grasp_ee.rotation_matrix = grasp.rotation_matrix @ T_ee_grasp[:3, :3].T

    pregrasp_pos, pregrasp_ori = compute_gripper_pose(grasp_ee, 0.1, quat=False)
    grasp_pos, grasp_ori = compute_gripper_pose(grasp_ee, 0, quat=False)

    pregrasp_ori_quat = Rot.from_matrix(pregrasp_ori).as_quat()
    grasp_ori_quat = Rot.from_matrix(grasp_ori).as_quat()
    
    pregrasp_gripper_pose = np.concatenate([pregrasp_pos, pregrasp_ori_quat])
    grasp_gripper_pose = np.concatenate([grasp_pos, grasp_ori_quat])

    return pregrasp_gripper_pose, grasp_gripper_pose


def convert_camera_name(camera_name):
    "convert between two different camera naming conventions in rlbench"
    if camera_name == 'cam_front':
        return 'front_camera'
    elif camera_name == 'cam_wrist':
        return 'wrist_camera'
    elif camera_name == 'cam_over_shoulder_left':
        return 'left_shoulder_camera'
    elif camera_name == 'cam_over_shoulder_right':
        return 'right_shoulder_camera'
    elif camera_name == 'cam_overhead':
        return 'overhead_camera'
    else:
        raise ValueError('camera name not recognized,\n available: cam_front, cam_wrist')

from typing import List
import open3d as o3d


def get_robot_pose(name) -> np.ndarray:
    "get robot pose in world frame, in [x, y, z, qx, qy, qz, qw]"
    from pyrep.objects.dummy import Dummy
    robot_name=name
    robot = Dummy(robot_name)
    robot_pose = robot.get_pose()
    return robot_pose
def pose_to_matrix(pose):
    """Converts pose [x, y, z, qx, qy, qz, qw] to a 4x4 transformation matrix."""
    position = np.array(pose[:3])
    orientation = Rot.from_quat(pose[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = orientation.as_matrix()
    matrix[:3, 3] = position
    return matrix

def create_obs_config(camera_names: List[str],
                       camera_resolution: List[int],
                       method_name: str
                    ):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=True,
        depth=True,
        depth_in_meters=True,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL
        )

    # cam_obs = []
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
        # cam_obs.append('%s_rgb' % n)
        # cam_obs.append('%s_pointcloud' % n)
    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config

def vis_pose(pos, ori, size=0.05):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0,0,0])
    T_w_obj = np.eye(4)
    T_w_obj[:3, :3] = ori
    T_w_obj[:3, 3] = pos
    frame.transform(T_w_obj)
    return frame

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


def draw_trajectory(trajectory, ambient_diffuse=[1, 0, 1], maxItemCount=9999):
    """
    Draws a debug line through a sequence of 3D waypoints using PyRep's drawing functions.
    Modified from pyrep arm_configuration_path.py
                            
    Args:
    trajectory (list or np.ndarray): A list (or array) of 3D waypoints.
    ambient_diffuse (list): specifying the color. Default is purple: [1.0, 0.0, 1.0].
    maxItemCount (int): Maximum number of drawing items allowed in the object.
        
    Returns:
        int: The handle to the created drawing object.
    """

    if len(trajectory) <= 0:
        raise RuntimeError("Can't visualise a trajectory with no points.")
    
    line_handle = sim.simAddDrawingObject(sim.sim_drawing_lines,
                                        size=3, 
                                        duplicateTolerance=0, 
                                        parentObjectHandle=-1, 
                                        maxItemCount=maxItemCount,
                                        ambient_diffuse=ambient_diffuse
                                        )
    # instantiate the line handle first
    sim.simAddDrawingObjectItem(line_handle, None)
    # draw the line segments
    prev_point = trajectory[0]
    for point in trajectory[1:]:
        # Concatenate the previous and current points into a flat list:
        # [prev_x, prev_y, prev_z, curr_x, curr_y, curr_z]
        segment = list(prev_point) + list(point)
        sim.simAddDrawingObjectItem(line_handle, segment)
        prev_point = point
    return line_handle



def visualize_affordance_in_mesh(mesh, T_o1c1, T_c2o1, affordance, scale):
    "use neus scale in this function, only for same object"
    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0)

    c1_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
    c1_axis.paint_uniform_color([1, 0, 0])
    c1_axis.transform(T_o1c1)

    # visualize cam2 in o1 frame
    T_o1c2 = np.linalg.inv(T_c2o1)
    c2_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
    c2_axis.transform(T_o1c2)
    
    # affordance in c1 
    affordance_vis = copy.deepcopy(affordance)
    affordance_vis.scale(scale, center=[0, 0, 0])

    # c2 frame
    mesh_c2 = copy.deepcopy(mesh)
    T_c2c1_R = np.eye(4)
    T_c2c1_R[:3, :3] = (T_c2o1 @ T_o1c1)[:3, :3]
    mesh_c2.transform(T_c2c1_R)
    
    affordance_vis_c2 = copy.deepcopy(affordance_vis)
    affordance_vis_c2.transform(T_c2c1_R)

    o3d.visualization.draw_geometries([mesh, mesh_c2, affordance_vis, affordance_vis_c2, world_axis, c1_axis, c2_axis]) 

def visualize_pointcloud(demo_pcd, target_pcd, T_o1c1, T_c2o1):
    # visualize to demo scale
    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)

    c1_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    c1_axis.paint_uniform_color([1, 0, 0]) # c1 in red
    c1_axis.transform(T_o1c1)

    T_o1c2 = np.linalg.inv(T_c2o1)
    c2_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    c2_axis.transform(T_o1c2)
    c2_axis.paint_uniform_color([0, 1, 0]) # c2 in green

    # o3d.visualization.draw_geometries([target_pcd, demo_pcd, c1_axis, c2_axis, world_axis]) 

    # # visualize demo pcd 
    demo_pcd_vis = copy.deepcopy(demo_pcd)
    target_pcd_vis = copy.deepcopy(target_pcd)
    # scale_factor = np.linalg.norm(np.asarray(demo_pcd_vis.points)) / np.linalg.norm(np.asarray(target_pcd_vis.points))
    # print(f"scale factor: {scale_factor}")
    # target_pcd_vis.scale(scale_factor, center=[0, 0, 0])

    # o3d.visualization.draw_geometries([target_pcd_vis, demo_pcd_vis, c1_axis, c2_axis, world_axis]) 

    demo_pcd_vis.transform(T_o1c1)
    target_pcd_vis.transform(T_o1c2)

    o3d.visualization.draw_geometries([target_pcd_vis, demo_pcd_vis, c1_axis, c2_axis, world_axis]) 

def visualize_affordance_with_scene(target_pcd, affordance_c2, corres_3d_c2):
    "visualize transferred affordance in target frame, scale = render scale or real scale"
    sphere_c2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sphere_c2.paint_uniform_color([1, 0, 0])
    sphere_c2.translate(corres_3d_c2.flatten())

    affordance_vis = visualize_3d_trajectory(affordance_c2, size=0.003, cmap_name="plasma", invert=False)

    o3d.visualization.draw_geometries([target_pcd, sphere_c2] + affordance_vis)


def visualize_affordance_in_pointcloud(demo_pcd, target_pcd, T_o1c1, T_c2o1, affordance_c2, corres_3d):
    "visualize transferred affordance in world frame, scale = render scale or real scale"
    # visualize some axises
    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)

    c1_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # c1_axis.paint_uniform_color([1, 0, 0]) # c1 in red
    c1_axis.transform(T_o1c1)

    T_o1c2 = np.linalg.inv(T_c2o1)
    c2_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    c2_axis.transform(T_o1c2)

    # visualize demo pcd 
    demo_pcd_vis = copy.deepcopy(demo_pcd)
    demo_pcd_vis.transform(T_o1c1)
    target_pcd_vis = copy.deepcopy(target_pcd)
    target_pcd_vis.transform(T_o1c2)
  
    # Rescale target_pts_c2 to similar scale to pcd_demo

    corres_c2 = corres_3d @ T_o1c2[:3, :3].T + T_o1c2[:3, 3]
    sphere_c2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    sphere_c2.paint_uniform_color([1, 0, 0])
    sphere_c2.translate(corres_c2.flatten())

    affordance_world = copy.deepcopy(affordance_c2)
    affordance_world = affordance_world @ T_o1c2[:3, :3].T + T_o1c2[:3, 3]
    affordance_vis = visualize_3d_trajectory(affordance_world, size=0.02, cmap_name="plasma", invert=False)

    ipdb.set_trace()
    o3d.visualization.draw_geometries([target_pcd_vis, c2_axis,  demo_pcd_vis, c1_axis ] + affordance_vis)


def backproject(depth, intrinsics, instance_mask, NOCS_convention=True):
    """backproject depth image to 3d points
    Args:
        depth: [h, w]
        intrinsics: [3, 3]
        instance_mask: [h, w]
    return: pts: [num_pixel, 3], idxs: [2, num_pixel]
    """
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

def get_pcd_with_color(obs, camera_name):
    "get point cloud backprojected from depth and color"
    key_name = convert_camera_name(camera_name)
    color = getattr(obs, f'{key_name[:-7]}_rgb')
    depth = getattr(obs, f'{key_name[:-7]}_depth')
    intrinsics = obs.misc[key_name+'_intrinsics']
    intrinsics[0, 0] = np.abs(intrinsics[0, 0])
    intrinsics[1, 1] = np.abs(intrinsics[1, 1])
    pts, colors = backproject_with_color(depth, color, intrinsics, depth<10, False)
    pcd_w_color = visualize_points(pts, colors)
    return pcd_w_color

def get_T_world_cam_gl(obs, camera):
    "get T_world_cam in OpenGL convention from observation"
    cam_key = convert_camera_name(camera)
    T_world_cam = obs.misc[cam_key+'_extrinsics'].copy()

    # converting coppliaSIM camera convention to OpenGL
    R_z_180 = np.array(
        [[ -1,  0,  0],
        [  0, -1,  0],
        [  0,  0,  1]])
    T_world_cam[:3, :3] = T_world_cam[:3, :3] @ R_z_180
    return T_world_cam


CAMERA_POSES_HZ = {
    "PickUpBottle":{
        'cam_overhead':{
        'pos': [1.35, 0, 1.13],
        'ori': [-180, -75, 90],
        },
        'cam_over_shoulder_left':{
            'pos': [1.35, -0.3, 1.13],
            'ori': [-110, -70, 160]
        },
        'cam_over_shoulder_right':{
            'pos': [1.35, 0.3, 1.13],
            'ori': [135, -75, 45]
        },
    },
    "PickUpCup":{
        'cam_overhead':{
        'pos': [1.35, 0, 1.13],
        'ori': [-180, -75, 90],
        },
        'cam_over_shoulder_left':{
            'pos': [1.35, -0.3, 1.13],
            'ori': [-110, -70, 160]
        },
        'cam_over_shoulder_right':{
            'pos': [1.35, 0.3, 1.13],
            'ori': [135, -75, 45]
        },
        },
    "PickUpMug":{
        'cam_overhead':{
        'pos': [1.35, 0, 1.13],
        'ori': [-180, -75, 90],
        },
        'cam_over_shoulder_left':{
            'pos': [1.35, -0.3, 1.13],
            'ori': [-110, -70, 160]
        },
        'cam_over_shoulder_right':{
            'pos': [1.35, 0.3, 1.13],
            'ori': [135, -75, 45]
        },
        },
    "PickUpBowl":{
        'cam_overhead':{
        'pos': [1.35, 0, 1.13],
        'ori': [-180, -75, 90],
        },
        'cam_over_shoulder_left':{
            'pos': [1.35, -0.3, 1.13],
            'ori': [-110, -70, 160]
        },
        'cam_over_shoulder_right':{
            'pos': [1.35, 0.3, 1.13],
            'ori': [135, -75, 45]
        },
        },


    "OpenDrawer":{
        'cam_over_shoulder_left':{
            'pos': [0.5, 1.6, 1.4],
            'ori': [105, -5, 0]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.1, 1.6, 1.4],
            'ori': [105, 5, 0]
        },
        'cam_overhead':{
            'pos': [0.2, 1.6, 1.7],
            'ori': [110, 0, 0]
        }
    },
    "CloseDrawer":{
        'cam_over_shoulder_left':{
            'pos': [0.5, 1.6, 1.4],
            'ori': [110, -5, 0]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.1, 1.6, 1.4],
            'ori': [110, 5, 0]
        },
        'cam_overhead':{
            'pos': [0.2, 1.6, 1.7],
            'ori': [110, 0, 0]
        }
    },

    "OpenMicrowave":{
        'cam_over_shoulder_left':{
            'pos': [0.8, -1.5, 1.2],
            'ori': [-100, -14, 180]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.2, -1.5, 1.2],
            'ori': [-100, 14, 180]
        },
        'cam_overhead':{
            'pos': [0.2, -1.6, 1.7],
            'ori': [-115, 0, -180]
        }
    },
    "CloseMicrowave":{
        'cam_over_shoulder_left':{
            'pos': [0.8, -1.5, 1.2],
            'ori': [-100, -14, 180]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.2, -1.5, 1.2],
            'ori': [-100, 14, 180]
        },
        'cam_overhead':{
            'pos': [0.2, -1.6, 1.7],
            'ori': [-115, 0, -180]
        }
    },

    "OpenDishwasher":{
        'cam_over_shoulder_left':{
            'pos': [0.8, -1.5, 1.2],
            'ori': [-94, -19, -180]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.2, -1.5, 1.2],
            'ori': [-95, 18, 180]
        },
        'cam_overhead':{
            'pos': [0.2, -1.6, 1.7],
            'ori': [-115, 0, -180]
        }
    },
    "CloseLaptop":{
        'cam_over_shoulder_left':{
            'pos': [0.8, -1.5, 1.2],
            'ori': [-100, -14, 180]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.2, -1.5, 1.2],
            'ori': [-100, 14, 180]
        },
        'cam_overhead':{
            'pos': [0.2, -1.6, 1.7],
            'ori': [-115, 0, -180]
        }
    },

    "DownToiletSeat":{
        'cam_over_shoulder_left':{
            'pos': [0.5, 1.6, 1.4],
            'ori': [105, -5, 0]
        },
        'cam_over_shoulder_right':{
            'pos': [-0.1, 1.6, 1.4],
            'ori': [105, 5, 0]
        },
        'cam_overhead':{
            'pos': [0.2, 1.6, 1.7],
            'ori': [110, 0, 0]
        }
    },
    "OpenSlideCabinet":{
        'cam_over_shoulder_left':{
        'pos': [-0.8, 0.2, 1.7],
        'ori': [150, 67, -60]
        },
        'cam_over_shoulder_right':{
        'pos': [-0.8, -0.2, 1.7],
        'ori': [-170, 70, -100]
        },
        'cam_overhead':{
        'pos': [-0.8, 0.0, 1.9],
        'ori': [-180, 55, -90]
        }
    },
    "CloseSlideCabinet":{
        'cam_over_shoulder_left':{
        'pos': [-1.0, 0.2, 1.7],
        'ori': [150, 74, -60]
        },
        'cam_over_shoulder_right':{
        'pos': [-1.0, -0.2, 1.7],
        'ori': [-170, 74, -100]
        },
        'cam_overhead':{
        'pos': [-1.2, 0.0, 1.9],
        'ori': [166, 70, -80]
        }
    },

    "OpenFridge":{
        'cam_over_shoulder_left':{
        'pos': [-1.0, 0.2, 1.7],
        'ori': [150, 67, -60]
        },
        'cam_over_shoulder_right':{
        'pos': [-1.0, -0.2, 1.7],
        'ori': [-170, 70, -100]
        },
        'cam_overhead':{
        'pos': [-1.2, 0.0, 1.9],
        'ori': [166, 70, -80]
        }
    },
    "OpenCabinet":{
        'cam_over_shoulder_left':{
        'pos': [-1.0, 0.2, 1.7],
        'ori': [150, 67, -60]
        },
        'cam_over_shoulder_right':{
        'pos': [-1.0, -0.2, 1.7],
        'ori': [-170, 70, -100]
        },
        'cam_overhead':{
        'pos': [-1.2, 0.0, 1.9],
        'ori': [166, 70, -80]
        }
    },
    "CloseCabinet":{
        'cam_over_shoulder_left':{
        'pos': [-1.0, 0.2, 1.7],
        'ori': [150, 67, -60]
        },
        'cam_over_shoulder_right':{
        'pos': [-1.0, -0.2, 1.7],
        'ori': [-170, 70, -100]
        },
        'cam_overhead':{
        'pos': [-1.2, 0.0, 1.9],
        'ori': [166, 70, -80]
        }
    },


}
    

def change_robot_renderability(robot_names=['Panda'], task=None):
    """
    Temporarily hide the robot to capture clean observation without robot occlusion.
    Uses PyRep's built-in get_visuals() method to find and hide robot visual objects.
    Returns the modified observation and a function to restore the robot.
    """
    visual_objects = []
    
    # Store original renderability states
    original_states = {}
    
    # Get robot from task if available
    if task is not None and hasattr(task, '_scene') and hasattr(task._scene, 'robot'):
        robot = task._scene.robot
        print(f"Processing robot from task scene")
        
        try:
            # Get visual objects from robot arm
            if hasattr(robot, 'arm') and hasattr(robot.arm, 'get_visuals'):
                arm_visuals = robot.arm.get_visuals()
                visual_objects.extend(arm_visuals)
                print(f"Found {len(arm_visuals)} arm visual objects")
            
            # Get visual objects from robot gripper
            if hasattr(robot, 'gripper') and hasattr(robot.gripper, 'get_visuals'):
                gripper_visuals = robot.gripper.get_visuals()
                visual_objects.extend(gripper_visuals)
                print(f"Found {len(gripper_visuals)} gripper visual objects")
                
        except Exception as e:
            print(f"Error getting robot visuals: {e}")
    
    # Fallback: try to find robot objects by name if task approach fails
    if len(visual_objects) == 0:
        print("Fallback: searching for robot objects by name")
        for name in robot_names:
            try:
                obj = Object.get_object(name)
                if obj is not None:
                    print(f"Found robot object: {name}")
                    # Try to get children and find visual objects
                    if hasattr(obj, 'get_children'):
                        children = obj.get_children()
                        for child in children:
                            if hasattr(child, 'is_renderable'):
                                visual_objects.append(child)
                                print(f"Added child object: {child.get_name() if hasattr(child, 'get_name') else 'unnamed'}")
            except Exception as e:
                print(f"Error getting robot object {name}: {e}")
                continue
    
    # Hide all found visual objects
    for obj in visual_objects:
        try:
            obj_name = obj.get_name() if hasattr(obj, 'get_name') else 'unnamed'
            original_states[obj_name] = {
                'renderable': obj.is_renderable()
            }
            obj.set_renderable(False)
            print(f"Hidden visual object: {obj_name}")
        except Exception as e:
            print(f"Error hiding object: {e}")
    
    print(f"Found and hidden {len(visual_objects)} visual objects")
    
    def restore_robot():
        """Restore robot visibility and renderability"""
        restored_count = 0
        for obj_name, states in original_states.items():
            try:
                obj = Object.get_object(obj_name)
                obj.set_renderable(states['renderable'])
                restored_count += 1
            except Exception as e:
                print(f"Error restoring object: {e}")
        print(f"Restored {restored_count} visual objects")
    
    return visual_objects, restore_robot

def get_clean_point_cloud(robot_names=['Panda'], obs=None, camera_name='cam_overhead', task=None):
    """
    Get clean point cloud without robot occlusion by temporarily hiding the robot.
    
    Args:
        robot_names: List of robot object names to hide
        obs: Observation object
        camera_name: Name of the camera to use
        task: Task object to refresh observation (required for getting new point cloud)
    
    Returns:
        clean_point_cloud: (N, 3) array of 3D points without robot occlusion
    """
    # Temporarily hide robot
    robot_objects, restore_robot = change_robot_renderability(robot_names=robot_names, task=task)
    
    try:
        # Refresh observation to get new point cloud without robot
        if task is not None:
            obs = task.get_observation()
            print("Refreshed observation after hiding robot")
        
        # convert camera name to attribute name
        if 'shoulder_left' in camera_name:
            obs_arrtibute_name = 'left_shoulder'
        elif 'shoulder_right' in camera_name:
            obs_arrtibute_name = 'right_shoulder'
        elif 'overhead' in camera_name:
            obs_arrtibute_name = 'overhead'
        else:
            raise ValueError('Invalid camera name')
        
        # Get the point cloud from the specified camera
        if hasattr(obs, f'{obs_arrtibute_name}_point_cloud'):
            point_cloud = getattr(obs, f'{obs_arrtibute_name}_point_cloud')
        else:
            print(f"No point cloud found in observation for {obs_arrtibute_name}, double check camera name")
            point_cloud = None
        
        # Reshape and return
        if point_cloud is not None:
            point_cloud = point_cloud.reshape(-1, 3)
            print(f"Captured clean point cloud from {obs_arrtibute_name}: {len(point_cloud)} points")
            return point_cloud
        else:
            return np.array([])
            
    finally:
        # Always restore robot visibility
        restore_robot()