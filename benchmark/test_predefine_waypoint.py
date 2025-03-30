import numpy as np
import ipdb
import cv2
import matplotlib.pyplot as plt # for debugging
import open3d as o3d
from omegaconf import OmegaConf

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.tasks import PickUpCup

# from affordance.affordance_transfer import AffordanceTransfer
from benchmark.helpers import visualize_points,\
    visualize_3d_trajectory, backproject_with_color,\
        preprocess_target_data
from benchmark.sim_utils import visualize_affordance_with_scene, visualize_affordance_in_pointcloud
from pyrep.backend import sim
# from transforms3d.quaternions import mat2quat, quat2mat # don't fuck using this
from thirdparty.graspNet.gsnet_wrapper import GSNetWrapper
import copy
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

def save_scene(pcd, tgt_pt, T_cw, home_pose, save_dir='./grasp_test.npz'):
    "save points ,target grasp point and T_cw for simple grasping test"
    data = {}
    data['pcd'] = pcd
    data['tgt_pt'] = tgt_pt
    data['T_cw'] = T_cw
    data['home_pose'] = home_pose
    np.savez(save_dir, **data)

def interpolate_trajectory(waypoints, num_points=5):
    interpolated = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        for t in np.linspace(0, 1, num_points):
            interpolated.append((1 - t) * start + t * end)
    return np.array(interpolated)

def transform_trajectory(affordance_cam, T_world_cam):
    affordance_trajectory = affordance_cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    # print(f'transformed affordance trajectory: {affordance_trajectory}')
    return affordance_trajectory

def visualize_affordance(T_world_cam, affordance_traj_world):
    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
    desk = o3d.geometry.TriangleMesh.create_box(width=1.2, height=1.2, depth=0.1)
    desk.translate([-0.6, -0.6, 0.7])
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    camera.transform(T_world_cam)
    camera_unchanged = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    T_world_cam_unchanged = obs.misc['front_camera_extrinsics']
    camera_unchanged.transform(T_world_cam_unchanged)
    affordance_vis = visualize_3d_trajectory(affordance_traj_world, size=0.02, cmap_name="plasma", invert=False)
    o3d.visualization.draw_geometries([world_axis, desk, camera_unchanged] + affordance_vis)

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

def vis_pose(pos, ori, size=0.05):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0,0,0])
    T_w_obj = np.eye(4)
    T_w_obj[:3, :3] = ori
    T_w_obj[:3, 3] = pos
    frame.transform(T_w_obj)
    return frame

def compute_gripper_pose(grasp, approach_dist, quat=True):
    grasp_ = copy.deepcopy(grasp)
    grasp_pos = grasp_.translation
    grasp_ori = grasp_.rotation_matrix # already in world frame
    grasp_depth = grasp_.depth
    grasp_width = grasp_.width
    grasp_height = grasp_.height
    # !
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

class Agent(object):
    def __init__(self, config_fp=None):
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
        self.descending_traj = None
        self.descending_idx = 0
        self.n_descend_steps = 5
        self.gripper_states = ['home', 'home', 'pregrasp', 'grasp', 'attach', 'postgrasp']
        

        if config_fp is not None:
            print(f'affordance transfer using config from: {config_fp}')
            self.config = OmegaConf.load(config_fp)
            self.affordance_transfer = AffordanceTransfer(self.config)
        else:
            print('No config file provided, affordance transfer not initialized')
        
        self.gsNet = GSNetWrapper(self.config)

    # def predict_action(self, batch):
    #     return np.random.uniform(size=(len(batch), 7))

    # def behaviour_cloning_loss(self, ground_truth_actions, predicted_actions):
    #     return 1
    def convert_camera_name(self, camera_name):
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

    def run_transfer(self, obs, cam_name='cam_front', tgt_obj_prompt='', DEBUG=False):
        """
        Run affordance transfer on the target object
        obs: observation from the environment
        tgt_obj_prompt: target object name, for grounded SAM prompt, e.g. 'cup.'
        DEBUG: visualize the affordance trajectory
        Return:
        corres_3d_c2: correspondences between affordance trajectory and object
        affordance_cam: affordance trajectory in camera frame
        affordance_trajectory: affordance trajectory in world frame
        """
        self.init_ori = obs.gripper_pose[3:7]
        self._get_images(obs, SAVE_IMAGES=False)
        self._get_camera_intrinsics_and_pose(obs, camera_name=cam_name)
        key_name = self.convert_camera_name(cam_name)
        tgt_rgb = self.rgbs[key_name[:-7]]
        tgt_depth = self.depths[key_name[:-7]]
        tgt_data = preprocess_target_data(tgt_rgb, tgt_depth, self.cam_K, 'kinect', obj_name=tgt_obj_prompt)
        self.T_co_optimized, corres_3d_c2, self.affordance_cam = self.affordance_transfer.run(tgt_data)

        if DEBUG:
            tgt_pts, colors = backproject_with_color(tgt_data['depth'], tgt_data['rgb'],
                                             tgt_data['camera_intrinsic'], tgt_data['mask'],
                                             NOCS_convention=False)
            tgt_pcd = visualize_points(tgt_pts, colors)
            # visualize backprojected object and affordance trajectory
            visualize_affordance_with_scene(tgt_pcd, self.affordance_cam, corres_3d_c2)

        #! convert rlbench camera convention to opengl convention, understand this!!!
        R_z_180 = np.array([[ -1,  0,  0],
                            [  0, -1,  0],
                            [  0,  0,  1]])
        # transfer the affordance trajectory to world frame for robot to execute
        self.trajectory_idx = 0
        self.T_world_cam[:3, :3] = self.T_world_cam[:3,:3] @ R_z_180
        self.affordance_trajectory = transform_trajectory(self.affordance_cam, self.T_world_cam)
        return corres_3d_c2, self.affordance_cam, self.affordance_trajectory

    def plan_gripper_trajectory(self, obs, affordance_traj_world):
        "plan smooth gripper trajectory based on affordance trajectory"

        current_gripper_pose =obs.gripper_pose[:7]
        # make it deeper
        current_gripper_pose[2]-=0.01

        offset = affordance_traj_world[0] - current_gripper_pose[:3]
        affordance_traj_world -= offset
        affordance_traj_world_downsampled = affordance_traj_world[::10]
        affordance_traj_world_downsampled = np.concatenate([affordance_traj_world[0].reshape(-1,3), affordance_traj_world_downsampled], axis=0)
        post_gripper_ori = current_gripper_pose[3:7]

        post_gripper_poses = np.concatenate((affordance_traj_world_downsampled, np.repeat(post_gripper_ori.reshape(-1, 4), affordance_traj_world_downsampled.shape[0], axis=0)), axis=1)
        post_gripper_poses = np.concatenate([post_gripper_poses, np.zeros((affordance_traj_world_downsampled.shape[0], 1))], axis=1)
        self.actions =  post_gripper_poses

    def act(self, obs, idx):
        """
        Given the current observation and a list/array of 3D waypoints (affordance_traj_world),
        return an action that includes a goal position (3D), current gripper orientation (4D), and
        a gripper command (1D). The action sequence first executes a pregrasp trajectory:
           1. Move above the target waypoint.
           2. Slowly descend to the target waypoint and close the gripper.
        Then, the postgrasp trajectory is executed.
        """
        action = self.actions[idx]
        return action
    
    def act_sparse(self, obs):
        if self.trajectory_idx >= len(self.actions):
            return self.actions[-1]

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

    def is_collision_free(self, start, end):
        """
        Stub function for collision checking between start and end.
        Returns True if the linear path is collision free.
        In practice, replace with your collision checking routine.
        """
        # For now, always assume the path is collision free.
        return True

    def generate_descending_traj(self, current_pos, target_pose):
        """
        Generates a descending trajectory from current_pos to target_pose with n_descend_steps.
        A simple linear interpolation is used. If collision checking fails for a segment,
        one might adjust the step size or re-plan (this is a placeholder).
        """
        # Generate linearly interpolated points (each is a 3D point)
        traj = np.linspace(current_pos, target_pose, self.n_descend_steps)
        
        # (Optional) Check each segment for collisions:
        for i in range(len(traj) - 1):
            if not self.is_collision_free(traj[i], traj[i+1]):
                # If a segment is not collision-free, you can modify this behavior.
                print(f"Collision detected between step {i} and {i+1}. Adjusting trajectory.")
                # Here, for simplicity, we return the trajectory up to the collision.
                return traj[:i+1]
        return traj

    def _get_images(self, obs, SAVE_IMAGES=False):
        "camera_name: cam_front cam_left_shoulder etc"
        camera_keys = ['front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist']
        self.rgbs = {key: getattr(obs, f'{key}_rgb') for key in camera_keys}
        self.depths = {key: getattr(obs, f'{key}_depth') for key in camera_keys}

        if SAVE_IMAGES:
            for key in camera_keys:
                cv2.imwrite(f'./outputs/rlbench/{key}_rgb.png', self.rgbs[key])
                cv2.imwrite(f'./outputs/rlbench/{key}_depth.png', (self.depths[key] * 1000).astype(np.float32))
            print(f'saved images to ./outputs/rlbench/')
    
    def _get_camera_intrinsics_and_pose(self, obs, camera_name='cam_front'):
        cam_key = self.convert_camera_name(camera_name)
        self.cam_K = obs.misc[cam_key+'_intrinsics']

        # fix the negative focal length
        self.cam_K[0, 0] = np.abs(self.cam_K[0,0])
        self.cam_K[1, 1] = np.abs(self.cam_K[1,1])
        self.T_world_cam = obs.misc[cam_key+'_extrinsics'].copy()

# setup environment
# TODO: add point clouds to the observation
front_camera = CameraConfig(image_size=(512, 512), depth_in_meters=True) 
wrist_camera = CameraConfig(image_size=(512, 512), depth_in_meters=True)
obs_config = ObservationConfig(front_camera=front_camera, wrist_camera=wrist_camera)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, ), 
        gripper_action_mode=Discrete()
        ),
    obs_config=obs_config,
    headless=False)
env.launch()

task = env.get_task(PickUpCup)
obs = None

# set up affordance agent
config_fp = './affordance/task_conf/cup0.yaml'
affordance_agent = Agent(config_fp)
episode_length = 100
num_try = 3
total_length =  episode_length * num_try
# task.move_to_grasp()

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
            corres_3d_c2, affordance_cam, affordance_traj_world = \
                affordance_agent.run_transfer(obs, cam_name='cam_front', tgt_obj_prompt=obj_name, DEBUG=False)
            
            task.move_to_grasp()
            obs=task.get_observation()
            affordance_agent.plan_gripper_trajectory(obs, affordance_traj_world)

        action = affordance_agent.act_sparse(obs)
        obs, reward, terminate = task.step(action)
    
        if terminate:
            if not reward:
                print('All fails condition are met, task terminated')
            else:
                print('Task Success!')
            break

print('Done')
env.shutdown()
