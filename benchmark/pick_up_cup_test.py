import numpy as np
import ipdb
import cv2
import matplotlib.pyplot as plt # for debugging
import open3d as o3d
from omegaconf import OmegaConf

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.tasks import PickUpCup

from affordance.affordance_transfer import AffordanceTransfer
from affordance.helpers import preprocess_target_data
from affordance.affordance_transfer import AffordanceTransfer
from affordance.helpers import hash_filename, read_image, visualize_points,\
resize_img, load_optimization_result, get_configs, interpolate_trajectory,\
visualize_3d_trajectory, pick_points_in_viewer, draw_line, backproject_with_color
from simulation.sim_utils import visualize_affordance_with_scene, visualize_affordance_in_pointcloud
from NeuS.models.utils import backproject
from pyrep.backend import sim
from transforms3d.quaternions import mat2quat, quat2mat


def interpolate_trajectory(waypoints, num_points=5):
    interpolated = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        for t in np.linspace(0, 1, num_points):
            interpolated.append((1 - t) * start + t * end)
    return np.array(interpolated)

def transform_trajectory(affordance_c2, T_world_cam):
    affordance_trajectory = affordance_c2 @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
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

        if config_fp is not None:
            print(f'affordance transfer using config from: {config_fp}')
            self.config = OmegaConf.load(config_fp)
            self.affordance_transfer = AffordanceTransfer(self.config)
        else:
            print('No config file provided, affordance transfer not initialized')

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
        affordance_c2: affordance trajectory in camera frame
        affordance_trajectory: affordance trajectory in world frame
        """
        self.init_ori = obs.gripper_pose[3:7]
        self._get_images(obs, SAVE_IMAGES=False)
        self._get_camera_intrinsics_and_pose(obs, camera_name=cam_name)
        key_name = self.convert_camera_name(cam_name)
        tgt_rgb = self.rgbs[key_name[:-7]]
        tgt_depth = self.depths[key_name[:-7]]
        tgt_data = preprocess_target_data(tgt_rgb, tgt_depth, self.cam_K, 'kinect', obj_name=tgt_obj_prompt)
        self.T_co_optimized, corres_3d_c2, affordance_c2 = self.affordance_transfer.run(tgt_data)

        if DEBUG:
            tgt_pts, colors = backproject_with_color(tgt_data['depth'], tgt_data['rgb'],
                                             tgt_data['camera_intrinsic'], tgt_data['mask'],
                                             NOCS_convention=False)
            tgt_pcd = visualize_points(tgt_pts, colors)
            # visualize backprojected object and affordance trajectory
            visualize_affordance_with_scene(tgt_pcd, affordance_c2, corres_3d_c2)

        #! convert rlbench camera convention to opengl convention
        R_z_180 = np.array([[ -1,  0,  0],
                            [  0, -1,  0],
                            [  0,  0,  1]])
        # transfer the affordance trajectory to world frame for robot to execute
        self.trajectory_idx = 0
        self.T_world_cam[:3, :3] = self.T_world_cam[:3,:3] @ R_z_180
        self.affordance_trajectory = transform_trajectory(affordance_c2, self.T_world_cam)
        return corres_3d_c2, affordance_c2, self.affordance_trajectory

    def dummy_act(self, obs, ori=None, affordance_trajectory=None):
        if affordance_trajectory is not None:
            self.affordance_trajectory = affordance_trajectory
        else:
            print('Using dummy affordance trajectory')
            self.affordance_trajectory = interpolate_trajectory(np.array([
                [3.81813741e-01, -8.13834369e-03,  1.2455574e+00],
                [5.1813741e-01, -8.13834369e-03,  1.1455574e+00],
                [3.81813741e-01, -8.13834369e-03,  1.0455574e+00],
                [5.1813741e-01, -8.13834369e-03,  0.95574e+00],
            ]))

        current_pos = obs.gripper_pose[:3]
        current_ori = obs.gripper_pose[3:7]
        distance = np.linalg.norm(
            current_pos - self.affordance_trajectory[self.trajectory_idx])
        if distance < 0.05: # index ++ if reached
            self.trajectory_idx += 1
            if self.trajectory_idx >= len(self.affordance_trajectory):
                self.trajectory_idx = len(self.affordance_trajectory) - 1
            print(f'move to position: {self.trajectory_idx}')
        goal_pos = self.affordance_trajectory[self.trajectory_idx]

        if ori is None:
            goal_ori = current_ori
        else:
            goal_ori_matrix = np.dot(ori, quat2mat(current_ori))
            goal_ori = mat2quat(goal_ori_matrix)
        gripper = [1.0]  # Always open
        return np.concatenate([goal_pos, goal_ori, gripper], axis=-1)

    def act(self, obs, affordance_traj_world):
        """
        Given the current observation and a list/array of 3D waypoints (affordance_traj_world),
        return an action that includes a goal position (3D), current gripper orientation (4D), and
        a gripper command (1D). The action sequence first executes a pregrasp trajectory:
           1. Move above the target waypoint.
           2. Slowly descend to the target waypoint and close the gripper.
        Then, the postgrasp trajectory is executed.
        """
        current_pos = obs.gripper_pose[:3]
        current_ori = obs.gripper_pose[3:7]
        
        # compute grasp orientation
        ori = o3d.geometry.get_rotation_matrix_from_axis_angle([ -np.pi / 2,0, 0])
        grasp_ori = mat2quat(np.dot(ori, quat2mat(self.init_ori)))
        
        # Phase 1: PREGRASP -- move to a point above the target (first waypoint)
        if self.phase == 'pregrasp':
            # Use the first waypoint in the trajectory as the target grasp point.
            target_wp = affordance_traj_world[0].copy()
            # Assume the z-axis is vertical; move up by the offset.
            pregrasp_wp = target_wp.copy()
            pregrasp_wp[2] += self.pregrasp_offset

            # Check if we are at the pregrasp position
            dist_to_pregrasp = np.linalg.norm(current_pos - pregrasp_wp)
            if dist_to_pregrasp > self.distance_threshold:
                goal_pos = pregrasp_wp
            else:
                # Once above the target, switch to the descend phase.
                print("Reached pregrasp position; switching to descend phase.")
                self.phase = 'descend'
                goal_pos = target_wp  # begin descending

            # Keep the gripper open during pregrasp
            gripper_command = [1.0]
            goal_ori = grasp_ori

        # --- Phase 2: DESCEND ---
        elif self.phase == 'descend':
            # If we haven't generated the descending trajectory yet, do it now.
            if self.descending_traj is None:
                target_wp = affordance_traj_world[0]
                self.descending_traj = self.generate_descending_traj(current_pos, target_wp)
                self.descending_idx = 0

            # Follow the descending trajectory step-by-step.
            goal_pos = self.descending_traj[self.descending_idx]
            distance_to_goal = np.linalg.norm(current_pos - goal_pos)
            
            if distance_to_goal < self.distance_threshold:
                # Move to the next intermediate waypoint if available.
                self.descending_idx += 1
                if self.descending_idx >= len(self.descending_traj):
                    # Final waypoint reached: switch phase.
                    print("Reached target grasp point; closing gripper and switching to postgrasp phase.")
                    self.gripper_closed = True
                    self.phase = 'postgrasp'
                    # Start postgrasp trajectory from index 1 (if available).
                    self.trajectory_idx = 1 if len(affordance_traj_world) > 1 else 0
                    goal_pos = affordance_traj_world[0]  # ensure staying at grasp point momentarily
                else:
                    goal_pos = self.descending_traj[self.descending_idx]
            else:
                # keep goal pose as it is 
                pass
            
            gripper_command = [1.0] # open
            goal_ori = grasp_ori

        # Phase 3: POSTGRASP
        elif self.phase == 'postgrasp':
            if self.trajectory_idx == 1:
                # visualize the affordance trajectory
                self.line_handle = draw_trajectory(affordance_traj_world)

            # Ensure that the trajectory index is within bounds.
            if self.trajectory_idx >= len(affordance_traj_world):
                self.trajectory_idx = len(affordance_traj_world) - 1

            affordance_wp = affordance_traj_world[self.trajectory_idx]
            distance = np.linalg.norm(current_pos - affordance_wp)

            if distance > self.distance_threshold:
                goal_pos = affordance_wp
            else:
                # Advance to the next waypoint.
                self.trajectory_idx += 1
                if self.trajectory_idx >= len(affordance_traj_world):
                    self.trajectory_idx = len(affordance_traj_world) - 1
                goal_pos = affordance_traj_world[self.trajectory_idx]

            # Keep the gripper closed during postgrasp.
            goal_ori = grasp_ori
            gripper_command = [0.0]
        else:  # do nothing
            goal_pos = current_pos
            goal_ori = current_ori
            gripper_command = [1.0]

        action = np.concatenate([goal_pos, goal_ori, gripper_command], axis=-1)
        return action
    
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
front_camera = CameraConfig(image_size=(512, 512), depth_in_meters=True) #! in meters
obs_config = ObservationConfig(front_camera=front_camera)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, ), gripper_action_mode=Discrete()),
    obs_config=obs_config,
    headless=False)
env.launch()

task = env.get_task(PickUpCup)
obs = None

# set up affordance agent
config_fp = './affordance/task_conf/cup1_to_cup_hz.yaml'
affordance_agent = Agent(config_fp)
episode_length = 100
num_try = 3
total_length =  episode_length * num_try

for i in range(total_length):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        affordance_agent.reset()
        obs = task.get_observation()
        print('<===== Task description: ====>\n', descriptions)
        obj_name = descriptions[0][11:]

        # run transfer based on the observation
        corres_3d_c2, affordance_c2, affordance_traj_world = \
            affordance_agent.run_transfer(obs, cam_name='cam_front', tgt_obj_prompt=obj_name, DEBUG=True)
        # ipdb.set_trace() # check the affordance trajectory in world frame
        # visualize_affordance(affordance_agent.T_world_cam, affordance_traj_world)
        # ipdb.set_trace() # ready for action

    action = affordance_agent.act(obs, affordance_traj_world)
    obs, reward, terminate = task.step(action)
    
    if terminate:
        if not reward:
            print('All fails condition are met, task terminated')
        else:
            print('Task Success!')
    

# # dummy action
# for i in range(total_length):
#     if i % episode_length == 0:
#         print('Reset Episode')
#         descriptions, obs = task.reset()
#         affordance_agent.trajectory_idx = 0 # re
#         # affordance_traj_rob = affordance_agent.run_transfer(obs, T_robot_cam=T_robot_cam)
#     # action = affordance_agent.act(obs, affordance_traj_rob)
#     action = affordance_agent.dummy_act(obs)
#     print(f'action from act: {action}')
#     obs, reward, terminate = task.step(action)

print('Done')
env.shutdown()
