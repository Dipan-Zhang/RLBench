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
from rlbench.tasks import SlideCabinetOpen

from affordance.affordance_transfer import AffordanceTransfer
from affordance.helpers import preprocess_target_data
from affordance.affordance_transfer import AffordanceTransfer
from affordance.helpers import hash_filename, read_image, visualize_points,\
resize_img, load_optimization_result, get_configs, interpolate_trajectory,\
visualize_3d_trajectory, pick_points_in_viewer, draw_line, backproject_with_color
from simulation.sim_utils import visualize_affordance_with_scene, visualize_affordance_in_pointcloud
from NeuS.models.utils import backproject
import transforms3d as t3d
from pyrep.backend import sim


ZOFFFSET = 2
def hide_robot_temporarily(robot_base_name='Panda'):
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


def restore_robot_position(robot_base_name):
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
        position_offset (list or tuple): [dx, dy, dz] added to the camera’s world position.
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
    
    # Convert the new matrix back to a flat 3x4 list.
    new_mat_3x4 = new_mat[:3, :].flatten().tolist()
    
    # Update the camera's transformation.
    sim.simSetObjectMatrix(cam_handle, -1, new_mat_3x4)
    
    print(f"Camera '{camera_name}' adjusted:")
    print("  New position:", new_t)
    print("  New rotation matrix:\n", new_R)


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

    def act(self, obs, affordance_traj_world):
        """
        Given the current observation and a list/array of 3D waypoints (affordance_traj_world),
        return an action that includes a goal position (3D), current gripper orientation (4D), and
        a gripper command (1D). The action sequence first executes a pregrasp trajectory:
           1. Move above the target waypoint.
           2. Slowly descend to the target waypoint and close the gripper.
        Then, the postgrasp trajectory is executed.
        """
        # Extract current gripper pose and orientation
        current_pose = obs.gripper_pose[:3]
        current_orientation = obs.gripper_pose[3:7]

        # Phase 1: PREGRASP -- move to a point above the target (first waypoint)
        if self.phase == 'pregrasp':
            # Use the first waypoint in the trajectory as the target grasp point.
            target_wp = affordance_traj_world[0].copy()
            # Assume the z-axis is vertical; move up by the offset.
            pregrasp_wp = target_wp.copy()
            pregrasp_wp[2] += self.pregrasp_offset

            # Check if we are at the pregrasp position
            dist_to_pregrasp = np.linalg.norm(current_pose - pregrasp_wp)
            if dist_to_pregrasp > self.distance_threshold:
                goal_position = pregrasp_wp
            else:
                # Once above the target, switch to the descend phase.
                print("Reached pregrasp position; switching to descend phase.")
                self.phase = 'descend'
                goal_position = target_wp  # begin descending

            # Keep the gripper open during pregrasp
            gripper_command = [1.0]

        # --- Phase 2: DESCEND ---
        elif self.phase == 'descend':
            # If we haven't generated the descending trajectory yet, do it now.
            if self.descending_traj is None:
                target_wp = affordance_traj_world[0]
                self.descending_traj = self.generate_descending_traj(current_pose, target_wp)
                self.descending_idx = 0

            # Follow the descending trajectory step-by-step.
            goal_position = self.descending_traj[self.descending_idx]
            distance_to_goal = np.linalg.norm(current_pose - goal_position)
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
                    goal_position = affordance_traj_world[0]  # ensure staying at grasp point momentarily
                    gripper_command = [0.0]  # Close gripper
                else:
                    goal_position = self.descending_traj[self.descending_idx]
                    gripper_command = [1.0]  # Keep gripper open until final descent
            else:
                gripper_command = [1.0]
   

        # Phase 3: POSTGRASP -- follow the affordance trajectory with gripper closed.
        elif self.phase == 'postgrasp':
            # Ensure that the trajectory index is within bounds.
            if self.trajectory_idx >= len(affordance_traj_world):
                self.trajectory_idx = len(affordance_traj_world) - 1

            affordance_wp = affordance_traj_world[self.trajectory_idx]
            distance = np.linalg.norm(current_pose - affordance_wp)

            if distance > self.distance_threshold:
                goal_position = affordance_wp
            else:
                # Advance to the next waypoint.
                self.trajectory_idx += 1
                if self.trajectory_idx >= len(affordance_traj_world):
                    self.trajectory_idx = len(affordance_traj_world) - 1
                print(f"Moving to postgrasp waypoint index: {self.trajectory_idx}")
                goal_position = affordance_traj_world[self.trajectory_idx]

            # Keep the gripper closed during postgrasp.
            gripper_command = [0.0]


        else:
            # Fallback: if for some reason phase is not recognized, do nothing.
            goal_position = current_pose
            gripper_command = [1.0]

        # Construct and return the action
        action = np.concatenate([goal_position, current_orientation, gripper_command], axis=-1)
        return action
    
    def reset(self):
        """Call this function at the start of each episode to reset the agent's state."""
        self.phase = 'pregrasp'
        self.trajectory_idx = 0
        self.gripper_closed = False
        self.descending_traj = None
        self.descending_idx = 0


    def is_collision_free(self, start, end):
        """
        Stub function for collision checking between start and end.
        Returns True if the linear path is collision free.
        In practice, replace with your collision checking routine.
        """
        # For now, always assume the path is collision free.
        return True

    def generate_descending_traj(self, current_pose, target_pose):
        """
        Generates a descending trajectory from current_pose to target_pose with n_descend_steps.
        A simple linear interpolation is used. If collision checking fails for a segment,
        one might adjust the step size or re-plan (this is a placeholder).
        """
        # Generate linearly interpolated points (each is a 3D point)
        traj = np.linspace(current_pose, target_pose, self.n_descend_steps)
        
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
            for key, rgb, depth in zip(camera_keys, self.rgbs, self.depths):
                cv2.imwrite(f'./outputs/rlbench/{key}_rgb.png', rgb)
                cv2.imwrite(f'./outputs/rlbench/{key}_depth.png', (depth * 1000).astype(np.float32))
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
left_shoulder_camera = CameraConfig(image_size=(512, 512), depth_in_meters=True)
right_shoulder_camera = CameraConfig(image_size=(512, 512), depth_in_meters=True)
obs_config = ObservationConfig(front_camera=front_camera,
                               left_shoulder_camera=left_shoulder_camera,
                               right_shoulder_camera=right_shoulder_camera,
                               )

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, ), gripper_action_mode=Discrete()),
    obs_config=obs_config,
    headless=False)
env.launch()

task = env.get_task(SlideCabinetOpen)
obs = None

# modify camera pose for this task
adjust_camera_pose('cam_front', [0, 0, -0.5], [-np.pi/8, 0, 0])
adjust_camera_pose('cam_over_shoulder_left', [-0.6, 0.0, -1.0], [-np.pi/4, -np.pi/16, 0]) # get overview of the workspace
adjust_camera_pose('cam_over_shoulder_right', [-0.3, -0.3, -1.0], [-np.pi/4, np.pi/16, 0])

# set up affordance agent
config_fp = './affordance/task_conf/cupboard1_to_cupboard2.yaml'
affordance_agent = Agent(config_fp)
episode_length = 100
num_try = 3
total_length =  episode_length * num_try

for i in range(total_length):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        hide_robot_temporarily('Panda')
        obs = task.get_observation()
        affordance_agent.reset()
        # run transfer based on the observation
        corres_3d_c2, affordance_c2, affordance_traj_world = \
            affordance_agent.run_transfer(obs, cam_name='cam_over_shoulder_left', tgt_obj_prompt='cabinet.', DEBUG=True)
        
        restore_robot_position('Panda')

    action = affordance_agent.act(obs, affordance_traj_world)
    print(f'action from act: {action}')
    obs, reward, terminate = task.step(action)


print('Done')
env.shutdown()
