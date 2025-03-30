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
from rlbench.tasks import PickUpCup, SlideCabinetOpen

from affordance.affordance_transfer import AffordanceTransfer
from affordance.helpers import preprocess_target_data
from affordance.affordance_transfer import AffordanceTransfer
from affordance.helpers import hash_filename, read_image, visualize_points,\
      resize_img, load_optimization_result, get_configs, interpolate_trajectory,\
visualize_3d_trajectory, pick_points_in_viewer, draw_line, backproject_with_color
from simulation.sim_utils import visualize_affordance_with_scene, visualize_affordance_in_pointcloud
from NeuS.models.utils import backproject
from pyrep.backend import sim
# from transforms3d.quaternions import mat2quat, quat2mat # don't fuck using this
from thirdparty.graspNet.gsnet_wrapper import GSNetWrapper
from thirdparty.graspNet.gsnet import GSNet
import copy
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from thirdparty.graspNet.utils.grasp_utils import crop_points


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

#########################
def test_depth(depth_dir):
    depth = cv2.imread(depth_dir, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000
    cam_K_dir = '/home/stud/zanr/data+ckpt/graspNet/scene_0090/kinect/camK.npy'
    cam_K = np.load(cam_K_dir)

    rgb_dir = depth_dir.replace('depth', 'rgb')
    rgb = cv2.imread(rgb_dir)
    # points, _ = backproject(depth, cam_K, depth>0, False)
    pts, colors = backproject_with_color(depth, rgb, cam_K, depth>0, False)
    pcd = visualize_points(pts, colors)
    # pcd = visualize_points(points, colors)
    return pts, pcd

def test_depth2(depth_idx):
    depth_dir = 'dataset/kinect/table1_far/depth/{:06}.png'.format(depth_idx)
    rgb_dir = 'dataset/kinect/table1_far/color/{:06}.jpg'.format(depth_idx)
    depth = cv2.imread(depth_dir, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000
    cam_K_dir = 'dataset/kinect/table1_far/camera_intrinsic.json'

    import json
    data = json.load(open(cam_K_dir))
    cam_K = np.array(data['intrinsic_matrix']).reshape(3, 3).T
    print(cam_K)

    rgb = cv2.imread(rgb_dir, -1)
    # points, _ = backproject(depth, cam_K, depth>0, False)
    pts, colors = backproject_with_color(depth, rgb, cam_K, depth>0, False)
    pcd = visualize_points(pts, colors)
    # pcd = visualize_points(points, colors)
    tgt_pt = pick_points_in_viewer(pcd)
    return tgt_pt, pcd

def get_pcd_in_wristcam(obs):
    points_wrist_world = obs.wrist_point_cloud
    T_wc = obs.misc['wrist_camera_extrinsics']
    T_cw = np.linalg.inv(T_wc)
    points_wrist_cam = points_wrist_world @ T_cw[:3, :3].T + T_cw[:3, 3]
    pcd = visualize_points(points_wrist_cam.reshape(-1, 3))
    tgt_pt = pick_points_in_viewer(pcd)
    return tgt_pt, pcd

#####################
def vis_grasp(pcd, gg, keep=20):
    gg = gg.nms()
    gg = gg.sort_by_score()
    if gg.__len__() > keep:
        gg = gg[:keep]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([pcd, *grippers])  


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
            # self.affordance_transfer = AffordanceTransfer(self.config)
        else:
            print('No config file provided, affordance transfer not initialized')
        
        self.raw_gsNet = GSNet(self.config['gsnet'])
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


    def try_raw_gsNet(self, tgt_pt, pcd):
        "plan smooth gripper trajectory based on affordance trajectory"
        pcs = np.asarray(pcd.points)
        dist_maks = pcs[:, 2] < 1.3
        pcs = pcs[dist_maks]

        gg = self.raw_gsNet.inference(pcs)
        vis_grasp(pcd, gg, keep=100)

    def try_raw_gsNet_wrapper(self, tgt_pt, pcd):
        "plan smooth gripper trajectory based on affordance trajectory"
        pcs = np.asarray(pcd.points)
        dist_maks = pcs[:, 2] < 1.3
        pcs = pcs[dist_maks]

        gg = self.gsNet.detect_grasp_gsnet(pcs)
        vis_grasp(pcd, gg, keep=100)
    
    def try_gsNet_wrapper(self, tgt_pt, pcd, top_down=True):

        best_grasp = self.gsNet.infer_best_grasp(pcd, tgt_pt.reshape(-1,1), max_dis=0.06, top_down=top_down)
        gripper_vis = best_grasp.to_open3d_geometry()
        o3d.visualization.draw_geometries([pcd, gripper_vis])

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
left_shoulder_camera = CameraConfig(image_size=(1280, 1080), depth_in_meters=True) 
obs_config = ObservationConfig(front_camera=front_camera, wrist_camera=wrist_camera, left_shoulder_camera=left_shoulder_camera)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(absolute_mode=True, ), gripper_action_mode=Discrete()),
    obs_config=obs_config,
    headless=False)
env.launch()

task = env.get_task(SlideCabinetOpen)
obs = None

# set up affordance agent
config_fp = './affordance/task_conf/cup0.yaml'
affordance_agent = Agent(config_fp)
episode_length = 220
num_try = 10
total_length =  episode_length * num_try

for i in range(num_try):
    for ii in range(episode_length):
        if i % episode_length == 0:
            idx = 0
            print('Reset Episode')
            descriptions, obs = task.reset()
            hide_robot_temporarily('Panda')
            obs = task.get_observation()

            affordance_agent.reset()
            print('<===== Task description: ====>\n', descriptions)
            obj_name = descriptions[0][11:]

            # get pointcloud 
            pts = obs.left_shoulder_point_cloud # WH3
            pts = pts.reshape(-1, 3) # N3
            pcd = visualize_points(pts)
            tgt_pt = pick_points_in_viewer(pcd)
            tgt_pt = tgt_pt.reshape(3,1).squeeze(-1)
            

            # optional crop
            ds_points, _, _ = crop_points(tgt_pt, np.asarray(pcd.points), thres=0.5)
            cropped_pcd = visualize_points(ds_points)
            affordance_agent.try_gsNet_wrapper(tgt_pt, pcd, top_down=True)

            restore_robot_position('Panda')

        terminate = True
        if terminate:
            break

print('Done')
env.shutdown()
