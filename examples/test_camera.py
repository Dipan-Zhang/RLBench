import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget, PickUpCup, OpenDrawerFixed
import matplotlib.pyplot as plt
from pyrep.backend import sim
import transforms3d as t3d

from pyrep.objects.vision_sensor import VisionSensor
from scipy.spatial.transform import Rotation as Rot

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

class ImitationLearning(object):
    "dummy agent for imitation learning"
    def predict_action(self, batch):
        return np.random.uniform(size=(len(batch), 7))

    def behaviour_cloning_loss(self, ground_truth_actions, predicted_actions):
        return 1


# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

# enable all observations/ sensors
obs_config = ObservationConfig()
obs_config.set_all(True)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=False)
env.launch()

# created from available tasks
# task = env.get_task(ReachTarget)
from typing import Tuple , List
task = env.get_task(OpenDrawerFixed)

il = ImitationLearning()

# live_demo: record 2 demos on the fly, includes a list of observations + actions
# # waypoints coming from copeliaSIM, within ttm file, get demo -> robot move
# demos = task.get_demos(5, live_demos=live_demos)  # -> List[List[Observation]]
# demos = task.get_demos(2, live_demos=live_demos)  # -> List[List[Observation]]

succ = task.move_to_grasp()
# import ipdb; ipdb.set_trace()
# demos = np.array(demos).flatten()
# task._scene.move_to_grasp_pose()

# adjust_camera_pose('cam_over_shoulder_left', [2, 1.5, 1.2], [100,0,0])
# adjust_camera_pose2('cam_over_shoulder_left', [2, 1.5, 1.2], [100,0,0])
set_camera_pose('cam_over_shoulder_left', [0.2, 1.5, 1.2], [100, 0, 0])
# An example of using the demos to 'train' using behaviour cloning loss.
for i in range(100):
    print("'training' iteration %d" % i)
    obs = task.get_observation()
    import ipdb; ipdb.set_trace()
    plt.imshow(obs.left_shoulder_rgb)
    plt.show()
    plt.savefig('test.png')
    batch = np.random.choice(demos, replace=False)
    batch_images = [obs.left_shoulder_rgb for obs in batch]
    predicted_actions = il.predict_action(batch_images)
    ground_truth_actions = [obs.joint_velocities for obs in batch]
    loss = il.behaviour_cloning_loss(ground_truth_actions, predicted_actions)

print('Done')
env.shutdown()
