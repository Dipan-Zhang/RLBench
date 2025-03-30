import numpy as np
import ipdb
import cv2

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget
from pyrep.objects.vision_sensor import VisionSensor

"borrowed from single_task_rl.py and imitation_learning.py"


class Agent(object):
    "dummy agent for imitation learning"

    def __init__(self):
        # create affordance_transfer object
        self.affordance_transfer = None
        self.affordance_trajectory = None
        self.trajectory_index = 0

    def predict_action(self, batch):
        return np.random.uniform(size=(len(batch), 7))

    def behaviour_cloning_loss(self, ground_truth_actions, predicted_actions):
        return 1

    def get_images(self, obs, SAVE_IMAGES=False):
        camera_save = ['front_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb', 'wrist_rgb']
        images = []
        for camera in camera_save:
            image = getattr(obs, camera)
            image = (image * 255).astype(np.uint8)
            images.append(image)
        if SAVE_IMAGES:
            for camera, image in zip(camera_save, images):
                cv2.imwrite(f'{camera}.png', image)
            print(f'saved images')
        return images

    def affordance_transfer(self, images):
        pass

        
    def dummy_act(self, obs):
        self.affordance_trajectory = np.array([
            [3.81813741e-01, -8.13834369e-03,  1.2455574e+00],
            [5.1813741e-01, -8.13834369e-03,  1.1455574e+00],
            # [6.1813741e-01, -8.13834369e-02,  1.0455574e+00],
            # [5.1813741e-01, -8.13834369e-02,  1.255574e+00]
        ])
        current_pose = obs.gripper_pose[:3]
        distance = np.linalg.norm(
            current_pose - self.affordance_trajectory[self.trajectory_index])
        if distance < 0.1:
            self.trajectory_index += 1
            if self.trajectory_index >= len(self.affordance_trajectory):
                self.trajectory_index = len(self.affordance_trajectory) - 1
            print(f'move to position: {self.trajectory_index}')
        goal_position = self.affordance_trajectory[self.trajectory_index]

        # goal_position = current_pose
        # goal_position = np.array([[ 5.1813741e-01, -8.13834369e-03,  1.1455574e+00]])
        current_orientation = obs.gripper_pose[3:7]
        gripper = [1.0]  # Always open
        return np.concatenate([goal_position, current_orientation, gripper], axis=-1)

    def act(self, obs):
        # goal_trajectory = affordance_transfer.transfer(obs)
        # gripper = [1.0]  # Always open
        # return np.concatenate([arm, gripper], axis=-1)
        pass



# enable all observations/ sensors
obs_config = ObservationConfig()
obs_config.set_all(True)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=False)
env.launch()


# created from available tasks
task = env.get_task(ReachTarget)

affordance_transfer = Agent()

training_steps = 120
episode_length = 40
obs = None
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        affordance_transfer.save_images(obs)
        print(descriptions)
    action = affordance_transfer.dummy_act(obs)
    print(f'action from dummy act: {action}')
    obs, reward, terminate = task.step(action)


print('Done')
env.shutdown()
