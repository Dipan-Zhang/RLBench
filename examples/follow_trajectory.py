import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget


class Affordance_transfer(object):
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
task = env.get_task(ReachTarget)

affordance_transfer = Affordance_transfer()

# get 2 demos from the task, within demo, includes a list of observations + actions
demos = task.get_demos(2, live_demos=live_demos)  # -> List[List[Observation]]
demos = np.array(demos).flatten()

# An example of using the demos to 'train' using behaviour cloning loss.
for i in range(100):
    print("'training' iteration %d" % i)
    batch = np.random.choice(demos, replace=False)
    batch_images = [obs.left_shoulder_rgb for obs in batch]
    predicted_actions = affordance_transfer.predict_action(batch_images)
    ground_truth_actions = [obs.joint_velocities for obs in batch]
    loss = affordance_transfer.behaviour_cloning_loss(ground_truth_actions, predicted_actions)

print('Done')
env.shutdown()
