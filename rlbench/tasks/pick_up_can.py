from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary


class PickUpCan(Task):

    def init_task(self) -> None:
        self.success_sensor = ProximitySensor('success')
        self.success_sensor.set_position([0.52,0.2,0.7561])
        self.coke_can = Shape('coke_can_resp')
        self.register_graspable_objects([self.coke_can])
        self.register_success_conditions([
            DetectedCondition(self.coke_can, self.success_sensor, negated=True),
            GraspedCondition(self.robot.gripper, self.coke_can),
        ])

    def init_episode(self, index: int) -> List[str]:
        # self.variation_index = index
        # target_color_name, target_rgb = colors[index]

        # random_idx = np.random.choice(len(colors))
        # while random_idx == index:
        #     random_idx = np.random.choice(len(colors))
        # _, other1_rgb = colors[random_idx]

        # self.bottle1_visual.set_color(target_rgb)

        # self.boundary.clear()
        # self.boundary.sample(self.success_sensor, min_distance=0.1)

        return ['pick up the bottle',
                'grasp the bottle and lift it',
                'lift the bottle']

    def variation_count(self) -> int:
        return 1
    
    # def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
    #     # return [0, 0, -3.14 / 8.], [0, 0, 3.14 / 8.]
    #     return [0, 0, 0], [0, 0, 0]
    
    def is_static_workspace(self) -> bool:
        return True