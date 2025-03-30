from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary


class PickUpBowl(Task):

    def init_task(self) -> None:
        self.bowl1 = Shape('bowl')
        self.bowl1_visual = Shape('bowl_visual')
        self.boundary = SpawnBoundary([Shape('boundary')]) # already defined in the scene
        self.success_sensor = ProximitySensor('success')
        self.register_graspable_objects([self.bowl1])
        self.register_success_conditions([
            DetectedCondition(self.bowl1, self.success_sensor, negated=True),
            GraspedCondition(self.robot.gripper, self.bowl1),
        ])

    def init_episode(self, index: int) -> List[str]:
        # self.variation_index = index
        # target_color_name, target_rgb = colors[index]

        # random_idx = np.random.choice(len(colors))
        # while random_idx == index:
        #     random_idx = np.random.choice(len(colors))
        # _, other1_rgb = colors[random_idx]

        # self.bowl1_visual.set_color(target_rgb)

        self.boundary.clear()
        self.boundary.sample(self.success_sensor, min_distance=0.1)

        return ['pick up the bowl',
                'grasp the bowl and lift it',
                'lift the bowl']

    def variation_count(self) -> int:
        return 1
    
    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -3.14 / 8.], [0, 0, 3.14 / 8.]
    

