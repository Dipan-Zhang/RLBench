from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary


class PickUpKnife(Task):

    def init_task(self) -> None:
        self.mug1 = Shape('mug1')
        self.mug1_visual = Shape('mug1_visual')
        self.boundary = SpawnBoundary([Shape('boundary')]) # already defined in the scene
        self.success_sensor = ProximitySensor('success')
        self.register_graspable_objects([self.mug1])
        self.register_success_conditions([
            DetectedCondition(self.mug1, self.success_sensor, negated=True),
            GraspedCondition(self.robot.gripper, self.mug1),
        ])

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index
        target_color_name, target_rgb = colors[index]

        random_idx = np.random.choice(len(colors))
        while random_idx == index:
            random_idx = np.random.choice(len(colors))
        _, other1_rgb = colors[random_idx]

        self.mug1_visual.set_color(target_rgb)

        self.boundary.clear()
        self.boundary.sample(self.success_sensor, min_distance=0.1)

        return ['pick up the %s mug' % target_color_name,
                'grasp the %s mug and lift it' % target_color_name,
                'lift the %s mug' % target_color_name]

    def variation_count(self) -> int:
        return len(colors)
    
    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        # return [0, 0, -3.14 / 8.], [0, 0, 3.14 / 8.]
        return [0, 0, 0], [0, 0, 0]
    

