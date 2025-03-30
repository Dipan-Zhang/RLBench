from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary


class TakeBowl(Task):

    def init_task(self) -> None:
        # self.cup1 = Shape('cup1')
        # self.cup2 = Shape('cup2')
        self.bowl1 = Shape.import_mesh('./tools/assets/bowl1.obj',scaling_factor=1)
        # self.cup1_visual = Shape('cup1_visual')
        # self.cup2_visual = Shape('cup2_visual')
        self.boundary = SpawnBoundary([Shape('boundary')]) # already defined in the scene
        self.success_sensor = ProximitySensor('success')
        self.register_graspable_objects([self.bowl1])
        self.register_success_conditions([
            DetectedCondition(self.bowl1, self.success_sensor, negated=True),
            GraspedCondition(self.robot.gripper, self.bowl1),
        ])

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index
        target_color_name, target_rgb = colors[index]

        random_idx = np.random.choice(len(colors))
        while random_idx == index:
            random_idx = np.random.choice(len(colors))
        _, other1_rgb = colors[random_idx]

        # self.cup1_visual.set_color(target_rgb)
        # self.cup2_visual.set_color(other1_rgb)

        self.boundary.clear()
        # self.boundary.sample(self.cup2, min_distance=0.1)
        self.boundary.sample(self.success_sensor, min_distance=0.1)

        return ['pick up the %s cup' % target_color_name,
                'grasp the %s cup and lift it' % target_color_name,
                'lift the %s cup' % target_color_name]

    def variation_count(self) -> int:
        return len(colors)

    def import_3d_model(self, model_path):
        print('Importing 3D model from:', model_path)
        # if not isfile(model_path):
        #     print('Model file does not exist!')
        #     return

        try:
            model = Shape.import_mesh(model_path,scaling_factor=1)
            model.set_parent(self.task.get_base())
            print('3D model imported successfully!')
        except Exception as e:
            print('Failed to import 3D model!')