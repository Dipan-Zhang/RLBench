from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from rlbench.backend.task import Task
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import JointCondition

class OpenDishwasher(Task):

    def init_task(self) -> None:
        self._dishwasher = Dummy('open_dishwasher')
        self._dishwasher.set_position([0.25, -0.1, 0.752])
        self._dishwasher.set_orientation([0, 0, -80])
        self.register_success_conditions([JointCondition(
                    Joint('dishwasher_door_joint'), np.deg2rad(30))])

    def init_episode(self, index: int) -> List[str]:
        return ['open the  dishwasher door',
                'open the dishwasher till fully open'
                ]

    def variation_count(self) -> int:
        return 1

    # def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
    #     return [0, 0, -3.14 / 2.], [0, 0, 3.14 / 2.]


    def is_static_workspace(self) -> bool:
        return True
