from typing import List, Tuple
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition, NothingGrasped
import numpy as np


class OpenCabinet(Task):

    def init_task(self):
        self.cabinet = Shape('cabinet')
        self.cabinet.set_position([0.65,0,1.275])
        self.left_joint = Joint('cabinet_door_hinge_left')



    def init_episode(self, index: int) -> List[str]:
        self.register_success_conditions(
            [JointCondition(self.left_joint, np.deg2rad(30)),])


        return ['slide left cabinet open',
                'open the left door',
            ]

    def variation_count(self) -> int :
        return 2

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, -3.14 / 4.], [0.0, 0.0, 3.14 / 4.]

    def is_static_workspace(self) -> bool:
        return True
