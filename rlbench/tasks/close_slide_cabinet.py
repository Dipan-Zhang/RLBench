from typing import List, Tuple
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition, NothingGrasped



class CloseSlideCabinet(Task):

    def init_task(self):
        self.cabinet = Shape('cabinet_base')
        self.cabinet.set_position([0.35, 0.1, 1.45])
        self.left_joint = Joint('left_joint')
        self.left_initial_waypoint = Dummy('waypoint0')
        self.left_close_waypoint = Dummy('waypoint1')
        self.left_far_waypoint = Dummy('waypoint2')

    def init_episode(self, index: int) -> List[str]:
        self.register_success_conditions(
            [JointCondition(self.left_joint, 0.1)])


        return ['slide left cabinet open',
                'open the left door',
            ]

    def variation_count(self) -> int :
        return 2

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, -3.14 / 4.], [0.0, 0.0, 3.14 / 4.]

    def is_static_workspace(self) -> bool:
        return True
