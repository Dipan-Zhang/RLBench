from typing import List, Tuple
import numpy as np
from pyrep.objects.joint import Joint
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task
from pyrep.objects.dummy import Dummy


class ToiletSeatUp(Task):

    def init_task(self) -> None:
        self.toilet_seat = Dummy('toilet_seat_up')
        self.toilet_seat.set_position([0.35, 0.1, 0.75])
        # self.waypoint0 = Dummy('waypoint0')
        # self.waypoint1 = Dummy('waypoint1')
        # self.waypoint0.set_position([0.138, 0.244, 0.9885])
        # self.waypoint1.set_position([0.19, 0.244, 0.9885])

        self.register_success_conditions([
            JointCondition(Joint('toilet_seat_up_revolute_joint'), 1.40)])

    def init_episode(self, index: int) -> List[str]:
        return ['lift toilet seat up',
                'put the toilet seat up',
                'leave the lid of the toilet seat in a upright position',
                'grip the edge of the toilet seat and lift it up to an '
                'upright position',
                'leave the toilet lid open',
                'raise the toilet seat']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, -np.pi / 4.], [0.0, 0.0, np.pi / 4.]

    def is_static_workspace(self) -> bool:
        return True