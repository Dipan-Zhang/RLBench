from typing import List, Tuple
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task


class OpenDrawer(Task):

    def init_task(self) -> None:
        self.drawer = Dummy('open_drawer')
        self.drawer_bottom = Shape('drawer_bottom')
        self.drawer.set_position([0.25, -0.2, 0.752])
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._waypoint1 = Dummy('waypoint1')
        # self.register_graspable_objects([self.drawer_bottom])

    def init_episode(self, index: int) -> List[str]:
        option = self._options[index] # possible grasping options
        self._waypoint1.set_position(self._anchors[index].get_position())
        self.register_success_conditions(
            [JointCondition(self._joints[index], 0.08)])
        return ['open %s drawer' % option,
                'grip the %s handle and pull the %s drawer open' % (
                    option, option),
                'slide the %s drawer open' % option]

    def variation_count(self) -> int:
        return 3

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, 0], [0, 0, 0]

    def is_static_workspace(self) -> bool:
        return True