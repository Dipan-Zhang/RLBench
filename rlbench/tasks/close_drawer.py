from typing import List, Tuple
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task


class CloseDrawer(Task):

    def init_task(self) -> None:
        self.drawer = Dummy('close_drawer')
        self.drawer.set_position([0.275, 0.0, 0.752])
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._waypoint1 = Dummy('waypoint1')

    def init_episode(self, index: int) -> List[str]:
        self._joints[index].set_joint_position(0.1)
        option = self._options[index]
        self._waypoint1.set_position(self._anchors[index].get_position())
        self.register_success_conditions(
            [JointCondition(self._joints[index], 0.06)])

        return ['close %s drawer' % (option,),
                'shut the %s drawer' % (option,),
                'slide the %s drawer shut' % (option,)]

    def variation_count(self) -> int:
        return 3

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
    
    # def is_static_workspace(self) -> bool:
    #     return True
