from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor

from rlbench.backend.task import Task
from rlbench.const import colors
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary

class SlideBlockToTargetTest(Task):

    def init_task(self) -> None:
        self.block = Shape('block')
        success_detector = ProximitySensor('success')
        self.target = Shape('target')
        self.boundary = SpawnBoundary([Shape('boundary')])
        success_condition = DetectedCondition(self.block, success_detector)
        self.register_success_conditions([success_condition])

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        block_name, block_rgb = colors[index]
        self.block.set_color(block_rgb)
        self.boundary.clear()
        self.boundary.sample(self.target)
        return ['slide the %s block to the target' % block_name,
                'move the %s block to the target' % block_name]
    

    def variation_count(self) -> int:
        return len(colors)

