from typing import List
from rlbench.backend.task import Task
from rlbench.const import colors


class ReachTargetTutorial(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions([
            DetectedCondition(self.robot.arm.get_tip(), success_sensor)
        ])

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)
        return [f'reach the {color_name} target' , 'reach the sphere']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> List[float], List[float]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]