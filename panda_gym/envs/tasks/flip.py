from typing import Any, Dict, Tuple, Union

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class Flip(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.2,
        obj_xy_range: float = 0.3,
    ):
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_xy_choices = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2])
        self.goal_z_range_low = -np.pi  # rotation here !
        self.goal_z_range_high = np.pi
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.goal = None  # will be generated when reset
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            texture="colored_cube.png",
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.5]),
            texture="colored_cube.png",
        )

    def get_goal(self) -> np.ndarray:
        if not isinstance(self.goal, np.ndarray):
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = np.array(self.sim.get_base_position("object"))
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        object_velocity = np.array(self.sim.get_base_velocity("object"))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        return object_rotation

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position, object_orientation = self._sample_object()
        self.sim.set_base_pose("target", np.array([0.0, 0.0, self.object_size / 2]), self.goal)
        self.sim.set_base_pose("object", object_position, object_orientation)

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal_x = np.random.choice(self.goal_xy_choices)
        goal_y = np.random.choice(self.goal_xy_choices)
        goal_z = np.random.uniform(self.goal_z_range_low, self.goal_z_range_high)
        goal = np.array([goal_x, goal_y, goal_z])
        return goal

    def _sample_object(self) -> Tuple[np.ndarray, np.ndarray]:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        object_rotation = np.zeros(3)
        return object_position, object_rotation

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d