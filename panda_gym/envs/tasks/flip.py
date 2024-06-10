from typing import Any, Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import angle_distance


class Flip(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.2,
        obj_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()

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
            position=np.array([0.0, 0.0, 3 * self.object_size / 2]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.5]),
            texture="colored_cube.png",
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object", "quaternion")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_rotation = np.array(self.sim.get_base_rotation("object", "quaternion"))
        return object_rotation

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position, object_orientation = self._sample_object()
        self.sim.set_base_pose("target", np.array([0.0, 0.0, 3 * self.object_size / 2]), self.goal)
        self.sim.set_base_pose("object", object_position, object_orientation)

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = R.random().as_quat()
        return goal

    def _sample_object(self) -> Tuple[np.ndarray, np.ndarray]:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        object_rotation = np.zeros(3)
        return object_position, object_rotation

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = angle_distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = angle_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
