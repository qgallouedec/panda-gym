from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class PickAndPlace(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
        table_length: float = 1.1,
        table_x_pos: float = -0.3,
        table_y_pos: float = 0,
        table_width: float = 0.7,
        second_table: bool = False,
        plane_Size: float = 3.0,

    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.table_length = table_length
        self.table_x_pos = table_x_pos
        self.table_y_pos = table_y_pos
        self.table_width = table_width
        self.plane_size = plane_Size
        self.second_table = second_table
        self.goal_y = 0
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4,size=self.plane_size)
        self.sim.create_table(length=self.table_length, width=self.table_width,  height=0.4, x_offset=self.table_x_pos, y_offset=self.table_y_pos)
        if self.second_table:
            self.goal_y = 2
            self.sim.create_table(length=self.table_length, width=self.table_width,  height=0.4, x_offset=0, y_offset=self.goal_y)
            
        
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.5]),
        )

        self.sim.create_box(
            body_name="hindsight_target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, -99]),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation
    
    def show_hindsight_goal(self):
        self.sim.set_base_pose("hindsight_target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def hide_hindsight_goal(self):
        self.sim.set_base_pose("hindsight_target", np.array([0.0, 0.0, -99]), np.array([0.0, 0.0, 0.0, 1.0]))


    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.hide_hindsight_goal()
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, self.goal_y, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        if self.second_table:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, self.table_y_pos, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        if self.second_table:
            noise[2] = 0.0
        object_position += noise
        return object_position
    

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)

    def compute_distance(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}):
            return distance(achieved_goal, desired_goal)