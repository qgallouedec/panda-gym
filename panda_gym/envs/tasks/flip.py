import numpy as np
from gym import utils

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Flip(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.05,
        obj_xy_range=0.3,
    ):
        self.sim = sim
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_xy_choices = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2])
        self.goal_z_range_low = -np.pi  # rotation here !
        self.goal_z_range_high = np.pi
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=[0, 0, 0], distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self):
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(
            length=1.1, width=0.7, height=0.4, x_offset=-0.3, friction=0.2
        )  # increase friction in order to allow fliping
        self.sim.create_box(
            body_name="object",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=1.0,
            position=[0.0, 0.0, self.object_size / 2],
            friction=5,  # increase friction. For some reason, it helps a lot learning
            texture="colored_cube.png",
        )
        self.sim.create_box(
            body_name="target",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=0.0,
            ghost=True,
            position=[0.0, 0.0, self.object_size / 2],
            rgba_color=[1.0, 1.0, 1.0, 0.5],
            texture="colored_cube.png",
        )

    def get_goal(self):
        return self.goal.copy()

    def get_obs(self):
        # position, rotation of the object
        object_position = np.array(self.sim.get_base_position("object"))
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        object_velocity = np.array(self.sim.get_base_velocity("object"))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self):
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        return object_rotation

    def reset(self):
        self.goal = self._sample_goal()
        object_position, object_orientation = self._sample_object()
        self.sim.set_base_pose("target", [0.0, 0.0, self.object_size / 2], self.goal)
        self.sim.set_base_pose("object", object_position, object_orientation)

    def _sample_goal(self):
        """Randomize goal."""
        goal_x = np.random.choice(self.goal_xy_choices)
        goal_y = np.random.choice(self.goal_xy_choices)
        goal_z = np.random.uniform(self.goal_z_range_low, self.goal_z_range_high)
        goal = np.array([goal_x, goal_y, goal_z])
        return goal

    def _sample_object(self):
        """Randomize start position of object."""
        object_position = [0.0, 0.0, self.object_size / 2]
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        object_rotation = [0, 0, 0]
        return object_position, object_rotation

    def is_success(self, achieved_goal, desired_goal):
        d = distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
