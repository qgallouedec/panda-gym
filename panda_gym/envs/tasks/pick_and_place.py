import numpy as np
from gym import utils

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class PickAndPlace(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_xy_range=0.3,
        goal_z_range=0.2,
        obj_xy_range=0.3,
        seed=None,
    ):
        self.sim = sim
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.np_random, self.seed = utils.seeding.np_random(seed)
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array(
            [goal_xy_range / 2, goal_xy_range / 2, goal_z_range]
        )
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target=[0, 0, 0], distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self):
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=2,
            position=[0.0, 0.0, self.object_size / 2],
            rgba_color=[0.9, 0.1, 0.1, 1],
            friction=5,  # increase friction. For some reason, it helps a lot learning
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
            position=[0.0, 0.0, 0.05],
            rgba_color=[0.9, 0.1, 0.1, 0.3],
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
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self):
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, [0, 0, 0, 1])
        self.sim.set_base_pose("object", object_position, [0, 0, 0, 1])

    def _sample_goal(self):
        """Randomize goal."""
        goal = [0.0, 0.0, self.object_size / 2]  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if np.random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self):
        """Randomize start position of object."""
        object_position = [0.0, 0.0, self.object_size / 2]
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal, desired_goal):
        d = distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
