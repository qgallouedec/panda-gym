import numpy as np

from panda_gym.envs.core import TaskEnv


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class SlideEnv(TaskEnv):
    def __init__(
        self,
        sim,
        goal_xy_range=0.5,
        goal_x_offset=0.5,
        obj_xy_range=0.3,
        distance_threshold=0.05,
        reward_type="sparse",
        seed=None,
    ):
        self.sim = sim
        self.goal_range_low = np.array(
            [-goal_xy_range / 2 + goal_x_offset, -goal_xy_range / 2, 0]
        )
        self.goal_range_high = np.array(
            [goal_xy_range / 2 + goal_x_offset, goal_xy_range / 2, 0]
        )
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.seed(seed)

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self):
        self.sim.create_box(
            body_name="plane",
            half_extents=[1.15, 0.7, 0.01],
            mass=0,
            position=[0.15, 0.0, -0.41],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0.15, 0.15, 0.15, 1.0],
        )
        self.sim.create_box(
            body_name="table",
            half_extents=[0.6, 0.35, 0.2],
            mass=0,
            position=[0.25, 0.0, -0.2],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0.8, 0.8, 0.8, 1],
        )
        self.sim.create_cylinder(
            body_name="object",
            mass=0.5,
            radius=0.03,
            height=0.03,
            position=[0.0, 0.0, 0.015],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0.2, 0.2, 0.2, 1],
            friction=0.1
        )
        self.sim.create_sphere(
            body_name="target",
            ghost=True,
            mass=0,
            radius=0.03,
            position=[1.0, 1.0, 1.0],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[1, 0, 0, 1],
        )

    def resample(self):
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, [0, 0, 0, 1])
        self.sim.set_base_pose("object", object_position, [0, 0, 0, 1])

    def _sample_goal(self):
        """Randomize goal."""
        goal = [0.0, 0.0, 0.015]  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return goal.copy()

    def _sample_object(self):
        """Randomize start position of object."""
        object_position = [0.0, 0.0, 0.015]  # z offset for the cube center
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position.copy()

    def _is_success(self, achieved_goal, desired_goal):
        """Returns whether the achieved goal match the desired goal."""
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
