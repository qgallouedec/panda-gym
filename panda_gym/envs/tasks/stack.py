import numpy as np

from panda_gym.envs.core import TaskEnv


def distance(a, b):
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)


class StackEnv(TaskEnv):
    def __init__(
        self,
        sim,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
        distance_threshold=0.05,
        reward_type="sparse",
        seed=None,
    ):
        self.sim = sim
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
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
            half_extents=[0.85, 0.7, 0.01],
            mass=0,
            position=[-0.5, 0.0, -0.41],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0.15, 0.15, 0.15, 1.0],
        )
        self.sim.create_box(
            body_name="table",
            half_extents=[0.25, 0.35, 0.2],
            mass=0,
            position=[0.0, 0.0, -0.2],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0.8, 0.8, 0.8, 1],
        )
        self.sim.create_box(
            body_name="object1",
            half_extents=[0.025, 0.025, 0.025],
            mass=0.5,
            position=[0.0, -0.1, 0.025],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0.9, 0.2, 0.2, 1],
        )
        self.sim.create_box(
            body_name="object2",
            half_extents=[0.025, 0.025, 0.025],
            mass=0.5,
            position=[0.0, 0.1, 0.025],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0.2, 0.9, 0.2, 1],
        )
        self.sim.create_sphere(
            body_name="target1",
            ghost=True,
            mass=0,
            radius=0.02,
            position=[0.0, -0.1, 0.05],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[1, 0, 0, 0.7],
        )
        self.sim.create_sphere(
            body_name="target2",
            ghost=True,
            mass=0,
            radius=0.02,
            position=[0.0, 0.1, 0.05],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0, 1, 0, 0.7],
        )

    def resample(self):
        self.goal = self._sample_goal()
        object1_position, object2_position = self._sample_objects()
        self.sim.set_base_pose("target1", self.goal[:3], [0, 0, 0, 1])
        self.sim.set_base_pose("target2", self.goal[3:], [0, 0, 0, 1])
        self.sim.set_base_pose("object1", object1_position, [0, 0, 0, 1])
        self.sim.set_base_pose("object2", object2_position, [0, 0, 0, 1])

    def _sample_goal(self):
        """Randomize goal."""
        goal1 = [0.0, 0.0, 0.025]  # z offset for the cube center
        goal2 = [0.0, 0.0, 0.075]  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal1 += noise
        goal2 += noise
        return np.concatenate((goal1, goal2))

    def _sample_objects(self):
        """Randomize start position of object."""
        while True: # make sure that cubes are distant enough
            object1_position = [0.0, 0.0, 0.025]  # z offset for the cube center
            object2_position = [0.0, 0.0, 0.025]  # z offset for the cube center
            noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
            noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
            object1_position += noise1
            object2_position += noise2
            if distance(object1_position, object2_position) > 0.1:
                return object1_position, object2_position

    def _is_success(self, achieved_goal, desired_goal):
        """Returns whether the achieved goal match the desired goal."""
        d = distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold * 2).astype(np.float32)
        else:
            return -d
