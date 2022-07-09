from typing import Any, Dict, Tuple

import gym
import numpy as np
from gym import spaces

from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

COLORS = np.array(
    [
        [0.94, 0.28, 0.44, 1.00],
        [1.00, 0.82, 0.4, 1.00],
        [0.02, 0.84, 0.63, 1.00],
        [0.07, 0.54, 0.7, 1.00],
        [0.03, 0.23, 0.3, 1.00],
    ]
)


class PandaNoTaskEnv(gym.Env):
    """Panda robot without any task. Reward is always 0.0.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, control_type: str = "ee", nb_objects: int = 0) -> None:
        self.sim = PyBullet(render=render)
        self.robot = Panda(self.sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        self.nb_objects = nb_objects
        self.object_size = 0.04
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        obs = self.reset()
        observation_shape = obs.shape
        self.observation_space = spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32)
        self.action_space = self.robot.action_space

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        for i in range(self.nb_objects):
            self.sim.create_box(
                body_name="object" + str(i),
                half_extents=np.ones(3) * self.object_size / 2,
                mass=1.0,
                position=np.array([self.object_size * i + 0.1, 0.1, self.object_size / 2]),
                rgba_color=COLORS[i % 5],
            )

    def get_objects_pos(self):
        observation = []
        for i in range(self.nb_objects):
            object_position = np.array(self.sim.get_base_position("object" + str(i)))
            object_rotation = np.array(self.sim.get_base_rotation("object" + str(i)))
            object_velocity = np.array(self.sim.get_base_velocity("object" + str(i)))
            object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object" + str(i)))
            observation.append(
                np.concatenate(
                    (
                        object_position,
                        object_rotation,
                        object_velocity,
                        object_angular_velocity,
                    )
                )
            )
        observation = np.concatenate(observation)
        return observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        obs = self.robot.get_obs()
        if self.nb_objects > 0:
            obs = np.concatenate((obs, self.get_objects_pos()))
        reward, done, info = 0.0, False, {}
        return obs, reward, done, info

    def reset(self):
        self.robot.reset()
        for i in range(self.nb_objects):
            self.sim.set_base_pose(
                "object" + str(i),  # add 0.1 : trick not to be between two cells
                np.array([self.object_size * i + 0.1, 0.1, self.object_size / 2]),
                np.array([0.0, 0.0, 0.0, 1.0]),
            )
        obs = self.robot.get_obs()
        if self.nb_objects > 0:
            obs = np.concatenate((obs, self.get_objects_pos()))
        return obs
