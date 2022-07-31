from typing import Any, Dict, Optional, Tuple

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
        border_size = 0.05
        self.sim.create_box(
            body_name="border0",
            half_extents=np.array([1.1, border_size, border_size]) / 2,
            mass=0.0,
            position=np.array([-0.3, 0.325, border_size / 2]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.95, 0.95, 0.95, 1]),
        )
        self.sim.create_box(
            body_name="border1",
            half_extents=np.array([1.1, border_size, border_size]) / 2,
            mass=0.0,
            position=np.array([-0.3, -0.325, border_size / 2]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.95, 0.95, 0.95, 1]),
        )
        self.sim.create_box(
            body_name="border2",
            half_extents=np.array([border_size, 0.6, border_size]) / 2,
            mass=0.0,
            position=np.array([0.225, 0.0, border_size / 2]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.95, 0.95, 0.95, 1]),
        )
        self.sim.create_box(
            body_name="border3",
            half_extents=np.array([border_size, 0.6, border_size]) / 2,
            mass=0.0,
            position=np.array([-0.225, 0.0, border_size / 2]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.95, 0.95, 0.95, 1]),
        )
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
            observation.append(np.concatenate((object_position,)))
        observation = np.concatenate(observation)
        return observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        obs = self.robot.get_obs()[[0, 1, 2, 6]]
        if self.nb_objects > 0:
            obs = np.concatenate((obs, self.get_objects_pos()))
        reward, done, info = 0.0, False, {}
        return obs, reward, done, info

    def reset(self):
        self.robot.reset()
        for i in range(self.nb_objects):
            self.sim.set_base_pose(
                "object" + str(i),
                np.array([self.object_size * i, 0.0, self.object_size / 2]),
                np.array([0.0, 0.0, 0.0, 1.0]),
            )
        obs = self.robot.get_obs()[[0, 1, 2, 6]]
        if self.nb_objects > 0:
            obs = np.concatenate((obs, self.get_objects_pos()))
        return obs

    def render(
        self,
        mode: str,
        width: int = 720,
        height: int = 480,
        target_position: Optional[np.ndarray] = None,
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.

        If mode is "human", make the rendering real-time. All other arguments are
        unused. If mode is "rgb_array", return an RGB array of the scene.

        Args:
            mode (str): "human" of "rgb_array". If "human", this method waits for the time necessary to have
                a realistic temporal rendering and all other args are ignored. Else, return an RGB array.
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Rool of the camera. Defaults to 0.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        target_position = target_position if target_position is not None else np.zeros(3)
        return self.sim.render(
            mode,
            width=width,
            height=height,
            target_position=target_position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )
