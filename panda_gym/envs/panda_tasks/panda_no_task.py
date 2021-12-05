from typing import Any, Dict, Tuple

import gym
import numpy as np
from gym import spaces
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


class PandaNoTaskEnv(gym.Env):
    """Panda robot without any task. Reward is always 0.0.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, control_type: str = "ee") -> None:
        self.sim = PyBullet(render=render)
        self.robot = Panda(self.sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        obs = self.reset()
        observation_shape = obs.shape
        self.observation_space = spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32)
        self.action_space = self.robot.action_space
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        obs = self.robot.get_obs()  # robot state
        reward, done, info = 0.0, False, {}
        return obs, reward, done, info

    def reset(self):
        self.robot.reset()
        obs = self.robot.get_obs()  # robot state
        return obs
