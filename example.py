import time

import gym
import numpy as np
from gym import spaces

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.reach import Reach
from panda_gym.pybullet import PyBullet


class DoublePandaRobot:
    """Example for double Panda arm robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        control_type: str = "ee",
    ) -> None:
        self.sim = sim
        base_positions = [np.array([-1, 0.0, 0.0]), np.array([1, 0.0, 0.0])]
        base_orientations = [np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0, 0.0])]
        assert len(base_positions) == 2
        self.robots = [Panda(sim, robot_id=i, block_gripper=block_gripper, base_position=base_positions[i],
                             control_type=control_type, base_orientation=base_orientations[i])
                       for i in range(len(base_positions))]
        self.n_action = sum([robot.action_space.shape[0] for robot in self.robots])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_action,), dtype=np.float32)

    def set_action(self, action: np.ndarray) -> None:
        self.robots[0].set_action(action[:int(self.n_action/2)])
        self.robots[1].set_action(action[int(self.n_action/2):])

    def get_obs(self) -> np.ndarray:
        obs = np.concatenate((self.robots[0].get_obs(), self.robots[1].get_obs()))
        return obs

    def reset(self) -> None:
        for robot in self.robots:
            robot.reset()


class DoublePandaEnv(RobotTaskEnv):
    """Reach task wih Panda robot.
    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "joints") -> None:
        sim = PyBullet(render=render)
        double_panda = DoublePandaRobot(sim, control_type='joint')
        task = DoublePandaTask(sim, reward_type=reward_type, get_ee_position=double_panda.robots[0].get_ee_position)
        super().__init__(double_panda, task)


class DoublePandaTask(Reach):
    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=2.4, width=0.7, height=0.4, x_offset=0)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )


if __name__ == '__main__':
    gym.envs.register(
        id='DoublePanda-v0',
        entry_point='example:DoublePandaEnv',
        max_episode_steps=150,
    )

    env = gym.make('DoublePanda-v0', render=True)

    env.reset()
    done = False
    i = 0
    while not done:
        action = env.action_space.sample()  # random action
        env.step(action)
        if i == 0:
            time.sleep(5)
        else:
            time.sleep(0.05)
        i += 1

    env.close()