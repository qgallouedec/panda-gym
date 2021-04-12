from gym import spaces
import numpy as np

from panda_gym.envs.robots import PandaEnv
from panda_gym.envs.tasks import StackEnv
from panda_gym.pybullet import PyBullet


class PandaStackEnv(PandaEnv, StackEnv):
    """Stack task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
    """

    def __init__(self, render=False, reward_type="sparse"):
        sim = PyBullet(render=render, n_substeps=20)
        StackEnv.__init__(self, sim, reward_type=reward_type)
        PandaEnv.__init__(self, sim, block_gripper=False, base_position=[-0.6, 0.0, 0.0])
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-np.inf, np.inf, shape=(31,)),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(6,)),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(6,)),
            )
        )

    def _get_obs(self):
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())

        # fingers opening
        fingers_width = self.get_fingers_width()

        # position, rotation of the object
        object1_position = np.array(self.sim.get_base_position("object1"))
        object1_rotation = np.array(self.sim.get_base_rotation("object1"))
        object1_velocity = np.array(self.sim.get_base_velocity("object1"))
        object1_angular_velocity = np.array(
            self.sim.get_base_angular_velocity("object1")
        )
        object2_position = np.array(self.sim.get_base_position("object2"))
        object2_rotation = np.array(self.sim.get_base_rotation("object2"))
        object2_velocity = np.array(self.sim.get_base_velocity("object2"))
        object2_angular_velocity = np.array(
            self.sim.get_base_angular_velocity("object2")
        )
        observation = np.concatenate(
            [
                ee_position,
                ee_velocity,
                [fingers_width],  # this is a float
                object1_position,
                object1_rotation,
                object1_velocity,
                object1_angular_velocity,
                object2_position,
                object2_rotation,
                object2_velocity,
                object2_angular_velocity,
            ]
        )
        achieved_goal = np.concatenate((object1_position, object2_position))
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.copy(),
        }

    def reset(self):
        with self.sim.no_rendering():
            self.set_joint_neutral()
            self.resample()
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        obs = self._get_obs()
        done = False
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        return obs, reward, done, info