from typing import Any

import gym
from gym import utils
import numpy as np


class RobotEnv(gym.Env):
    """Base class for robot env.

    Args:
        sim (Any): The simulation engine.
        body_name (str): The name of the robot within the simulation.
        ee_link (int): Link index of the end-effector
        file_name (str): Path of the urdf file.
        base_position (x, y, z): Position of the base of the robot.
        seed (int, optional): Seed. Defaults to None.
    """

    def __init__(self, sim, body_name, ee_link, file_name, base_position, seed=None):

        self.sim = sim  # sim engine
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(round(1.0 / self.sim.dt)),
        }
        self.ee_link = ee_link
        self.body_name = body_name
        self.seed(seed)
        with self.sim.no_rendering():
            self._load_robot(file_name, base_position)
            self._env_setup()
            self._viewer_setup()

    def seed(self, seed=None):
        """Seed setup."""
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def _load_robot(self, file_name, base_position):
        """Load the robot.

        Args:
            file_name (str): The file name of the robot.
            base_position (x, y, z): The position of the robot.
        """
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            useFixedBase=True,
        )

    def _env_setup(self):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Direct the camera to the gripper position."""
        target = self.sim.get_link_position("panda", self.ee_link)
        self.sim.place_visualizer(target=target, distance=1.1, yaw=48, pitch=-14)

    def get_link_position(self, link):
        """Returns the position of a link as (x, y, z)"""
        return self.sim.get_link_position(self.body_name, link)

    def get_ee_position(self):
        """Returns the position of the ned-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_link_velocity(self, link):
        """Returns the velocity of a link as (vx, vy, vz)"""
        return self.sim.get_link_velocity(self.body_name, link)

    def get_ee_velocity(self):
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    def control_joints(self, target_angles):
        """Control the joints of the robot."""
        self.sim.control_joints(
            body=self.body_name,
            joints=self.JOINT_INDICES,
            target_angles=target_angles,
            forces=self.JOINT_FORCES,
        )


class TaskEnv(gym.GoalEnv):
    def seed(self, seed=None):
        """Setup the seed."""
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def render(self, *args, **kwargs):
        return self.sim.render(*args, **kwargs)
