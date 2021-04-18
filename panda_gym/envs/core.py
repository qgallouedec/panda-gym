import gym
from gym import utils, spaces
import numpy as np


class PyBulletRobot:
    """Base class for robot env.

    Args:
        sim (Any): The simulation engine.
        body_name (str): The name of the robot within the simulation.
        ee_link (int): Link index of the end-effector
        file_name (str): Path of the urdf file.
        base_position (x, y, z): Position of the base of the robot.
        seed (int, optional): Seed. Defaults to None.
    """

    def __init__(self, sim, body_name, file_name, base_position):
        self.sim = sim  # sim engine
        self.body_name = body_name
        with self.sim.no_rendering():
            self._load_robot(file_name, base_position)
            self.setup()

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

    def setup(self):
        """Called once in en constructor."""
        pass

    def set_action(self, action):
        """Perform the action."""
        raise NotImplementedError

    def get_obs(self):
        """Return the observation associated to the robot."""
        raise NotImplementedError

    def reset(self):
        """Reset the robot."""
        raise NotImplementedError

    def get_link_position(self, link):
        """Returns the position of a link as (x, y, z)"""
        return self.sim.get_link_position(self.body_name, link)

    def get_link_velocity(self, link):
        """Returns the velocity of a link as (vx, vy, vz)"""
        return self.sim.get_link_velocity(self.body_name, link)

    def control_joints(self, target_angles):
        """Control the joints of the robot."""
        self.sim.control_joints(
            body=self.body_name,
            joints=self.JOINT_INDICES,
            target_angles=target_angles,
            forces=self.JOINT_FORCES,
        )


class RobotTaskEnv(gym.GoalEnv):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, seed=None):
        obs = self.reset()
        observation_shape = obs["observation"].shape
        achieved_goal_shape = obs["achieved_goal"].shape
        desired_goal_shape = obs["achieved_goal"].shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-np.inf, np.inf, shape=observation_shape),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=achieved_goal_shape),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=desired_goal_shape),
            )
        )
        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward
        self.render = self.sim.render
        self.seed(seed)

    def _get_obs(self):
        robot_obs = self.robot.get_obs()  # robot state
        task_obs = self.task.get_obs()  # object position, velococity, etc...
        observation = np.concatenate([robot_obs, task_obs])

        achieved_goal = self.task.get_achieved_goal()

        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal(),
        }

    def reset(self):
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        return self._get_obs()

    def step(self, action):
        self.robot.set_action(action)
        self.sim.step()
        obs = self._get_obs()
        done = False
        info = {
            "is_success": self.task.is_success(
                obs["achieved_goal"], self.task.get_goal()
            ),
        }
        reward = self.task.compute_reward(
            obs["achieved_goal"], self.task.get_goal(), info
        )
        return obs, reward, done, info

    def seed(self, seed=None):
        """Setup the seed."""
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        self.sim.close()


class Task:
    """To be completed."""

    def get_goal(self):
        """Return the current goal."""
        raise NotImplementedError

    def get_obs(self):
        """Return the observation associated to the task."""
        raise NotImplementedError

    def get_achieved_goal(self):
        """Return the achieved goal."""
        raise NotImplementedError

    def reset(self):
        """Reset the task: sample a new goal"""
        pass

    def is_success(self, achieved_goal, desired_goal):
        """Returns whether the achieved goal match the desired goal."""
        raise NotImplementedError

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward associated to the achieved and the desired goal."""
        raise NotImplementedError
