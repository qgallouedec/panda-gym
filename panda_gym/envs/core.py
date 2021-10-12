from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import gym
import gym.spaces
import gym.utils.seeding
import numpy as np

from panda_gym.pybullet import PyBullet


class PyBulletRobot(ABC):
    """Base class for robot env.

    Args:
        sim (PyBullet): The simulation engine.
        body_name (str): The name of the robot within the simulation.
        file_name (str): Path of the urdf file.
        base_position (np.ndarray): Position of the base of the robot as (x, y, z).
    """

    @property
    @abstractmethod
    def JOINT_INDICES(self):
        ...

    @property
    @abstractmethod
    def JOINT_FORCES(self):
        ...

    def __init__(
        self, sim: PyBullet, body_name: str, file_name: str, base_position: np.ndarray, action_space: gym.spaces.Space
    ) -> None:
        self.sim = sim  # sim engine
        self.body_name = body_name
        with self.sim.no_rendering():
            self._load_robot(file_name, base_position)
            self.setup()
        self.action_space = action_space

    def _load_robot(self, file_name: str, base_position: np.ndarray) -> None:
        """Load the robot.

        Args:
            file_name (str): The URDF file name of the robot.
            base_position (np.ndarray): The position of the robot, as (x, y, z).
        """
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            useFixedBase=True,
        )

    def setup(self) -> None:
        """Called after robot loading."""
        pass

    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Set the action. Must be call just before sim.step().

        Args:
            action (np.ndarray): The action.
        """

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the robot.

        Returns:
            np.ndarray: The observation.
        """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the robot and return the observation.

        Returns:
            np.ndarray: The observation.
        """

    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_link_position(self.body_name, link)

    def get_link_velocity(self, link: int) -> np.ndarray:
        """Returns the velocity of a link as (vx, vy, vz)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Velocity as (vx, vy, vz)
        """
        return self.sim.get_link_velocity(self.body_name, link)

    def control_joints(self, target_angles: np.ndarray) -> None:
        """Control the joints of the robot.

        Args:
            target_angles (np.ndarray): The target angles. The length of the array must equal to the number of joints.
        """
        self.sim.control_joints(
            body=self.body_name,
            joints=self.JOINT_INDICES,
            target_angles=target_angles,
            forces=self.JOINT_FORCES,
        )


class Task(ABC):
    """To be completed."""

    def __init__(self, sim: PyBullet) -> None:
        self.sim = sim

    @abstractmethod
    def get_goal(self) -> np.ndarray:
        """Return the current goal."""

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the task."""

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Return the achieved goal."""

    def reset(self) -> None:
        """Reset the task: sample a new goal"""
        pass

    def seed(self, seed: Optional[int]) -> int:
        """Sets the random seed.

        Args:
            seed (Optional[int]): The desired seed. Leave None to generate one.

        Returns:
            int: The seed.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return seed

    @abstractmethod
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        """Returns whether the achieved goal match the desired goal."""

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]
    ) -> Union[np.ndarray, float]:
        """Compute reward associated to the achieved and the desired goal."""


class RobotTaskEnv(gym.GoalEnv):
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        robot (PyBulletRobot): The robot.
        task (Task): The task.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, robot: PyBulletRobot, task: Task) -> None:
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        self.robot = robot
        self.task = task
        self.seed()  # required for init; can be changed later
        obs = self.reset()
        observation_shape = obs["observation"].shape
        achieved_goal_shape = obs["achieved_goal"].shape
        desired_goal_shape = obs["achieved_goal"].shape
        self.observation_space = gym.spaces.Dict(
            dict(
                observation=gym.spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float64),
                desired_goal=gym.spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float64),
                achieved_goal=gym.spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float64),
            )
        )
        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs()  # robot state
        task_obs = self.task.get_obs()  # object position, velococity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal()
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal(),
        }

    def reset(self) -> Dict[str, np.ndarray]:
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        obs = self._get_obs()
        done = False
        info = {"is_success": self.task.is_success(obs["achieved_goal"], self.task.get_goal())}
        reward = self.task.compute_reward(obs["achieved_goal"], self.task.get_goal(), info)
        assert isinstance(reward, float)  # needed for pytype cheking
        return obs, reward, done, info

    def seed(self, seed: Optional[int] = None) -> int:
        """Setup the seed."""
        return self.task.seed(seed)

    def close(self) -> None:
        self.sim.close()

    def render(
        self,
        mode,
        width: int = 720,
        height: int = 480,
        target_position: np.ndarray = np.zeros(3),
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll=0,
    ):
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
