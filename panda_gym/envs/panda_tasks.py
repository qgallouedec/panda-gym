import warnings
from typing import Optional

import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.flip import Flip
from panda_gym.envs.tasks.pick_and_place import PickAndPlace
from panda_gym.envs.tasks.push import Push
from panda_gym.envs.tasks.reach import Reach
from panda_gym.envs.tasks.slide import Slide
from panda_gym.envs.tasks.stack import Stack
from panda_gym.pybullet import PyBullet


class PandaFlipEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        render (bool, optional): Deprecated: This argument is deprecated and will be removed in a future
            version. Use the render_mode argument instead.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        render: Optional[bool] = None,
    ) -> None:
        if render is not None:
            warnings.warn(
                "The 'render' argument is deprecated and will be removed in "
                "a future version. Use the 'render_mode' argument instead.",
                DeprecationWarning,
            )
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Flip(sim, reward_type=reward_type)
        super().__init__(robot, task)


class PandaPickAndPlaceEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        render (bool, optional): Deprecated: This argument is deprecated and will be removed in a future
            version. Use the render_mode argument instead.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        render: Optional[bool] = None,
    ) -> None:
        if render is not None:
            warnings.warn(
                "The 'render' argument is deprecated and will be removed in "
                "a future version. Use the 'render_mode' argument instead.",
                DeprecationWarning,
            )
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlace(sim, reward_type=reward_type)
        super().__init__(robot, task)


class PandaPushEnv(RobotTaskEnv):
    """Push task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        render (bool, optional): Deprecated: This argument is deprecated and will be removed in a future
            version. Use the render_mode argument instead.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        render: Optional[bool] = None,
    ) -> None:
        if render is not None:
            warnings.warn(
                "The 'render' argument is deprecated and will be removed in "
                "a future version. Use the 'render_mode' argument instead.",
                DeprecationWarning,
            )
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Push(sim, reward_type=reward_type)
        super().__init__(robot, task)


class PandaReachEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        render (bool, optional): Deprecated: This argument is deprecated and will be removed in a future
            version. Use the render_mode argument instead.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        render: Optional[bool] = None,
    ) -> None:
        if render is not None:
            warnings.warn(
                "The 'render' argument is deprecated and will be removed in "
                "a future version. Use the 'render_mode' argument instead.",
                DeprecationWarning,
            )
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Reach(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(robot, task)


class PandaSlideEnv(RobotTaskEnv):
    """Slide task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        render (bool, optional): Deprecated: This argument is deprecated and will be removed in a future
            version. Use the render_mode argument instead.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        render: Optional[bool] = None,
    ) -> None:
        if render is not None:
            warnings.warn(
                "The 'render' argument is deprecated and will be removed in "
                "a future version. Use the 'render_mode' argument instead.",
                DeprecationWarning,
            )
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Slide(sim, reward_type=reward_type)
        super().__init__(robot, task)


class PandaStackEnv(RobotTaskEnv):
    """Stack task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        render (bool, optional): Deprecated: This argument is deprecated and will be removed in a future
            version. Use the render_mode argument instead.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        render: Optional[bool] = None,
    ) -> None:
        if render is not None:
            warnings.warn(
                "The 'render' argument is deprecated and will be removed in "
                "a future version. Use the 'render_mode' argument instead.",
                DeprecationWarning,
            )
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Stack(sim, reward_type=reward_type)
        super().__init__(robot, task)
