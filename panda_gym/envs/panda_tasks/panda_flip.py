from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots import Panda
from panda_gym.envs.tasks import Flip


class PandaFlipEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render=False, reward_type="sparse", control_type="ee"):
        self.sim = PyBullet(render=render)
        self.robot = Panda(
            self.sim,
            block_gripper=False,
            base_position=[-0.6, 0.0, 0.0],
            control_type=control_type,
            fingers_friction=0.1,
        )
        self.task = Flip(self.sim, reward_type=reward_type)
        RobotTaskEnv.__init__(self)
