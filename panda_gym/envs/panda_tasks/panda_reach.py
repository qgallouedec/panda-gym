from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots import Panda
from panda_gym.envs.tasks import Reach


class PandaReachEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
    """

    def __init__(self, render=False, reward_type="sparse"):
        self.sim = PyBullet(render=render)
        self.robot = Panda(self.sim, block_gripper=True, base_position=[-0.6, 0.0, 0.0])
        self.task = Reach(
            self.sim,
            reward_type=reward_type,
            get_ee_position=self.robot.get_ee_position,
        )
        RobotTaskEnv.__init__(self)
