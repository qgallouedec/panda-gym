from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots import Panda
from panda_gym.envs.tasks import Reach
from panda_gym.pybullet import PyBullet


class PandaReachEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse"):
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=True, base_position=[-0.6, 0.0, 0.0])
        task = Reach(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(robot, task)
