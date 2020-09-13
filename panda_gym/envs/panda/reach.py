from gym import utils
from panda_gym.envs import panda_env

MODEL_JSON_PATH = 'reach.json'


class PandaReachEnv(panda_env.PandaEnv, utils.EzPickle):
    # position of the robot base along the three axis
    def __init__(self, render=False, reward_type='sparse'):
        initial_qpos = {}
        panda_env.PandaEnv.__init__(
            self, MODEL_JSON_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, render=render)
        utils.EzPickle.__init__(self)
