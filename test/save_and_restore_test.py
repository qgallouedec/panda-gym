import numpy as np
import pybullet
import pytest

from panda_gym.envs.panda_tasks.panda_reach import PandaReachEnv


def test_save_and_restore_state():
    env = PandaReachEnv()
    obs_original = env.reset()
    state_id = env.save_state()

    a = env.action_space.sample()
    env.step(a)

    env.restore_state(state_id)
    obs_restored = env._get_obs()
    assert np.all(obs_original["achieved_goal"] == obs_restored["achieved_goal"])
    assert np.all(obs_original["observation"] == obs_restored["observation"])


def test_remove_state():
    env = PandaReachEnv()
    env.reset()
    state_id = env.save_state()
    env.remove_state(state_id)
    with pytest.raises(pybullet.error):
        env.restore_state(state_id)
