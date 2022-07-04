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

    obs_restored = env.restore_state(state_id)
    assert(np.all(obs_original["achieved_goal"] == obs_restored["achieved_goal"]))
    assert(np.all(obs_original["observation"] == obs_restored["observation"]))
   
def test_remove_state():
    env = PandaReachEnv()
    env.reset()
    state_id = env.save_state()
    with pytest.raises(pybullet.error):
        env.remove_state(state_id)
