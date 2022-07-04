import numpy as np

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
    
