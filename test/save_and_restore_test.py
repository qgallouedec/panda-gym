import gymnasium as gym
import numpy as np
import pybullet
import pytest

import panda_gym


def test_save_and_restore_state():
    env = gym.make("PandaReach-v3")
    env.reset()

    state_id = env.unwrapped.save_state()

    # Perform the action
    action = env.action_space.sample()
    observation1, _, _, _, _ = env.step(action)

    # Restore and perform the same action
    env.reset()
    env.unwrapped.restore_state(state_id)
    observation2, _, _, _, _ = env.step(action)

    # The observations in both cases should be equals
    assert np.all(observation1["achieved_goal"] == observation2["achieved_goal"])
    assert np.all(observation1["observation"] == observation2["observation"])
    assert np.all(observation1["desired_goal"] == observation2["desired_goal"])


def test_remove_state():
    env = gym.make("PandaReach-v3")
    env.reset()
    state_id = env.unwrapped.save_state()
    env.unwrapped.remove_state(state_id)
    with pytest.raises(pybullet.error):
        env.unwrapped.restore_state(state_id)
