import gymnasium as gym
import numpy as np
import pybullet
import pytest

import panda_gym


def test_save_and_restore_state():
    env = gym.make("PandaReach-v2")
    env.reset()

    state_id = env.save_state()

    # Perform the action
    action = env.action_space.sample()
    next_observation1, reward, done, info = env.step(action)

    # Restore and perform the same action
    env.reset()
    env.restore_state(state_id)
    next_observation2, reward, done, info = env.step(action)

    # The observations in both cases should be equals
    assert np.all(next_observation1["achieved_goal"] == next_observation2["achieved_goal"])
    assert np.all(next_observation1["observation"] == next_observation2["observation"])
    assert np.all(next_observation1["desired_goal"] == next_observation2["desired_goal"])


def test_remove_state():
    env = gym.make("PandaReach-v2")
    env.reset()
    state_id = env.save_state()
    env.remove_state(state_id)
    with pytest.raises(pybullet.error):
        env.restore_state(state_id)
