import gymnasium as gym
import pytest

import panda_gym


def run_env(env):
    """Tests running panda gym envs."""
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    env.close()
    # check that it allows to be closed multiple times
    env.close()


@pytest.mark.parametrize("env_id", panda_gym.ENV_IDS)
def test_env(env_id):
    """Tests running panda gym envs."""
    env = gym.make(env_id)
    run_env(env)
