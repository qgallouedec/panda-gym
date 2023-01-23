import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper

import panda_gym


def test_pixel_observation_wrapper():
    env = gym.make("PandaReach-v3", render_mode="rgb_array")
    env = PixelObservationWrapper(env)

    observation, _ = env.reset()
    assert observation["pixels"].shape == (480, 720, 3)

    for _ in range(100):
        action = env.action_space.sample()  # random action
        observation, _, terminated, truncated, _ = env.step(action)
        assert observation["pixels"].shape == (480, 720, 3)
        if terminated or truncated:
            observation, _ = env.reset()

    env.close()


def test_new_render_shape():
    env = gym.make("PandaReach-v3", render_mode="rgb_array", render_width=84, render_height=84)

    env.reset()
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        image = env.render()
        assert image.shape == (84, 84, 3)
        if terminated or truncated:
            env.reset()

    env.close()
