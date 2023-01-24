import gymnasium as gym

import panda_gym


def test_render():
    env = gym.make("PandaReach-v3", render_mode="rgb_array")
    env.reset()

    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        img = env.render()
        assert img.shape == (480, 720, 3)
        if terminated or truncated:
            env.reset()

    env.close()


def test_new_render_shape():
    env = gym.make("PandaReach-v3", render_mode="rgb_array", render_height=48, render_width=84)

    env.reset()
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        image = env.render()
        assert image.shape == (48, 84, 3)
        if terminated or truncated:
            env.reset()

    env.close()
