import gymnasium as gym
import numpy as np

import panda_gym


def test_seed_reach():
    final_observations = []
    env = gym.make("PandaReach-v3")
    actions = [
        np.array([-0.931, 0.979, -0.385]),
        np.array([-0.562, 0.391, -0.532]),
        np.array([0.042, 0.254, -0.624]),
        np.array([0.465, 0.745, 0.284]),
        np.array([-0.237, 0.995, -0.425]),
        np.array([0.67, 0.472, 0.972]),
    ]
    for _ in range(2):
        env.reset(seed=12345)
        for action in actions:
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
        final_observations.append(observation)

    assert np.allclose(final_observations[0]["observation"], final_observations[1]["observation"])
    assert np.allclose(final_observations[0]["achieved_goal"], final_observations[1]["achieved_goal"])
    assert np.allclose(final_observations[0]["desired_goal"], final_observations[1]["desired_goal"])


def test_seed_push():
    final_observations = []
    env = gym.make("PandaPush-v3")
    actions = [
        np.array([0.925, 0.352, -0.014]),
        np.array([0.400, -0.018, -0.042]),
        np.array([0.308, 0.189, -0.943]),
        np.array([-0.556, 0.209, 0.907]),
        np.array([-0.862, -0.243, 0.835]),
        np.array([-0.552, -0.262, 0.317]),
    ]
    for _ in range(2):
        env.reset(seed=6789)
        for action in actions:
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
        final_observations.append(observation)

    assert np.allclose(final_observations[0]["observation"], final_observations[1]["observation"])
    assert np.allclose(final_observations[0]["achieved_goal"], final_observations[1]["achieved_goal"])
    assert np.allclose(final_observations[0]["desired_goal"], final_observations[1]["desired_goal"])


def test_seed_slide():
    final_observations = []
    env = gym.make("PandaSlide-v3")
    actions = [
        np.array([0.245, 0.786, 0.329]),
        np.array([-0.414, 0.343, -0.839]),
        np.array([0.549, 0.047, -0.857]),
        np.array([0.744, -0.507, 0.092]),
        np.array([-0.202, -0.939, -0.945]),
        np.array([-0.97, -0.616, 0.472]),
    ]
    for _ in range(2):
        env.reset(seed=13795)
        for action in actions:
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
        final_observations.append(observation)
    assert np.allclose(final_observations[0]["observation"], final_observations[1]["observation"])
    assert np.allclose(final_observations[0]["achieved_goal"], final_observations[1]["achieved_goal"])
    assert np.allclose(final_observations[0]["desired_goal"], final_observations[1]["desired_goal"])


def test_seed_pick_and_place():
    final_observations = []
    env = gym.make("PandaPickAndPlace-v3")
    actions = [
        np.array([0.429, -0.287, 0.804, -0.592]),
        np.array([0.351, -0.136, 0.296, -0.223]),
        np.array([-0.187, 0.706, -0.988, 0.972]),
        np.array([-0.389, -0.249, 0.374, -0.389]),
        np.array([-0.191, -0.297, -0.739, 0.633]),
        np.array([0.093, 0.242, -0.11, -0.949]),
    ]
    for _ in range(2):
        env.reset(seed=794512)
        for action in actions:
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
        final_observations.append(observation)

    assert np.allclose(final_observations[0]["observation"], final_observations[1]["observation"])
    assert np.allclose(final_observations[0]["achieved_goal"], final_observations[1]["achieved_goal"])
    assert np.allclose(final_observations[0]["desired_goal"], final_observations[1]["desired_goal"])


def test_seed_stack():
    final_observations = []
    env = gym.make("PandaStack-v3")
    actions = [
        np.array([-0.609, 0.73, -0.433, 0.76]),
        np.array([0.414, 0.327, 0.275, -0.196]),
        np.array([-0.3, 0.589, -0.712, 0.683]),
        np.array([0.772, 0.333, -0.537, -0.253]),
        np.array([0.784, -0.014, -0.997, -0.118]),
        np.array([-0.12, -0.958, -0.744, -0.98]),
    ]
    for _ in range(2):
        env.reset(seed=657894)
        for action in actions:
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
        final_observations.append(observation)
    assert np.allclose(final_observations[0]["observation"], final_observations[1]["observation"])
    assert np.allclose(final_observations[0]["achieved_goal"], final_observations[1]["achieved_goal"])
    assert np.allclose(final_observations[0]["desired_goal"], final_observations[1]["desired_goal"])
