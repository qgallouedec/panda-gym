import gym
import panda_gym
import numpy as np


def test_seed_reach():
    env = gym.make("PandaReach-v1")
    env.seed(12345)
    env.reset()
    actions = [
        np.array([-0.931, 0.979, -0.385]),
        np.array([-0.562, 0.391, -0.532]),
        np.array([0.042, 0.254, -0.624]),
        np.array([0.465, 0.745, 0.284]),
        np.array([-0.237, 0.995, -0.425]),
        np.array([0.67, 0.472, 0.972]),
    ]
    for action in actions:
        state, _, _, _ = env.step(action)
    # check somes values from the final state
    assert state["observation"][2] == 0.10261139602330989
    assert state["achieved_goal"][1] == 0.13984260601763496
    assert state["desired_goal"][0] == 0.006393300281407527


def test_seed_push():
    env = gym.make("PandaPush-v1")
    env.seed(6789)
    env.reset()
    actions = [
        np.array([0.925, 0.352, -0.014]),
        np.array([0.400, -0.018, -0.042]),
        np.array([0.308, 0.189, -0.943]),
        np.array([-0.556, 0.209, 0.907]),
        np.array([-0.862, -0.243, 0.835]),
        np.array([-0.552, -0.262, 0.317]),
    ]
    for action in actions:
        state, _, _, _ = env.step(action)
    # check somes values from the final state
    assert state["observation"][2] == 0.07347989437569474
    assert state["achieved_goal"][1] == 0.06545984927027101
    assert state["desired_goal"][0] == 0.04850898672097903


def test_seed_slide():
    env = gym.make("PandaSlide-v1")
    env.seed(13795)
    env.reset()
    actions = [
        np.array([0.245, 0.786, 0.329]),
        np.array([-0.414, 0.343, -0.839]),
        np.array([0.549, 0.047, -0.857]),
        np.array([0.744, -0.507, 0.092]),
        np.array([-0.202, -0.939, -0.945]),
        np.array([-0.97, -0.616, 0.472]),
    ]
    for action in actions:
        state, _, _, _ = env.step(action)
    # check somes values from the final state
    assert state["observation"][2] == 0.02201876710050804
    assert state["achieved_goal"][1] == 0.02719424543671162
    assert state["desired_goal"][0] == 0.4518336277440088


def test_seed_pick_and_place():
    env = gym.make("PandaPickAndPlace-v1")
    env.seed(794512)
    env.reset()
    actions = [
        np.array([0.429, -0.287, 0.804, -0.592]),
        np.array([0.351, -0.136, 0.296, -0.223]),
        np.array([-0.187, 0.706, -0.988, 0.972]),
        np.array([-0.389, -0.249, 0.374, -0.389]),
        np.array([-0.191, -0.297, -0.739, 0.633]),
        np.array([0.093, 0.242, -0.11, -0.949]),
    ]
    for action in actions:
        state, _, _, _ = env.step(action)
    # check somes values from the final state
    assert state["observation"][2] == 0.09952820768754468
    assert state["achieved_goal"][1] == -0.04404695023965532
    assert state["desired_goal"][0] == -0.07605634245602978


def test_seed_stack():
    env = gym.make("PandaStack-v1")
    env.seed(657894)
    env.reset()
    actions = [
        np.array([-0.609, 0.73, -0.433, 0.76]),
        np.array([0.414, 0.327, 0.275, -0.196]),
        np.array([-0.3, 0.589, -0.712, 0.683]),
        np.array([0.772, 0.333, -0.537, -0.253]),
        np.array([0.784, -0.014, -0.997, -0.118]),
        np.array([-0.12, -0.958, -0.744, -0.98]),
    ]
    for action in actions:
        state, _, _, _ = env.step(action)
    # check somes values from the final state
    assert state["observation"][2] == 0.03127797734509673
    assert state["achieved_goal"][1] == -0.010945185098696963
    assert state["desired_goal"][0] == 0.05333860128652357
