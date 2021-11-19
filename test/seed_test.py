import gym
import panda_gym
import numpy as np


def test_seed_reach():
    env = gym.make("PandaReach-v2")
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
    expected_state = {
        "observation": np.array([-0.00562165, 0.13984261, 0.1026114, 0.57252472, 0.16387926, 0.35602099]),
        "achieved_goal": np.array([-0.00562165, 0.13984261, 0.1026114]),
        "desired_goal": np.array([0.0063933, -0.0785642, 0.26941446]),
    }
    assert np.allclose(state["observation"], expected_state["observation"])
    assert np.allclose(state["achieved_goal"], expected_state["achieved_goal"])
    assert np.allclose(state["desired_goal"], expected_state["desired_goal"])


def test_seed_push():
    env = gym.make("PandaPush-v2")
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
    expected_state = {
        "observation": np.array(
            [
                -2.42257369e-02,
                8.08462671e-03,
                7.34798944e-02,
                -3.25473688e-01,
                -1.05371741e-01,
                1.91825476e-01,
                -5.56523060e-02,
                6.54558515e-02,
                1.99899273e-02,
                -1.30230788e-10,
                -3.63555618e-06,
                -1.81331223e-05,
                3.16133907e-06,
                -1.74732558e-10,
                3.16147875e-06,
                8.54144872e-09,
                1.58067667e-04,
                -1.15699379e-10,
            ]
        ),
        "achieved_goal": np.array([-0.05565231, 0.06545585, 0.01998993]),
        "desired_goal": np.array([0.04850899, -0.04495698, 0.02]),
    }
    assert np.allclose(state["observation"], expected_state["observation"])
    assert np.allclose(state["achieved_goal"], expected_state["achieved_goal"])
    assert np.allclose(state["desired_goal"], expected_state["desired_goal"])


def test_seed_slide():
    env = gym.make("PandaSlide-v2")
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
    expected_state = {
        "observation": np.array(
            [
                -9.53984287e-04,
                -4.27160227e-02,
                2.20187671e-02,
                -1.05556234e00,
                -2.46236634e-01,
                -7.07535426e-01,
                6.16437932e-02,
                2.72341968e-02,
                1.49899685e-02,
                -1.75389061e-05,
                -1.17214020e-05,
                6.81697625e-04,
                8.41885028e-06,
                -9.79033996e-06,
                2.02787221e-06,
                6.37290778e-04,
                5.49748686e-04,
                -4.94256257e-06,
            ]
        ),
        "achieved_goal": np.array([0.06164379, 0.0272342, 0.01498997]),
        "desired_goal": np.array([0.45183363, 0.04100129, 0.03]),
    }
    assert np.allclose(state["observation"], expected_state["observation"])
    assert np.allclose(state["achieved_goal"], expected_state["achieved_goal"])
    assert np.allclose(state["desired_goal"], expected_state["desired_goal"])


def test_seed_pick_and_place():
    env = gym.make("PandaPickAndPlace-v2")
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
    expected_state = {
        "observation": np.array(
            [
                2.80928255e-02,
                8.27207566e-03,
                9.95282077e-02,
                -1.56790883e-02,
                1.45471370e-01,
                -1.80584228e-01,
                8.67016959e-03,
                8.22368084e-02,
                -4.40469502e-02,
                1.99894128e-02,
                4.48807556e-06,
                -3.66759335e-05,
                -2.32015619e-05,
                -6.61656071e-07,
                1.09536696e-05,
                -1.21293651e-09,
                7.80705244e-11,
                -6.06878665e-08,
                -9.66072195e-05,
            ]
        ),
        "achieved_goal": np.array([0.08223681, -0.04404695, 0.01998941]),
        "desired_goal": np.array([-0.07605634, 0.02172643, 0.20853403]),
    }
    assert np.allclose(state["observation"], expected_state["observation"])
    assert np.allclose(state["achieved_goal"], expected_state["achieved_goal"])
    assert np.allclose(state["desired_goal"], expected_state["desired_goal"])


def test_seed_stack():
    env = gym.make("PandaStack-v2")
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
    expected_state = {
        "observation": np.array(
            [
                5.40996189e-02,
                4.59357277e-02,
                3.12779773e-02,
                -3.42651042e-01,
                -1.13286196e00,
                -9.92534442e-01,
                1.14918990e-03,
                6.52296198e-02,
                -1.09451851e-02,
                1.99899066e-02,
                -1.55383507e-08,
                -4.68830546e-06,
                -1.86308753e-05,
                4.01648869e-06,
                2.50407832e-08,
                3.99006056e-06,
                4.18980251e-09,
                1.99501357e-04,
                -1.79175223e-06,
                9.90341818e-02,
                9.23175415e-02,
                1.99889895e-02,
                6.74546584e-06,
                -1.10591593e-05,
                -1.03259781e-05,
                9.45574685e-06,
                6.04001409e-06,
                4.39861731e-05,
                -3.00748088e-04,
                4.71467318e-04,
                -1.78927334e-06,
            ]
        ),
        "achieved_goal": np.array([0.06522962, -0.01094519, 0.01998991, 0.09903418, 0.09231754, 0.01998899]),
        "desired_goal": np.array([0.0533386, 0.04649341, 0.02, 0.0533386, 0.04649341, 0.06]),
    }
    assert np.allclose(state["observation"], expected_state["observation"])
    assert np.allclose(state["achieved_goal"], expected_state["achieved_goal"])
    assert np.allclose(state["desired_goal"], expected_state["desired_goal"])
