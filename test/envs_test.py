import gymnasium as gym

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


def test_reach():
    env = gym.make("PandaReach-v3")
    run_env(env)


def test_slide():
    env = gym.make("PandaSlide-v3")
    run_env(env)


def test_push():
    env = gym.make("PandaPush-v3")
    run_env(env)


def test_pickandplace():
    env = gym.make("PandaPickAndPlace-v3")
    run_env(env)


def test_stack():
    env = gym.make("PandaStack-v3")
    run_env(env)


def test_flip():
    env = gym.make("PandaFlip-v3")
    run_env(env)


def test_dense_reach():
    env = gym.make("PandaReachDense-v3")
    run_env(env)


def test_dense_slide():
    env = gym.make("PandaSlideDense-v3")
    run_env(env)


def test_dense_push():
    env = gym.make("PandaPushDense-v3")
    run_env(env)


def test_dense_pickandplace():
    env = gym.make("PandaPickAndPlaceDense-v3")
    run_env(env)


def test_dense_stack():
    env = gym.make("PandaStackDense-v3")
    run_env(env)


def test_dense_flip():
    env = gym.make("PandaFlipDense-v3")
    run_env(env)


def test_reach_joints():
    env = gym.make("PandaReachJoints-v3")
    run_env(env)


def test_slide_joints():
    env = gym.make("PandaSlideJoints-v3")
    run_env(env)


def test_push_joints():
    env = gym.make("PandaPushJoints-v3")
    run_env(env)


def test_pickandplace_joints():
    env = gym.make("PandaPickAndPlaceJoints-v3")
    run_env(env)


def test_stack_joints():
    env = gym.make("PandaStackJoints-v3")
    run_env(env)


def test_flip_joints():
    env = gym.make("PandaFlipJoints-v3")
    run_env(env)


def test_dense_reach_joints():
    env = gym.make("PandaReachJointsDense-v3")
    run_env(env)


def test_dense_slide_joints():
    env = gym.make("PandaSlideDense-v3")
    run_env(env)


def test_dense_push_joints():
    env = gym.make("PandaPushJointsDense-v3")
    run_env(env)


def test_dense_pickandplace_joints():
    env = gym.make("PandaPickAndPlaceJointsDense-v3")
    run_env(env)


def test_dense_stack_joints():
    env = gym.make("PandaStackJointsDense-v3")
    run_env(env)


def test_dense_flip_joints():
    env = gym.make("PandaFlipJointsDense-v3")
    run_env(env)
