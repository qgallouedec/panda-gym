import gym
import panda_gym


def run_env(env):
    """Tests running panda gym envs."""
    done = False
    env.reset()
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
    env.close()


def test_reach():
    env = gym.make("PandaReach-v2")
    run_env(env)


def test_slide():
    env = gym.make("PandaSlide-v2")
    run_env(env)


def test_push():
    env = gym.make("PandaPush-v2")
    run_env(env)


def test_pickandplace():
    env = gym.make("PandaPickAndPlace-v2")
    run_env(env)


def test_stack():
    env = gym.make("PandaStack-v2")
    run_env(env)


def test_flip():
    env = gym.make("PandaFlip-v2")
    run_env(env)


def test_dense_reach():
    env = gym.make("PandaReachDense-v2")
    run_env(env)


def test_dense_slide():
    env = gym.make("PandaSlideDense-v2")
    run_env(env)


def test_dense_push():
    env = gym.make("PandaPushDense-v2")
    run_env(env)


def test_dense_pickandplace():
    env = gym.make("PandaPickAndPlaceDense-v2")
    run_env(env)


def test_dense_stack():
    env = gym.make("PandaStackDense-v2")
    run_env(env)


def test_dense_flip():
    env = gym.make("PandaFlipDense-v2")
    run_env(env)


def test_reach_joints():
    env = gym.make("PandaReachJoints-v2")
    run_env(env)


def test_slide_joints():
    env = gym.make("PandaSlideJoints-v2")
    run_env(env)


def test_push_joints():
    env = gym.make("PandaPushJoints-v2")
    run_env(env)


def test_pickandplace_joints():
    env = gym.make("PandaPickAndPlaceJoints-v2")
    run_env(env)


def test_stack_joints():
    env = gym.make("PandaStackJoints-v2")
    run_env(env)


def test_flip_joints():
    env = gym.make("PandaFlipJoints-v2")
    run_env(env)


def test_dense_reach_joints():
    env = gym.make("PandaReachJointsDense-v2")
    run_env(env)


def test_dense_slide_joints():
    env = gym.make("PandaSlideDense-v2")
    run_env(env)


def test_dense_push_joints():
    env = gym.make("PandaPushJointsDense-v2")
    run_env(env)


def test_dense_pickandplace_joints():
    env = gym.make("PandaPickAndPlaceJointsDense-v2")
    run_env(env)


def test_dense_stack_joints():
    env = gym.make("PandaStackJointsDense-v2")
    run_env(env)


def test_dense_flip_joints():
    env = gym.make("PandaFlipJointsDense-v2")
    run_env(env)
