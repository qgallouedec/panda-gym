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
    env = gym.make("PandaReach-v1")
    run_env(env)


def test_slide():
    env = gym.make("PandaSlide-v1")
    run_env(env)


def test_push():
    env = gym.make("PandaPush-v1")
    run_env(env)


def test_pickandplace():
    env = gym.make("PandaPickAndPlace-v1")
    run_env(env)


def test_stack():
    env = gym.make("PandaStack-v1")
    run_env(env)


def test_dense_reach():
    env = gym.make("PandaReachDense-v1")
    run_env(env)


def test_dense_slide():
    env = gym.make("PandaSlideDense-v1")
    run_env(env)


def test_dense_push():
    env = gym.make("PandaPushDense-v1")
    run_env(env)


def test_dense_pickandplace():
    env = gym.make("PandaPickAndPlaceDense-v1")
    run_env(env)


def test_dense_stack():
    env = gym.make("PandaStackDense-v1")
    run_env(env)
