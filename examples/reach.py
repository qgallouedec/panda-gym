import gym
import panda_gym

env = gym.make("PandaReach-v1", render=True)

obs = env.reset()
for _ in range(50):
    env.render()
    action = env.action_space.sample()
    env.step()
