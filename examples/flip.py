import gym
import panda_gym

env = gym.make("PandaFlip-v2", render=True)

obs = env.reset()
for _ in range(50):
    env.render()
    action = env.action_space.sample()
    env.step(action)

env.close()
