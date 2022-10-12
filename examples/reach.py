import gymnasium as gym

import panda_gym

env = gym.make("PandaReach-v2", render=True)

obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
