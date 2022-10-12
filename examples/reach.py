import gymnasium as gym

import panda_gym

env = gym.make("PandaReach-v2", render=True)

observation, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()

env.close()
