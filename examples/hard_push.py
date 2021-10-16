import gym
import panda_gym
import numpy as np

env = gym.make("PandaPush-v2", render=True)

obs = env.reset()
for _ in range(10):
    desired_position = obs["achieved_goal"][0:3] - np.array([0.1, 0.0, 0.0])
    action = 5.0 * (desired_position - obs["observation"][0:3])
    obs, reward, done, info = env.step(action)
    env.render()

for _ in range(20):
    action = np.array([0.2, 0.0, 0.0])
    env.step(action)
    env.render()

env.close()
