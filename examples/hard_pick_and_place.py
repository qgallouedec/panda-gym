import gym
import panda_gym
import numpy as np

env = gym.make("PandaPickAndPlace-v2", render=True)

obs = env.reset()

# place above
for _ in range(15):
    action = np.zeros(4)
    desired_position = obs["achieved_goal"][0:3] + np.array([0.0, 0.0, 0.1])
    action[:3] = 5.0 * (desired_position - obs["observation"][0:3])
    obs, reward, done, info = env.step(action)
    env.render()

# open fingers
for _ in range(3):
    action = np.array([0.0, 0.0, 0.0, 0.2])
    obs, reward, done, info = env.step(action)
    env.render()

# descent
for _ in range(10):
    action = np.zeros(4)
    desired_position = obs["achieved_goal"][0:3] + np.array([0.0, 0.0, 0.0])
    action[:3] = 5.0 * (desired_position - obs["observation"][0:3])
    obs, reward, done, info = env.step(action)
    env.render()

# close fingers
for _ in range(3):
    action = np.array([0.0, 0.0, 0.0, -0.1])
    obs, reward, done, info = env.step(action)
    env.render()

# rise
for _ in range(15):
    action = np.array([0.0, 0.0, 0.3, -0.05])
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
