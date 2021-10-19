import gym
import numpy as np
import panda_gym

env = gym.make("PandaSlide-v2", render=True)
obs = env.reset()

# place behind
for _ in range(10):
    desired_position = obs["achieved_goal"][0:3] - np.array([0.1, 0.0, 0.0])
    action = 5.0 * (desired_position - obs["observation"][0:3])
    obs, reward, done, info = env.step(action)
    env.render()

# hit
for _ in range(10):
    action = np.array([0.3, 0.0, 0.0])
    env.step(action)
    env.render()

# wait
for _ in range(30):
    action = np.array([0.0, 0.0, 0.0])
    env.step(action)
    env.render()

env.close()
