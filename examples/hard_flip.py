import gym
import panda_gym
import numpy as np

env = gym.make("PandaFlip-v2", render=True)

obs = env.reset()
for _ in range(15):
    action = np.zeros(4)
    desired_position = obs["observation"][7:10] + np.array([0.0, 0.0, 0.1])
    action[:3] = 5.0 * (desired_position - obs["observation"][0:3])
    obs, reward, done, info = env.step(action)
    env.render()

for _ in range(6):
    action = np.array([0.0, 0.0, -0.5, 0.1])
    env.step(action)
    env.render()

action = np.array([0.0, 0.0, 0.0, -0.1])
env.step(action)
env.render()

for _ in range(10):
    action = np.array([0.0, 0.0, 0.2, -0.03])
    env.step(action)
    env.render()

for _ in range(100):
    action = np.array([0.0, 0.0, 0.0, -0.001])
    env.step(action)
    env.render()

env.close()
