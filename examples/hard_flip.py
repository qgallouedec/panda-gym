import gym
import panda_gym
import numpy as np

env = gym.make("PandaFlip-v2", render=True)

obs = env.reset()

# place above
for _ in range(15):
    action = np.zeros(4)
    desired_position = obs["observation"][7:10] + np.array([0.02, 0.0, 0.1])
    action[:3] = 5.0 * (desired_position - obs["observation"][0:3])
    obs, reward, done, info = env.step(action)
    env.render()

# descent
for _ in range(6):
    action = np.array([0.0, 0.0, -0.5, 0.1])
    env.step(action)
    env.render()

# grasp
action = np.array([0.0, 0.0, 0.0, -0.3])
env.step(action)
env.render()

# ascent (should flip the cube)
for _ in range(10):
    action = np.array([0.0, 0.0, 0.2, -0.0002])
    env.step(action)
    env.render()

# don't move
for _ in range(10):
    action = np.array([0.0, 0.0, 0.0, 0.0])
    env.step(action)
    env.render()

env.close()
