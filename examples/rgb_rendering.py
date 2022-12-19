import gymnasium as gym
from numpngw import write_apng  # pip install numpngw

import panda_gym

env = gym.make("PandaStack-v3", render_mode="rgb_array")
images = []


observation, info = env.reset()
images.append(env.render())

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    images.append(env.render())

    if terminated or truncated:
        observation, info = env.reset()
        images.append(env.render())

env.close()

write_apng("stack.png", images, delay=40)
