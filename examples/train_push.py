import gymnasium as gym
from stable_baselines3 import DDPG, HerReplayBuffer

import panda_gym

env = gym.make("PandaPush-v3")

model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)

model.learn(total_timesteps=100000)
