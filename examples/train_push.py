# WARNING, This file will not be functional until stable-baselines3 is compatible
# with gymnasium. See https://github.com/DLR-RM/stable-baselines3/pull/780 for more information.
import sys
import gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import DDPG, HerReplayBuffer

import panda_gym

env = gymnasium.make("PandaPush-v3")

model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)

model.learn(total_timesteps=100000)
