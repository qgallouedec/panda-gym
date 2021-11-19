import gym
import panda_gym
from stable_baselines3 import DDPG, HerReplayBuffer

env = gym.make("PandaPush-v2")

model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)

model.learn(total_timesteps=100000)
