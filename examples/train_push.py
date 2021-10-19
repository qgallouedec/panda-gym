"""Optimize hyperparameters for PandaPickAndPlace-v1

# Usage :
Can be run in parallel on many workers:
$ python examples/optimize_panda_object.py >> out.log 2>&1 &
"""

import gym
import panda_gym
from stable_baselines3 import DDPG, HerReplayBuffer

env = gym.make("PandaPushJoints-v2")

model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=True)
model.learn(300000)


env = gym.make("PandaPushJoints-v2", render=True)

for _ in range(100):
    obs = env.reset()
    done = False
    while not done:
        action = model.predict(obs)[0]
        obs, reward, done, info = env.step(action)
        env.render()
