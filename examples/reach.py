import gym
import panda_gym

env = gym.make("PandaReach-v2", render=True)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
