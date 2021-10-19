import gym
import panda_gym

env = gym.make("PandaPushJoints-v2", render=True)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
