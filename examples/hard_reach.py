import gym
import panda_gym

env = gym.make("PandaPush-v2", render=True)
images = []

for _ in range(10):
    obs = env.reset()
    done = False
    images.append(env.render('rgb_array'))

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        images.append(env.render('rgb_array'))

env.close()

from numpngw import write_apng

write_apng('reach.png', images, delay=40)