import gym
import panda_gym

for task in ['Reach', 'Slide', 'Push', 'PickAndPlace', 'Stack']:
    env = gym.make("Panda{}-v1".format(task), render=True)

    obs = env.reset()
    for _ in range(50):
        env.render()
        action = env.action_space.sample()
        env.step(action)
    
    env.close()
