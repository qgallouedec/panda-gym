import gymnasium as gym
from stable_baselines3 import DDPG, HerReplayBuffer

import panda_gym
import imageio

env = gym.make("PandaMobilePickAndPlace-v3",render_mode = 'human')

model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer,
             replay_buffer_kwargs= {'n_sampled_goal':4},
             verbose=1,learning_starts=200, learning_rate=0.000005,train_freq=5)



def render(name = 'movie'):
    images = []

    observation, info = env.reset()
    images.append(env.render())
    print('rendering')
    for _ in range(150):
        action, _states = model.predict(observation, deterministic=False)
        observation, reward, terminated, truncated, info = env.step(action)
        images.append(env.render())

        if terminated or truncated:
            print('truncated:',truncated)
            print('terminated:',terminated)
            observation, info = env.reset()
            images.append(env.render())
            break



    imageio.mimsave('movie/'+name+'.gif', images)

model.learn(total_timesteps=800000)
render('lastbutnotleast')


# while True:
#     try:
#         name = input()
#         render(name=name)
#     except KeyboardInterrupt:
#         print('stop rendering')
#         env.close()
#         break
