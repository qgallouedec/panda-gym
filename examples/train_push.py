import gym
import panda_gym
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("PandaReach-v2")

model = DDPG(policy="MultiInputPolicy",env=env, verbose=1)
#     replay_buffer_class=HerReplayBuffer,
#     verbose=True,
#     gamma=0.99,
#     learning_rate=0.00042,
#     batch_size=256,
#     tau=0.001,
#     train_freq=1,
#     policy_kwargs=dict(net_arch=[256, 256]),
#     replay_buffer_kwargs=dict(n_sampled_goal=2),
# )
# # Train the agent
model.learn(total_timesteps=int(3e4))
# # Save the agent
# model.save("ddpg_push")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
# model = DDPG.load("ddpg_push", env=env)

# # Evaluate the agent
# # NOTE: If you use wrappers with your environment that modify rewards,
# #       this will be reflected here. To evaluate with original rewards,
# #       wrap environment in a "Monitor" wrapper before other wrappers.
# # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# # print(mean_reward)
# # Enjoy trained agent

# for _ in range(100):
#     obs = env.reset()
#     done = False
#     while not done:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
#         env.render()
#     print(reward)

