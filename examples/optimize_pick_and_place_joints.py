"""Optimize hyperparameters for PandaPickAndPlace-v1

# Usage :
Can be run in parallel on many workers:
$ python examples/optimize_panda_object.py >> out.log 2>&1 &
"""

import gym
import numpy as np
import optuna
import panda_gym
from stable_baselines3 import HerReplayBuffer, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def objective(trial: optuna.Study):
    env = gym.make("PandaPickAndPlaceJoints-v2")

    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
    tau = trial.suggest_categorical("tau", [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    train_freq = trial.suggest_categorical("train_freq", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    arch_depth = trial.suggest_categorical("arch_depth", [2, 3, 4])
    arch_width = trial.suggest_categorical("arch_width", [32, 64, 128, 256])
    net_arch = [arch_width] * arch_depth
    n_sampled_goal = trial.suggest_categorical("n_sampled_goal", [1, 2, 3, 4, 5, 6])
    goal_selection_strategy = trial.suggest_categorical("goal_selection_strategy", ["future", "episode"])
    online_sampling = trial.suggest_categorical("online_sampling", [True, False])
    action_noise_cls = trial.suggest_categorical("action_noise_cls", ["Normal", "OrnsteinUhlenbeck", "None"])
    action_noise_cls = {"Normal": NormalActionNoise, "OrnsteinUhlenbeck": OrnsteinUhlenbeckActionNoise, "None": None}[
        action_noise_cls
    ]
    action_noise_sigma = trial.suggest_loguniform("action_noise_std", 1e-5, 1)
    if action_noise_cls is not None:
        action_noise = action_noise_cls(
            mean=np.zeros(env.action_space.shape), sigma=np.ones(env.action_space.shape) * action_noise_sigma
        )
    else:
        action_noise = None
    policy_delay = trial.suggest_categorical("policy_delay", [1, 2, 3, 5, 10])
    target_policy_noise = trial.suggest_loguniform("target_policy_noise", 1e-5, 1)
    target_noise_clip = trial.suggest_loguniform("target_noise_clip", 1e-5, 1)

    all_successes = []
    for _ in range(3):
        model = TD3(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            action_noise=action_noise,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=online_sampling,
            ),
            policy_kwargs=dict(net_arch=net_arch),
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
        )
        model.learn(1000000)

        # test
        successes = []
        for _ in range(100):
            obs = env.reset()
            done = False
            while not done:
                action = model.predict(obs)[0]
                obs, reward, done, info = env.step(action)
            successes.append(info.get("is_success", 0.0))
        all_successes.append(np.mean(successes))

    return np.median(all_successes)


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///optimize_panda_joints.db",
        study_name="PandaPickAndPlaceJoints",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10)
