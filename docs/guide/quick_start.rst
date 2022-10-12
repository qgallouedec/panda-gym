.. _quick_start:

Quick Start
===========

Once ``panda-gym`` installed, you can start the "Reach" task by executing the following lines.

.. code-block:: python

    import gymnasium as gym
    import panda_gym

    env = gym.make('PandaReach-v3', render=True)

    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample() # random action
        observation, reward, terminated, truncated, info = env.step(action)
        env.render() # wait the right amount of time to make the rendering real-time

        if terminated or truncated:
            observation, info = env.reset()
            env.render() # wait the right amount of time to make the rendering real-time
    

Obviously, since the chosen actions are random, you will not see any learning. To access the section dedicated to the learning of the tasks, refer to the section :ref:`Train with stable-baselines3<train_with_sb3>`.
