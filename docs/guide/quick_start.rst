.. _quick_start:

Quick Start
===========

Once ``panda-gym`` installed, you can start the "Reach" task by executing the following lines.

.. code-block:: python

    import gym
    import panda_gym

    env = gym.make('PandaReach-v2', render=True)

    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample() # random action
        obs, reward, done, info = env.step(action)
        env.render() # wait the right amount of time to make the rendering real-time
    

Obviously, since the chosen actions are random, you will not see any learning. To access the section dedicated to the learning of the tasks, refer to the section :ref:`Train with stable-baselines3<train_with_sb3>`.
