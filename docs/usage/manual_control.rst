.. _manual_control:

Manual control
==============

It is possible to manually control the robot, giving it deterministic actions, depending on the observations. For example, for the realization of the task Reach, here is a possibility for the realization of the task.

.. code-block:: python

    import gym
    import panda_gym

    env = gym.make("PandaReach-v2", render=True)
    obs = env.reset()
    done = False

    while not done:
        current_position = obs["observation"][0:3]
        desired_position = obs["desired_goal"][0:3]
        action = 5.0 * (desired_position - current_position)
        obs, reward, done, info = env.step(action)
        env.render()

    env.close()

The result is as follows.

.. image:: https://gallouedec.com/uploads/img/manual_reach.png
  :alt: Manual push rendering