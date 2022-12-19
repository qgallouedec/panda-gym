.. _manual_control:

Manual control
==============

It is possible to manually control the robot, giving it deterministic actions, depending on the observations. For example, for the realization of the task Reach, here is a possibility for the realization of the task.

.. code-block:: python

    import gymnasium as gym
    import panda_gym

    env = gym.make("PandaReach-v3", render_mode="human")
    observation, info = env.reset()

    for _ in range(1000):
        current_position = observation["observation"][0:3]
        desired_position = observation["desired_goal"][0:3]
        action = 5.0 * (desired_position - current_position)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

The result is as follows.

.. image:: https://gallouedec.com/uploads/img/manual_reach.png
  :alt: Manual push rendering