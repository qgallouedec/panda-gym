.. _custom_env:

Custom environment
==================

A customized environment is the junction of a **task** and a **robot**. 
You can choose to :ref:`define your own task<custom_task>`, or use one of the tasks present in the package. Similarly, you can choose to :ref:`define your own robot<custom_robot>`, or use one of the robots present in the package.

Then, you have to inherit from the :py:class:`RobotTaskEnv<panda_gym.envs.core.RobotTaskEnv>` class, in the following way.

.. code-block:: python

    from panda_gym.envs.core import RobotTaskEnv
    from panda_gym.pybullet import PyBullet


    class MyRobotTaskEnv(RobotTaskEnv):
        """My robot-task environment."""

        def __init__(self, render_mode):
            sim = PyBullet(render_mode=render_mode)
            robot = MyRobot(sim)
            task = MyTask(sim)
            super().__init__(robot, task)

That's it.

Test it
-------

You can now test your environment by running the following code.

.. code-block:: python

    env = MyRobotTaskEnv(render_mode="human")

    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample() # random action
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

