.. _custom_task:

Custom task
===========

Prerequisites
-------------

To create your own robot, you will need its URDF file.

Code
----

To define your own task, you need to inherit from :py:class:`Task<panda_gym.envs.core.Task>`, and define the following 5 methods:

- ``reset()``: how the task is reset; you must define `self.goal` in this function
- ``get_obs()``: returns the observation
- ``get_achieved_goal()``: returns the achieved goal
- ``is_success(achieved_goal, desired_goal, info)``: returns whether the task is successful
- ``compute_reward(achieved_goal, desired_goal, info)``: returns the reward

For the purpose of the example, let's consider here a very simple task, consisting in moving a cube toward a target position. The goal position is sampled within a volume of 10 m x 10 m x 10 m. 

.. code-block:: python

    import numpy as np

    from panda_gym.envs.core import Task
    from panda_gym.utils import distance


    class MyTask(Task):
        def __init__(self, sim):
            super().__init__(sim)
            # create an cube
            self.sim.create_box(body_name="object", half_extents=np.array([1, 1, 1]), mass=1.0, position=np.array([0.0, 0.0, 0.0]))

        def reset(self):
            # randomly sample a goal position
            self.goal = np.random.uniform(-10, 10, 3)
            # reset the position of the object
            self.sim.set_base_pose("object", position=np.array([0.0, 0.0, 0.0]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))

        def get_obs(self):
            # the observation is the position of the object
            observation = self.sim.get_base_position("object")
            return observation

        def get_achieved_goal(self):
            # the achieved goal is the current position of the object
            achieved_goal = self.sim.get_base_position("object")
            return achieved_goal

        def is_success(self, achieved_goal, desired_goal, info={}):  # info is here for consistency 
            # compute the distance between the goal position and the current object position
            d = distance(achieved_goal, desired_goal)
            # return True if the distance is < 1.0, and False otherwise
            return np.array(d < 1.0, dtype=bool)

        def compute_reward(self, achieved_goal, desired_goal, info={}):  # info is here for consistency
            # for this example, reward = 1.0 if the task is successful, 0.0 otherwise
            return self.is_success(achieved_goal, desired_goal, info).astype(np.float32)
            


Obviously, you have to adapt the example to your task.

Test it
-------

The task is ready. To test it, execute the following code.

.. code-block:: python

    from panda_gym.pybullet import PyBullet

    sim = PyBullet(render_mode="human")
    task = MyTask(sim)

    task.reset()
    print(task.get_obs())
    print(task.get_achieved_goal())
    print(task.is_success(task.get_achieved_goal(), task.get_goal()))
    print(task.compute_reward(task.get_achieved_goal(), task.get_goal()))
