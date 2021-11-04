.. _custom_robot:

Custom robot
============

Prerequisites
-------------

To create your own robot, you will need its URDF file.

Code
----

To define your own robot, you need to inherit from `PyBulletRobot`, and define at least 3 methods:

- `set_action(action)`: what the robot must do with the action.
- `get_obs()`: returns the observation.
- `reset()`: how the robot is reset.

For the purpose of the example, we consider here that our URDF file defines a very simple robot, consisting of two links and a single joint.  

.. code-block:: python

    import numpy as np
    from gym import spaces

    from panda_gym.envs.core import PyBulletRobot
    from panda_gym.pybullet import PyBullet


    class MyRobot(PyBulletRobot):
        """My robot"""

        def __init__(self, sim):
            action_dim = 1 # = number of joints; here, 1 joint, so dimension = 1
            action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
            super().__init__(
                sim,
                body_name="my_robot",  # choose the name you want
                file_name="my_robot.urdf",  # the path of the URDF file
                base_position=np.zeros(3),  # the position of the base
                action_space=action_space,
                joint_indices=np.array([0]),  # list of the indices, as defined in the URDF
                joint_forces=np.array([1.0]),  # force applied when robot is controled (Nm)
            )

        def set_action(self, action):
            self.control_joints(target_angles=action)

        def get_obs(self):
            return self.get_joint_angle(joint=0)

        def reset(self):
            # Sets the neutral position of the joint. Here, at angle=0
            neutral_angle = np.array([0.0])
            self.set_joint_angles(angles=neutral_angle)


Obviously, you have to adapt the example to your robot, especially concerning the number and indeces of the joints, as well as the force applied for the control.
You can also use other types of control, for example using inverse dynamics with the parent class function , and any 

You can also use other types of control, using all the methods of the parent class :py:class:`PyBulletRobot<panda_gym.envs.core.PyBulletRobot>` and the simulation instance :py:class:`PyBullet<panda_gym.pybullet.PyBullet>`. For example for inverse kinematics you can use the method :py:meth:`PyBulletRobot.inverse_kinematics<panda_gym.envs.core.PyBulletRobot.inverse_kinematics>`.

Test it
-------

The robot is ready. To see it move, execute the following code.

.. code-block:: python

    from panda_gym.pybullet import PyBullet

    sim = PyBullet(render=True)
    robot = MyRobot(sim)

    for _ in range(50):
        robot.set_action(np.array([1.0]))
        sim.step()
        sim.render()
