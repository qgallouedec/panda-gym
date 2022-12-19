.. _custom_robot:

Custom robot
============

Prerequisites
-------------

To create your own robot, you will need a URDF file describing the robot.

Code
----

To define your own robot, you need to define the following methods and attributes:

- the joint indices,
- the joint forces,
- the ``set_action(action)`` method,
- the ``get_obs()`` method and
- the ``reset()`` method.

For the purpose of the example, let's use a very simple robot, whose URDF file is given below. It consists in two links and a single joint.

.. code-block:: xml

    <?xml version="1.0"?>
    <robot name="my_robot">
        <link name="link0"> ... </link>
        <link name="link1"> ... </link>
        <joint name="joint0" type="continuous">
            <parent link="link0" />
            <child link="link1" />
            ...
        </joint>
    </robot>


Joint indices
~~~~~~~~~~~~~

The first step is to identify the joints you want to be able to control with the agent. These joints will be identified by their index in the URDF file. Here, it is the index 0 joint that you want to control (it is also the only one in the URDF file). For the following, you will use ``joint_indices=np.array([0])``.

Joint forces
~~~~~~~~~~~~~

For each joint, you must define a maximum force. This data is usually found in the technical specifications of the robot, and sometimes in the URDF file (``<limit effort="1.0"/>`` for a maximum effort of 1.0 Nm). Here, let's consider that the maximum effort is 1.0 Nm. 
For the following, you will use ``joint_forces=np.array([1.0])``.

``set_action`` method
~~~~~~~~~~~~~~~~~~~~~

The ``set_action`` method specify what the robot must do with the action. In the example, the robot only uses the action as a target angle for its single joint. Thus:

.. code-block:: python

    def set_action(self, action):
        self.control_joints(target_angles=action)


``get_obs`` method
~~~~~~~~~~~~~~~~~~

The ``get_obs`` method returns the observation associated with the robot. In the example, the robot only returns the position of it single joint.

.. code-block:: python

    def get_obs(self):
        return self.get_joint_angle(joint=0)


``reset`` method
~~~~~~~~~~~~~~~~

The ``reset`` method specify how to reset the robot. In the example, the robot resets its single joint to an angle of 0.

.. code-block:: python

    def reset(self):
        neutral_angle = np.array([0.0])
        self.set_joint_angles(angles=neutral_angle)

Full code
~~~~~~~~~

You now have everything you need to define your custom robot. You only have to inherit the class :py:class:`PyBulletRobot<panda_gym.envs.core.PyBulletRobot>` in the following way.

.. code-block:: python

    import numpy as np
    from gymnasium import spaces

    from panda_gym.envs.core import PyBulletRobot


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
            neutral_angle = np.array([0.0])
            self.set_joint_angles(angles=neutral_angle)


Obviously, you have to adapt the example to your robot, especially concerning the number and the indices of the joints, as well as the forces applied for the control.

You can also use other types of control, using all the methods of the parent class :py:class:`PyBulletRobot<panda_gym.envs.core.PyBulletRobot>` and the simulation instance :py:class:`PyBullet<panda_gym.pybullet.PyBullet>`. For example for inverse kinematics you can use the method :py:meth:`PyBulletRobot.inverse_kinematics<panda_gym.envs.core.PyBulletRobot.inverse_kinematics>`.

Test it
-------

The robot is ready. To see it move, execute the following code.

.. code-block:: python

    from panda_gym.pybullet import PyBullet

    sim = PyBullet(render_mode="human")
    robot = MyRobot(sim)

    for _ in range(50):
        robot.set_action(np.array([1.0]))
        sim.step()

To see how to use this robot to define a new environment, see the :ref:`custom environment<custom_env>` section. 