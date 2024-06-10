.. _environments:

Environments
============

``panda-gym`` includes:

- 1 robot: 
    - the Franka Emika Panda robot,
- 6 tasks:
    - **Reach**: the robot must place its end-effector at a target position,
    - **Push**: the robot has to push a cube to a target position,
    - **Slide**: the robot has to slide an object to a target position,
    - **Pick and place**: the robot has to pick up and place an object at a target position,
    - **Stack**: the robot has to stack two cubes at a target position,
    - **Flip**: the robot must flip the cube to a target orientation,
- 2 control modes:
    - **End-effector displacement control**: the action corresponds to the displacement of the end-effector.
    - **Joints control**: the action corresponds to the individual motion of each joint,
- 2 reward types:
    - **Sparse**: the environment return a reward if and only if the task is completed,
    - **Dense**: the closer the agent is to completing the task, the higher the reward.

By default, the reward is sparse and the control mode is the end-effector displacement.
The complete set of environments present in the package is presented in the following list.

Sparse reward, end-effector control (default setting)
-----------------------------------------------------

* ``PandaReach-v3``
* ``PandaPush-v3``
* ``PandaSlide-v3``
* ``PandaPickAndPlace-v3``
* ``PandaStack-v3``
* ``PandaFlip-v3``

Dense reward, end-effector control
----------------------------------

* ``PandaReachDense-v3``
* ``PandaPushDense-v3``
* ``PandaSlideDense-v3``
* ``PandaPickAndPlaceDense-v3``
* ``PandaStackDense-v3``
* ``PandaFlipDense-v3``

Sparse reward, joints control
-----------------------------

* ``PandaReachJoints-v3``
* ``PandaPushJoints-v3``
* ``PandaSlideJoints-v3``
* ``PandaPickAndPlaceJoints-v3``
* ``PandaStackJoints-v3``
* ``PandaFlipJoints-v3``

Dense reward, joints control
----------------------------

* ``PandaReachJointsDense-v3``
* ``PandaPushJointsDense-v3``
* ``PandaSlideJointsDense-v3``
* ``PandaPickAndPlaceJointsDense-v3``
* ``PandaStackJointsDense-v3``
* ``PandaFlipJointsDense-v3``
