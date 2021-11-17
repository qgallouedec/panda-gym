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

Sparce reward, end-effector control (default setting)
-----------------------------------------------------

* ``PandaReach-v2``
* ``PandaPush-v2``
* ``PandaSlide-v2``
* ``PandaPickAndPlace-v2``
* ``PandaStack-v2``
* ``PandaFlip-v2``

Dense reward, end-effector control
----------------------------------

* ``PandaReachDense-v2``
* ``PandaPushDense-v2``
* ``PandaSlideDense-v2``
* ``PandaPickAndPlaceDense-v2``
* ``PandaStackDense-v2``
* ``PandaFlipDense-v2``

Sparce reward, joints control
-----------------------------

* ``PandaReachJoints-v2``
* ``PandaPushJoints-v2``
* ``PandaSlideJoints-v2``
* ``PandaPickAndPlaceJoints-v2``
* ``PandaStackJoints-v2``
* ``PandaFlipJoints-v2``

Dense reward, joints control
----------------------------

* ``PandaReachJointsDense-v2``
* ``PandaPushJointsDense-v2``
* ``PandaSlideJointsDense-v2``
* ``PandaPickAndPlaceJointsDense-v2``
* ``PandaStackJointsDense-v2``
* ``PandaFlipJointsDense-v2``
